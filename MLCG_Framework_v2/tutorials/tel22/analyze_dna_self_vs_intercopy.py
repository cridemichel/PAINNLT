#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

import analyze_conditional_noise as cn
import analyze_force_source_decomposition as fs


def residue_repeat_period(signatures: Sequence[Tuple[str, int]]) -> int:
    n = len(signatures)
    for p in range(1, n + 1):
        if n % p:
            continue
        if list(signatures) == list(signatures[:p]) * (n // p):
            return p
    return n


def make_copy_index(topology: Path, output: Path, manifest: Path) -> None:
    u = mda.Universe(str(topology))
    dna_res = [r for r in u.residues if str(r.resname) in fs.DNA_RESNAMES]
    if not dna_res:
        raise RuntimeError("No DNA residues found")
    signatures = [(str(r.resname), int(len(r.atoms))) for r in dna_res]
    period = residue_repeat_period(signatures)
    if period == len(signatures):
        raise RuntimeError("Could not detect repeated TEL22 copies from DNA residue signatures")
    if len(signatures) % period:
        raise RuntimeError("DNA residue count is not divisible by detected repeat period")
    copies = len(signatures) // period
    if copies < 2:
        raise RuntimeError("Need at least two repeated DNA copies")

    blocks: List[np.ndarray] = []
    for ci in range(copies):
        block = dna_res[ci * period:(ci + 1) * period]
        sig = [(str(r.resname), int(len(r.atoms))) for r in block]
        if sig != signatures[:period]:
            raise RuntimeError(f"Copy {ci} residue signature differs from copy 0")
        idx = np.sort(np.concatenate([np.asarray(r.atoms.indices, dtype=np.int64) for r in block]))
        blocks.append(idx)

    dna_all = np.sort(np.concatenate(blocks))
    if len(np.unique(dna_all)) != len(dna_all):
        raise RuntimeError("Copy atom groups overlap")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        fs.write_ndx_group(fh, "FS_DNA_ALL", dna_all)
        for ci, idx in enumerate(blocks):
            fs.write_ndx_group(fh, f"FS_COPY_{ci:02d}", idx)

    info = {
        "topology": str(topology),
        "dna_residues": len(dna_res),
        "residues_per_copy": period,
        "copies": copies,
        "dna_atoms_total": int(len(dna_all)),
        "atoms_per_copy": [int(len(x)) for x in blocks],
        "residue_signature_per_copy": [[name, nat] for name, nat in signatures[:period]],
        "groups": {"FS_DNA_ALL": int(len(dna_all)), **{f"FS_COPY_{i:02d}": int(len(v)) for i, v in enumerate(blocks)}},
    }
    manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


def load_targets(topology: Path, trr: Path):
    times, forces, torques, signature = fs.mapped_generalized_forces(topology, trr)
    return np.asarray(times), np.asarray(forces), np.asarray(torques), signature


def load_self_targets(copy_dir: Path, manifest: Dict, reference_times: np.ndarray):
    copies = int(manifest["copies"])
    period = int(manifest["residues_per_copy"])
    all_f = []
    all_t = []
    reference_sig = None
    for ci in range(copies):
        tag = f"copy_{ci:02d}"
        times, f, t, sig = load_targets(copy_dir / f"{tag}.gro", copy_dir / f"{tag}_rerun.trr")
        if len(times) != len(reference_times) or not np.allclose(times, reference_times, atol=1e-4, rtol=0.0):
            raise RuntimeError(f"{tag}: rerun times do not match DNA-all times")
        if f.shape[1:] != (period, 3) or t.shape != f.shape:
            raise RuntimeError(f"{tag}: unexpected mapped target shape {f.shape}")
        if reference_sig is None:
            reference_sig = sig
        elif sig != reference_sig:
            raise RuntimeError(f"{tag}: residue signature differs from copy_00")
        all_f.append(f)
        all_t.append(t)
    # concatenate residue blocks in the original copy order
    self_f = np.concatenate(all_f, axis=1)
    self_t = np.concatenate(all_t, axis=1)
    return self_f, self_t, reference_sig


def geometry_with_targets(dataset: Path, raw_indices, targets: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    wanted = {int(v): i for i, v in enumerate(raw_indices)}
    descriptors=[]; frame_ids=[]; copy_ids=[]
    rotated = {name: {"forces": [], "torques": []} for name in targets}
    with dataset.open("rb") as fh:
        nframes = cn.I32.unpack(cn.read_exact(fh, cn.I32.size))[0]
        if nframes <= 0:
            raise ValueError("dataset contains no frames")
        first = cn.read_frame(fh)
        period = cn.detect_repeat_period(first.molecules)
        ncopies = len(first.molecules) // period
        if ncopies < 2 or period * ncopies != len(first.molecules):
            raise ValueError("invalid repeated-copy topology")
        ref_block = first.molecules[:period]
        ref_xyz = cn.unwrap_copy_geometry(ref_block, first.box)
        labels = cn.infer_residue_labels(ref_block)
        rigid_mask = np.asarray([m.nsites > 1 for m in ref_block], dtype=bool)
        signature_ref = [cn.molecule_signature(m) for m in first.molecules]
        for name,(f,t) in targets.items():
            if f.shape[1:] != (len(first.molecules),3) or t.shape != f.shape:
                raise ValueError(f"{name}: target shape {f.shape} incompatible with dataset")

        def consume(frame, fi):
            if fi not in wanted:
                return
            ti = wanted[fi]
            if [cn.molecule_signature(m) for m in frame.molecules] != signature_ref:
                raise ValueError(f"frame {fi}: molecule/site topology changed")
            for ci in range(ncopies):
                lo=ci*period; hi=lo+period
                block=frame.molecules[lo:hi]
                xyz=cn.unwrap_copy_geometry(block, frame.box)
                r=cn.kabsch_row(xyz, ref_xyz)
                descriptors.append((xyz@r).reshape(-1).astype(np.float32))
                frame_ids.append(fi); copy_ids.append(ci)
                for name,(f,t) in targets.items():
                    rotated[name]["forces"].append(f[ti,lo:hi,:]@r)
                    rotated[name]["torques"].append(t[ti,lo:hi,:]@r)

        consume(first,0)
        for fi in range(1,nframes):
            consume(cn.read_frame(fh),fi)
        if fh.read(1):
            raise ValueError("unexpected trailing bytes after dataset")

    expected = len(raw_indices) * ncopies
    if len(descriptors) != expected:
        raise RuntimeError(f"selected sample count mismatch: got {len(descriptors)}, expected {expected}")
    out = {
        "descriptors": np.asarray(descriptors,np.float32),
        "frame_ids": np.asarray(frame_ids,np.int32),
        "copy_ids": np.asarray(copy_ids,np.int16),
        "period": period,
        "copies": ncopies,
        "labels": labels,
        "rigid_mask": rigid_mask,
        "sites_per_copy": int(ref_xyz.shape[0]),
    }
    for name in targets:
        out[name] = {
            "forces": np.asarray(rotated[name]["forces"],np.float32),
            "torques": np.asarray(rotated[name]["torques"],np.float32),
        }
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa=np.asarray(a,dtype=np.float64).ravel(); bb=np.asarray(b,dtype=np.float64).ravel()
    den=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    return float(np.dot(aa,bb)/den) if den>0 else math.nan


def source_scale_report(data, source: str):
    f=data[source]["forces"]
    rigid=data["rigid_mask"]
    t=data[source]["torques"][:,rigid,:]
    return {"force_component_rms":cn.rms_components(f),"torque_component_rms":cn.rms_components(t)}


def pair_report(source: str, name: str, pairs: np.ndarray, data, force_rms: float, torque_rms: float):
    return cn.pair_metrics(name,pairs,data["descriptors"],data[source]["forces"],data[source]["torques"],data["labels"],data["rigid_mask"],force_rms,torque_rms)


def run_pair_analysis(source: str, pair_sets: Dict[str,np.ndarray], random_sets: Dict[str,np.ndarray], data):
    scale=source_scale_report(data,source)
    reports={}; arrays={}
    for name,pairs in pair_sets.items():
        near,near_arr=pair_report(source,name,pairs,data,scale["force_component_rms"],scale["torque_component_rms"])
        rand,rand_arr=pair_report(source,name+"_random_control",random_sets[name],data,scale["force_component_rms"],scale["torque_component_rms"])
        item={"nearest":near,"random_control":rand}
        if near.get("pairs",0) and rand.get("pairs",0):
            nf=near["force_half_pair_difference_mse_fraction_of_target_mse"]; rf=rand["force_half_pair_difference_mse_fraction_of_target_mse"]
            nt=near["torque_half_pair_difference_mse_fraction_of_target_mse"]; rt=rand["torque_half_pair_difference_mse_fraction_of_target_mse"]
            item["nearest_vs_random_force_half_mse_ratio"]=float(nf/rf) if rf>0 else math.nan
            item["nearest_vs_random_torque_half_mse_ratio"]=float(nt/rt) if np.isfinite(rt) and rt>0 else math.nan
        reports[name]=item
        arrays[name]=(near_arr,rand_arr)
    return scale,reports,arrays


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--make-index",action="store_true")
    ap.add_argument("--topology",default="md.gro")
    ap.add_argument("--index-output")
    ap.add_argument("--index-manifest")
    ap.add_argument("--dataset",default="tel22_dataset.bin")
    ap.add_argument("--config",default="tel22_training_config.json")
    ap.add_argument("--raw-topology",default="md.gro")
    ap.add_argument("--raw-trr",default="md.trr")
    ap.add_argument("--dna-all-topology")
    ap.add_argument("--dna-all-rerun-trr")
    ap.add_argument("--copy-dir")
    ap.add_argument("--copy-manifest")
    ap.add_argument("--same-copy-gap-frames",type=int,default=20)
    ap.add_argument("--seed",type=int,default=20260811)
    ap.add_argument("--output-json")
    ap.add_argument("--output-csv")
    args=ap.parse_args()

    if args.make_index:
        if not args.index_output or not args.index_manifest:
            ap.error("--make-index requires --index-output and --index-manifest")
        make_copy_index(Path(args.topology),Path(args.index_output),Path(args.index_manifest))
        return

    required=["dna_all_topology","dna_all_rerun_trr","copy_dir","copy_manifest","output_json","output_csv"]
    missing=[x for x in required if not getattr(args,x)]
    if missing:
        ap.error("missing analysis arguments: "+", ".join("--"+x.replace("_","-") for x in missing))

    cfg=json.loads(Path(args.config).read_text())
    cutoff=float(cfg["cutoff"])
    manifest=json.loads(Path(args.copy_manifest).read_text())
    times,all_f,all_t,all_sig=load_targets(Path(args.dna_all_topology),Path(args.dna_all_rerun_trr))
    self_f,self_t,self_sig=load_self_targets(Path(args.copy_dir),manifest,times)
    if all_f.shape != self_f.shape or all_t.shape != self_t.shape:
        raise RuntimeError(f"DNA-all {all_f.shape} and assembled self {self_f.shape} target shapes differ")
    period=int(manifest["residues_per_copy"]); copies=int(manifest["copies"])
    if len(all_sig) != period*copies or len(self_sig) != period:
        raise RuntimeError("rerun residue counts do not match copy manifest")
    expected_sig=self_sig*copies
    if all_sig != expected_sig:
        raise RuntimeError("DNA-all residue signature does not equal repeated single-copy signature")

    inter_f=all_f-self_f
    inter_t=all_t-self_t
    raw_indices=fs.raw_time_to_frame_indices(Path(args.raw_topology),Path(args.raw_trr),times)
    data=geometry_with_targets(Path(args.dataset),raw_indices,{
        "dna_all":(all_f,all_t),"self":(self_f,self_t),"intercopy":(inter_f,inter_t)})
    if data["copies"] != copies or data["period"] != period:
        raise RuntimeError("dataset repeated-copy topology does not match rerun manifest")

    desc=data["descriptors"]; copy_ids=data["copy_ids"]; frame_ids=data["frame_ids"]
    all_idx=np.arange(len(desc),dtype=np.int64)
    pair_sets={
        "nearest_different_copy":cn.nearest_different_copy(desc,copy_ids,all_idx),
        "nearest_same_copy_gap":cn.nearest_same_copy_gap(desc,copy_ids,frame_ids,args.same_copy_gap_frames),
    }
    rng=np.random.default_rng(args.seed)
    random_sets={
        "nearest_different_copy":cn.random_control_pairs(rng,len(pair_sets["nearest_different_copy"]),all_idx,copy_ids,frame_ids,"different_copy",args.same_copy_gap_frames),
        "nearest_same_copy_gap":cn.random_control_pairs(rng,len(pair_sets["nearest_same_copy_gap"]),all_idx,copy_ids,frame_ids,"same_copy_gap",args.same_copy_gap_frames),
    }

    pair_analyses={}; scales={}; array_cache={}
    for source in ("dna_all","self","intercopy"):
        scale,reports,arrays=run_pair_analysis(source,pair_sets,random_sets,data)
        scales[source]=scale; pair_analyses[source]=reports; array_cache[source]=arrays

    rigid=data["rigid_mask"]
    source_decomposition={
        "force":{
            "self_over_dna_all_rms":float(scales["self"]["force_component_rms"]/scales["dna_all"]["force_component_rms"]),
            "intercopy_over_dna_all_rms":float(scales["intercopy"]["force_component_rms"]/scales["dna_all"]["force_component_rms"]),
            "cosine_self_vs_dna_all":cosine(data["self"]["forces"],data["dna_all"]["forces"]),
            "cosine_intercopy_vs_dna_all":cosine(data["intercopy"]["forces"],data["dna_all"]["forces"]),
            "cosine_self_vs_intercopy":cosine(data["self"]["forces"],data["intercopy"]["forces"]),
        },
        "torque_dg_multisite":{
            "self_over_dna_all_rms":float(scales["self"]["torque_component_rms"]/scales["dna_all"]["torque_component_rms"]),
            "intercopy_over_dna_all_rms":float(scales["intercopy"]["torque_component_rms"]/scales["dna_all"]["torque_component_rms"]),
            "cosine_self_vs_dna_all":cosine(data["self"]["torques"][:,rigid,:],data["dna_all"]["torques"][:,rigid,:]),
            "cosine_intercopy_vs_dna_all":cosine(data["intercopy"]["torques"][:,rigid,:],data["dna_all"]["torques"][:,rigid,:]),
            "cosine_self_vs_intercopy":cosine(data["self"]["torques"][:,rigid,:],data["intercopy"]["torques"][:,rigid,:]),
        }
    }

    rows=[]
    for source in ("dna_all","self","intercopy"):
        for name,pairs in pair_sets.items():
            for control,pp,arr in [("nearest",pairs,array_cache[source][name][0]),("random",random_sets[name],array_cache[source][name][1])]:
                for k,(i,j) in enumerate(pp):
                    rows.append({"source":source,"pair_set":name,"control":control,"sample_i":int(i),"sample_j":int(j),
                                 "frame_i":int(frame_ids[i]),"frame_j":int(frame_ids[j]),"copy_i":int(copy_ids[i]),"copy_j":int(copy_ids[j]),
                                 "geometry_rmsd_nm":float(arr["geom"][k]),"force_pair_rms":float(arr["force_pair"][k]),
                                 "torque_pair_rms":float(arr["torque_pair"][k])})

    report={
        "definition":{
            "dna_all":"DNA-only rerun containing all repeated TEL22 copies",
            "self":"for each TEL22 copy, a separate single-copy rerun in the original box/PME settings; assembled back in original copy order",
            "intercopy":"DNA-all generalized force/torque minus the corresponding single-copy self rerun",
            "geometry":"retained CG geometry of one TEL22 copy, centered and Kabsch-aligned exactly as in prior conditional-noise diagnostics",
            "question":"After removing solvent/ions and all other TEL22 copies, are instantaneous intramolecular DNA forces predictable from the retained CG mapping?",
            "guardrail":"Single-copy PME includes the self/periodic contribution of that copy in the original box. Intercopy is defined diagnostically by DNA-all minus self; do not use it as a production force target without a separate statistical-mechanical derivation."},
        "inputs":{"sampled_frames":len(raw_indices),"times_ps":[float(x) for x in times],"raw_dataset_frame_indices":[int(x) for x in raw_indices],
                  "cutoff_nm":cutoff,"same_copy_min_gap_frames":args.same_copy_gap_frames,"seed":args.seed},
        "counts":{"copy_samples":int(len(desc)),"copies_per_frame":copies,"residues_per_copy":period,"sites_per_copy":int(data["sites_per_copy"])},
        "target_scale":scales,
        "source_decomposition":source_decomposition,
        "pair_analyses":pair_analyses,
        "interpretation_guardrails":[
            "Use the self nearest/random ratios as the primary diagnostic: self has no water, ions, or other TEL22 copies.",
            "A self ratio near 1 means retained CG geometry barely reduces differences in instantaneous intramolecular DNA forces at the tested resolution.",
            "A self ratio well below 1 means removing inter-copy interactions restores substantial predictability.",
            "Compare DNA-all and intercopy on the exact same geometric pair sets to quantify the concentration/inter-copy confounder.",
            "These pair metrics are diagnostics, not rigorous conditional-variance lower bounds unless geometry separation tends to zero and hidden variables are independently sampled."],
    }

    outj=Path(args.output_json); outc=Path(args.output_csv); outj.parent.mkdir(parents=True,exist_ok=True); outc.parent.mkdir(parents=True,exist_ok=True)
    outj.write_text(json.dumps(report,indent=2,allow_nan=True)+"\n")
    fields=["source","pair_set","control","sample_i","sample_j","frame_i","frame_j","copy_i","copy_j","geometry_rmsd_nm","force_pair_rms","torque_pair_rms"]
    with outc.open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)

    print("======================================================")
    print(" TEL22 DNA SELF vs INTER-COPY DIAGNOSTIC")
    print("======================================================")
    print(f"frames={len(raw_indices)} copies/frame={copies} samples={len(desc)}")
    for source in ("dna_all","self","intercopy"):
        s=scales[source]
        print(f"[{source}] RMS F={s['force_component_rms']:.3f} kJ/(mol nm), T={s['torque_component_rms']:.3f} kJ/mol")
        for name,item in pair_analyses[source].items():
            print(f"  {name}: pairs={item['nearest'].get('pairs',0)} "
                  f"F near/random={item.get('nearest_vs_random_force_half_mse_ratio',math.nan):.3f} "
                  f"T near/random={item.get('nearest_vs_random_torque_half_mse_ratio',math.nan):.3f}")
    print(f"JSON: {outj}")
    print(f"CSV:  {outc}")


if __name__ == "__main__":
    main()
