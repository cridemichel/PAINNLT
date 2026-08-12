#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import MDAnalysis as mda
import numpy as np

import analyze_conditional_noise as cn
import analyze_force_source_decomposition as fs


def load_dna_only_targets(topology: Path, trr: Path):
    times, forces, torques, signature = fs.mapped_generalized_forces(topology, trr)
    return np.asarray(times), np.asarray(forces), np.asarray(torques), signature


def selected_geometry_with_external_targets(dataset: Path, cutoff: float, raw_indices, target_f, target_t):
    wanted = {int(v): i for i, v in enumerate(raw_indices)}
    descriptors=[]; forces=[]; torques=[]; frame_ids=[]; copy_ids=[]; contacts_out=[]
    with dataset.open('rb') as fh:
        nframes = cn.I32.unpack(cn.read_exact(fh, cn.I32.size))[0]
        if nframes <= 0:
            raise ValueError('dataset contains no frames')
        first = cn.read_frame(fh)
        period = cn.detect_repeat_period(first.molecules)
        ncopies = len(first.molecules)//period
        if ncopies < 2 or period*ncopies != len(first.molecules):
            raise ValueError('invalid repeated-copy topology')
        ref_block = first.molecules[:period]
        ref_xyz = cn.unwrap_copy_geometry(ref_block, first.box)
        labels = cn.infer_residue_labels(ref_block)
        rigid_mask = np.asarray([m.nsites > 1 for m in ref_block], dtype=bool)
        signature_ref = [cn.molecule_signature(m) for m in first.molecules]
        if target_f.shape[1:] != (len(first.molecules), 3):
            raise ValueError(f'external force shape {target_f.shape} incompatible with {len(first.molecules)} residues')
        if target_t.shape != target_f.shape:
            raise ValueError('force/torque external target shape mismatch')

        def consume(frame, fi):
            if fi not in wanted:
                return
            ti = wanted[fi]
            if [cn.molecule_signature(m) for m in frame.molecules] != signature_ref:
                raise ValueError(f'frame {fi}: molecule/site topology changed')
            ext = cn.cross_copy_contact_flags(frame, period, cutoff)
            for ci in range(ncopies):
                lo=ci*period; hi=lo+period
                block=frame.molecules[lo:hi]
                xyz=cn.unwrap_copy_geometry(block, frame.box)
                r=cn.kabsch_row(xyz, ref_xyz)
                descriptors.append((xyz@r).reshape(-1).astype(np.float32))
                forces.append(target_f[ti,lo:hi,:]@r)
                torques.append(target_t[ti,lo:hi,:]@r)
                frame_ids.append(fi); copy_ids.append(ci); contacts_out.append(bool(ext[ci]))

        consume(first,0)
        for fi in range(1,nframes):
            frame=cn.read_frame(fh)
            consume(frame,fi)
        if fh.read(1):
            raise ValueError('unexpected trailing bytes after dataset')

    expected=len(raw_indices)*ncopies
    if len(descriptors)!=expected:
        raise RuntimeError(f'selected sample count mismatch: got {len(descriptors)}, expected {expected}')
    return {
        'descriptors':np.asarray(descriptors,np.float32),
        'forces':np.asarray(forces,np.float32),
        'torques':np.asarray(torques,np.float32),
        'frame_ids':np.asarray(frame_ids,np.int32),
        'copy_ids':np.asarray(copy_ids,np.int16),
        'contacts':np.asarray(contacts_out,bool),
        'period':period,'copies':ncopies,'frames_total':nframes,'labels':labels,
        'rigid_mask':rigid_mask,'sites_per_copy':int(ref_xyz.shape[0])}


def pair_report(name,pairs,data,force_rms,torque_rms):
    return cn.pair_metrics(name,pairs,data['descriptors'],data['forces'],data['torques'],data['labels'],data['rigid_mask'],force_rms,torque_rms)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='tel22_dataset.bin')
    ap.add_argument('--config',default='tel22_training_config.json')
    ap.add_argument('--raw-topology',default='md.gro')
    ap.add_argument('--raw-trr',default='md.trr')
    ap.add_argument('--dna-topology',required=True)
    ap.add_argument('--dna-rerun-trr',required=True)
    ap.add_argument('--same-copy-gap-frames',type=int,default=20)
    ap.add_argument('--seed',type=int,default=20260811)
    ap.add_argument('--output-json',required=True)
    ap.add_argument('--output-csv',required=True)
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text())
    cutoff=float(cfg['cutoff'])
    times,target_f,target_t,signature=load_dna_only_targets(Path(args.dna_topology),Path(args.dna_rerun_trr))
    raw_indices=fs.raw_time_to_frame_indices(Path(args.raw_topology),Path(args.raw_trr),times)
    data=selected_geometry_with_external_targets(Path(args.dataset),cutoff,raw_indices,target_f,target_t)

    expected_sig=[]
    # All residues in this TEL22 mapping are DNA; validate residue order/count against rerun mapping.
    for _copy in range(data['copies']):
        expected_sig.extend(data['labels'])
    rerun_labels=[x[0] for x in signature]
    if rerun_labels != expected_sig:
        raise RuntimeError('DNA-only rerun residue order does not match repeated TEL22 dataset topology')

    desc=data['descriptors']; forces=data['forces']; torques=data['torques']; rigid=data['rigid_mask']
    frame_ids=data['frame_ids']; copy_ids=data['copy_ids']; contacts=data['contacts']
    force_rms=cn.rms_components(forces)
    torque_rms=cn.rms_components(torques[:,rigid,:])
    all_idx=np.arange(len(desc),dtype=np.int64)
    isolated_idx=np.flatnonzero(~contacts)

    pair_sets={
      'nearest_different_copy_all':cn.nearest_different_copy(desc,copy_ids,all_idx),
      'nearest_different_copy_isolated':cn.nearest_different_copy(desc,copy_ids,isolated_idx) if len(isolated_idx)>=2 else np.empty((0,2),dtype=np.int64),
      'nearest_same_copy_gap':cn.nearest_same_copy_gap(desc,copy_ids,frame_ids,args.same_copy_gap_frames),
    }
    rng=np.random.default_rng(args.seed); reports={}; rows=[]
    for name,pairs in pair_sets.items():
        near,near_arr=pair_report(name,pairs,data,force_rms,torque_rms)
        if 'different_copy' in name:
            candidates=isolated_idx if name.endswith('isolated') else all_idx; mode='different_copy'
        else:
            candidates=all_idx; mode='same_copy_gap'
        random_pairs=cn.random_control_pairs(rng,len(pairs),candidates,copy_ids,frame_ids,mode,args.same_copy_gap_frames)
        rand,rand_arr=pair_report(name+'_random_control',random_pairs,data,force_rms,torque_rms)
        item={'nearest':near,'random_control':rand}
        if near.get('pairs',0) and rand.get('pairs',0):
            nf=near['force_half_pair_difference_mse_fraction_of_target_mse']; rf=rand['force_half_pair_difference_mse_fraction_of_target_mse']
            nt=near['torque_half_pair_difference_mse_fraction_of_target_mse']; rt=rand['torque_half_pair_difference_mse_fraction_of_target_mse']
            item['nearest_vs_random_force_half_mse_ratio']=float(nf/rf) if rf>0 else math.nan
            item['nearest_vs_random_torque_half_mse_ratio']=float(nt/rt) if np.isfinite(rt) and rt>0 else math.nan
        reports[name]=item
        for control,pp,aa in [('nearest',pairs,near_arr),('random',random_pairs,rand_arr)]:
            for k,(i,j) in enumerate(pp):
                rows.append({'pair_set':name,'control':control,'sample_i':int(i),'sample_j':int(j),
                  'frame_i':int(frame_ids[i]),'frame_j':int(frame_ids[j]),'copy_i':int(copy_ids[i]),'copy_j':int(copy_ids[j]),
                  'contact_i':int(contacts[i]),'contact_j':int(contacts[j]),'geometry_rmsd_nm':float(aa['geom'][k]),
                  'force_pair_rms':float(aa['force_pair'][k]),'torque_pair_rms':float(aa['torque_pair'][k])})

    report={
      'definition':{
        'geometry':'same CG copy descriptor/Kabsch alignment as analyze_conditional_noise.py',
        'target':'GROMACS DNA-only rerun mapped per residue: force=sum atomistic DNA-only forces; torque=sum (r-COM)xF; rotated by same Kabsch transform',
        'question':'Does removing water/K/Cl make instantaneous DNA forces substantially more predictable from the retained CG geometry?',
        'local_noise_proxy':'MSE(target_i-target_j)/(2*MSE(target)); interpret via nearest/random ratio, not as a rigorous lower bound.'},
      'inputs':{'sampled_frames':len(raw_indices),'times_ps':[float(x) for x in times],'raw_dataset_frame_indices':[int(x) for x in raw_indices],
                'cutoff_nm':cutoff,'same_copy_min_gap_frames':args.same_copy_gap_frames,'seed':args.seed},
      'counts':{'copy_samples':int(len(desc)),'isolated_copy_samples':int(len(isolated_idx)),'isolated_copy_fraction':float(len(isolated_idx)/len(desc)),
                'copies_per_frame':int(data['copies']),'residues_per_copy':int(data['period']),'sites_per_copy':int(data['sites_per_copy'])},
      'target_scale':{'dna_only_force_component_rms_kj_mol_nm':force_rms,'dna_only_torque_component_rms_kj_mol':torque_rms},
      'pair_analyses':reports,
      'interpretation_guardrails':['Compare primarily nearest/random ratios.','Ratios near 1 mean retained CG geometry barely reduces DNA-only target differences at the tested resolution.','Ratios well below 1 mean DNA-only targets are substantially more predictable after removing solvent/ions.','The isolated subset may be small; report its pair count before interpreting it.']}

    outj=Path(args.output_json); outc=Path(args.output_csv); outj.parent.mkdir(parents=True,exist_ok=True); outc.parent.mkdir(parents=True,exist_ok=True)
    outj.write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    fields=['pair_set','control','sample_i','sample_j','frame_i','frame_j','copy_i','copy_j','contact_i','contact_j','geometry_rmsd_nm','force_pair_rms','torque_pair_rms']
    with outc.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)

    print('======================================================')
    print(' TEL22 DNA-ONLY CONDITIONAL-NOISE DIAGNOSTIC')
    print('======================================================')
    print(f'frames={len(raw_indices)} copies/frame={data["copies"]} samples={len(desc)} isolated={len(isolated_idx)}')
    print(f'DNA-only RMS: force={force_rms:.3f} kJ/(mol nm), torque={torque_rms:.3f} kJ/mol')
    for name,item in reports.items():
        near=item['nearest']; rand=item['random_control']
        print(f'[{name}] pairs={near.get("pairs",0)} geomP50={near.get("geometry_rmsd_nm",{}).get("p50",math.nan):.4f} nm '
              f'F near/random={item.get("nearest_vs_random_force_half_mse_ratio",math.nan):.3f} '
              f'T near/random={item.get("nearest_vs_random_torque_half_mse_ratio",math.nan):.3f}')
    print(f'JSON: {outj}')
    print(f'CSV:  {outc}')

if __name__=='__main__':
    main()
