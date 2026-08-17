#!/usr/bin/env python3
"""Matched replica matrix for conservative-in-loop dihedral IBI priors.

The number of IBI priors is inferred from the IBI report and the number of
paired replicas is supplied externally. Existing step-38 samples are reused
only when their exact prior/seed cell belongs to the configured matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_MD = ROOT / "simulation" / "run_cg_md.py"
VALIDATE_STRUCTURE = ROOT / "ibi" / "validate_runtime_structure.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _check_conservative_active_dihedrals(priors_path: Path) -> dict[str, Any]:
    data = json.loads(priors_path.read_text())
    active = []
    bad = []
    for idx, entry in enumerate(data.get("dihedrals", [])):
        if str(entry.get("ibi_mode", "")).lower() != "ibi":
            continue
        active.append(idx)
        if str(entry.get("type", "")).lower() != "conservative_spline":
            bad.append(idx)
        elif str(entry.get("ibi_runtime_representation", "")) != "conservative_spline":
            bad.append(idx)
    require(bool(active), f"No active IBI dihedrals in {priors_path}")
    return {
        "active_entries": len(active),
        "nonconservative_entries": bad,
        "pass": not bad,
    }


def build_plan(
    *,
    ibi_report_path: Path,
    final_sampling_report_path: Path,
    final_sample_npz: Path,
    dataset: Path,
    config: Path,
    rb_info: Path,
    ibi_config: Path,
    outdir: Path,
    replicas: int,
) -> dict[str, Any]:
    require(replicas >= 2, "Configured replica count must be at least 2")
    ibi = json.loads(ibi_report_path.read_text())
    sampling = json.loads(final_sampling_report_path.read_text())
    metrics = sorted(list(ibi.get("metrics", [])), key=lambda row: int(row["iteration"]))

    require(bool(ibi.get("conservative_dihedrals_in_loop")), "IBI report is not conservative-in-loop")
    require(ibi.get("dihedral_runtime_representation") == "conservative_spline", "Unexpected runtime representation")
    require(bool(metrics), "IBI report contains no completed sampling iterations")
    iterations = [int(row["iteration"]) for row in metrics]
    require(iterations == list(range(1, len(metrics) + 1)), "IBI metric iterations must be contiguous from 1")

    dt = float(ibi["dt_ps"])
    burn = int(ibi["burn_in_steps"])
    prod = int(ibi["production_steps"])
    interval = int(ibi["sample_interval"])
    kT = float(ibi["kT"])
    neighbor = str(ibi["neighbor_search"])
    base_v = int(ibi["velocity_seed"])
    base_t = int(ibi["thermostat_seed"])

    final_priors = _path(ibi["final_priors"])
    priors: dict[str, Path] = {}
    existing_samples: dict[tuple[str, int], Path] = {}
    for j, metric in enumerate(metrics):
        label = f"U{j}"
        priors[label] = _path(metric["source_priors"])
        replica = int(metric["iteration"])
        if 1 <= replica <= replicas:
            existing_samples[(label, replica)] = _path(metric["sample"])
    final_label = f"U{len(metrics)}"
    priors[final_label] = final_priors

    require(sampling.get("kind") == "matched_final_ibi_sampling_protocol", "Missing matched final-sampling metadata")
    require(bool(sampling.get("matched_to_ibi_loop", False)), "Final sample was not marked matched to the IBI loop")
    require(_path(sampling["source_priors"]) == final_priors, "Final sampling source priors mismatch")
    final_iteration = len(metrics) + 1
    require(int(sampling["sampled_iteration"]) == final_iteration, "Final sample iteration mismatch")
    require(int(sampling["velocity_seed"]) == base_v + final_iteration - 1, "Final sample velocity seed mismatch")
    require(int(sampling["thermostat_seed"]) == base_t + final_iteration - 1, "Final sample thermostat seed mismatch")
    require(int(sampling["burn_in_steps"]) == burn and int(sampling["production_steps"]) == prod, "Final burn/production protocol mismatch")
    require(math.isclose(float(sampling["dt_ps"]), dt, rel_tol=0.0, abs_tol=1.0e-15), "Final dt mismatch")
    require(math.isclose(float(sampling["kT"]), kT, rel_tol=0.0, abs_tol=1.0e-12), "Final kT mismatch")
    require(str(sampling["neighbor_search"]) == neighbor, "Final neighbor-search mismatch")
    require(sampling.get("checkpoint_used") is False and sampling.get("ml_active") is False, "Final sample must use no checkpoint and no ML")
    if final_iteration <= replicas:
        existing_samples[(final_label, final_iteration)] = final_sample_npz.resolve()

    for path in [*priors.values(), *existing_samples.values(), dataset, config, rb_info, ibi_config]:
        require(path.is_file(), f"Missing required artifact: {path}")
    prior_checks = {label: _check_conservative_active_dihedrals(path) for label, path in priors.items()}
    require(all(row["pass"] for row in prior_checks.values()), f"Non-conservative active dihedral prior found: {prior_checks}")

    labels = list(priors)
    rows = []
    for prior_label in labels:
        for replica in range(1, replicas + 1):
            velocity_seed = base_v + replica - 1
            thermostat_seed = base_t + replica - 1
            key = (prior_label, replica)
            reused_step38 = key in existing_samples
            if reused_step38:
                sample = existing_samples[key]
                run_dir = sample.parent
            else:
                run_dir = outdir / "runs" / prior_label / f"seedpair_{replica:02d}_v{velocity_seed}_t{thermostat_seed}"
                sample = run_dir / "trajectory.npz"
            structure_report = outdir / "structure" / prior_label / f"seedpair_{replica:02d}.json"
            rows.append({
                "prior": prior_label, "replica": replica,
                "velocity_seed": velocity_seed, "thermostat_seed": thermostat_seed,
                "priors": str(priors[prior_label]), "sample_npz": str(sample),
                "run_dir": str(run_dir), "structure_report": str(structure_report),
                "reused_step38": reused_step38, "needs_new_md": not reused_step38,
            })

    return {
        "schema_version": 2,
        "kind": "conservative_dihedral_ibi_matched_replica_matrix_plan",
        "ibi_report": str(ibi_report_path), "final_sampling_report": str(final_sampling_report_path),
        "dataset": str(dataset), "config": str(config), "rb_info": str(rb_info), "ibi_config": str(ibi_config),
        "protocol": {"dt_ps": dt, "burn_in_steps": burn, "production_steps": prod, "total_steps": burn + prod,
                     "sample_interval": interval, "kT": kT, "init_kT": kT, "neighbor_search": neighbor,
                     "checkpoint_used": False, "ml_active": False,
                     "starting_state": "target_dataset_initial_frame_plus_initialized_velocities"},
        "replica_count": replicas, "prior_labels": labels,
        "seed_pairs": [{"replica": r, "velocity_seed": base_v+r-1, "thermostat_seed": base_t+r-1} for r in range(1, replicas+1)],
        "priors": {label: str(path) for label, path in priors.items()}, "prior_checks": prior_checks, "rows": rows,
        "reused_step38_samples": sum(1 for row in rows if row["reused_step38"]),
        "new_md_runs": sum(1 for row in rows if row["needs_new_md"]),
    }


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    require(arr.size > 0 and np.all(np.isfinite(arr)), "Non-finite or empty statistic input")
    sd = _sample_sd(list(arr))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "sd": sd,
        "sem": float(sd / math.sqrt(arr.size)) if arr.size else math.nan,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(x) for x in arr],
    }


def _delta_hint(values: list[float], *, comparison: str) -> str:
    n = len(values)
    if all(x < 0.0 for x in values):
        return f"{comparison}_lower_L1_in_all_{n}_seed_pairs"
    if all(x > 0.0 for x in values):
        return f"{comparison}_higher_L1_in_all_{n}_seed_pairs"
    return f"{comparison}_mixed_sign_across_{n}_seed_pairs"


def summarize_matrix(plan: dict[str, Any], reports: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    labels = list(plan["prior_labels"])
    replicas = int(plan["replica_count"])
    expected = {(u, r) for u in labels for r in range(1, replicas + 1)}
    require(set(reports) == expected, "Replica matrix is incomplete")

    prior_stats: dict[str, Any] = {}
    group_names: set[str] | None = None
    for prior in labels:
        vals=[]; per_group: dict[str,list[float]]={}
        for replica in range(1, replicas+1):
            report=reports[(prior,replica)]
            val=float(report.get("mean_l1_by_kind",{}).get("dihedral",np.nan))
            require(np.isfinite(val), f"Missing dihedral mean L1 for {prior} replica {replica}")
            vals.append(val)
            groups={key:float(row["distribution_l1"]) for key,row in report.get("groups",{}).items() if str(row.get("kind"))=="dihedral"}
            require(bool(groups), f"Missing dihedral group metrics for {prior} replica {replica}")
            names=set(groups)
            if group_names is None: group_names=names
            require(names==group_names, f"Dihedral group set changed for {prior} replica {replica}")
            for key,value in groups.items(): per_group.setdefault(key,[]).append(value)
        prior_stats[prior]={"overall_dihedral_mean_l1":_stats(vals),"groups":{k:_stats(v) for k,v in sorted(per_group.items())}}

    paired={}
    comparisons=[]
    for i in range(len(labels)-1): comparisons.append((labels[i],labels[i+1],f"{labels[i+1]}_minus_{labels[i]}"))
    if len(labels)>1: comparisons.append((labels[0],labels[-1],f"{labels[-1]}_minus_{labels[0]}"))
    seen=set()
    for a,b,name in comparisons:
        if name in seen: continue
        seen.add(name); deltas=[]; per_seed=[]; group_deltas={k:[] for k in sorted(group_names or [])}
        for replica in range(1,replicas+1):
            va=float(reports[(a,replica)]["mean_l1_by_kind"]["dihedral"]); vb=float(reports[(b,replica)]["mean_l1_by_kind"]["dihedral"])
            d=vb-va; deltas.append(d); per_seed.append({"replica":replica,"delta_l1":d})
            ga=reports[(a,replica)]["groups"]; gb=reports[(b,replica)]["groups"]
            for key in group_deltas: group_deltas[key].append(float(gb[key]["distribution_l1"])-float(ga[key]["distribution_l1"]))
        paired[name]={"sign_convention":f"{b}-{a}; negative means lower/better L1 for {b}","overall":_stats(deltas),"per_seed":per_seed,
                      "groups":{k:_stats(v) for k,v in group_deltas.items()},"sign_hint":_delta_hint(deltas,comparison=name)}
    final_comp=f"{labels[-1]}_minus_{labels[0]}"
    final_deltas=paired[final_comp]["overall"]["values"]
    return {"schema_version":2,"framework":"MLCG_Framework_v2","kind":"conservative_dihedral_ibi_matched_replica_matrix",
            "test_only":True,"matrix_complete":True,"protocol":plan["protocol"],"replica_count":replicas,"seed_pairs":plan["seed_pairs"],
            "priors":plan["priors"],"prior_statistics":prior_stats,"paired_differences":paired,
            "diagnostic_hint":_delta_hint(final_deltas,comparison=f"{labels[-1]}_vs_{labels[0]}"),"infrastructure_pass":True,"promotion_ready":False,
            "notes":["All configured priors are sampled with the same configured velocity/thermostat seed pairs and dataset-start protocol.",
                     "Existing conservative-in-loop samples are reused only for exact prior/seed cells.",
                     "Replica count and any decision threshold are model-dependent; this report imposes no universal significance or promotion threshold.",
                     "Negative paired delta means the later prior has lower structural L1 for the same seed pair."]}


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def execute_matrix(
    *,
    plan: dict[str, Any],
    pypresso: Path,
    mode: str,
    outdir: Path,
    output: Path,
) -> dict[str, Any] | None:
    plan_path = outdir / "replica_matrix_plan.json"
    outdir.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    print("[STEP 39 -- MATCHED CONSERVATIVE DIHEDRAL IBI REPLICA MATRIX]")
    print(f"[MATRIX] priors={','.join(plan['prior_labels'])} replicas={plan['replica_count']} cells={len(plan['rows'])}")
    print(f"[MATRIX] reused_step38={plan['reused_step38_samples']} new_md={plan['new_md_runs']}")
    p = plan["protocol"]
    print(f"[PROTOCOL] burn-in={p['burn_in_steps']} production={p['production_steps']} dt={p['dt_ps']} ps")
    print(f"[NOTE] Same {plan['replica_count']} seed pairs are applied to every prior; negative paired delta means lower/better L1.")

    if mode == "dry-run":
        for row in plan["rows"]:
            action = "REUSE step38" if row["reused_step38"] else "NEW MD"
            print(
                f"[PLAN] {row['prior']} seedpair={row['replica']} "
                f"v={row['velocity_seed']} t={row['thermostat_seed']} {action}"
            )
        print(f"[PLAN] New integration work: {plan['new_md_runs'] * p['total_steps']} steps")
        print(f"[DONE] plan: {plan_path}")
        return None

    require(pypresso.is_file(), f"pypresso not found: {pypresso}")
    dataset = _path(plan["dataset"])
    config = _path(plan["config"])
    rb_info = _path(plan["rb_info"])
    ibi_config = _path(plan["ibi_config"])
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    sample_manifest = []

    for row in plan["rows"]:
        prior = row["prior"]
        replica = int(row["replica"])
        priors = _path(row["priors"])
        sample = _path(row["sample_npz"])
        run_dir = _path(row["run_dir"])
        structure_report = _path(row["structure_report"])

        if row["reused_step38"]:
            require(sample.is_file(), f"Step-38 sample disappeared: {sample}")
            action = "REUSE step38"
        elif sample.is_file() and mode == "resume":
            action = "REUSE step39"
        else:
            if sample.exists() and mode == "run":
                raise ValueError(f"Existing generated sample blocks --run: {sample}; use --resume or remove step-39 output")
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(pypresso), str(RUN_MD),
                "--config", str(config), "--priors", str(priors), "--rb_info", str(rb_info),
                "--dataset", str(dataset), "--dt", str(p["dt_ps"]),
                "--steps", str(p["total_steps"]), "--log_interval", str(p["sample_interval"]),
                "--sample_start_step", str(p["burn_in_steps"]), "--sample_npz", str(sample),
                "--kT", str(p["kT"]), "--init_kT", str(p["init_kT"]),
                "--velocity_seed", str(row["velocity_seed"]),
                "--thermostat_seed", str(row["thermostat_seed"]),
                "--neighbor_search", str(p["neighbor_search"]), "--no_log",
            ]
            print(
                f"[RUN] {prior} seedpair={replica} v={row['velocity_seed']} "
                f"t={row['thermostat_seed']}"
            , flush=True)
            _run_command(command, cwd=run_dir, log_path=run_dir / "run.log")
            require(sample.is_file(), f"MD did not produce sample: {sample}")
            action = "NEW MD"

        structure_report.parent.mkdir(parents=True, exist_ok=True)
        validate_cmd = [
            sys.executable, str(VALIDATE_STRUCTURE),
            "--dataset", str(dataset), "--priors", str(priors),
            "--sample-npz", str(sample), "--ibi-config", str(ibi_config),
            "--output", str(structure_report),
        ]
        _run_command(validate_cmd, cwd=structure_report.parent, log_path=structure_report.with_suffix(".log"))
        structure = json.loads(structure_report.read_text())
        l1 = float(structure["mean_l1_by_kind"]["dihedral"])
        reports[(prior, replica)] = structure
        sample_manifest.append({
            **row,
            "action": action,
            "sample_sha256": sha256_file(sample),
            "structure_report_sha256": sha256_file(structure_report),
            "dihedral_mean_l1": l1,
        })
        print(f"[RESULT] {prior} seedpair={replica} action={action} dihedral_mean_L1={l1:.6f}")

    report = summarize_matrix(plan, reports)
    report["plan"] = str(plan_path)
    report["samples"] = sample_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for prior in plan["prior_labels"]:
        stats = report["prior_statistics"][prior]["overall_dihedral_mean_l1"]
        print(f"[PRIOR] {prior} mean_L1={stats['mean']:.6f} SD={stats['sd']:.6f} values=" + ",".join(f"{x:.6f}" for x in stats["values"]))
    for name, item in report["paired_differences"].items():
        stats=item["overall"]; hint=item["sign_hint"]
        print(f"[PAIRED] {name} mean={stats['mean']:+.6f} SD={stats['sd']:.6f} deltas=" + ",".join(f"{x:+.6f}" for x in stats["values"]) + f" hint={hint}")
    print(f"[HINT] {report['diagnostic_hint']} (diagnostic only, n={plan['replica_count']} paired replicas)")
    print("[FINAL] infrastructure_pass=True promotion_ready=False")
    print(f"[DONE] report: {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "run", "resume"), required=True)
    parser.add_argument("--ibi-report", type=Path, required=True)
    parser.add_argument("--final-sampling-report", type=Path, required=True)
    parser.add_argument("--final-sample-npz", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rb-info", type=Path, required=True)
    parser.add_argument("--ibi-config", type=Path, required=True)
    parser.add_argument("--pypresso", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--model-config-provenance", type=Path, default=None)
    args = parser.parse_args()

    inputs = [
        args.ibi_report, args.final_sampling_report, args.final_sample_npz,
        args.dataset, args.config, args.rb_info, args.ibi_config,
    ]
    for path in inputs:
        require(path.expanduser().is_file(), f"Missing required artifact: {path}")

    plan = build_plan(
        ibi_report_path=args.ibi_report.resolve(),
        final_sampling_report_path=args.final_sampling_report.resolve(),
        final_sample_npz=args.final_sample_npz.resolve(),
        dataset=args.dataset.resolve(), config=args.config.resolve(), rb_info=args.rb_info.resolve(),
        ibi_config=args.ibi_config.resolve(), outdir=args.outdir.resolve(), replicas=args.replicas,
    )
    report = execute_matrix(
        plan=plan, pypresso=args.pypresso.resolve(), mode=args.mode,
        outdir=args.outdir.resolve(), output=args.output.resolve(),
    )
    if report is not None and args.model_config_provenance is not None:
        provenance=args.model_config_provenance.resolve()
        require(provenance.is_file(), f"Missing model config provenance: {provenance}")
        report["model_config_provenance"] = str(provenance)
        report["model_config_provenance_sha256"] = sha256_file(provenance)
        args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
