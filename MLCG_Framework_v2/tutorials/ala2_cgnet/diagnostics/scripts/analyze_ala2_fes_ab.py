#!/usr/bin/env python3
"""Compare matched Ala2 prior-only and prior+PaiNN free-energy surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KBT_KJ_MOL = 0.008314462618 * 300.0


def minimum_image(vector: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return vector
    return vector - box * np.round(vector / box)


def dihedral_angles(
    coordinates: np.ndarray, indices: tuple[int, int, int, int], box: np.ndarray | None = None
) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)[:, indices, :]
    if points.ndim != 3 or points.shape[1:] != (4, 3):
        raise ValueError("Coordinates must have shape (frames, particles, 3)")
    box_value = None if box is None else np.asarray(box, dtype=np.float64)
    b0 = minimum_image(points[:, 0] - points[:, 1], box_value)
    b1 = minimum_image(points[:, 2] - points[:, 1], box_value)
    b2 = minimum_image(points[:, 3] - points[:, 2], box_value)
    norm = np.linalg.norm(b1, axis=1)
    if np.any(norm <= 1.0e-12):
        raise ValueError("Degenerate central bond in dihedral calculation")
    b1_unit = b1 / norm[:, None]
    v = b0 - np.sum(b0 * b1_unit, axis=1)[:, None] * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=1)[:, None] * b1_unit
    if np.any(np.linalg.norm(v, axis=1) <= 1.0e-12) or np.any(
        np.linalg.norm(w, axis=1) <= 1.0e-12
    ):
        raise ValueError("Degenerate plane in dihedral calculation")
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1_unit, v) * w, axis=1)
    return np.arctan2(y, x)


def load_runtime_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "complete" not in data.files or int(np.asarray(data["complete"]).item()) != 1:
            raise ValueError(f"Incomplete structured trajectory: {path}")
        sites = np.asarray(data["sites"], dtype=np.float64)
        site_molecule = np.asarray(data["site_molecule"], dtype=int)
        site_index = np.asarray(data["site_index"], dtype=int)
        box = np.asarray(data["box"], dtype=np.float64)
    if sites.ndim != 3 or sites.shape[2] != 3:
        raise ValueError(f"Invalid sites array in {path}: {sites.shape}")
    order = []
    for molecule in range(5):
        matches = np.flatnonzero((site_molecule == molecule) & (site_index == 0))
        if len(matches) != 1:
            raise ValueError(f"Expected one site 0 for Ala2 molecule {molecule} in {path}")
        order.append(int(matches[0]))
    if box.shape != (3,) or np.any(box <= 0.0):
        raise ValueError(f"Invalid periodic box in {path}")
    return sites[:, order, :], box


def load_optional_cgnet(path: Path, units: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        coordinates = loaded
    else:
        try:
            if "coordinates_nm" in loaded.files:
                coordinates = loaded["coordinates_nm"]
                units = "nm"
            elif "coordinates" in loaded.files:
                coordinates = loaded["coordinates"]
            elif "sites" in loaded.files:
                coordinates = loaded["sites"]
            else:
                raise ValueError("CGnet NPZ needs coordinates_nm, coordinates, or sites")
        finally:
            loaded.close()
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (5, 3):
        raise ValueError(f"CGnet samples must have shape (frames, 5, 3); got {coordinates.shape}")
    if units == "angstrom":
        coordinates = coordinates * 0.1
    return coordinates


def torsions(coordinates: np.ndarray, box: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    return (
        dihedral_angles(coordinates, (0, 1, 2, 3), box),
        dihedral_angles(coordinates, (1, 2, 3, 4), box),
    )


def histogram(phi: np.ndarray, psi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _, _ = np.histogram2d(phi, psi, bins=(edges, edges))
    return counts.astype(np.float64)


def probability(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64) + pseudocount
    return values / np.sum(values)


def js_divergence(reference: np.ndarray, model: np.ndarray) -> float:
    midpoint = 0.5 * (reference + model)
    return float(
        0.5 * np.sum(reference * np.log(reference / midpoint))
        + 0.5 * np.sum(model * np.log(model / midpoint))
    )


def surface_metrics(
    reference_counts: np.ndarray,
    model_counts: np.ndarray,
    pseudocount: float,
    min_reference_count: int,
) -> dict[str, float | int]:
    reference = probability(reference_counts, pseudocount)
    model = probability(model_counts, pseudocount)
    mask = reference_counts >= min_reference_count
    if not np.any(mask):
        raise ValueError("No reference bins satisfy min-reference-count")
    reference_fes = -np.log(reference)
    model_fes = -np.log(model)
    optimal_shift = float(np.mean(reference_fes[mask] - model_fes[mask]))
    difference = model_fes[mask] + optimal_shift - reference_fes[mask]
    return {
        "js_divergence_nats": js_divergence(reference, model),
        "fes_mse_kbt2": float(np.mean(difference * difference)),
        "fes_rmse_kbt": float(np.sqrt(np.mean(difference * difference))),
        "optimal_model_shift_kbt": optimal_shift,
        "reference_support_bins": int(np.count_nonzero(mask)),
        "reference_mass_covered_by_sampled_bins": float(
            np.sum(reference[model_counts > 0.0])
        ),
    }


def bootstrap_delta(
    reference_counts: np.ndarray,
    baseline_counts: list[np.ndarray],
    candidate_counts: list[np.ndarray],
    pseudocount: float,
    samples: int,
) -> dict[str, float]:
    if len(baseline_counts) != len(candidate_counts) or len(baseline_counts) < 2:
        raise ValueError("Bootstrap requires at least two matched replica pairs")
    rng = np.random.default_rng(20260901)
    deltas = np.empty(samples, dtype=np.float64)
    count = len(baseline_counts)
    reference = probability(reference_counts, pseudocount)
    for sample in range(samples):
        chosen = rng.integers(0, count, size=count)
        baseline = probability(
            sum((baseline_counts[i] for i in chosen), np.zeros_like(reference_counts)),
            pseudocount,
        )
        candidate = probability(
            sum((candidate_counts[i] for i in chosen), np.zeros_like(reference_counts)),
            pseudocount,
        )
        deltas[sample] = js_divergence(reference, baseline) - js_divergence(
            reference, candidate
        )
    return {
        "mean_js_improvement_nats": float(np.mean(deltas)),
        "ci95_low_nats": float(np.percentile(deltas, 2.5)),
        "ci95_high_nats": float(np.percentile(deltas, 97.5)),
        "resamples": int(samples),
    }


def fes_for_plot(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    values = -KBT_KJ_MOL * np.log(probability(counts, pseudocount))
    return values - np.min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prior-samples", required=True, nargs="+", type=Path)
    parser.add_argument("--ml-samples", required=True, nargs="+", type=Path)
    parser.add_argument("--cgnet-samples", nargs="+", type=Path)
    parser.add_argument("--cgnet-units", choices=("angstrom", "nm"), default="angstrom")
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument("--min-reference-count", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()

    if len(args.prior_samples) != len(args.ml_samples) or len(args.prior_samples) < 2:
        raise ValueError("Provide at least two matched prior/ML trajectory pairs")
    if (
        args.bins < 8
        or args.pseudocount <= 0.0
        or args.min_reference_count < 1
        or args.bootstrap_samples < 1
    ):
        raise ValueError("Invalid histogram settings")

    with np.load(args.reference, allow_pickle=False) as data:
        reference_coordinates = np.asarray(data["coordinates_nm"], dtype=np.float64)
    reference_phi, reference_psi = torsions(reference_coordinates)
    edges = np.linspace(-np.pi, np.pi, args.bins + 1)
    reference_counts = histogram(reference_phi, reference_psi, edges)

    prior_counts = []
    ml_counts = []
    per_replica = []
    for replica, (prior_path, ml_path) in enumerate(zip(args.prior_samples, args.ml_samples)):
        prior_coordinates, prior_box = load_runtime_sample(prior_path)
        ml_coordinates, ml_box = load_runtime_sample(ml_path)
        prior_phi, prior_psi = torsions(prior_coordinates, prior_box)
        ml_phi, ml_psi = torsions(ml_coordinates, ml_box)
        prior_hist = histogram(prior_phi, prior_psi, edges)
        ml_hist = histogram(ml_phi, ml_psi, edges)
        prior_counts.append(prior_hist)
        ml_counts.append(ml_hist)
        prior_metrics = surface_metrics(
            reference_counts, prior_hist, args.pseudocount, args.min_reference_count
        )
        ml_metrics = surface_metrics(
            reference_counts, ml_hist, args.pseudocount, args.min_reference_count
        )
        per_replica.append(
            {
                "replica": replica,
                "prior_frames": int(len(prior_phi)),
                "ml_frames": int(len(ml_phi)),
                "prior": prior_metrics,
                "prior_plus_painn": ml_metrics,
                "js_improvement_nats": float(
                    prior_metrics["js_divergence_nats"] - ml_metrics["js_divergence_nats"]
                ),
            }
        )

    prior_total = sum(prior_counts, np.zeros_like(reference_counts))
    ml_total = sum(ml_counts, np.zeros_like(reference_counts))
    aggregate_prior = surface_metrics(
        reference_counts, prior_total, args.pseudocount, args.min_reference_count
    )
    aggregate_ml = surface_metrics(
        reference_counts, ml_total, args.pseudocount, args.min_reference_count
    )
    js_improvement = float(
        aggregate_prior["js_divergence_nats"] - aggregate_ml["js_divergence_nats"]
    )
    fes_mse_improvement = float(
        aggregate_prior["fes_mse_kbt2"] - aggregate_ml["fes_mse_kbt2"]
    )
    bootstrap = bootstrap_delta(
        reference_counts,
        prior_counts,
        ml_counts,
        args.pseudocount,
        args.bootstrap_samples,
    )
    if js_improvement > 0.0 and bootstrap["ci95_low_nats"] > 0.0:
        verdict = "painn_improves_fes"
    elif js_improvement < 0.0 and bootstrap["ci95_high_nats"] < 0.0:
        verdict = "painn_worsens_fes"
    else:
        verdict = "inconclusive_at_current_sampling"

    model_surfaces = [
        ("Reference atomistico", reference_counts),
        ("Solo prior", prior_total),
        ("Prior + PaiNN", ml_total),
    ]
    cgnet_metrics = None
    if args.cgnet_samples is not None:
        cgnet_counts_per_replica = []
        cgnet_per_replica = []
        for replica, sample_path in enumerate(args.cgnet_samples):
            cgnet_coordinates = load_optional_cgnet(sample_path, args.cgnet_units)
            cgnet_phi, cgnet_psi = torsions(cgnet_coordinates)
            counts = histogram(cgnet_phi, cgnet_psi, edges)
            metrics = surface_metrics(
                reference_counts, counts, args.pseudocount, args.min_reference_count
            )
            cgnet_counts_per_replica.append(counts)
            cgnet_per_replica.append(
                {
                    "replica": replica,
                    "frames": int(len(cgnet_phi)),
                    "source": str(sample_path.resolve()),
                    "metrics": metrics,
                }
            )
        cgnet_counts = sum(cgnet_counts_per_replica, np.zeros_like(reference_counts))
        aggregate_cgnet = surface_metrics(
            reference_counts, cgnet_counts, args.pseudocount, args.min_reference_count
        )
        cgnet_metrics = {
            "aggregate": aggregate_cgnet,
            "frames": int(sum(item["frames"] for item in cgnet_per_replica)),
            "replicas": len(cgnet_per_replica),
            "per_replica": cgnet_per_replica,
            "comparison": {
                "js_improvement_vs_prior_nats_positive_is_cgnet_better": float(
                    aggregate_prior["js_divergence_nats"]
                    - aggregate_cgnet["js_divergence_nats"]
                ),
                "js_improvement_vs_painn_nats_positive_is_cgnet_better": float(
                    aggregate_ml["js_divergence_nats"]
                    - aggregate_cgnet["js_divergence_nats"]
                ),
                "fes_mse_improvement_vs_prior_kbt2_positive_is_cgnet_better": float(
                    aggregate_prior["fes_mse_kbt2"] - aggregate_cgnet["fes_mse_kbt2"]
                ),
                "fes_mse_improvement_vs_painn_kbt2_positive_is_cgnet_better": float(
                    aggregate_ml["fes_mse_kbt2"] - aggregate_cgnet["fes_mse_kbt2"]
                ),
            },
        }
        if len(cgnet_counts_per_replica) == len(prior_counts):
            vs_prior = bootstrap_delta(
                reference_counts,
                prior_counts,
                cgnet_counts_per_replica,
                args.pseudocount,
                args.bootstrap_samples,
            )
            vs_painn = bootstrap_delta(
                reference_counts,
                ml_counts,
                cgnet_counts_per_replica,
                args.pseudocount,
                args.bootstrap_samples,
            )
            cgnet_metrics["paired_replica_bootstrap_vs_prior"] = vs_prior
            cgnet_metrics["paired_replica_bootstrap_vs_painn"] = vs_painn
            cgnet_vs_painn = cgnet_metrics["comparison"][
                "js_improvement_vs_painn_nats_positive_is_cgnet_better"
            ]
            if cgnet_vs_painn > 0.0 and vs_painn["ci95_low_nats"] > 0.0:
                cgnet_metrics["scientific_verdict_vs_painn"] = "cgnet_improves_fes_over_painn"
            elif cgnet_vs_painn < 0.0 and vs_painn["ci95_high_nats"] < 0.0:
                cgnet_metrics["scientific_verdict_vs_painn"] = "painn_improves_fes_over_cgnet"
            else:
                cgnet_metrics["scientific_verdict_vs_painn"] = (
                    "inconclusive_at_current_sampling"
                )
        else:
            cgnet_metrics["bootstrap_note"] = (
                "CGnet replica count differs from the matched prior/PaiNN count"
            )
        model_surfaces.append(("CGnet ufficiale", cgnet_counts))

    report = {
        "schema_version": 1,
        "status": "pass",
        "scientific_verdict": verdict,
        "reference_frames": int(len(reference_phi)),
        "matched_replicas": len(prior_counts),
        "histogram": {
            "bins_per_dimension": args.bins,
            "range_radians": [-float(np.pi), float(np.pi)],
            "pseudocount_per_bin": args.pseudocount,
            "min_reference_count_for_fes_mse": args.min_reference_count,
        },
        "aggregate": {
            "prior_only": aggregate_prior,
            "prior_plus_painn": aggregate_ml,
            "js_improvement_nats_positive_is_better": js_improvement,
            "fes_mse_improvement_kbt2_positive_is_better": fes_mse_improvement,
        },
        "paired_replica_bootstrap": bootstrap,
        "per_replica": per_replica,
        "cgnet_external": cgnet_metrics,
        "literature_comparison": {
            "metric_alignment": (
                "fes_mse_kbt2 uses the paper's shifted squared free-energy-surface "
                "difference, restricted to bins supported by the 10000-frame public subset"
            ),
            "paper_training_frames": 1000000,
            "paper_simulation_protocol": "100 independent simulations x 1000000 steps",
            "dynamics_note": (
                "The paper used overdamped Langevin dynamics; ESPResSo uses inertial Langevin. "
                "The equilibrium Boltzmann distribution is comparable, not the kinetics."
            ),
            "public_subset_frames": 10000,
            "force_mse_noise_floor_context_kcal2_mol2_angstrom2": 381.0,
            "warning": (
                "381 is the offset used to expose changes above the irreducible force-matching "
                "noise; it is not a published CGnet score"
            ),
        },
    }
    if args.training_report is not None:
        report["training_diagnostic"] = json.loads(args.training_report.read_text())

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(1, len(model_surfaces), figsize=(4.4 * len(model_surfaces), 4.0), constrained_layout=True)
    if len(model_surfaces) == 1:
        axes = [axes]
    image = None
    for axis, (title, counts) in zip(axes, model_surfaces):
        surface = np.minimum(fes_for_plot(counts, args.pseudocount), 20.0)
        image = axis.imshow(
            surface.T,
            origin="lower",
            extent=(-180.0, 180.0, -180.0, 180.0),
            vmin=0.0,
            vmax=20.0,
            cmap="viridis",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel(r"$\phi$ (gradi)")
        axis.set_ylabel(r"$\psi$ (gradi)")
    assert image is not None
    figure.colorbar(image, ax=axes, label=r"$\Delta F$ (kJ/mol)", shrink=0.82)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=180)
    plt.close(figure)

    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    if cgnet_metrics is not None:
        print(json.dumps({"official_cgnet": cgnet_metrics}, indent=2, sort_keys=True))
    print(f"[PASS] FES A/B analysis complete; scientific verdict: {verdict}")


if __name__ == "__main__":
    main()
