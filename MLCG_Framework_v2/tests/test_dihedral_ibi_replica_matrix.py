#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from dihedral_ibi_replica_matrix import build_plan, summarize_matrix  # noqa: E402


def _structure(value: float, offset: float = 0.0) -> dict:
    return {
        "mean_l1_by_kind": {"dihedral": value},
        "groups": {
            "dihedrals:a": {"kind": "dihedral", "distribution_l1": value + offset},
            "dihedrals:b": {"kind": "dihedral", "distribution_l1": value - offset},
        },
    }


class DihedralIbiReplicaMatrixTests(unittest.TestCase):
    def test_summary_uses_paired_seed_differences(self):
        plan = {
            "protocol": {"dt_ps": 0.0005},
            "replica_count": 3,
            "prior_labels": ["U0", "U1", "U2"],
            "seed_pairs": [
                {"replica": 1, "velocity_seed": 10, "thermostat_seed": 20},
                {"replica": 2, "velocity_seed": 11, "thermostat_seed": 21},
                {"replica": 3, "velocity_seed": 12, "thermostat_seed": 22},
            ],
            "priors": {"U0": "u0", "U1": "u1", "U2": "u2"},
        }
        reports = {}
        u0 = [0.50, 0.60, 0.55]
        u1 = [0.54, 0.59, 0.57]
        u2 = [0.48, 0.55, 0.52]
        for replica in (1, 2, 3):
            reports[("U0", replica)] = _structure(u0[replica - 1], 0.02)
            reports[("U1", replica)] = _structure(u1[replica - 1], 0.02)
            reports[("U2", replica)] = _structure(u2[replica - 1], 0.02)
        result = summarize_matrix(plan, reports)
        self.assertAlmostEqual(result["prior_statistics"]["U0"]["overall_dihedral_mean_l1"]["mean"], 0.55)
        delta = result["paired_differences"]["U2_minus_U0"]["overall"]
        self.assertEqual(delta["values"], [-0.020000000000000018, -0.04999999999999993, -0.030000000000000027])
        self.assertAlmostEqual(delta["mean"], -1.0 / 30.0)
        self.assertIn("lower_L1_in_all_3_seed_pairs", result["diagnostic_hint"])
        self.assertFalse(result["promotion_ready"])

    def test_plan_reuses_only_step38_diagonal_and_schedules_six_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("dataset.bin", "config.json", "rb.json", "settings.json"):
                (root / name).write_text("{}\n")
            priors = []
            samples = []
            for idx in range(3):
                pdir = root / f"u{idx}"
                pdir.mkdir()
                p = pdir / "cg_priors.json"
                p.write_text(json.dumps({
                    "bonds": [], "angles": [],
                    "dihedrals": [{
                        "type": "conservative_spline", "ibi_mode": "ibi",
                        "ibi_runtime_representation": "conservative_spline",
                    }],
                }) + "\n")
                priors.append(p)
                s = root / f"sample{idx}.npz"
                s.write_bytes(b"sample")
                samples.append(s)
            ibi = root / "ibi_report.json"
            ibi.write_text(json.dumps({
                "conservative_dihedrals_in_loop": True,
                "dihedral_runtime_representation": "conservative_spline",
                "dt_ps": 0.0005, "burn_in_steps": 500, "production_steps": 2000,
                "sample_interval": 10, "kT": 2.49, "neighbor_search": "link-cell",
                "velocity_seed": 380001, "thermostat_seed": 380101,
                "final_priors": str(priors[2]),
                "metrics": [
                    {"iteration": 1, "source_priors": str(priors[0]), "sample": str(samples[0])},
                    {"iteration": 2, "source_priors": str(priors[1]), "sample": str(samples[1])},
                ],
            }) + "\n")
            sampling = root / "final_sampling.json"
            sampling.write_text(json.dumps({
                "kind": "matched_final_ibi_sampling_protocol", "matched_to_ibi_loop": True,
                "source_priors": str(priors[2]), "sampled_iteration": 3,
                "velocity_seed": 380003, "thermostat_seed": 380103,
                "burn_in_steps": 500, "production_steps": 2000, "dt_ps": 0.0005,
                "kT": 2.49, "neighbor_search": "link-cell",
                "checkpoint_used": False, "ml_active": False,
            }) + "\n")
            plan = build_plan(
                ibi_report_path=ibi, final_sampling_report_path=sampling,
                final_sample_npz=samples[2], dataset=root / "dataset.bin",
                config=root / "config.json", rb_info=root / "rb.json",
                ibi_config=root / "settings.json", outdir=root / "out", replicas=3,
            )
            reused = {(r["prior"], r["replica"]) for r in plan["rows"] if r["reused_step38"]}
            self.assertEqual(reused, {("U0", 1), ("U1", 2), ("U2", 3)})
            self.assertEqual(plan["reused_step38_samples"], 3)
            self.assertEqual(plan["new_md_runs"], 6)
            self.assertEqual([x["velocity_seed"] for x in plan["seed_pairs"]], [380001, 380002, 380003])


if __name__ == "__main__":
    unittest.main()
