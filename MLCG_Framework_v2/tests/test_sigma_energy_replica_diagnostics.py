#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from sigma_energy_replica_diagnostics import (  # noqa: E402
    aggregate_sigma,
    bootstrap_fixed_effects_slope,
    collect_existing_complete_replicas,
    fixed_effects_loglog_fit,
    prefix_metrics,
    summarize_duration,
)


class SigmaEnergyReplicaDiagnosticTests(unittest.TestCase):
    @staticmethod
    def synthetic_rows(exponent=2.0):
        dts = [0.001, 0.0005, 0.00025, 0.000125]
        amplitudes = [0.5, 2.0, 7.0, 19.0]
        rows = []
        for replica, amp in enumerate(amplitudes):
            for dt in dts:
                rows.append({
                    "replica_index": replica,
                    "dt_ps": dt,
                    "sigma_E": amp * dt ** exponent,
                })
        return rows

    def test_fixed_effects_absorbs_replica_prefactor(self):
        fit = fixed_effects_loglog_fit(self.synthetic_rows())
        self.assertAlmostEqual(fit["exponent_p"], 2.0, places=12)
        self.assertGreater(fit["within_loglog_r2"], 0.999999999999)

    def test_aggregate_and_summary_recover_second_order(self):
        rows = self.synthetic_rows()
        aggregate = aggregate_sigma(rows)
        self.assertEqual(len(aggregate), 4)
        summary = summarize_duration(
            rows, bootstrap_samples=100, bootstrap_seed=1234,
            second_order_p_min=1.7, second_order_p_max=2.3, second_order_r2_min=0.95,
        )
        self.assertAlmostEqual(summary["fit_mean_sigma"]["exponent_p"], 2.0, places=12)
        self.assertAlmostEqual(summary["fit_geometric_mean_sigma"]["exponent_p"], 2.0, places=12)
        self.assertAlmostEqual(summary["fixed_effects_fit"]["exponent_p"], 2.0, places=12)

    def test_bootstrap_is_deterministic_and_contains_true_slope(self):
        rows = self.synthetic_rows()
        a = bootstrap_fixed_effects_slope(rows, samples=100, seed=99)
        b = bootstrap_fixed_effects_slope(rows, samples=100, seed=99)
        self.assertEqual(a, b)
        self.assertLessEqual(a["p025"], 2.0)
        self.assertGreaterEqual(a["p975"], 2.0)


    def test_two_replicas_keep_fixed_effects_but_disable_bootstrap(self):
        rows = [row for row in self.synthetic_rows() if row["replica_index"] < 2]
        summary = summarize_duration(
            rows, bootstrap_samples=1000, bootstrap_seed=1234,
            second_order_p_min=1.7, second_order_p_max=2.3, second_order_r2_min=0.95,
        )
        self.assertAlmostEqual(summary["fixed_effects_fit"]["exponent_p"], 2.0, places=12)
        boot = summary["fixed_effects_fit"]["bootstrap"]
        self.assertEqual(boot["status"], "disabled_too_few_replicas")
        self.assertEqual(boot["requested_samples"], 1000)
        self.assertEqual(boot["samples"], 0)
        self.assertIsNone(boot["p025"])
        self.assertIsNone(boot["p975"])

    def test_step28_supports_analysis_only_recovery(self):
        source = ROOT.joinpath("simulation", "sigma_energy_replica_diagnostics.py").read_text(encoding="utf-8")
        self.assertIn("--analyze-existing", source)
        self.assertIn("Existing-trace analysis requires at least two complete", source)
        self.assertIn("existing_complete_replicas", source)

    def test_analyze_existing_skips_incomplete_replica(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_checkpoint = root / "base.npz"
            base_checkpoint.write_bytes(b"base")
            dts = [0.001, 0.0005, 0.00025]
            for replica in (0, 1, 2):
                rep = root / "replicas" / f"replica_{replica:02d}"
                (rep / "nve").mkdir(parents=True)
                (rep / "nvt_checkpoint.npz").write_bytes(b"checkpoint")
                for dt in dts:
                    if replica == 2 and dt == 0.00025:
                        continue
                    (rep / "nve" / f"energy_dt_{dt:.9g}.csv").write_text("dummy\n", encoding="utf-8")

            args = SimpleNamespace(
                replicas=3,
                output_dir=root,
                seed_base=280000,
                base_checkpoint=base_checkpoint,
                replica_equilibration_dt=0.0005,
                kT=2.49,
                dts=dts,
                durations=[0.002],
            )

            def fake_validate(path, **kwargs):
                replica = int(path.parent.name.split("_")[-1])
                return {"checkpoint_sha256": f"sha-{replica}"}

            def fake_read(path):
                dt = float(path.stem.removeprefix("energy_dt_"))
                n = int(round(0.002 / dt))
                times = np.arange(n + 1, dtype=float) * dt
                energies = 10.0 + (dt ** 2) * np.arange(n + 1, dtype=float)
                return times, energies

            with patch("sigma_energy_replica_diagnostics.validate_replica_checkpoint", side_effect=fake_validate), \
                 patch("sigma_energy_replica_diagnostics.read_energy_csv", side_effect=fake_read):
                replicas, observations, skipped = collect_existing_complete_replicas(
                    args, expected_hashes={}, eq_steps=2000, max_duration=0.002
                )

            self.assertEqual([item["replica_index"] for item in replicas], [0, 1])
            self.assertEqual(len(observations), 2 * 3 * 1)
            self.assertEqual([item["replica_index"] for item in skipped], [2])
            self.assertIn("missing NVE energy trace", skipped[0]["reason"])

    def test_prefix_metrics_uses_exact_prefix_and_population_sigma(self):
        dt = 0.001
        times = np.arange(0, 1001, dtype=float) * dt
        energies = 10.0 + np.arange(times.size, dtype=float) * 1e-6
        metrics = prefix_metrics(times, energies, dt_ps=dt, duration_ps=0.25)
        self.assertEqual(metrics["samples"], 251)
        self.assertAlmostEqual(metrics["duration_ps"], 0.25, places=14)
        self.assertAlmostEqual(metrics["sigma_E"], float(np.std(energies[:251], ddof=0)), places=15)

    def test_step28_keeps_sigma_raw_and_reuses_prefixes(self):
        source = ROOT.joinpath("tutorials", "tel22_IBI", "28_diagnose_sigma_energy_replicas.sh").read_text(encoding="utf-8")
        self.assertIn("SIGMA_REPLICA_DURATIONS_PS", source)
        self.assertIn("SIGMA_REPLICA_DTS", source)
        self.assertIn("load_model_dependent_config step28", source)
        cfg = __import__("json").loads(ROOT.joinpath("tutorials", "tel22_IBI", "model_dependent_workflow_config.json").read_text())
        self.assertEqual(cfg["sections"]["step28"]["SIGMA_REPLICA_DURATIONS_PS"], [0.125, 0.25, 0.5, 1, 2])
        self.assertEqual(cfg["sections"]["step28"]["SIGMA_REPLICA_DTS"], [0.001, 0.0005, 0.00025, 0.000125])
        self.assertIn("shorter sigma(E) windows are prefixes", source)
        self.assertIn("raw std(E_tot), ddof=0, with no detrending", source)


if __name__ == "__main__":
    unittest.main()
