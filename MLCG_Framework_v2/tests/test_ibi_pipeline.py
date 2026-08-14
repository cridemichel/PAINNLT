#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "ibi"))

from build_dbi_priors import build_initial_dbi_priors, load_continuation_priors  # noqa: E402
from geometry_io import (  # noqa: E402
    pool_requested,
    read_sampled_distributions,
    read_target_distributions,
)
from framework_utils import resolve_referenced_path  # noqa: E402
from run_ibi_loop import run_ibi  # noqa: E402
from ibi_core import histogram_density  # noqa: E402
from convergence import summarize_convergence  # noqa: E402
from validate_ibi_priors import validate_priors  # noqa: E402


def write_dataset(path: Path, distances):
    distances = np.asarray(distances, dtype=float)
    with path.open("wb") as handle:
        handle.write(struct.pack("i", len(distances)))
        for r in distances:
            handle.write(struct.pack("i", 2))  # molecules
            handle.write(struct.pack("i", 2))  # physical sites
            handle.write(struct.pack("3f", 10.0, 10.0, 10.0))
            for mol, center in enumerate(([2.0, 2.0, 2.0], [2.0 + float(r), 2.0, 2.0])):
                handle.write(struct.pack("i", mol))
                handle.write(struct.pack("i", 1))
                handle.write(struct.pack("3f", *center))
                handle.write(struct.pack("3f", 0.0, 0.0, 0.0))
                handle.write(struct.pack("3f", 0.0, 0.0, 0.0))
                handle.write(struct.pack("i", mol))
                handle.write(struct.pack("3f", *center))


def write_sample(path: Path, distances):
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    com = np.zeros((n, 2, 3), dtype=float)
    com[:, 0, :] = [2.0, 2.0, 2.0]
    com[:, 1, :] = np.column_stack((2.0 + distances, np.full(n, 2.0), np.full(n, 2.0)))
    sites = com.copy()
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        complete=np.asarray(1, dtype=np.int8),
        steps=np.arange(n, dtype=np.int64),
        com=com,
        sites=sites,
        site_molecule=np.asarray([0, 1], dtype=np.int32),
        site_index=np.asarray([0, 0], dtype=np.int32),
        box=np.asarray([10.0, 10.0, 10.0]),
    )


class IBIPipelineTests(unittest.TestCase):
    def test_dataset_and_runtime_sampling_use_same_geometry_definition(self):
        priors = {
            "bonds": [{
                "name": "shared",
                "type": "ibi",
                "mol_i": 0,
                "mol_j": 1,
                "site_i": -1,
                "site_j": -1,
            }]
        }
        distances = np.linspace(0.85, 1.15, 31)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            sample = tmp / "sample.npz"
            write_dataset(dataset, distances)
            write_sample(sample, distances)
            target = read_target_distributions(dataset, priors)[0][0]
            simulated = read_sampled_distributions(sample, priors)[0][0]
            self.assertTrue(np.allclose(target, distances, atol=2.0e-7))
            self.assertTrue(np.allclose(simulated, distances, atol=1.0e-12))

    def test_initial_dbi_builder_converts_only_requested_entries(self):
        distances = 1.0 + 0.08 * np.sin(np.linspace(0.0, 8.0 * np.pi, 240, endpoint=False))
        seed = {
            "bonds": [
                {
                    "name": "backbone",
                    "type": "ibi",
                    "mol_i": 0,
                    "mol_j": 1,
                    "site_i": -1,
                    "site_j": -1,
                },
                {
                    "name": "fixed",
                    "type": "harmonic",
                    "mol_i": 0,
                    "mol_j": 1,
                    "site_i": 0,
                    "site_j": 0,
                    "k": 10.0,
                    "r0": 1.0,
                },
            ],
            "angles": [],
            "dihedrals": [],
        }
        config = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "histogram_smoothing_sigma": 0.5,
            "bond": {
                "hist_min": 0.75,
                "hist_max": 1.25,
                "hist_edges": 51,
                "table_min": 0.5,
                "table_max": 1.8,
                "table_points": 401,
                "left_guard": 0.65,
                "right_guard": 1.35,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            priors_path = tmp / "seed.json"
            config_path = tmp / "ibi.json"
            outdir = tmp / "generated"
            write_dataset(dataset, distances)
            priors_path.write_text(json.dumps(seed))
            config_path.write_text(json.dumps(config))

            result = build_initial_dbi_priors(
                dataset,
                priors_path,
                outdir,
                ibi_config=config_path,
            )
            converted = result["priors"]
            self.assertEqual(result["generated"], 1)
            self.assertEqual(converted["bonds"][0]["type"], "tabulated")
            self.assertEqual(converted["bonds"][0]["ibi_mode"], "ibi")
            self.assertEqual(converted["bonds"][1]["type"], "harmonic")
            table = resolve_referenced_path(
                converted["bonds"][0]["file"], result["output_priors"]
            )
            self.assertTrue(table.is_file())
            data = np.loadtxt(table)
            self.assertEqual(data.shape, (401, 3))
            self.assertTrue(np.isfinite(data).all())

    def test_initial_dbi_builder_copies_fixed_tabulated_references(self):
        distances = 1.0 + 0.05 * np.sin(np.linspace(0.0, 6.0 * np.pi, 200, endpoint=False))
        seed = {
            "bonds": [
                {
                    "name": "backbone", "type": "ibi",
                    "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
                },
                {
                    "name": "fixed", "type": "tabulated",
                    "mol_i": 0, "mol_j": 1, "site_i": 0, "site_j": 0,
                    "file": "fixed_input.dat", "min": 0.5, "max": 1.8,
                },
            ],
            "angles": [],
            "dihedrals": [],
        }
        config = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "bond": {
                "hist_min": 0.75, "hist_max": 1.25, "hist_edges": 41,
                "table_min": 0.5, "table_max": 1.8, "table_points": 201,
                "left_guard": 0.65, "right_guard": 1.35,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            priors_path = tmp / "seed.json"
            config_path = tmp / "ibi.json"
            fixed_input = tmp / "fixed_input.dat"
            outdir = tmp / "generated"
            write_dataset(dataset, distances)
            fixed_x = np.linspace(0.5, 1.8, 11)
            np.savetxt(fixed_input, np.column_stack((fixed_x, fixed_x * 0.0, fixed_x * 0.0)))
            priors_path.write_text(json.dumps(seed))
            config_path.write_text(json.dumps(config))

            result = build_initial_dbi_priors(
                dataset, priors_path, outdir, ibi_config=config_path
            )
            fixed_entry = result["priors"]["bonds"][1]
            copied = resolve_referenced_path(fixed_entry["file"], result["output_priors"])
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.parent.resolve(), outdir.resolve())
            self.assertNotEqual(copied.resolve(), fixed_input.resolve())
            self.assertTrue(np.allclose(np.loadtxt(copied), np.loadtxt(fixed_input)))

    def test_histogram_rejects_silent_range_truncation(self):
        with self.assertRaisesRegex(ValueError, "excludes 1/3 samples"):
            histogram_density(np.asarray([0.5, 1.0, 1.5]), np.linspace(0.75, 1.5, 6))

    def test_pool_rejects_mixed_ibi_and_dbi_mode(self):
        priors = {
            "bonds": [
                {"name": "same", "type": "ibi"},
                {"name": "same", "type": "dbi"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "mixes IBI and DBI"):
            pool_requested(priors, {0: [1.0], 1: [1.1]}, "bonds")

    def test_referenced_table_path_is_relative_to_priors_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            priors = tmp / "nested" / "priors.json"
            expected = priors.parent / "tables" / "bond.dat"
            self.assertEqual(
                resolve_referenced_path("tables/bond.dat", priors).resolve(),
                expected.resolve(),
            )

    def test_runtime_source_exposes_structured_site_sampling(self):
        source = (ROOT / "simulation" / "run_cg_md.py").read_text()
        self.assertIn('"--sample_npz"', source)
        self.assertIn('site_molecule=', source)
        self.assertIn('site_index=', source)
        self.assertIn('sample_site_keys = sorted(mol_vs_parts)', source)

    def test_runtime_sample_reader_rejects_incomplete_npz(self):
        priors = {
            "bonds": [{
                "name": "shared", "type": "ibi",
                "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
            }]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "partial.npz"
            com = np.zeros((1, 2, 3), dtype=float)
            np.savez_compressed(
                path, schema_version=np.asarray(1, dtype=np.int32),
                complete=np.asarray(0, dtype=np.int8), steps=np.asarray([0]),
                com=com, sites=com.copy(),
                site_molecule=np.asarray([0, 1], dtype=np.int32),
                site_index=np.asarray([0, 0], dtype=np.int32),
                box=np.asarray([10.0, 10.0, 10.0]),
            )
            with self.assertRaisesRegex(ValueError, "not marked complete"):
                read_sampled_distributions(path, priors)

    def test_preprocessing_and_runtime_diagnostic_wire_all_tabulated_kernels(self):
        builder = (ROOT / "preprocessing" / "build_cg_dataset.py").read_text()
        diagnostic = (ROOT / "simulation" / "diagnose_tabulated_prior_parity.py").read_text()
        for name in (
            "tabulated_distance_forces",
            "tabulated_angle_forces",
            "tabulated_dihedral_forces",
        ):
            self.assertIn(name, builder)
            self.assertIn(name, diagnostic)

    def test_runtime_diagnostic_reuses_single_espresso_system(self):
        diagnostic = (ROOT / "simulation" / "diagnose_tabulated_prior_parity.py").read_text()
        self.assertEqual(diagnostic.count("espressomd.System("), 1)
        self.assertIn("def reset_system(system, positions):", diagnostic)
        self.assertIn("evaluate_bond(system)", diagnostic)
        self.assertIn("evaluate_angle(system)", diagnostic)
        self.assertIn("evaluate_dihedral(system)", diagnostic)

    def test_convergence_summary_combines_parent_and_continuation_reports(self):
        def make_report(path, iteration, mean_value, source_dir):
            source_dir.mkdir(parents=True)
            (source_dir / "cg_priors.json").write_text("{}")
            report = {
                "metrics": [{
                    "iteration": iteration,
                    "source_priors": str(source_dir / "cg_priors.json"),
                    "groups": {
                        "bonds:test": {
                            "kind": "bond", "mode": "ibi", "distribution_l1": mean_value,
                        }
                    },
                }]
            }
            path.write_text(json.dumps(report))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            parent = tmp / "parent.json"
            continuation = tmp / "continuation.json"
            make_report(parent, 5, 0.25, tmp / "parent_source")
            make_report(continuation, 6, 0.20, tmp / "continuation_source")
            result = summarize_convergence(
                continuation,
                tmp / "summary.json",
                tmp / "best",
                previous_reports=[parent],
                overwrite=False,
            )
            self.assertEqual([row["sampling_iteration"] for row in result["iterations"]], [5, 6])
            self.assertEqual(result["best_sampling_iteration"], 6)
            self.assertAlmostEqual(result["best_mean_l1"], 0.20)
            self.assertTrue((tmp / "best" / "cg_priors.json").is_file())

    def test_continuation_loader_preserves_evaluated_table(self):
        distances = 1.0 + 0.05 * np.sin(np.linspace(0.0, 6.0 * np.pi, 200, endpoint=False))
        seed = {
            "bonds": [{
                "name": "backbone", "type": "ibi",
                "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
            }],
            "angles": [],
            "dihedrals": [],
        }
        config = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "bond": {
                "hist_min": 0.75, "hist_max": 1.25, "hist_edges": 41,
                "table_min": 0.5, "table_max": 1.8, "table_points": 201,
                "left_guard": 0.65, "right_guard": 1.35,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            seed_path = tmp / "seed.json"
            cfg_path = tmp / "ibi.json"
            generated = tmp / "generated"
            write_dataset(dataset, distances)
            seed_path.write_text(json.dumps(seed))
            cfg_path.write_text(json.dumps(config))
            initial = build_initial_dbi_priors(dataset, seed_path, generated, ibi_config=cfg_path)
            priors_path = Path(initial["output_priors"])
            entry = initial["priors"]["bonds"][0]
            table_path = resolve_referenced_path(entry["file"], priors_path)
            table = np.loadtxt(table_path)
            table[:, 1] += 0.12345
            np.savetxt(table_path, table)

            resumed = load_continuation_priors(dataset, priors_path, ibi_config=cfg_path)
            state = resumed["groups"]["bonds"]["backbone"]
            self.assertTrue(np.allclose(state["grid"], table[:, 0]))
            self.assertTrue(np.allclose(state["energy"], table[:, 1]))
            self.assertTrue(np.allclose(state["force"], table[:, 2]))
            self.assertEqual(state["mode"], "ibi")

    def test_continuation_loader_allows_conservative_spline_only_when_opted_in(self):
        distances = 1.0 + 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, 120, endpoint=False))
        seed = {
            "bonds": [{
                "name": "backbone", "type": "ibi",
                "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
            }],
            "angles": [],
            "dihedrals": [],
        }
        config = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "bond": {
                "hist_min": 0.75, "hist_max": 1.25, "hist_edges": 41,
                "table_min": 0.5, "table_max": 1.8, "table_points": 201,
                "left_guard": 0.65, "right_guard": 1.35,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            seed_path = tmp / "seed.json"
            cfg_path = tmp / "ibi.json"
            generated = tmp / "generated"
            write_dataset(dataset, distances)
            seed_path.write_text(json.dumps(seed))
            cfg_path.write_text(json.dumps(config))
            initial = build_initial_dbi_priors(dataset, seed_path, generated, ibi_config=cfg_path)
            tabulated_priors = Path(initial["output_priors"])
            converted = json.loads(tabulated_priors.read_text())
            entry = converted["bonds"][0]
            source_table = np.loadtxt(resolve_referenced_path(entry["file"], tabulated_priors))
            spline_path = generated / "bond_conservative_backbone.dat"
            # Conservative bond files store dU/dr, whereas the evaluated IBI table
            # stores the radial force -dU/dr.
            np.savetxt(spline_path, np.column_stack((source_table[:, 0], source_table[:, 1], -source_table[:, 2])))
            entry["type"] = "conservative_spline"
            entry["file"] = spline_path.name
            entry["spline_schema"] = "pchip_hermite_v1"
            conservative_priors = generated / "cg_priors_conservative.json"
            conservative_priors.write_text(json.dumps(converted) + "\n")

            with self.assertRaisesRegex(ValueError, "not fully tabulated"):
                load_continuation_priors(dataset, conservative_priors, ibi_config=cfg_path)

            diagnostic = load_continuation_priors(
                dataset, conservative_priors, ibi_config=cfg_path, allow_conservative_spline=True
            )
            state = diagnostic["groups"]["bonds"]["backbone"]
            self.assertEqual(state["representation"], "conservative_spline")
            self.assertTrue(np.allclose(state["grid"], source_table[:, 0]))
            self.assertTrue(np.allclose(state["energy"], source_table[:, 1]))
            self.assertTrue(np.allclose(state["force"], source_table[:, 2]))

    def test_resume_driver_uses_offset_and_skips_dbi_reinitialization(self):
        distances = 1.0 + 0.05 * np.sin(np.linspace(0.0, 6.0 * np.pi, 200, endpoint=False))
        seed = {
            "bonds": [{
                "name": "backbone", "type": "ibi",
                "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
            }],
            "angles": [],
            "dihedrals": [],
        }
        config_override = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "histogram_smoothing_sigma": 0.5,
            "update_smoothing_sigma": 0.5,
            "bond": {
                "hist_min": 0.75, "hist_max": 1.25, "hist_edges": 41,
                "table_min": 0.5, "table_max": 1.8, "table_points": 201,
                "left_guard": 0.65, "right_guard": 1.35,
            },
            "simulation": {"dt": 0.001, "steps": 100, "log_interval": 10, "burn_in_steps": 20},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            seed_path = tmp / "seed.json"
            cfg_path = tmp / "ibi.json"
            config_path = tmp / "config.json"
            rb_path = tmp / "rb.json"
            dbi_dir = tmp / "dbi"
            outdir = tmp / "continued"
            fake_pypresso = tmp / "fake_pypresso.py"
            write_dataset(dataset, distances)
            seed_path.write_text(json.dumps(seed))
            cfg_path.write_text(json.dumps(config_override))
            config_path.write_text("{}")
            rb_path.write_text("{}")
            initial = build_initial_dbi_priors(dataset, seed_path, dbi_dir, ibi_config=cfg_path)
            resume_path = Path(initial["output_priors"])
            fake_pypresso.write_text("""#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
args = sys.argv[2:]
def value(flag): return args[args.index(flag) + 1]
path = Path(value('--sample_npz'))
start = int(value('--sample_start_step'))
steps = int(value('--steps'))
interval = int(value('--log_interval'))
recorded = np.arange(start, steps + 1, interval, dtype=np.int64)
n = len(recorded)
r = 1.0 + 0.03 * np.sin(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False))
com = np.zeros((n, 2, 3), dtype=float)
com[:, 0, :] = [2.0, 2.0, 2.0]
com[:, 1, :] = np.column_stack((2.0 + r, np.full(n, 2.0), np.full(n, 2.0)))
path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(path, schema_version=np.asarray(1, dtype=np.int32), complete=np.asarray(1, dtype=np.int8),
                    steps=recorded, com=com, sites=com.copy(),
                    site_molecule=np.array([0, 1], dtype=np.int32), site_index=np.array([0, 0], dtype=np.int32),
                    box=np.array([10.0, 10.0, 10.0]))
""")
            fake_pypresso.chmod(0o755)

            report = run_ibi(
                dataset=dataset,
                resume_priors=resume_path,
                config=config_path,
                rb_info=rb_path,
                outdir=outdir,
                iterations=1,
                iteration_offset=5,
                pypresso=fake_pypresso,
                ibi_config=cfg_path,
                neighbor_search="verlet",
            )
            self.assertEqual(report["run_mode"], "continuation")
            self.assertEqual(report["iteration_offset"], 5)
            self.assertEqual(report["metrics"][0]["iteration"], 6)
            self.assertTrue((outdir / "resume_start" / "cg_priors.json").is_file())
            self.assertTrue((outdir / "sampling" / "iteration_006" / "metrics.json").is_file())
            self.assertFalse((outdir / "iteration_000").exists())

    def test_read_only_validation_preserves_priors_and_reports_independent_l1(self):
        distances = 1.0 + 0.05 * np.sin(np.linspace(0.0, 6.0 * np.pi, 200, endpoint=False))
        seed = {
            "bonds": [{
                "name": "backbone", "type": "ibi",
                "mol_i": 0, "mol_j": 1, "site_i": -1, "site_j": -1,
            }],
            "angles": [],
            "dihedrals": [],
        }
        config_override = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "histogram_smoothing_sigma": 0.5,
            "update_smoothing_sigma": 0.5,
            "bond": {
                "hist_min": 0.75, "hist_max": 1.25, "hist_edges": 41,
                "table_min": 0.5, "table_max": 1.8, "table_points": 201,
                "left_guard": 0.65, "right_guard": 1.35,
            },
            "simulation": {"dt": 0.001, "steps": 100, "log_interval": 10, "burn_in_steps": 20},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            seed_path = tmp / "seed.json"
            cfg_path = tmp / "ibi.json"
            config_path = tmp / "config.json"
            rb_path = tmp / "rb.json"
            dbi_dir = tmp / "dbi"
            validation_dir = tmp / "validation"
            fake_pypresso = tmp / "fake_pypresso.py"
            reference_summary = tmp / "summary.json"
            write_dataset(dataset, distances)
            seed_path.write_text(json.dumps(seed))
            cfg_path.write_text(json.dumps(config_override))
            config_path.write_text("{}")
            rb_path.write_text("{}")
            initial = build_initial_dbi_priors(dataset, seed_path, dbi_dir, ibi_config=cfg_path)
            priors_path = Path(initial["output_priors"])
            table_path = resolve_referenced_path(initial["priors"]["bonds"][0]["file"], priors_path)
            prior_before = priors_path.read_bytes()
            table_before = table_path.read_bytes()
            reference_summary.write_text(json.dumps({
                "best_sampling_iteration": 5,
                "iterations": [{
                    "sampling_iteration": 5, "mean_l1": 0.25, "max_l1": 0.25,
                    "mean_l1_by_kind": {"bond": 0.25},
                }],
            }))
            fake_pypresso.write_text("""#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
args = sys.argv[2:]
def value(flag): return args[args.index(flag) + 1]
path = Path(value('--sample_npz'))
start = int(value('--sample_start_step'))
steps = int(value('--steps'))
interval = int(value('--log_interval'))
recorded = np.arange(start, steps + 1, interval, dtype=np.int64)
n = len(recorded)
r = 1.0 + 0.03 * np.sin(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False))
com = np.zeros((n, 2, 3), dtype=float)
com[:, 0, :] = [2.0, 2.0, 2.0]
com[:, 1, :] = np.column_stack((2.0 + r, np.full(n, 2.0), np.full(n, 2.0)))
path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(path, schema_version=np.asarray(1, dtype=np.int32), complete=np.asarray(1, dtype=np.int8),
                    steps=recorded, com=com, sites=com.copy(),
                    site_molecule=np.array([0, 1], dtype=np.int32), site_index=np.array([0, 0], dtype=np.int32),
                    box=np.array([10.0, 10.0, 10.0]))
""")
            fake_pypresso.chmod(0o755)

            report = validate_priors(
                dataset=dataset, priors=priors_path, config=config_path, rb_info=rb_path,
                pypresso=fake_pypresso, outdir=validation_dir, ibi_config=cfg_path,
                reference_summary=reference_summary, velocity_seed=123, thermostat_seed=456,
            )
            self.assertTrue(report["source_priors_unchanged"])
            self.assertGreater(report["mean_l1"], 0.0)
            self.assertEqual(report["velocity_seed"], 123)
            self.assertEqual(report["thermostat_seed"], 456)
            self.assertIsNotNone(report["comparison_to_reference_best"])
            self.assertEqual(priors_path.read_bytes(), prior_before)
            self.assertEqual(table_path.read_bytes(), table_before)
            self.assertTrue((validation_dir / "validation_report.json").is_file())

    def test_read_only_validation_rejects_output_inside_source_prior_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            priors = tmp / "best" / "cg_priors.json"
            priors.parent.mkdir()
            priors.write_text(json.dumps({"bonds": [], "angles": [], "dihedrals": []}))
            for name in ("target.bin", "config.json", "rb.json", "pypresso"):
                (tmp / name).write_text("")
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                validate_priors(
                    dataset=tmp / "target.bin", priors=priors, config=tmp / "config.json",
                    rb_info=tmp / "rb.json", pypresso=tmp / "pypresso",
                    outdir=priors.parent / "validation",
                )

    def test_iterative_driver_writes_self_contained_final_priors(self):
        distances = 1.0 + 0.08 * np.sin(np.linspace(0.0, 8.0 * np.pi, 240, endpoint=False))
        seed = {
            "bonds": [
                {
                    "name": "backbone",
                    "type": "ibi",
                    "mol_i": 0,
                    "mol_j": 1,
                    "site_i": -1,
                    "site_j": -1,
                },
                {
                    "name": "fixed",
                    "type": "tabulated",
                    "mol_i": 0,
                    "mol_j": 1,
                    "site_i": 0,
                    "site_j": 0,
                    "file": "fixed_input.dat",
                    "min": 0.5,
                    "max": 1.8,
                },
            ],
            "angles": [],
            "dihedrals": [],
        }
        config_override = {
            "min_count": 1,
            "relative_density_threshold": 1.0e-8,
            "min_support_points": 4,
            "histogram_smoothing_sigma": 0.5,
            "update_smoothing_sigma": 0.5,
            "bond": {
                "hist_min": 0.75,
                "hist_max": 1.25,
                "hist_edges": 41,
                "table_min": 0.5,
                "table_max": 1.8,
                "table_points": 401,
                "left_guard": 0.65,
                "right_guard": 1.35,
            },
            "simulation": {
                "dt": 0.001,
                "steps": 200,
                "log_interval": 10,
                "burn_in_steps": 40,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "target.bin"
            seed_path = tmp / "seed.json"
            config_path = tmp / "config.json"
            rb_path = tmp / "rb.json"
            ibi_cfg = tmp / "ibi.json"
            fake_pypresso = tmp / "fake_pypresso.py"
            outdir = tmp / "ibi_out"
            fixed_input = tmp / "fixed_input.dat"
            write_dataset(dataset, distances)
            fixed_x = np.linspace(0.5, 1.8, 11)
            np.savetxt(fixed_input, np.column_stack((fixed_x, fixed_x * 0.0, fixed_x * 0.0)))
            seed_path.write_text(json.dumps(seed))
            config_path.write_text("{}")
            rb_path.write_text("{}")
            ibi_cfg.write_text(json.dumps(config_override))
            fake_code = """#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
args = sys.argv[2:]
def value(flag):
    return args[args.index(flag) + 1]
path = Path(value('--sample_npz'))
start = int(value('--sample_start_step'))
steps = int(value('--steps'))
interval = int(value('--log_interval'))
recorded = np.arange(start, steps + 1, interval, dtype=np.int64)
n = len(recorded)
r = 1.0 + 0.05 * np.sin(np.linspace(0.0, 4.0 * np.pi, n, endpoint=False))
com = np.zeros((n, 2, 3), dtype=float)
com[:, 0, :] = [2.0, 2.0, 2.0]
com[:, 1, :] = np.column_stack((2.0 + r, np.full(n, 2.0), np.full(n, 2.0)))
path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(path, schema_version=np.asarray(1, dtype=np.int32),
                    complete=np.asarray(1, dtype=np.int8),
                    steps=recorded, com=com, sites=com.copy(),
                    site_molecule=np.array([0, 1], dtype=np.int32),
                    site_index=np.array([0, 0], dtype=np.int32),
                    box=np.array([10.0, 10.0, 10.0]))
print('fake sampling complete')
"""
            fake_pypresso.write_text(fake_code)
            fake_pypresso.chmod(0o755)

            report = run_ibi(
                dataset=dataset,
                seed_priors=seed_path,
                config=config_path,
                rb_info=rb_path,
                outdir=outdir,
                iterations=1,
                pypresso=fake_pypresso,
                ibi_config=ibi_cfg,
                neighbor_search="verlet",
                overwrite=False,
            )
            self.assertEqual(report["iterations_completed"], 1)
            final_path = Path(report["final_priors"])
            final = json.loads(final_path.read_text())
            self.assertEqual(final["bonds"][0]["type"], "tabulated")
            self.assertEqual(final["bonds"][0]["ibi_mode"], "ibi")
            table = resolve_referenced_path(final["bonds"][0]["file"], final_path)
            self.assertTrue(table.is_file())
            fixed_table = resolve_referenced_path(final["bonds"][1]["file"], final_path)
            self.assertTrue(fixed_table.is_file())
            self.assertEqual(fixed_table.parent.resolve(), (outdir / "final").resolve())
            self.assertTrue((outdir / "sampling" / "iteration_001" / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
