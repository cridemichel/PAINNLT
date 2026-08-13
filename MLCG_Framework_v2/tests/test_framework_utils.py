#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))

from framework_utils import (  # noqa: E402
    input_hashes,
    nonconservative_prior_entries,
    rigid_body_quaternion,
    save_checkpoint,
    sha256_file,
    validate_checkpoint,
    validate_model_manifest,
    validate_wca_exclusion_policy,
    wca_topology_exclusion_pairs,
    wca_direct_bonded_site_exclusions,
)


class FakeParticle:
    def __init__(self, pid, ptype, mol_id, is_virtual):
        self.id = pid
        self.type = ptype
        self.mol_id = mol_id
        self.is_virtual = is_virtual


class FakeParticleList:
    def __init__(self, particles):
        self._particles = particles

    def __len__(self):
        return len(self._particles)

    def by_id(self, pid):
        return self._particles[pid]


class FakeSystem:
    def __init__(self):
        self.box_l = np.asarray([4.0, 5.0, 6.0])
        self.part = FakeParticleList([
            FakeParticle(0, 4, 0, False),
            FakeParticle(1, 0, 0, True),
            FakeParticle(2, 4, 1, False),
            FakeParticle(3, 0, 1, True),
        ])


def quat_to_body_to_space_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class FrameworkUtilsTests(unittest.TestCase):
    def test_principal_frame_orientation_roundtrip(self):
        body = np.asarray([
            [0.20, 0.00, 0.00],
            [-0.10, 0.15, 0.00],
            [-0.05, -0.08, 0.12],
        ])
        angle = 0.73
        axis = np.asarray([1.0, 2.0, -0.5])
        axis /= np.linalg.norm(axis)
        cross = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
        center = np.asarray([3.9, 0.1, 5.8])
        box = np.asarray([4.0, 5.0, 6.0])
        sites = center + (rotation @ body.T).T
        sites %= box
        rb_data = {
            "body_frame": "principal_axes",
            "sites": {
                "a": {"type": 0, "relative_pos_nm": body[0].tolist()},
                "b": {"type": 1, "relative_pos_nm": body[1].tolist()},
                "c": {"type": 2, "relative_pos_nm": body[2].tolist()},
            },
        }
        quat = rigid_body_quaternion(center, sites, box, rb_data)
        recovered_body_to_space = quat_to_body_to_space_matrix(quat)
        self.assertTrue(np.allclose(recovered_body_to_space, rotation, atol=1e-7))

    def test_manifest_and_checkpoint_roundtrip(self):
        config = {
            "architecture_variant": "painn_canonical_context_silu_v2",
            "num_species": 3,
            "hidden_channels": 16,
            "n_layers": 2,
            "num_rbf": 12,
            "cutoff": 0.8,
            "toxvaerd_alpha": 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model = directory / "model.pt"
            dataset = directory / "dataset.bin"
            config_path = directory / "config.json"
            priors = directory / "priors.json"
            rb_info = directory / "rb.json"
            model.write_bytes(b"model")
            dataset.write_bytes(b"dataset")
            config_path.write_text(json.dumps(config))
            priors.write_text("{}")
            rb_info.write_text("{}")
            manifest = {
                "schema_version": 3,
                "framework": "MLCG_Framework_v2",
                "energy_gauge": "isolated_species_zero_v1",
                "architecture": {
                    "variant": config["architecture_variant"],
                    "num_species": config["num_species"],
                    "hidden_channels": config["hidden_channels"],
                    "n_layers": config["n_layers"],
                    "num_rbf": config["num_rbf"],
                    "cutoff": config["cutoff"],
                    "toxvaerd_alpha": config["toxvaerd_alpha"],
                },
                "model_file_size_bytes": model.stat().st_size,
                "model_sha256": sha256_file(model),
            }
            Path(f"{model}.manifest.json").write_text(json.dumps(manifest))
            validate_model_manifest(model, config)

            hashes = input_hashes(
                dataset=dataset, config=config_path, priors=priors, rb_info=rb_info, model=model
            )
            system = FakeSystem()
            checkpoint_path = directory / "checkpoint.npz"
            n = len(system.part)
            save_checkpoint(
                checkpoint_path,
                system=system,
                pos=np.zeros((n, 3)),
                vel=np.zeros((n, 3)),
                quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
                omega=np.zeros((n, 3)),
                hashes=hashes,
                config=config,
                dt=0.001,
                kT=2.49,
            )
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                validate_checkpoint(
                    checkpoint,
                    system=system,
                    expected_hashes=hashes,
                    expected_config=config,
                )

    def test_tabulated_nve_guard_classification(self):
        priors = {
            "bonds": [
                {"type": "harmonic"},
                {"type": "morse"},
                {"type": "tabulated"},
            ],
            "angles": [{"type": "tabulated"}],
            "dihedrals": [{"type": "cosine"}],
        }
        self.assertEqual(
            nonconservative_prior_entries(priors),
            ["bond[2]=tabulated", "angle[0]=tabulated"],
        )

    def test_wca_topology_exclusions(self):
        # Runtime consumes the explicit pair lists stored at preprocessing time;
        # it must not reinterpret arbitrary bonded restraints as topology.
        priors = {
            "bonds": [
                {"mol_i": 0, "mol_j": 1, "type": "harmonic", "exclude_wca": True},
                {"mol_i": 0, "mol_j": 3, "type": "morse", "exclude_wca": False},
            ],
            "wca_exclusions": {
                "exclude_12": True,
                "exclude_13": True,
                "scope": "molecule_pair_all_sites",
                "pair_source": "explicit_topology_pairs_v2",
                "direct_pairs": [[0, 1], [1, 2], [2, 3]],
                "one_three_pairs": [[0, 2], [1, 3]],
                "direct_pair_count": 3,
                "one_three_pair_count": 2,
            },
        }
        validate_wca_exclusion_policy(priors)
        direct, one_three = wca_topology_exclusion_pairs(priors, 5)
        self.assertEqual(direct, {(0, 1), (1, 2), (2, 3)})
        self.assertEqual(one_three, {(0, 2), (1, 3)})
        self.assertNotIn((0, 3), direct)

    def test_selective_wca_12_maps_only_explicit_bonded_sites(self):
        priors = {
            "bonds": [
                {
                    "mol_i": 0, "mol_j": 1, "site_i": 2, "site_j": 3,
                    "type": "harmonic", "exclude_wca": True,
                },
                {
                    "mol_i": 2, "mol_j": 1, "site_i": 5, "site_j": 4,
                    "type": "harmonic", "exclude_wca": True,
                },
            ],
            "wca_exclusions": {
                "exclude_12": True,
                "exclude_13": True,
                "scope": "molecule_pair_all_sites",
                "pair_source": "explicit_topology_pairs_v2",
                "direct_pairs": [[0, 1], [1, 2]],
                "one_three_pairs": [],
                "direct_pair_count": 2,
                "one_three_pair_count": 0,
            },
        }
        mapped = wca_direct_bonded_site_exclusions(priors, 3)
        self.assertEqual(mapped[(0, 1)], {(2, 3)})
        # Bond was stored as mol 2 -> mol 1, so site ordering is normalized.
        self.assertEqual(mapped[(1, 2)], {(4, 5)})

    def test_selective_wca_12_refuses_untraceable_direct_pair(self):
        priors = {
            "bonds": [
                {
                    "mol_i": 0, "mol_j": 1,
                    "type": "harmonic", "exclude_wca": True,
                }
            ],
            "wca_exclusions": {
                "exclude_12": True,
                "exclude_13": True,
                "scope": "molecule_pair_all_sites",
                "pair_source": "explicit_topology_pairs_v2",
                "direct_pairs": [[0, 1]],
                "one_three_pairs": [],
                "direct_pair_count": 1,
                "one_three_pair_count": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "site_i/site_j"):
            wca_direct_bonded_site_exclusions(priors, 2)

    def test_wca_topology_exclusion_pair_count_is_validated(self):
        priors = {
            "wca_exclusions": {
                "exclude_12": True,
                "exclude_13": True,
                "scope": "molecule_pair_all_sites",
                "pair_source": "explicit_topology_pairs_v2",
                "direct_pairs": [[0, 1]],
                "one_three_pairs": [],
                "direct_pair_count": 2,
                "one_three_pair_count": 0,
            },
        }
        with self.assertRaises(ValueError):
            wca_topology_exclusion_pairs(priors, 3)

    def test_wca_exclusion_policy_rejects_legacy_priors(self):
        with self.assertRaises(ValueError):
            validate_wca_exclusion_policy({"bonds": []})


if __name__ == "__main__":
    unittest.main()
