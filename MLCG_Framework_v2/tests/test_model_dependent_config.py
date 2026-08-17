from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorials" / ("tel" + "22_IBI")
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "ibi"))

from model_dependent_config import load_config, resolved_values  # noqa: E402
from ibi_core import DEFAULT_IBI_SETTINGS, load_ibi_settings  # noqa: E402


def test_model_config_covers_all_model_dependent_ibi_workflow_steps():
    data = load_config(TUTORIAL / "model_dependent_workflow_config.json")
    expected = {"common", *{f"step{i}" for i in range(11, 40) if i != 20}}
    # Step 20 only installs the generic ESPResSo kernel and has no model-specific policy.
    assert expected <= set(data["sections"])


def test_environment_override_is_explicitly_resolved(monkeypatch):
    data = load_config(TUTORIAL / "model_dependent_workflow_config.json")
    monkeypatch.setenv("IBI_DIHEDRAL_REPLICA_COUNT", "7")
    values, sources = resolved_values(data, ["common", "step39"], preserve_env=True)
    assert values["IBI_DIHEDRAL_REPLICA_COUNT"] == "7"
    assert sources["IBI_DIHEDRAL_REPLICA_COUNT"] == "environment_override"


def test_explicit_ibi_settings_are_authoritative_not_silently_merged(tmp_path):
    config = tmp_path / "ibi.json"
    config.write_text(json.dumps({"kT": 3.1}) + "\n")
    loaded = load_ibi_settings(config)
    assert loaded == {"kT": 3.1}
    assert "alpha" not in loaded
    # Low-level no-file use still exposes the documented generic fixture defaults.
    assert load_ibi_settings(None) == copy.deepcopy(DEFAULT_IBI_SETTINGS)


def test_model_dependent_workflow_wrappers_load_external_config():
    data = load_config(TUTORIAL / "model_dependent_workflow_config.json")
    configured_keys = {key for section in data["sections"].values() for key in section}
    for step in range(11, 40):
        if step == 20:
            continue
        matches = list(TUTORIAL.glob(f"{step:02d}_*.sh"))
        matches += list((TUTORIAL / "diagnostics" / "scripts").glob(f"{step:02d}_*.sh"))
        assert matches, f"missing wrapper for step {step}"
        source = matches[0].read_text()
        assert "model_config.sh" in source, matches[0].name
        assert "load_model_dependent_config" in source, matches[0].name
        assert "write_model_dependent_provenance" in source, matches[0].name
        prefix = source.split("load_model_dependent_config", 1)[0]
        for key in configured_keys:
            assert f"${{{key}}}" not in prefix, f"{matches[0].name} reads {key} before loading model config"


def test_generic_diagnostics_do_not_embed_tel22_calibrated_candidate_names():
    generic = [
        ROOT / "simulation" / "finalize_promoted_ibi_certification.py",
        ROOT / "simulation" / "finalize_dihedral_ibi_test.py",
        ROOT / "simulation" / "finalize_dihedral_update_localization.py",
        ROOT / "ibi" / "angle_regularization_diagnostics.py",
        ROOT / "ibi" / "generate_dihedral_update_localization_candidates.py",
        ROOT / "simulation" / "ibi_timestep_range_diagnostics.py",
    ]
    text = "\n".join(path.read_text() for path in generic)
    assert "smooth_0p0075" not in text
    assert "default_candidate_specs" not in text
    assert "DEFAULT_CANDIDATES" not in text
    assert "old_tel22" not in text
