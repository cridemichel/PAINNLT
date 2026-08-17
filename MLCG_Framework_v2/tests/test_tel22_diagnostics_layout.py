import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tutorials" / "organize_tel22_diagnostics.py"
spec = importlib.util.spec_from_file_location("organize_tel22_diagnostics", MODULE_PATH)
layout = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(layout)


def test_production_artifacts_are_not_in_diagnostic_move_plan():
    forbidden = {
        "ibi_conservative",
        "ibi_run_16ps",
        "ibi_run_16ps_continue",
        "tel22_dataset.bin",
        "tel22_dataset_ibi_residual.bin",
        "tel22_model.pt",
        "tel22_model_ibi.pt",
        "cg_priors.json",
        "md.trr",
        "md_whole.trr",
        "md.gro",
        "md.tpr",
    }
    assert forbidden.isdisjoint(layout.IBI_DIAGNOSTIC_PATHS)
    assert forbidden.isdisjoint(layout.TEL22_DIAGNOSTIC_PATHS)
    assert set(layout.PROTECTED_GROMACS).isdisjoint(layout.IBI_DIAGNOSTIC_PATHS)


def test_move_plan_moves_diagnostics_but_not_gromacs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tutorial = Path(tmpdir) / "tel22_IBI"
        tutorial.mkdir()
        (tutorial / "23_certify_conservative_ibi_nve.sh").write_text("#!/bin/sh\n")
        (tutorial / "nve_certification_conservative_ibi_only").mkdir()
        (tutorial / "nve_certification_conservative_ibi_only" / "energy.csv").write_text("x\n")
        (tutorial / "md.trr").write_bytes(b"gromacs-trajectory")
        before = layout.protected_fingerprints(tutorial)
        plan = layout.move_plan(
            tutorial,
            ("23_certify_conservative_ibi_nve.sh",),
            {"nve_certification_conservative_ibi_only": "diagnostics/nve/nve_certification_conservative_ibi_only"},
        )
        layout.preflight(plan)
        layout.apply_plan(plan, True)
        after = layout.protected_fingerprints(tutorial)
        assert before == after
        assert (tutorial / "diagnostics/scripts/23_certify_conservative_ibi_nve.sh").is_file()
        assert (tutorial / "diagnostics/nve/nve_certification_conservative_ibi_only/energy.csv").is_file()
        assert (tutorial / "md.trr").read_bytes() == b"gromacs-trajectory"


def test_model_config_routes_nonpipeline_outputs_below_diagnostics():
    cfg = json.loads((ROOT / "tutorials" / ("tel" + "22_IBI") / "model_dependent_workflow_config.json").read_text())
    sec = cfg["sections"]
    assert sec["step11"]["DBI_OUTDIR"].startswith("diagnostics/ibi/")
    assert sec["step15"]["IBI_VALIDATION_OUTDIR"].startswith("diagnostics/ibi/")
    assert sec["step17"]["MULTISEED_OUTDIR"].startswith("diagnostics/ml/")
    assert sec["step18"]["POSTIBI_RUNTIME_OUTDIR"].startswith("diagnostics/ml/")
    assert sec["step23"]["NVE_OUTPUT_DIR"].startswith("diagnostics/nve/")
    assert sec["step29"]["IBI_TIMESTEP_OUTPUT_DIR"].startswith("diagnostics/ibi/")
    assert sec["step33"]["IBI_ANGLE_FINAL_OUTPUT_DIR"].startswith("diagnostics/ibi/")
    assert sec["step35"]["IBI_DIHEDRAL_TEST_OUT"].startswith("diagnostics/ibi/")
    assert sec["step39"]["IBI_DIHEDRAL_REPLICA_OUT"].startswith("diagnostics/ibi/")
    assert sec["step35"]["IBI_DIHEDRAL_SETTINGS"] == "diagnostics/ibi/ibi_dihedral_test_settings.json"
    assert sec["step34"]["IBI_PROMOTION_CURRENT_DIR"] == "ibi_conservative"


def test_diagnostic_scripts_are_physically_separate_from_pipeline_root():
    tel22 = ROOT / "tutorials" / ("tel" + "22")
    ibi = ROOT / "tutorials" / ("tel" + "22_IBI")
    for name in layout.TEL22_DIAGNOSTIC_SCRIPTS:
        assert not (tel22 / name).exists()
        assert (tel22 / "diagnostics" / "scripts" / name).is_file()
    for name in layout.IBI_DIAGNOSTIC_SCRIPTS:
        assert not (ibi / name).exists()
        assert (ibi / "diagnostics" / "scripts" / name).is_file()


def test_relocated_model_config_scripts_resolve_tutorial_root():
    ibi_scripts = ROOT / "tutorials" / ("tel" + "22_IBI") / "diagnostics" / "scripts"
    for name in (
        "11_build_dbi_preview.sh",
        "17_benchmark_training_multiseed.sh",
        "23_certify_conservative_ibi_nve.sh",
        "30_diagnose_regularize_ibi_angles.sh",
        "35_test_conservative_ibi_dihedrals.sh",
    ):
        text = (ibi_scripts / name).read_text()
        assert "TUTORIAL_DIR=" in text
        assert 'source "${TUTORIAL_DIR}/model_config.sh"' in text
        assert 'cd "${TUTORIAL_DIR}"' in text


def test_historical_and_optional_outputs_have_reviewed_destinations():
    paths = layout.IBI_DIAGNOSTIC_PATHS
    assert paths["ibi_dbi_preview"] == "diagnostics/ibi/ibi_dbi_preview"
    assert paths["ibi_run"] == "diagnostics/historical/ibi_run"
    assert paths["training_multiseed_benchmark"] == "diagnostics/ml/training_multiseed_benchmark"
    assert paths["ibi_ml_ab_validation"] == "diagnostics/ml/ibi_ml_ab_validation"
    assert paths["archive"] == "diagnostics/historical/phase3_archive"


def test_reviewed_junk_does_not_include_live_trainer_build_or_pipeline_outputs():
    junk = set(layout.REPO_JUNK)
    for live in (
        "training/build",
        "training/cg_dataset.bin",
        "training/best_cg_model.pt",
        "training/cg_training_log.csv",
    ):
        assert live not in junk
    assert "training/build_test" in junk
    assert "training/best_cg_model_old.pt" in junk
    assert "repair_certify_nve_inline_sampling_v3_20260813.py" in junk
    assert "repair_nve_sigma_final_20260813.py" in junk
    assert "create_zip_per_chatgpt.sh" in junk
    assert all(not rel.endswith(("md.trr", "md_whole.trr", "md.gro", "md.tpr")) for rel in junk)


def test_cleanup_generated_mode_cannot_delete_gromacs_products():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tutorials = tmp / "tutorials"
        tutorials.mkdir()
        shutil.copy2(ROOT / "tutorials" / "cleanup_tel22_artifacts.sh", tutorials / "cleanup_tel22_artifacts.sh")
        for name in (("tel" + "22"), ("tel" + "22_IBI")):
            d = tutorials / name
            d.mkdir()
            (d / "md.trr").write_bytes(b"protected-gromacs")
            (d / "energy.csv").write_text("generated-cg\n")
        subprocess.run(["bash", str(tutorials / "cleanup_tel22_artifacts.sh"), "--run", "--generated"], check=True)
        for name in (("tel" + "22"), ("tel" + "22_IBI")):
            d = tutorials / name
            assert (d / "md.trr").read_bytes() == b"protected-gromacs"
            assert not (d / "energy.csv").exists()
