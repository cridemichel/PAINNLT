#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROD_CPP = ROOT / "simulation" / "espresso_plugin" / "PaiNN_ML_Potential.cpp"

sys.path.insert(0, str(HERE))
from export_shared_case import ESPRESSO_BOUNDARY_MARGIN_NM, translate_state_for_espresso  # noqa: E402


def main() -> int:
    source = PROD_CPP.read_text(encoding="utf-8")
    runner = (HERE / "run_mlcg_certification.py").read_text(encoding="utf-8")
    assert 'p.add_argument("--worker-dt"' in runner
    assert "one fresh ESPResSo Python process per dt" in runner
    assert "subprocess.run(" in runner
    assert "if args.worker_dt is not None:" in runner
    assert "MLCG_SYNTHETIC_PAINN_BENCHMARK_OVERRIDE" not in source
    assert "void PaiNN_ML_Potential::calculate_forces" in source
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "override.cpp"
        subprocess.run(
            [sys.executable, str(HERE / "make_espresso_benchmark_override.py"), "--source", str(PROD_CPP), "--output", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        generated = out.read_text(encoding="utf-8")
        assert "BEGIN MLCG_SYNTHETIC_PAINN_BENCHMARK_OVERRIDE" in generated
        assert "calculate_synthetic_benchmark_forces" in generated
        assert "atom_energies.sum()" in generated
        assert "atom_energies.sum() * bench.energy_scale_kcal" in generated
        assert "harmonic_energy_kcal" in generated
        assert "t_positions" in generated
        assert "model->energy_scale.fill_(1.0)" in generated
        assert "model->cutoff_radius = g_synthetic_benchmark_case->cutoff_A" in generated
        assert "FORCE_KCAL_A_TO_KJ_NM = 41.84" in generated
        assert "espresso_translation_A" in generated
        assert "- bench.espresso_translation_A[0]" in generated
        assert generated.count("void PaiNN_ML_Potential::calculate_forces") == 1
    # Regression: the synthetic TorchMD state contains small negative coordinates
    # around equilibrium sites at zero. ESPResSo folds those coordinates across
    # the periodic box unless the whole state is rigidly translated first.
    eq = np.array([[0.0, 0.0, 0.0], [0.4, 0.8, 1.2]], dtype=np.float64)
    pos = np.array([[-0.03, 0.02, -0.01], [0.42, 0.77, 1.23]], dtype=np.float64)
    eq_shifted, pos_shifted, shift = translate_state_for_espresso(eq, pos)
    assert np.all(np.minimum(eq_shifted.min(axis=0), pos_shifted.min(axis=0)) >= ESPRESSO_BOUNDARY_MARGIN_NM)
    assert np.all(shift > 0.0)
    np.testing.assert_allclose(pos_shifted - eq_shifted, pos - eq, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        pos_shifted[1] - pos_shifted[0], pos[1] - pos[0], rtol=0.0, atol=1.0e-15
    )

    # The actual TorchMD constant is rounded; the two physical conversions agree
    # to much better than one part per million.
    assert abs((100.0 / 48.88821) / math.sqrt(4.184) - 1.0) < 1.0e-6
    print("[PASS] MLCG/ESPResSo synthetic PaiNN parity benchmark self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
