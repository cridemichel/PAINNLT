#!/usr/bin/env python3
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "simulation" / "espresso_plugin"
CPP = (PLUGIN / "PaiNN_ML_Potential.cpp").read_text(encoding="utf-8")
HEADER = (PLUGIN / "PaiNN_ML_Potential.hpp").read_text(encoding="utf-8")
PYX = (PLUGIN / "painn.pyx").read_text(encoding="utf-8")
RUNNER = (ROOT / "simulation" / "run_cg_md.py").read_text(encoding="utf-8")
WRAPPER = ROOT / "tutorials" / "tel22" / "diagnostics" / "scripts" / "10_profile_painn_runtime.sh"


class PaiNNRuntimeProfilingSourceTests(unittest.TestCase):
    def test_profiling_is_opt_in_and_disabled_by_default(self):
        self.assertIn("bool enabled = false;", HEADER)
        self.assertIn("const bool profile_enabled = m_profile.enabled;", CPP)
        self.assertIn("if (profile_this_call)", CPP)
        self.assertNotIn("configure_profiling(true", CPP)

    def test_profile_json_object_keys_are_valid_cpp_string_literals(self):
        self.assertIn('out << "\\\"graph\\\":{";', CPP)
        self.assertIn('out << "\\\"allocation_churn_indicators\\\":{";', CPP)
        self.assertNotIn('out << "\\\"graph":{";', CPP)
        self.assertNotIn('out << "\\\"allocation_churn_indicators":{";', CPP)

    def test_profile_stages_cover_current_runtime_churn(self):
        for token in (
            "node_index_ms",
            "neighbor_traversal_ms",
            "edge_pack_ms",
            "tensor_inputs_ms",
            "forward_ms",
            "energy_scalar_ms",
            "autograd_ms",
            "force_to_cpu_ms",
            "force_scatter_ms",
            "unattributed_cleanup_mean",
            "host_payload_lower_bound_bytes_sum",
        ):
            self.assertIn(token, CPP)
        self.assertIn("temporary_cpp_containers_per_call", CPP)

    def test_python_interface_exposes_snapshot_without_changing_activation_defaults(self):
        self.assertIn("configure_painn_profiling", PYX)
        self.assertIn("get_painn_profile", PYX)
        self.assertIn('precision: str = "float32"', PYX)
        self.assertIn('device: str = "auto"', PYX)

    def test_runner_profiles_only_when_explicitly_requested(self):
        self.assertIn("--painn_profile_report", RUNNER)
        self.assertIn("--painn_profile_warmup_calls", RUNNER)
        self.assertIn('if args.painn_profile_report is not None:', RUNNER)
        self.assertIn('if args.device != "cpu":', RUNNER)
        self.assertIn("integration_wall_seconds", RUNNER)

    def test_tel22_wrapper_is_diagnostic_only_and_uses_certified_baseline_inputs(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("tel22_model.pt", source)
        self.assertIn("equilibrated.npz", source)
        self.assertIn("--nve", source)
        self.assertIn("--no_log", source)
        self.assertIn("--painn_profile_report", source)
        self.assertIn("PROFILE_DEVICE:-cpu", source)
        self.assertNotIn("--disable_ml", source)


if __name__ == "__main__":
    unittest.main()
