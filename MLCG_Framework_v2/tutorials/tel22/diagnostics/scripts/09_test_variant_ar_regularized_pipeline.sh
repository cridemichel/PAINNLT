#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TRAINING_CONFIG_SOURCE="${TRAINING_CONFIG_SOURCE:-${TEL22_DIR}/diagnostics/configs/tel22_training_config_variant_ar_30ep.json}"
export VARIANT_A_RUN_DIR="${VARIANT_AR_RUN_DIR:-${VARIANT_A_RUN_DIR:-${TEL22_DIR}/diagnostics/smoke/variant_ar_regularized_30ep}}"
export PIPELINE_ALLOW_EARLY_STOP="${PIPELINE_ALLOW_EARLY_STOP:-1}"

exec bash "${SCRIPT_DIR}/08_test_variant_a_pipeline_15ep.sh" "$@"
