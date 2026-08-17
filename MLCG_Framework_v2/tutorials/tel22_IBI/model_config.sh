#!/usr/bin/env bash
# Shared loader. Model-dependent choices live in JSON, not in workflow code.
# Later sections override earlier ones explicitly; environment variables may
# override configured values and that source is recorded by provenance.
load_model_dependent_config() {
  local root cfg
  local sections=("$@")
  if (( ${#sections[@]} == 0 )); then
    echo "[ERROR] load_model_dependent_config requires at least one section" >&2
    return 2
  fi
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  cfg="${IBI_MODEL_DEPENDENT_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_dependent_workflow_config.json}"
  [[ -f "${cfg}" ]] || { echo "[ERROR] Missing model-dependent workflow config: ${cfg}" >&2; return 1; }
  eval "$("${PYTHON_BIN:-python3}" "${root}/simulation/model_dependent_config.py" export-shell --config "${cfg}" --sections common "${sections[@]}" --preserve-env)"
}
write_model_dependent_provenance() {
  local output="$1" root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  "${PYTHON_BIN:-python3}" "${root}/simulation/model_dependent_config.py" provenance \
    --config "${MODEL_DEPENDENT_CONFIG_PATH}" --sections ${MODEL_DEPENDENT_CONFIG_SECTIONS} --output "${output}"
}
