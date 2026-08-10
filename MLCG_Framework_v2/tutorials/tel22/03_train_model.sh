#!/bin/bash
set -euo pipefail

echo "======================================================"
echo " 03. TRAINING NEURAL NETWORK "
echo "======================================================"

if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

if [ ! -f "rigid_bodies_info.json" ] || [ ! -f "tel22_topology.json" ]; then
    echo "Errore: rigid_bodies_info.json o tel22_topology.json non trovato."
    echo "Esegui prima ./02_build_dataset.sh con i sorgenti corretti."
    exit 1
fi

# Do not overwrite a tuned/ablated configuration on every launch.
# Create the historical TEL22 baseline only when no config exists yet.
if [ ! -f "tel22_training_config.json" ]; then
    cat << 'JSON' > tel22_training_config.json
{
    "architecture_variant": "painn_canonical_context_silu_v2",
    "num_species": 8,
    "hidden_channels": 64,
    "n_layers": 2,
    "num_rbf": 32,
    "cutoff": 1.2616,
    "toxvaerd_alpha": 0.1,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 16,
    "torque_weight": 0.5,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 20,
    "reduce_lr_patience": 6,
    "weight_decay": 0.0,
    "lipschitz_lambda": 0.0,
    "diagnostic_overfit_frames": 0,
    "physical_validation_only": true,
    "include_decoys_in_train": false,
    "shuffle_each_epoch": true,
    "split_seed": 42,
    "validation_fraction": 0.2
}
JSON
    echo "[INFO] Creato tel22_training_config.json con il profilo TEL22 sicuro/canonico."
else
    echo "[INFO] Uso tel22_training_config.json esistente senza riscriverlo."
fi

python3 - <<'PY'
import json
with open("tel22_training_config.json") as handle:
    c = json.load(handle)
with open("tel22_topology.json") as handle:
    topology = json.load(handle)
with open("rigid_bodies_info.json") as handle:
    rb_info = json.load(handle)
print(
    "[INFO] Config: "
    f"variant={c.get('architecture_variant')} | "
    f"hidden={c.get('hidden_channels')} | layers={c.get('n_layers')} | "
    f"rbf={c.get('num_rbf')} | cutoff={c.get('cutoff')} nm | "
    f"epochs={c.get('epochs')} | batch={c.get('batch_size')} | "
    f"torque_weight={c.get('torque_weight')} | physical_val={c.get('physical_validation_only')} | "
    f"include_decoys={c.get('include_decoys_in_train')} | shuffle_each_epoch={c.get('shuffle_each_epoch')}"
)
if int(c.get("diagnostic_overfit_frames", 0)) > 0:
    raise SystemExit(
        "[ERROR] tel22_training_config.json contiene diagnostic_overfit_frames > 0. "
        "Non usare una config tiny-set per il training di produzione."
    )
if c.get("architecture_variant") != "painn_canonical_context_silu_v2":
    raise SystemExit("[ERROR] architecture_variant non corrisponde al PaiNN canonico compilato.")
if c.get("physical_validation_only") is not True:
    raise SystemExit("[ERROR] Il training TEL22 di produzione richiede physical_validation_only=true.")
if c.get("include_decoys_in_train", False):
    raise SystemExit(
        "[ERROR] I legacy decoy whole-frame non mascherati sono disabilitati nel training di produzione."
    )
if c.get("shuffle_each_epoch") is not True:
    raise SystemExit("[ERROR] Il training TEL22 richiede shuffle_each_epoch=true.")
if float(topology.get("decoy_target_fraction", 0.0)) != 0.0:
    raise SystemExit(
        "[ERROR] tel22_topology.json abilita legacy decoy senza loss mask. "
        "Applica la patch e rigenera il dataset con ./02_build_dataset.sh."
    )
for resname, info in rb_info.items():
    sites = info.get("sites", {})
    if len(sites) == 1:
        site_name, site = next(iter(sites.items()))
        offset = site.get("relative_pos_nm", [0.0, 0.0, 0.0])
        norm2 = sum(float(x) * float(x) for x in offset)
        if norm2 > 1.0e-12:
            raise SystemExit(
                f"[ERROR] {resname}/{site_name} ha offset one-site non nullo. "
                "Rigenera dataset e rigid_bodies_info.json con ./02_build_dataset.sh."
            )
PY

echo "Avvio l'addestramento C++ (force+torque normalizzati)..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
../../training/build/train_painn \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config.json

echo "Addestramento completato e modello salvato in tel22_model.pt!"
