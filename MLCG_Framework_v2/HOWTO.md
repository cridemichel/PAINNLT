# MLCG Framework v2 - uso generico

Il framework implementa force matching coarse-grained con prior analitici e un
potenziale residuo PaiNN. Il **core non dipende da TEL22**: nomi dei residui,
mapping atomistico-CG, tipi dei siti, topologia bonded e corpi rigidi sono dati
di input.

## 1. Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Il preprocessing richiede una topologia leggibile da MDAnalysis e una
traiettoria che contenga le forze atomistiche.

## 2. Configurazione del mapping

`preprocessing/topology_config.json` è un template neutro. Sostituisci `MOL`,
`CG_A`, `CG_B` e i nomi atomici con quelli del sistema reale. Le chiavi
principali sono:

- `mapping.mapping_method`: `COM`, `COG` o `ATOM`;
- `mapping.residues`: mapping per `resname`;
- `mapping.site_types`: interi non negativi usati come specie PaiNN;
- `bonds`, `angles`, `dihedrals`: prior/topologia molecolare;
- `rigid_bodies`: opzionale, per geometrie multi-site rigidificate;
- parametri WCA e temperatura.

I file `cg_priors.json` e `rigid_bodies_info.json` sono **output generati**, non
configurazioni sorgente e non devono essere versionati.

## 3. Costruzione dataset

```bash
python3 preprocessing/build_cg_dataset.py \
  --topology /path/system.tpr \
  --trajectory /path/trajectory.trr \
  --config /path/topology_config.json \
  --output work/cg_dataset.bin
```

Per default gli output ausiliari vengono scritti accanto al dataset:

```text
work/cg_dataset.bin
work/cg_priors.json
work/rigid_bodies_info.json
```

È possibile specificare path diversi con `--priors-output` e
`--rb-info-output`.

## 4. Training

Compila il trainer:

```bash
cd training
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . -j
```

Copia/adatta `training/cg_model_config.json`. In particolare:

- `num_species` deve essere maggiore del massimo `site_type` presente nel dataset;
- `cutoff`, capacità PaiNN e iperparametri sono proprietà del problema, non del framework;
- `torque_weight=0` disabilita di fatto il contributo rotazionale;
- `diagnostic_overfit_frames=0` è il valore normale di produzione.

Esegui:

```bash
training/build/train_painn \
  work/cg_dataset.bin \
  work/cg_model.pt \
  /path/cg_model_config.json
```

## 5. ESPResSo

Dopo aver integrato il plugin in `simulation/espresso_plugin/`, equilibration e
produzione ricevono tutti gli artefatti esplicitamente:

```bash
pypresso simulation/equilibrate.py \
  --model work/cg_model.pt \
  --config /path/cg_model_config.json \
  --priors work/cg_priors.json \
  --rb_info work/rigid_bodies_info.json \
  --dataset work/cg_dataset.bin \
  --out_checkpoint work/equilibrated.npz

pypresso simulation/run_cg_md.py \
  --model work/cg_model.pt \
  --config /path/cg_model_config.json \
  --priors work/cg_priors.json \
  --rb_info work/rigid_bodies_info.json \
  --dataset work/cg_dataset.bin \
  --checkpoint work/equilibrated.npz
```

## 6. Test del framework

I test del core sono chemistry-agnostic e non dipendono da TEL22:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

I test C++/plugin che richiedono LibTorch o un ESPResSo patchato vanno eseguiti
nell'ambiente corrispondente.

## 7. Tutorial TEL22

`tutorials/tel22/` contiene soltanto un esempio applicativo minimale. Assume che
la traiettoria AA con forze sia già disponibile e mostra i passaggi dataset ->
training -> equilibration -> CG MD. Non è importato dal core e può essere
rimosso senza modificare il framework.
