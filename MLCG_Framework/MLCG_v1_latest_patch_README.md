# Patch per MLCG Framework v1

Base verificata: `MLCG_framework_v1.zip`

SHA256 della base:

```text
479645d816c6cde3aebcec2e24a4d344cec5a9d975292451b068a9a29cbc6646
```

Patch:

```text
MLCG_v1_latest_sources_fixes.patch
SHA256: dea2b1dbd1074b4d9a068230a523deeb749f299aff7d7896f45adeb6244fa0b1
```

## Modifiche incluse

- Correzione del mismatch tra griglia degli istogrammi IBI (299 punti) e griglia delle tabelle ESPResSo (2001 punti).
- Update IBI calcolato sulla distribuzione osservabile grezza; il Jacobiano viene usato solo per il PMF DBI iniziale.
- Proiezione spline dell'update dalla griglia degli istogrammi alla griglia tabulata.
- Riappliacazione conservativa delle pareti angolari dopo ogni update.
- Convenzioni distinte per `TabulatedDistance`/dihedral e `TabulatedAngle`.
- Tabelle uniformi e salvate a precisione `%.16e`.
- Dominio bond coerente `0.01 ... 3.0` e dihedral `0 ... 2*pi`.
- Traiettoria IBI site-aware in formato NPZ: COM, virtual sites, mapping sito-molecola e box.
- Analisi IBI delle coordinate richieste da `site_i`, `site_j`, `site_k`, `site_l`.
- Correzione del tipo CG della molecola: viene conservato il primo tipo di sito, non l'ultimo.
- Rimozione della seconda configurazione WCA con offset `+1` in `equilibrate.py`.
- Uso di `ibi_priors/cg_priors_final.json` nella produzione e nello scaling.
- Lo script di scaling elimina `energy.csv` prima di ogni run e fallisce esplicitamente se ESPResSo termina con errore.

## File modificati

```text
ibi/run_ibi_loop.py
simulation/equilibrate.py
simulation/run_cg_md.py
tutorials/tel22_IBI/06_run_espresso.sh
tutorials/tel22_IBI/run_ibi_scaling.py
```

## Applicazione

Dalla root di `MLCG_Framework`:

```bash
git status
git switch -c fix-ibi-latest

git apply --check ~/Downloads/MLCG_v1_latest_sources_fixes.patch
git apply ~/Downloads/MLCG_v1_latest_sources_fixes.patch
```

## Verifiche già eseguite

- `git apply --check`: PASS sulla base caricata.
- `python -m py_compile` sui file Python modificati: PASS.
- Test sintetico DBI/IBI: PASS.
- Shape test: istogrammi `(299,)`, tabelle `(2001,)`, update proiettato correttamente.
- Pareti angolari: gradiente ESPResSo negativo a sinistra e positivo a destra.
- Controllo griglia uniforme prima del salvataggio.

## Dopo l'applicazione

Le tabelle, il dataset residuo, il modello e il checkpoint precedenti non sono compatibili con la nuova pipeline. Rigenerare nell'ordine:

```text
DBI/IBI tables
-> ibi_priors/cg_priors_final.json
-> dataset residuo
-> modello PaiNN
-> checkpoint equilibrato
-> test priors-only NVE
-> test IBI + PaiNN NVE
```

Questa patch non modifica il kernel tabulato di ESPResSo. L'interpolazione lineare separata di energia e forza può ancora introdurre un floor numerico; la soluzione esatta richiede un interpolatore comune conservativo nel core ESPResSo.
