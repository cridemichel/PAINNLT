# Correzioni per `MLCG_Framework` v1 (versione con IBI)

## Convenzione usata

- `MLCG_Framework_v2`: riferimento funzionante, senza IBI.
- `MLCG_Framework`: v1 da correggere, con DBI/IBI.

L'obiettivo iniziale non è ottimizzare gli iperparametri IBI, ma rendere coerenti:

1. la PaiNN addestrata e quella eseguita nel plugin ESPResSo;
2. la definizione delle distribuzioni target e simulate;
3. la geometria usata dal preprocessing e dalla simulazione;
4. energia, forza e griglia delle tabelle;
5. l'Hamiltoniana usata in preprocessing, IBI, equilibrazione e produzione.

## Patch pronta

La patch combinata è:

```text
MLCG_Framework_v1_combined_local_fixes.patch
```

Si applica alla radice di `MLCG_Framework`:

```bash
cd ~/WORK/NEURAL_NETWORKS/PAINNLT/MLCG_Framework
git status
git switch -c fix-v1-ibi

git apply --check ~/Downloads/MLCG_Framework_v1_combined_local_fixes.patch
git apply ~/Downloads/MLCG_Framework_v1_combined_local_fixes.patch
```

Dopo la patch occorre ricompilare il plugin ESPResSo/Cython e il training C++.

La patch è stata verificata con `git apply --check`, `python -m py_compile` e test sintetici delle tabelle. Non è stato possibile compilare il plugin contro la specifica build locale di ESPResSo/LibTorch.

---

# Correzioni locali comprese nella patch

## 1. Allineamento del costruttore PaiNN fra training e plugin

### Bug

Il training costruisce:

```cpp
PaiNNModel(..., cutoff, apply_envelope, use_bias, toxvaerd_alpha);
```

Il plugin costruiva:

```cpp
PaiNNModel(..., cutoff, toxvaerd_alpha);
```

Poiché il sesto argomento dopo `cutoff` è un `bool`, `toxvaerd_alpha=0.1` veniva convertito in `true`. Il plugin attivava quindi `apply_envelope`, applicando il cutoff custom una seconda volta.

### Correzione

Propagare esplicitamente:

```cpp
bool apply_envelope,
bool use_bias,
double toxvaerd_alpha
```

attraverso:

- `PaiNN_ML_Potential.hpp`;
- `PaiNN_ML_Potential.cpp`;
- `painn.pyx`;
- `run_cg_md.py`;
- `equilibrate.py`;
- `verify_energy_scaling.py`;
- parity evaluator.

Per riprodurre la v2:

```json
{
  "apply_envelope": false,
  "use_bias": false,
  "toxvaerd_alpha": 0.1
}
```

## 2. Fallimento esplicito se il checkpoint non viene caricato

Il plugin catturava l'eccezione di `torch::load` e continuava con pesi casuali. La patch aggiunge:

```cpp
catch (const c10::Error& e) {
    std::cerr << e.what() << "\n";
    throw;
}
```

## 3. Rimozione del soft-clip non lineare dell'energia totale

Il force matching non determina l'offset additivo dell'energia. Se si applica una funzione non lineare `g(E)` prima del gradiente,

```math
F = -g'(E+C)\nabla E,
```

le forze dipendono dall'offset arbitrario `C`.

La patch usa:

```cpp
auto total_energy = energy.sum();
m_last_energy = total_energy.item<double>();

auto grads = torch::autograd::grad(
    {total_energy},
    {t_r_ij},
    {torch::ones_like(total_energy)},
    false,
    false
);
```

Questa modifica va verificata prima in single rank. L'energia MPI richiede una politica esplicita di ownership degli atomi/archi.

## 4. Distribuzione target corretta nell'IBI

La correzione Jacobiana serve solo per il PMF iniziale:

```math
U_0(r)=-k_BT\log[p(r)/r^2],
```

ma l'update deve confrontare lo stesso osservabile:

```math
U_{n+1}=U_n+\alpha k_BT\log[p_n(q)/p_{target}(q)].
```

La patch separa:

- `pmf_hist`: corretto per il Jacobiano;
- `target_hist`: istogramma grezzo della coordinata, normalizzato come quello simulato.

## 5. Normalizzazione robusta degli istogrammi

Target e simulazione vengono:

- convertiti in array finiti;
- regolarizzati nei bin vuoti;
- normalizzati usando lo stesso `dx`.

Questo evita rapporti dipendenti dalla normalizzazione numerica introdotta dal clipping.

## 6. Tabelle su griglia realmente uniforme

Il codice precedente salvava i centri dei bin e poi aggiungeva gli endpoint. I due intervalli alle estremità risultavano dimezzati, mentre ESPResSo interpreta le `N` energie e forze come campionate uniformemente fra `min` e `max`.

La patch:

- genera `len(centers)+1` punti uniformi sul dominio completo;
- interpola energia e forza mediante un unico interpolante;
- usa i domini:
  - bond: `[0, 5]`;
  - angle: `[0, pi]`;
  - dihedral: `[0, 2*pi]`.

Il limite inferiore del bond può essere portato a un piccolo valore positivo se richiesto dalla specifica configurazione fisica.

## 7. Periodicità dei dihedrals

La patch:

- porta target e simulazione in `[0, 2*pi)`;
- usa una derivata periodica centrale;
- evita l'integrazione non periodica della forza;
- riscampiona con spline periodica;
- impone:

```python
energy[-1] = energy[0]
force[-1] = force[0]
```

Se il massimo della forza supera il limite, scala l'intero profilo energetico invece di tagliare i singoli campioni, preservando `F=-dV/dphi`.

## 8. Accesso corretto ai virtual sites

`site_i`, `site_j`, ecc. sono indici della lista dei siti, non tipi chimici. La forma corretta è:

```python
pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i][1]
```

La patch corregge bond/angle/dihedral e la raccolta delle distribuzioni.

## 9. Correzione del loop Kabsch

La forma errata:

```python
for mol_idx, (m_type, r_name) in enumerate(mol_resnames):
```

viene sostituita con:

```python
for mol_idx, r_name in enumerate(mol_resnames):
```

## 10. Correzione delle forze dihedrali analitiche

La formula originale non coincideva con le differenze finite. La patch usa una formula coerente con la convenzione signed-dihedral di `get_dihedral()`.

Nel test sintetico usato per l'audit, l'errore massimo per componente passa da circa `4.30` a circa `5.4e-10` in double precision.

## 11. `force_cap` solo nel warmup IBI

Il cap viene disattivato prima della produzione usata per gli istogrammi:

```python
system.force_cap = 0.0
```

Altrimenti l'IBI converge verso una Hamiltoniana con forze troncate.

## 12. Il generatore di dataset residuo non può essere un `copy2`

`ibi/generate_residual_dataset.py` dichiarava di sottrarre le forze IBI, ma copiava byte per byte il dataset originale. La patch lo rende un errore esplicito.

Il dataset residuo deve essere generato direttamente da:

```bash
preprocessing/build_cg_dataset.py --priors cg_priors.json ...
```

## 13. DBI con zero iterazioni

Con `--iterations 0`, il codice ora salva il JSON aggiornato prima di terminare.

## 14. Sicurezza per il runner IBI COM-only

Il runner IBI attuale crea una sola particella COM per molecola. La patch interrompe l'esecuzione se trova `site_i`, `site_j`, ecc. diversi da `-1`.

Questo non è il supporto completo ai virtual sites: impedisce soltanto di produrre tabelle fisicamente sbagliate in silenzio.

---

# Correzione strutturale ancora necessaria

## Runner IBI site-aware

Per il caso TEL22, la parte iterativa deve essere rifattorizzata. Il runner deve costruire lo stesso sistema usato in `simulation/run_cg_md.py`:

1. COM rigido per ogni molecola;
2. virtual sites con le stesse posizioni relative;
3. mappa `(mol_idx, site_idx) -> particle_id`;
4. bond/angle/dihedral applicati ai particle ID corretti;
5. stessi tipi, WCA ed exclusions;
6. stessa politica di box e PBC;
7. traiettoria che conserva COM e virtual sites;
8. analisi delle coordinate con lo stesso resolver usato per il target.

La soluzione raccomandata è estrarre la costruzione del sistema in un modulo condiviso, anziché duplicare nuovamente la logica dentro una stringa Python generata.

Finché questo refactoring non viene fatto:

- `--iterations 0` (DBI) è utilizzabile anche con siti;
- l'IBI iterativa è utilizzabile solo con priors COM-only;
- il caso TEL22 site-specific deve fermarsi con un errore esplicito.

---

# Modifiche da fare solo dopo il test di parità

## A. Bias nei layer vettoriali

In una PaiNN rigorosamente equivariante, le trasformazioni dei canali vettoriali non devono introdurre un bias cartesiano:

```cpp
linear_v = torch::nn::Linear(
    torch::nn::LinearOptions(dim, dim).bias(false)
);
linear_u = torch::nn::Linear(
    torch::nn::LinearOptions(dim, dim).bias(false)
);
```

Questa modifica cambia la struttura del checkpoint e richiede riaddestramento. Non va mischiata alla diagnosi IBI iniziale.

## B. Torque loss

Nella v1:

```cpp
float torque_weight = 0.0f;
```

Per corpi rigidi multi-sito questo disabilita la supervisione orientazionale. Rendere `torque_weight` configurabile, normalizzare separatamente force e torque loss, quindi riaddestrare.

## C. Hard-core aggiuntivo fra COM

La v1 di produzione aggiunge un WCA fra tutti i COM che la v2 funzionante non usa. Se questo termine non è incluso nell'Hamiltoniana target e nella sottrazione dei priors, va rimosso o reso esplicito/configurabile.

## D. Box, WCA ed exclusions

La politica deve essere identica in:

- preprocessing;
- DBI/IBI;
- equilibrazione;
- produzione.

In particolare non devono coesistere box hardcoded differenti, tipi shiftati di uno, o esclusioni 1-2/1-3 applicate soltanto in uno stadio.

---

# Sequenza di validazione

1. Applicare la patch e ricompilare.
2. Disabilitare IBI e usare gli stessi priors analitici della v2.
3. Confrontare su uno snapshot:

```math
E_{v1}=E_{v2},\qquad F_{v1}=F_{v2}.
```

4. Confrontare training C++ e plugin ESPResSo con gli stessi pesi e archi.
5. Eseguire finite differences di ogni prior tabulato.
6. Controllare uniformità della griglia e periodicità del dihedral.
7. Eseguire DBI-only.
8. Implementare il runner IBI site-aware.
9. Eseguire una sola iterazione IBI e verificare che l'errore delle distribuzioni diminuisca.
10. Rigenerare il dataset residuo e riaddestrare.
11. Eseguire NVE con `dt`, `dt/2`, `dt/4`; per un integratore del secondo ordine, l'ampiezza dell'errore energetico deve mostrare l'andamento atteso finché domina l'errore di discretizzazione.
