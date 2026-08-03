# Audit tecnico della v1 `MLCG_Framework` con priors IBI

## Ambito e limiti

Ho confrontato staticamente i due alberi contenuti in `mlcg.zip`, seguendo il flusso

\[
\text{traiettoria atomistica}
\rightarrow \text{mapping CG}
\rightarrow \text{priors}
\rightarrow \text{dataset residuo}
\rightarrow \text{PaiNN}
\rightarrow \text{forze}
\rightarrow \text{ESPResSo}.
\]

**Nota sui nomi.** Nell'archivio ricevuto, l'albero che contiene `ibi/run_ibi_loop.py`, `ibi_priors/` e `tutorials/tel22_IBI/` è `MLCG_Framework`; `MLCG_Framework_v2` non contiene IBI. Nel report lo chiamo **albero IBI**, indipendentemente dal nome effettivo sul Mac.

Le linee citate si riferiscono ai file originali dell'archivio. Ho validato l'applicabilità delle patch e la sintassi dei file Python modificati. Non ho potuto compilare il plugin contro la specifica build locale di ESPResSo/LibTorch né eseguire il caso TEL22, perché l'archivio non contiene il runtime, le traiettorie atomistiche complete e tutti gli input di simulazione.

---

# Esito sintetico

Il problema non sembra essere un semplice tuning dell'IBI. Nei diversi stadi vengono usate **Hamiltoniane o rappresentazioni geometriche non equivalenti**. Le regressioni più importanti sono:

| Priorità | Problema | Effetto |
|---|---|---|
| P0 | Argomenti PaiNN disallineati tra training e plugin | cutoff custom applicato due volte in ESPResSo |
| P0 | IBI confronta istogrammi con Jacobiani diversi | update sistematicamente distorto |
| P0 | target IBI site-specific, runner IBI COM-only | l'IBI ottimizza coordinate diverse dal target |
| P0 | dominio delle tabelle diverso tra IBI e produzione | stesso array interpretato su griglie diverse |
| P0 | Kabsch dei corpi rigidi non eseguito | sottrazione dei priors su geometria errata |
| P0 | torque loss disattivata | orientazione dei corpi multi-sito non vincolata |
| P0 | soft-clip non lineare dell'energia totale | le forze dipendono dall'offset arbitrario dell'energia |
| P0 | box/WCA/exclusions/force cap incoerenti | cambia lo stato termodinamico tra gli stadi |
| P1 | bias nei layer vettoriali | rottura formale dell'equivarianza SO(3) |
| P1 | formula delle forze dihedrali errata | sottrazione non coerente con il potenziale |

Questi problemi sono sufficienti, anche separatamente, a spiegare perché la v1 con IBI (`MLCG_Framework`) non riproduca la stabilità della v2 senza IBI (`MLCG_Framework_v2`).

---

# Problemi bloccanti

## P0.1 — Regressione certa nel costruttore PaiNN: cutoff applicato due volte in ESPResSo

### Evidenza

Il costruttore dell'albero IBI è stato esteso:

`training/PaiNN_Architecture.hpp:73`

```cpp
PaiNNModelImpl(...,
               double cutoff = 5.0,
               bool env = false,
               bool ubias = false,
               double t_alpha = 0.1)
```

La RBF contiene già il cutoff custom:

`training/PaiNN_Architecture.hpp:86-99`

```cpp
return rbf * tox_cutoff.unsqueeze(1);
```

Se `env=true`, `PaiNNMessage` lo applica una seconda volta:

`training/PaiNN_Architecture.hpp:16-21`

```cpp
if (m_apply_envelope) {
    w = w * tox_cutoff.unsqueeze(1);
}
```

Il training passa correttamente tutti gli argomenti:

`training/train_painn.cpp:323-352`

```cpp
PaiNNModel model(..., cutoff,
                 apply_envelope,
                 use_bias,
                 toxvaerd_alpha);
```

Il plugin ESPResSo usa invece ancora la vecchia firma:

`simulation/espresso_plugin/PaiNN_ML_Potential.cpp:14-18`

```cpp
model = PaiNNModel(..., cutoff, toxvaerd_alpha);
```

Un `double` come `0.1` viene convertito implicitamente in `true` e assegnato a `env`; `t_alpha` resta al default `0.1`.

### Conseguenza

Con i default dell'albero IBI:

- training: una moltiplicazione per il cutoff;
- ESPResSo: due moltiplicazioni, quindi \(f_c(r)^2\);
- un `toxvaerd_alpha` diverso da `0.1` passato da Python non raggiunge il modello.

Per

\[
f_c(r)=\frac{x^4}{x^4+\alpha^4},\qquad x=\frac{r_c-r}{r_c},
\]

con \(r_c=1\) e \(\alpha=0.1\):

| \(r\) | \(f_c(r)\) | \(f_c(r)^2\) |
|---:|---:|---:|
| 0.90 | 0.5000 | 0.2500 |
| 0.95 | 0.05882 | 0.003460 |
| 0.99 | \(9.999\times10^{-5}\) | \(9.998\times10^{-9}\) |

È quindi una differenza sostanziale vicino al cutoff.

### Stato

Corretto nella patch meccanica propagando esplicitamente `apply_envelope`, `use_bias` e `toxvaerd_alpha` attraverso Python/Cython/C++. Anche il default di `eval_parity` è stato allineato al training.

---

## P0.2 — L'IBI confronta distribuzioni definite rispetto a misure diverse

### Evidenza

In `ibi/run_ibi_loop.py:204-235` il target viene corretto per il Jacobiano:

```python
if jacobian_type == "bond":
    hist = hist / bin_centers**2
elif jacobian_type == "angle":
    hist = hist / np.sin(bin_centers)
```

La funzione restituisce poi questo istogramma corretto come `P_target`. Durante le iterazioni, invece, `hist_sim` è l'istogramma grezzo della coordinata (`ibi/run_ibi_loop.py:670-695`).

### Perché è errato

Per il PMF iniziale è corretto rimuovere la misura geometrica, per esempio

\[
p_r(r)\propto r^2e^{-\beta U(r)},
\qquad
U_0(r)=-k_BT\log\frac{p_r(r)}{r^2}+C.
\]

L'update IBI deve però confrontare lo stesso osservabile:

\[
U_{n+1}(q)=U_n(q)+\alpha k_BT
\log\frac{p_n(q)}{p_{\rm target}(q)}.
\]

Il Jacobiano è lo stesso per target e simulazione e si cancella nel rapporto. Confrontare \(p_n(r)\) con \(p_{\rm target}(r)/r^2\) introduce un termine spurio dipendente da \(r\).

### Stato

Corretto nella patch: il Jacobiano viene usato solo per il PMF iniziale; `P_target` resta l'istogramma grezzo normalizzato esattamente come quello simulato.

---

## P0.3 — Il target IBI usa siti specifici; il runner IBI simula soltanto i centri di massa

### Evidenza

L'estrazione del target rispetta `site_i`, `site_j`, ecc.:

`ibi/run_ibi_loop.py:93-131`

```python
pos_i = frame_centers[i] if site_i == -1 else frame_sites[i][site_i]
```

Il runner temporaneo invece:

- salva soltanto i centri (`run_ibi_loop.py:481-483`);
- crea una particella per molecola (`307-320`);
- collega i termini usando gli indici delle molecole (`357-395`);
- misura bond/angle/dihedral sui COM (`642-668`).

Nel tutorial TEL22 incluso, 210 dei 390 bond sono site-specific e il mapping `DG` contiene sei siti.

### Conseguenza

L'IBI cerca di riprodurre una distribuzione tra siti virtuali simulando una coordinata tra COM. Nessun numero di iterazioni può correggere questa differenza geometrica.

### Correzione necessaria

Il runner IBI deve costruire lo stesso modello usato in produzione:

1. particella COM rigida per molecola;
2. virtual sites da `rigid_bodies_info.json`;
3. termini bonded applicati ai siti richiesti;
4. stessi tipi, WCA ed exclusions;
5. stessa box e PBC.

È una modifica strutturale e non è inclusa nella patch meccanica.

---

## P0.4 — Il tipo assegnato alla molecola multi-sito è quello dell'ultimo sito

`ibi/run_ibi_loop.py:78-87` salva `site_type` dopo il loop sui siti:

```python
frame_types.append(site_type if num_sites > 0 else 0)
```

La variabile contiene il tipo dell'ultimo sito, non del primo come afferma il commento. Per un mapping multi-sito, anche la rappresentazione COM-only usa quindi il tipo WCA sbagliato.

**Stato:** corretto nella patch, salvando esplicitamente `first_site_type`. La correzione non sostituisce il refactoring richiesto in P0.3.

---

## P0.5 — La stessa tabella viene usata su domini diversi

Per i bond, `ibi/run_ibi_loop.py:506-513` usa:

```python
bins = np.linspace(0.0, 5.0, 300)
```

Le coordinate tabulate sono quindi i centri dei bin, circa \(0.00836\ldots4.99164\). Poco dopo il JSON viene però impostato a:

```python
b["min"] = 0.01
b["max"] = 3.0
```

Il runner IBI legge gli endpoint effettivi del file (`357-363`), mentre produzione e sottrazione del prior usano i metadati JSON:

- `simulation/run_cg_md.py:206-216`;
- `preprocessing/build_cg_dataset.py:914-923`.

ESPResSo riceve gli array e li interpreta su una griglia uniforme tra `min` e `max`; quindi la stessa riga dell'array corrisponde a distanze diverse nei tre stadi.

**Stato:** la patch imposta `min` e `max` agli endpoint effettivi. Va aggiunto anche un assert che la griglia del file sia uniforme.

---

## P0.6 — I dihedrals target e simulati usano intervalli diversi

Il target viene convertito da \([ -\pi,\pi ]\) a \([0,2\pi)\) (`run_ibi_loop.py:572-575`), mentre i valori simulati restano nel dominio di `atan2`, \([ -\pi,\pi ]\), e vengono istogrammati con bin \([0,2\pi]\).

La parte negativa viene quindi persa o assegnata fuori dominio.

**Stato:** corretto nella patch con `np.mod(phi, 2.0*np.pi)` prima dell'istogramma.

---

## P0.7 — Le tabelle dihedrali non sono realmente periodiche

In modalità periodica il codice usa uno smoothing `mode="wrap"`, ma:

- la derivata è calcolata con `np.gradient` ordinario;
- `enforce_consistency_and_cap` reintegra linearmente da un minimo;
- non impone \(V(0)=V(2\pi)\) e \(F(0)=F(2\pi)\).

Riferimenti: `ibi/run_ibi_loop.py:165-255`.

Un controllo sintetico ha mostrato un salto di forza agli endpoint di circa `1.25` dopo DBI e `0.70` dopo un update IBI.

### Correzione necessaria

Usare una rappresentazione periodica coerente, preferibilmente:

- spline cubica periodica; oppure
- espansione di Fourier;
- derivazione e integrazione circolari;
- test automatico di chiusura di energia e forza.

Non incluso nella patch meccanica.

---

## P0.8 — La produzione dell'IBI viene campionata con `force_cap` attivo

Nel runner generato (`ibi/run_ibi_loop.py:430-443`):

```python
system.force_cap = 1000.0
# system.force_cap = 0  # Leave it capped just in case
```

Gli istogrammi usati dall'IBI appartengono quindi al sistema con forze troncate. Quando il cap viene tolto in produzione, l'Hamiltoniana cambia.

**Stato:** la patch conserva il cap nelle sole fasi di rilassamento e lo disattiva prima del campionamento IBI.

---

## P0.9 — Il Kabsch per ricostruire i siti rigidi non viene eseguito correttamente

`preprocessing/build_cg_dataset.py:775`:

```python
for mol_idx, (m_type, r_name) in enumerate(mol_resnames):
```

`mol_resnames` è una lista di stringhe. Una stringa di due caratteri come `"DA"` viene spacchettata in `m_type="D"`, `r_name="A"`; il lookup in `rigid_bodies_info` fallisce. Con nomi di lunghezza diversa l'unpacking può anche sollevare un'eccezione.

### Conseguenza

La geometria ideale orientata, destinata alla sottrazione dei priors site-specific, non viene ricostruita. Il dataset residuo usa quindi siti rumorosi o geometricamente diversi da quelli rigidi della simulazione ESPResSo.

**Stato:** corretto nella patch con:

```python
for mol_idx, r_name in enumerate(mol_resnames):
```

---

## P0.10 — In più punti un indice di sito viene confrontato con un tipo di sito

Nella sottrazione di angle e dihedral (`preprocessing/build_cg_dataset.py:951-962` e `1028-1045`) il codice cerca il sito così:

```python
for st, sp in frame_sites[i]:
    if st == site_i:
        pos_i = sp
```

Nel resto della pipeline `site_i` è un **indice nella lista dei siti**, mentre `st` è il **tipo chimico** del sito. I due coincidono solo accidentalmente.

Anche il passaggio iniziale che raccoglie le distribuzioni angle/dihedral usava sempre i COM (`450-465`) ignorando gli indici site-specific.

### Conseguenza

Il prior è stimato su una coordinata e sottratto su un'altra. In alcuni casi viene selezionato il sito sbagliato, in altri si ricade sul COM.

**Stato:** corretto nella patch con accesso diretto `frame_sites[mol][site_index][1]`, sia nel passaggio DBI sia nella sottrazione.

---

## P0.11 — La loss sui momenti torcenti è calcolata ma moltiplicata per zero

`training/train_painn.cpp:367-370`:

```cpp
float torque_weight = 0.0f;
```

Il codice costruisce correttamente `pred_mol_torques`, la maschera per molecole multi-sito e `loss_t`, ma la loss totale contiene sempre zero volte il termine torcente.

### Conseguenza

Per un corpo rigido multi-sito, la sola forza netta

\[
\mathbf F_m=\sum_{a\in m}\mathbf f_a
\]

non determina la distribuzione delle forze sui siti e non vincola

\[
\boldsymbol\tau_m=\sum_{a\in m}
(\mathbf r_a-\mathbf R_m)\times\mathbf f_a.
\]

Due campi di forza possono avere la stessa risultante ma torque completamente diverso. Questo è particolarmente rilevante per TEL22.

### Correzione raccomandata

Rendere `torque_weight` configurabile e normalizzare forza e torque su scale fisiche separate. Prima di riaddestrare, verificare che i torque target siano calcolati sulla stessa geometria rigida usata in simulazione.

Non incluso automaticamente nella patch, perché modifica l'obiettivo di training e richiede retraining.

---

## P0.12 — Soft-clip non lineare applicato all'energia totale aggregata

Nel plugin, `forward_with_rij` aggrega già le energie atomiche per batch (`PaiNN_Architecture.hpp:160-166`). Poiché `t_batch` è tutto zero (`PaiNN_ML_Potential.cpp:150`), `energy` ha un solo elemento: l'energia dell'intero sistema.

Il plugin applica poi (`PaiNN_ML_Potential.cpp:153-176`):

```cpp
e_capped = g(energy);
forces = -grad(e_capped.sum(), r_ij);
```

### Perché è problematico

Il force matching non fissa l'offset additivo dell'energia:

\[
E_\theta(\mathbf R)\quad\text{e}\quad E_\theta(\mathbf R)+C
\]

producono le stesse forze. Dopo un mapping non lineare \(g\), invece:

\[
-\nabla g(E+C)=-g'(E+C)\nabla E,
\]

quindi le forze dipendono dall'offset arbitrario \(C\). Inoltre il clipping è globale e non estensivo: aumentando il numero di particelle si entra più facilmente nella regione di saturazione.

La riga:

```cpp
e_capped.slice(0, 0, num_local_ml_particles)
```

non rimuove energie di ghost: il tensore ha dimensione uno, non una energia atomica per particella.

### Stato

Fornita una patch **opzionale separata** che differenzia l'energia grezza. Va applicata solo dopo un test di parità a singolo rank. La contabilizzazione corretta dell'energia MPI richiede energie atomiche o una progettazione esplicita della domain decomposition.

---

## P0.13 — La box della simulazione viene modificata silenziosamente

`simulation/run_cg_md.py:67-74` e `simulation/equilibrate.py:58-65` leggono la box dal dataset e poi impongono:

```python
min_box = 11.0
system.box_l = [max(b, min_box) for b in box_dim]
```

Il dataset incluso ha box \((2,2,2)\) nm. Portarla a \((11,11,11)\) riduce la densità di un fattore

\[
\left(\frac{11}{2}\right)^3=166.375.
\]

Questo non è un dettaglio numerico: cambia completamente lo stato termodinamico e le distribuzioni non bonded.

### Correzione raccomandata

La box deve essere quella del target, oppure il dataset deve essere esplicitamente riscalato con una procedura documentata. Se il cutoff richiede una box più grande, bisogna rivedere cutoff, PBC e sistema target; non alterare la box in modo implicito.

---

## P0.14 — WCA ed exclusion policy cambiano tra preprocessing, IBI, equilibrazione e produzione

Sono presenti policy diverse:

- preprocessing: WCA principalmente tra molecole diverse;
- runner IBI: modello COM semplificato con propria configurazione;
- `equilibrate.py`: exclusions 1–2 e 1–3;
- `run_cg_md.py`: messaggio dichiara 1–2/1–3, ma il blocco mostrato costruisce soprattutto exclusions intramolecolari;
- plugin: salta sempre coppie della stessa molecola e dipende anche dalla exclusion list di ESPResSo.

Inoltre `equilibrate.py:258-286` configurava WCA una seconda volta usando tipi traslati `1+i`, `1+j`, dopo una prima configurazione 0-based.

### Conseguenza

Il prior non bonded sottratto dal target non coincide con quello usato per generare l'IBI o con quello applicato in MD. La rete apprende quindi il residuo di un Hamiltoniano e viene eseguita dentro un altro.

**Stato:** la patch elimina la seconda configurazione WCA in `equilibrate.py` e usa il cutoff di configurazione per l'interazione dummy. Resta necessario definire una sola funzione condivisa per tipi, mixing rules ed exclusions in tutti gli stadi.

---

# Problemi importanti ma non specifici della regressione IBI

## P1.1 — I layer lineari sulle feature vettoriali hanno bias

`training/PaiNN_Architecture.hpp:37-49`:

```cpp
linear_v = torch::nn::Linear(dim, dim);
linear_u = torch::nn::Linear(dim, dim);
```

Il bias predefinito è attivo. Applicando la `Linear` al tensore `[N,3,F]`, lo stesso bias viene aggiunto a ciascuna componente cartesiana. Sotto una rotazione \(R\):

\[
L(Rv)=R v W^T+b,
\qquad
R L(v)=R v W^T+R b,
\]

che non coincidono in generale. Un bias non nullo introduce una direzione cartesiana privilegiata.

### Nota

Il problema è comune ai due alberi e non spiega da solo la differenza v1-v2. Correggerlo richiede `bias(false)` e retraining; un checkpoint esistente non è direttamente equivalente.

---

## P1.2 — La norma usata nel mixing differisce dalla PaiNN canonica

Nel mixing (`PaiNN_Architecture.hpp:50`) viene passato:

```cpp
(v_v * v_v).sum(1)
```

cioè il quadrato della norma, non la norma \(\sqrt{\sum_\alpha v_\alpha^2+\varepsilon}\) usata nella formulazione canonica. La quantità resta invariante, quindi non rompe da sola l'equivarianza, ma cambia la parametrizzazione e può peggiorare la scala numerica. È una personalizzazione da documentare e validare, non un bug certo.

---

## P1.3 — La formula delle forze dihedrali non passa il finite-difference test

`preprocessing/build_cg_dataset.py:287-308` e la formula inline `1083-1092` implementano le forze analitiche dei dihedrals.

Per un sistema generico e

\[
U(\phi)=K[1-\cos(n\phi-\phi_0)],
\]

il confronto con differenze finite centrali ha mostrato un errore massimo per componente di circa `4.30` nelle unità del test. La somma delle forze è zero, ma le singole forze hanno segni/combinazioni non corretti.

### Conseguenza

La sottrazione del prior non coincide con \(-\nabla U\), quindi il dataset residuo non è energy-consistent. Nel tutorial TEL22 incluso non risultano dihedrals attivi, ma è un bug del framework generale.

---

## P1.4 — Fallimento nel caricamento del modello: la MD continua con pesi casuali

`simulation/espresso_plugin/PaiNN_ML_Potential.cpp:20-55` intercetta l'eccezione di `torch::load`, stampa l'errore e continua. Il plugin resta quindi attivo con il modello appena inizializzato.

**Stato:** la patch rilancia l'eccezione e interrompe immediatamente la simulazione.

---

## P1.5 — `generate_residual_dataset.py` non sottrae alcun prior

Nonostante i messaggi stampati, `ibi/generate_residual_dataset.py:39-45` esegue soltanto:

```python
shutil.copy2(args.dataset, args.output)
```

Questo può essere corretto soltanto se `build_cg_dataset.py` ha già sottratto esattamente tutti i priors, inclusi quelli IBI finali. Il nome e i log dello script sono fuorvianti e possono causare una doppia assunzione sulla pipeline.

---

## P1.6 — Riproducibilità incompleta del training

Il training imposta seed PyTorch, ma lo split usa sorgenti di casualità non necessariamente allineate; inoltre batch size e alcuni pesi sono hard-coded. Per un confronto v1-v2 servono:

- seed esplicito unico;
- split salvato su file;
- stessa precisione;
- stessa permutazione dei frame;
- parametri di loss nel JSON.

---

## P1.7 — Energia e domain decomposition MPI non sono ancora definite correttamente

Il plugin costruisce un grafo comprendendo particelle locali/ghost e restituisce una sola energia aggregata. Non è possibile attribuire quella quantità ai soli atomi locali tramite slicing. Per più rank servono almeno:

- una convenzione unica di ownership degli archi;
- energie atomiche o contributi edge con ownership definita;
- comunicazione delle feature per ogni layer PaiNN, non soltanto halo iniziale;
- riduzione globale dell'energia senza doppi conteggi;
- verifica delle forze sui confini di dominio.

La patch opzionale è quindi dichiaratamente un percorso di parità **single-rank**, non una soluzione MPI completa.

---

# Osservazioni sui dati inclusi

Gli artefatti binari presenti nei due alberi sono byte-identici:

- stesso `cg_dataset.bin`;
- stesso `best_cg_model.pt`.

Il primo frame del dataset incluso contiene:

- 1001 frame totali;
- 221 molecole;
- 221 siti, cioè un sito per molecola;
- box \((2,2,2)\) nm;
- un solo tipo di sito, `0`.

Questi dati non esercitano il percorso multi-sito del tutorial TEL22. Un test che passa sul dataset incluso non valida Kabsch, torque, virtual sites o IBI site-specific.

---

# Patch prodotte

## 1. Patch meccanica a rischio contenuto

`mlcg_safe_mechanical_fixes.patch` corregge:

- firma del costruttore PaiNN e propagazione degli argomenti;
- default `apply_envelope` di parity test;
- stop immediato se il checkpoint non viene caricato;
- separazione tra istogramma osservabile e densità Jacobian-corrected;
- tipo del primo sito;
- disattivazione del force cap in produzione IBI;
- wrapping dei dihedrals simulati;
- dominio della tabella coerente con le coordinate salvate;
- protezione dei debug ID specifici di TEL22;
- loop Kabsch;
- uso corretto degli indici di sito;
- rimozione della configurazione WCA duplicata;
- uso del cutoff configurato per le interazioni dummy.

La patch è stata verificata con `git apply --check` e i file Python modificati passano `python -m py_compile`.

## 2. Patch opzionale sul soft-clip

`mlcg_optional_remove_global_energy_softclip.patch` rimuove il mapping non lineare dell'energia globale e calcola le forze dall'energia grezza. Deve essere applicata soltanto dopo:

1. parity test C++ training vs plugin;
2. esecuzione single-rank;
3. controllo del range delle forze senza force cap;
4. NVE a più timestep.

## 3. Sorgente già patchato

`MLCG_Framework_safe_fixes_source.zip` contiene una copia source-only dell'albero IBI con la patch meccanica già applicata. Non include dataset binari, checkpoint o directory di build.

---

# Ordine consigliato delle correzioni

1. **Allineare training e plugin** con la patch del costruttore.
2. Eseguire un test su uno snapshot con IBI disabilitata:
   \[
   E_{\rm training}=E_{\rm plugin},\qquad
   \mathbf F_{\rm training}=\mathbf F_{\rm plugin}.
   \]
3. Rimuovere o validare il soft-clip globale in single-rank.
4. Riscrivere il runner IBI usando COM rigidi e virtual sites reali.
5. Unificare box, WCA, exclusions, tipi e tabelle tramite un'unica configurazione condivisa.
6. Implementare tabelle periodiche energy/force-consistent.
7. Correggere e testare le forze dihedrali.
8. Attivare una torque loss normalizzata e riaddestrare.
9. Valutare `bias(false)` nei layer vettoriali e riaddestrare un modello realmente equivarante.
10. Solo alla fine ripetere IBI e test NVE.

---

# Test minimi di accettazione

## Parità numerica

Per lo stesso snapshot, checkpoint e neighbor list:

- energia C++ training vs ESPResSo;
- forze per particella;
- valori di cutoff/RBF per ogni edge;
- output dopo ogni message/update block.

Usare inizialmente `float64`, single-rank e nessun force cap.

## Priors

Per ogni bond, angle e dihedral:

\[
\mathbf F^{\rm analytic}
\stackrel{?}{=}
-\nabla_{\mathbf R} U
\]

con finite differences centrali. Controllare anche continuità ai nodi della tabella e ai bordi periodici.

## Simmetrie

- invarianza traslazionale dell'energia;
- equivarianza rotazionale delle forze;
- \(\sum_i\mathbf F_i\approx0\);
- \(\sum_i(\mathbf r_i-\mathbf R_{\rm COM})\times\mathbf F_i\approx0\) per sistemi isolati.

## IBI

A ogni iterazione salvare e confrontare, con gli stessi bin:

- istogramma grezzo target;
- istogramma grezzo simulato;
- PMF iniziale Jacobian-corrected;
- update \(\Delta U\);
- distanza KL/Jensen–Shannon per ogni coordinata;
- percentuale di campioni fuori dominio;
- frequenza di attivazione del cap.

## Dinamica

Eseguire NVE con \(\Delta t\), \(\Delta t/2\), \(\Delta t/4\). Per un integratore del secondo ordine, le fluttuazioni devono scalare approssimativamente come \(\Delta t^2\), senza drift lineare sistematico. Il test va fatto separatamente con:

1. soli priors;
2. sola PaiNN;
3. priors + PaiNN;
4. IBI + PaiNN.

---

# Conclusione

Il primo bug da correggere è la firma del costruttore: produce una differenza certa tra il modello addestrato e quello eseguito in ESPResSo. Subito dopo, il runner IBI va reso geometricamente identico alla simulazione di produzione. Senza questi due interventi, qualunque tuning di smoothing, damping o numero di iterazioni IBI ottimizza un sistema diverso da quello finale e non può fornire una diagnosi affidabile.
