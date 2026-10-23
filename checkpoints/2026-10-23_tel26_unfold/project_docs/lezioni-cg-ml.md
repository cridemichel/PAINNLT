# Metodologia per un nuovo modello CG-ML — ricetta validata sul TEL26

Aggiornato al 5 ottobre 2026, dal lavoro su MLCG_Framework_v2 per il TEL26 ibrido (2JPZ) su Leonardo.

**Stato: ricetta validata.** Il TEL26 è stato portato da zero a un modello che:
- riproduce la struttura AA: P(r) intra 0,985 contro un tetto di 0,9955; nucleo e tetradi al livello AA, FES (rmsd_core, Q) al tetto, RMSF delle tetradi 0,97–1,00×;
- è stazionario su 2 × 20 ns;
- è **28–140× più efficiente dell'AA per GPU**, a seconda dell'osservabile (rmsd_core il minimo; traslazione e siti del nucleo il massimo).

Limite dichiarato: i loop non hanno gli stati metastabili dell'AA, ma il riferimento AA stesso non è convergente sui loop.

Questa pagina dice cosa fare, in che ordine e con quali script, per il prossimo sistema. La storia di come ci si è arrivati è in fondo.

**Tempi attesi.**
- Un altro G-quadruplex con lo stesso mapping: **da qualche giorno a due settimane**, quasi tutte di calcolo (catena RE, validazione). Il TEL26 ha richiesto settimane perché ogni passo era una domanda aperta.
- Un sistema qualitativamente diverso (duplex, proteina): più tempo, perché i prior fisici vanno ripensati. L'infrastruttura resta la stessa.

---

## Passo 0 — La domanda e i dati all-atom (prima che partano le produzioni AA)

1. **Il CG ha senso solo se è almeno 10–100× più efficiente dell'AA** sugli osservabili che contano.
   - Per struttura locale e dinamica su ns–µs di un sistema di queste dimensioni basta l'AA.
   - Il CG ripaga su scale che l'AA non raggiunge: unfolding o folding (µs–ms), aggregazione di molte copie, sistemi molto più grandi.
   - Scrivere subito quali osservabili decidono: struttura del nucleo, loop, trasporto, contatti fra copie.
2. **Chiedere i dati AA giusti:**
   - **forze e posizioni nello stesso file, ogni ~1 ps** (`nstxout = nstfout = 500` a dt 2 fs). Con forze ogni 20 ps il residuo è al ~99 % rumore;
   - **almeno 10 copie**, e corse abbastanza lunghe da far rilassare i gradi di libertà lenti. Sul TEL26 i loop hanno τ 4–10 ns e metà della varianza fra copie già in 136 ns: per i loop servirebbero più copie, corse più lunghe o un campionamento accelerato;
   - ns/day dell'AA su 1 GPU con tutti i core del nodo e con 1/4 del nodo, per il confronto per GPU e per budget.
3. **Misurare il tetto del riferimento**: metà della traiettoria contro l'altra metà (`--ref-range 0:0.5` con `@ref:0.5:1`). Sotto il tetto nessun modello può fare meglio. Senza tetto "accurato" non ha soglia.

## Passo 1 — Configurazione del sistema (`system.json`)

Tutto ciò che dipende dal sistema sta in un file JSON nella cartella del sistema, letto da `tutorials/system_config.py`. La ricerca segue quest'ordine: `MLCG_SYSTEM`, poi la cartella corrente, poi la cartella dello script. Non ci sono valori di ripiego nel codice: senza file lo script si ferma.

Campi:
- `name`, `sequence` (una copia), `copies`;
- `sites` (nomi dei siti per tipo di residuo, nell'ordine del dataset);
- `tetrads` (residui 1-based, dal basso verso l'alto della pila), `outer_tetrads`, `hoogsteen_site`;
- `loop_groups`, `contact_pairs`;
- `file_prefix`, `reference_dataset`;
- `aa` (`frame_dt_ps`, `ns_day`), `md` (`dt_ps`, `gamma`, `kT`), `production` (`model`, `prior_set`).

Esempio completo: `tutorials/tel26/system.json`.

Gli script di analisi 46–52 e `tune_state_contacts.py` leggono già da qui sequenza, tetradi, sito di Hoogsteen, gruppi dei loop, coppie di contatto e default (`--nuc`, `--aa-dt`, `kT`). I loader validano la configurazione: tetradi fatte di G, gruppi dentro la sequenza, siti dichiarati per ogni tipo di residuo.

**Da generalizzare ancora:** i nomi `tel26_*` negli script di stadio (`_prior_set.sh`, `05_run_espresso.sh`, `06_relent.sh`) e negli script di fit dei prior. Il prossimo sistema copia la cartella `tel26` e adatta questi nomi, finché non leggono `file_prefix`.

## Passo 2 — Mapping e topologia

- **Siti:** G con 6 siti (S, B1–B5); A e T con un sito solo (S). Un corpo rigido per residuo.
- **Registro delle tetradi** con tre verifiche indipendenti. Attenzione alla PBC nelle distanze (trappola nota: rovina le distanze ma non il registro).
- **Nocciolo repulsivo per ogni coppia**, anche quelle legate. Due timine consecutive senza WCA sono collassate a 0,02 nm.

## Passo 3 — Prior fisici, in quest'ordine

Solo termini che corrispondono a interazioni fisiche. Le reti elastiche migliorano la g(r) ma sovravincolano (84 vincoli su 66 gradi di libertà del nucleo). I parametri si stimano dal riferimento, classe per classe e mediando sulle copie.

| | termine | script | effetto misurato sul TEL26 (P(r) intra) |
|---|---|---|---|
| a | legami e angoli (Boltzmann), WCA | `build_cg_dataset.py` | base |
| b | Hoogsteen B3–B3 nel piano (Morse sito-sito) | `fit_tetrad_site_morse.py` | B3–B3 0,13 → 0,72 |
| c | impilamento B5–B5 fra guanine sovrapposte (`tract`) e torsione B3–B5–B5–B3 | `--stacking --twist` | B3 e B5 +0,03 |
| d | diedri di backbone sul sito 0, **con smorzamento angolare (CBT)** | `fit_backbone_dihedrals.py`, `apply_cbt.py` | 0,944 → 0,958; loop a posto |
| e | secondo legame di Hoogsteen N2–H···N7 (B2–B4) | `fit_hoogsteen_hbond.py` | 0,958 → 0,971 |
| f | cappucci persistenti delle basi dei loop | `find_loop_contacts.py --copy-tol … --max-per-base 3 --face-tetrads …` | → 0,976 (lp1/lp2) |
| g | (solo se giustificati) contatti di stato deboli | `add_state_contacts.py`, `tune_state_contacts.py` | vedi sotto |

- **Il diedro di backbone va sempre smorzato.** Con tre siti quasi allineati il diedro a coseno dà forze ~K/sin θ: è la causa di tutte le "esplosioni" del ML e del "bond broken". Rimedio: g(θ) = 1 per sin θ ≥ 0,3, poi raccordo; non il sin³ di Bulacu.
- **Ogni prior nuovo con un test runtime ↔ dataset alla precisione di macchina** (`compare_prior_parity.py`, `diagnose_*.py`). Una convenzione diversa non dà errori: dà fisica sbagliata.
- **Le varianti per le sole corse con i soli prior** si fanno con `derive_prior_set.py --base <set> --set <nuovo>`, senza ricostruire il dataset.
- **Contatti di stato.** Per contatti presenti solo in parte delle copie o del tempo, `find_loop_contacts.py` li esclude per costruzione (`--copy-tol`). `add_state_contacts.py` li aggiunge come Morse stretti, con D piccolo regolato sulla popolazione AA:

  D ← D + kT·[logit f_AA − logit f_CG]

  La regolazione si ferma quando il contatto entra nell'errore AA. Con `ALLOW_ML_ON_DERIVED_PRIORS=1` il residuo RE gira sopra l'insieme derivato.

  Con un solo sito per T e A gli appaiamenti di registro (T1·A15 contro T2·A15) non si separano con contatti fra coppie: sul TEL26 rafforzare T1–A15 ha trascinato anche T2 (T2–A15 0,55 → 0,61, AA 0,24). Se i loop contano, valutare più siti per le basi dei loop.

  **Usarli solo se la popolazione AA è ben determinata e il CG ne dista più di ~2σ.** Sul TEL26 (T1–A15, T1–A21, A9–G18) il modello senza contatti era già entro l'errore AA, che è ±0,1 con 10 copie non ergodiche. I contatti spostavano le popolazioni solo dentro le barre d'errore: lasciati da parte.

## Passo 4 — Verifica con i soli prior e iterazione

- **Il giudice è la P(r) intra per canale e per coppia di tipi** (`44_tel22_rdf.py --types --all-pairs`), sulla seconda metà di corse termostatate, rispetto al tetto. L'inter con 10 copie e 50–100 ps non si giudica: il riferimento contro sé stesso dà solo 0,85.
- **Iterare i parametri dei contatti sulla dinamica completa** (`iterate_contacts.py`, stadio `iterate`, catene in parallelo con `AFTER`). Converge in 2–3 iterazioni. Il fit su una sola distanza ignora gli altri termini (doppio conteggio).
- **Leggere gli scarti.** Un termine fisico mancante si vede come scarto concentrato su un gruppo di coppie di tipi. Termini locali a convergenza non garantiscono la g(r): controllare le coppie fra basi **non** a contatto, cioè l'orientazione relativa.
- **Prima la sopravvivenza** (`E_kin`, `max_f`, `min_dist`), poi la struttura.

## Passo 5 — Residuo ML a entropia relativa, non force matching

- **Con prior già tarati sulla struttura il force matching può solo peggiorarla.** Sul TEL26 andava da 0,976 a 0,934–0,940 su cinque insiemi di prior, e aggregava le copie.
- **Perché qui fallisce e altrove funziona.** CGnet e CGSchNet hanno ~10⁶ frame, prior deboli e un giudizio su 1–2 coordinate collettive. Qui ~7 000 frame al 99 % di rumore e la g(r) sito-sito come giudice.
- **Entropia relativa** (`training/train_relent.cpp`, `06_relent.sh`):
  - partenza da correzione nulla (`--zero-init`), così la prima corsa riproduce i soli prior;
  - regione di fiducia ESS/N ≥ 0,5 con passo adattivo: senza, si ha una falsa convergenza;
  - ripesatura per copia (`RE_RESIDUES_PER_COPY`);
  - profilo MD `fast` (dt 4 fs, verlet, grafo PaiNN sul device), 9,3× più veloce del `legacy` con la stessa struttura; γ 20 per avere campioni canonici;
  - ~30 iterazioni. Sul TEL26: P(r) intra 0,976 → 0,985, B3–B3 0,837 → 0,911, rmsd_core 0,102 → 0,077 nm.
- **Validare "alla pari":** stesso stato iniziale, stessa finestra, stesso protocollo. Il totale su tratti brevi deriva anche a modello fisso.
- La rete d64 a 2 strati basta. Reti più larghe o con cutoff più lungo non hanno aiutato.

## Passo 6 — Produzione

- **Default** (`05_run_espresso.sh`): dt 4 fs, γ 2, liste di Verlet, grafo PaiNN sul device, energia completa ogni 10 registrazioni, `LOG_INTERVAL=250` (1 ps).
- **Due repliche per GPU** (`run_replicas_per_gpu.sh`): +50 % di produzione aggregata, stessa fisica. MPS non serve, 4 repliche non convengono.
- **Produzioni lunghe** a segmenti con dipendenze: `chain_replicas.sh` (`PREFIX`, `START_FROM`, `NSEG`, `NS_PER_SEG`, `DRY_RUN=1`).
- **Numeri TEL26** (860 bead, A100): 10,3 ms/passo con una corsa sola; **54 ns/day aggregati** con 2 repliche e un frame ogni ps. Variabilità fra nodi ~20 %.
- **Ripresa da checkpoint con un insieme di prior diverso:** `--allow_checkpoint_mismatch`. Con più marker virtuali vengono ripristinati solo i corpi rigidi, per id.

## Passo 7 — Validazione contro l'AA

| cosa | script | criterio |
|---|---|---|
| struttura della copia | `46_tel26_structure.py` | rmsd_core, Q, Rg, FES (rmsd_core, Q), RMSF per regione e per residuo, rispetto al tetto; primi contro ultimi 10 ns per la stazionarietà |
| tempi di rilassamento ed efficienza | `47_tel26_timescales.py` | α = τ_AA/τ_CG per osservabile; efficienza per GPU = ns/day_CG · α / ns/day_AA |
| trasporto per ora di GPU | `48_tel26_msd_wall.py` | MSD traslazionale e angolare per ora di GPU contro AA |
| loop | `49`–`52` | coda lunga delle ACF e varianza fra copie; stati (PCA + k-means), strutture rappresentative e PDB; contatti per copia con errore |

- **Efficienza: due numeri.**
  - Di **trasporto**: MSD per ora di GPU; per la traslazione coincide con α × rapporto dei throughput.
  - Di **campionamento**: dai τ di decorrelazione.

  Il numero principale è il **minimo fra gli osservabili confrontabili**. Con γ basso la rotazione è inerziale e l'MSD angolare sovrastima di ~7×. La traslazione CG è fissata in buona parte dal termostato (D ≈ kT/(Nγ)).
- **Un α grande non è sempre un guadagno.** Sul TEL26, su Rg e loop si ha α ~2 400, ma i tempi a 1/e misurano processi diversi: rilassamento lento nell'AA, fluttuazioni di ps in un solo bacino nel CG. Prima di usare un α controllare con lo script 49 che l'AA non abbia una componente lenta assente nel CG.
- **Campionamento per i tempi.** Con frame ogni 1 ps i τ sotto ~1 ps non sono risolti: per i tempi del nucleo usare corse con frame ogni 0,5 ps.

---

## Dati all-atom: dettagli

- **Forze più frequenti aiutano contro il rumore, non contro la copertura.** A 1 ps la configurazione CG è quasi la stessa, ma la fluttuazione δF dei gradi di libertà eliminati si decorrela in ~0,1–1 ps; mediare ~20 campioni riduce la varianza del bersaglio di ~20×. Prima di chiedere produzioni lunghe:
  1. verificare che il limite sia il rumore (esperimento con metà dei dati);
  2. misurare il tempo di correlazione di δF con una corsa corta a `nstfout` di pochi passi.
- **Copie non ergodiche** (lezione dai loop del TEL26): popolazioni stimate su poche copie bloccate in stati diversi hanno errori di ±0,1–0,2. Va misurata la varianza fra copie (script 49) prima di usare una popolazione come target.

## Criteri di valutazione: dettagli

- **Mai l'errore sulle forze né la skill del trainer.** Sul TEL22 le metriche sulle forze ordinavano i modelli **al contrario** della struttura.
- **Checkpoint scelti sulla dinamica** (g(r), stabilità), non sulla loss di validazione.
- **Distanze fra basi a contatto giuste non bastano:** controllare le coppie fra basi non a contatto.

## Trappole tecniche, da controllare subito

- **Termostato:** in ESPResSo γ è un attrito e τ = m/γ. Con masse di ~300 amu γ = 1 dà τ ≈ 300 ps: su corse brevi il termostato non agisce. γ ≈ 20 per l'equilibrazione e per i campioni canonici dell'RE; γ 2 in produzione, perché la struttura è invariata e i moti sono più rapidi.
- **Elettrostatica in ESPResSo:**
  - le esclusioni di particella **non** si applicano al Coulomb/DH;
  - serve `check_neutrality=False` con controioni impliciti;
  - il cutoff DH va nel `cutoff_regular` della decomposizione ibrida.
- **Ogni variante di prior ha il suo dataset** (il residuo dipende dai prior). Per le corse con i soli prior il dataset non serve.
- **Nomi derivati dal modello e dall'insieme** per tutti i file di corsa. File fissi condivisi fra job paralleli si mescolano in silenzio.
- **Leonardo:**
  - sul login `python3` è quello di sistema (3.6): prima `source hpc/env_leonardo.sh`;
  - `pypresso` sul login vuole `LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6)`;
  - la copia di lavoro è in `/leonardo_work/...`, non sotto `$HOME`;
  - `sbatch` va lanciato da una shell in cui le variabili usate nel `--wrap` (`$T`, `$S`) sono definite;
  - non ricompilare mentre girano job che usano `pypresso`.
- **Diedri a coseno in ESPResSo:** con tre siti quasi allineati ESPResSo ferma la corsa con "bond broken" (`install_dihedral_undefined_zero.py`, poi CBT). "bond partner not found" è invece un partner fuori dalla cella: leggere il messaggio esatto prima di scegliere la correzione.
- **Leggere la diagnosi dei crash prima di curare** (`crash_checkpoint.npz`, `diagnose_crash.py`). Due volte la cura ipotizzata era sbagliata.
- **Due job nella stessa cartella non devono scrivere file a nome fisso:** lock o nomi derivati dal job.
- **Filo conduttore dei difetti:** ognuno emergeva uno stadio dopo quello in cui era stato introdotto. Controllare l'output di ogni stadio prima di lanciare il successivo.

---

## Storia delle scoperte (TEL26, settembre–ottobre 2026)

Per il dettaglio numerico: `pipeline-tel26.md` (prior e RE) ed `efficienza-cg-aa.md` (prestazioni, efficienza, loop).

- **Termostato:** con γ = 1 gli stati ML uscivano a 1,7–2,2 volte l'equipartizione. Sembrava instabilità del modello.
- **Prior sito-sito:** i Morse sito-sito sulle B3 al posto dei COM–COM portano B3–B3 intra da 0,13 a 0,72.
- **Termini locali a convergenza non danno la g(r):** contatti e torsioni riprodotti classe per classe portano la P(r) intra solo da 0,938 a 0,944. Lo scarto residuo era nei loop e nell'impilamento oltre i primi vicini.
- **Diedri di backbone:** 0,944 → 0,958, loop a posto. Iterarli non serve.
- **Secondo legame di Hoogsteen** (B2–B4): 0,958 → 0,971. Costo: corsa ~2× più lenta allora (marker da ~240 a ~480), irrilevante dopo le ottimizzazioni.
- **Esplosioni del ML:** dovute al diedro a coseno singolare, raggiunto raddrizzando i loop. Corrette con lo smorzamento angolare (lp1).
- **Force matching:** peggiora la struttura su cinque insiemi di prior. Escluse come cause la quantità di dati, le coppie del corpo rigido, la parità prior runtime/dataset e il raggio della rete. Causa: base incompleta e segnale al 99 % rumore.
- **Entropia relativa:** il primo residuo ML che migliora la struttura (28/9). Il criterio intra 0,985 è raggiunto con it30 (29/9); catena re1 ferma per convergenza a it33.
- **Prestazioni** (1–4/10), sempre con la stessa Hamiltoniana in FP32:
  - grafo PaiNN sul device (23 → 6,7 ms);
  - liste di Verlet (resto dello step 14 → 6 ms);
  - log vettoriale ed energia ogni K registrazioni;
  - fusione della MLP a livello di nodo (forward −37 %);
  - due repliche per GPU.

  Da 1,4 a 54 ns/day aggregati. Il CUDA graph è corretto ma più lento: lasciato spento.
- **Efficienza** (3–4/10): 2 × 20 ns, per GPU 28× (rmsd_core) – 140× (traslazione, nucleo). Indicatore MSD per ora di GPU, con la distinzione fra trasporto e campionamento.
- **Loop** (4–5/10):
  - nessun processo lento nel CG (script 49);
  - stati AA ben riprodotti dove l'AA è ergodico (loop 1, coda 3'); assenti o diffusi dove non lo è (loop 2, loop 3, coda 5');
  - contatti di stato: il modello era già entro l'errore AA;
  - conclusione: per migliorare i loop serve prima un riferimento AA convergente.
