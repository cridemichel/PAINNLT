# TEL26 unfoldable con ioni impliciti — piano incrementale

Avviato il 6/10/2026. Cartella: `MLCG_Framework_v2/tutorials/tel26_unfold`. Dati AA: corse di Giulia di un singolo TEL26 ad alta temperatura, `/leonardo_work/IscrB_G4MES/GiuliaT/AllAtom`. Corse AA nostre e copie degli input: `/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold` (input in `input/`, estrazioni in `dnak/`, dataset CG in `cg/`).

Scelta di Cristiano: partire dal modello nativo TEL26 (lp2 + re1_it30) e introdurre per gradi la possibilità di unfolding, mantenendo gli ioni impliciti.

## Obiettivo finale (7/10)

Usare il modello unfoldable per **stimare la temperatura di transizione (Tm), anche in presenza di ligandi**.

Conseguenze sul progetto:
- **Serve la termodinamica di equilibrio vicino a Tm** (~330–350 K), non solo struttura e cammino a 400 K. ΔG_unf(T) va giusto in valore e in derivata (ΔH, ΔS).
- **Bersaglio di calibrazione: dati sperimentali** (Tm e ΔH da DSC/UV melting del TEL26 ibrido in K⁺, in funzione di [K⁺]). Non è garantito che il campo di forza AA dia la Tm giusta. L'AA fissa struttura, stati e cammino; l'esperimento fissa le energie libere.
- **Hamiltoniana CG dipendente da T e [K⁺]**: termini nativi con D(T) = ΔH − TΔS per famiglia, termine per il K⁺ legato nel canale. Tm si calcola in equilibrio: replica exchange in T nel CG e ripesatura multi-istogramma (MBAR con u_k(x) = β_k U(x; T_k), perché U dipende da T). Non con rampe di riscaldamento.
- **Ligandi**:
  - modello CG del ligando, con interazioni con i siti DNA tarate su simulazioni AA ligando–G4 (pose, PMF di legame, mappate e RE) e/o costanti di legame sperimentali;
  - validazione su ΔTm misurati con FRET-melting per ligandi noti;
  - riferimento analitico a due stati: 1/Tm′ − 1/Tm ≈ (R/ΔH_unf)·ln[(1 + K_F[L]) / (1 + K_U[L])];
  - il CG aggiunge legame agli intermedi, siti multipli, competizione con il K⁺, dipendenza da concentrazione.
- **Residuo ML**: non deve spostare ΔG_unf. Va limitato alla struttura locale del nativo o escluso dal modello per la Tm, e verificato con e senza.

## Fase 0 — Dati AA a 400 K (6–7/10)

Inventario: `00_inventory_aa.sh` (`MLCG_COMMIT_MSG_unfold_inventory`; output `inventory_aa.txt`). Il job si è fermato a 1 h su gmx check dell'xtc da 6 GB; i parametri sono completi.

- **Sistema**:
  - un solo TEL26 ibrido 2JPZ, stessa sequenza (TTAGGG)4TT e 843 atomi, come nel riferimento a 10 copie;
  - il nome di molecola `5mvb` viene da acpype;
  - 28 074 SPC/E e 25 K⁺: i soli controioni, nessun sale;
  - box 9,55 nm iniziale, 9,77 nm a 400 K (densità −7 %, liquido); [K⁺] ≈ 45 mM.
- **Campo di forza**: AMBER convertito con acpype. C5′ e C3′ hanno tipi atomici dedicati (CJ, C7) e le torsioni α/γ/ε/ζ hanno fasi arbitrarie (per esempio γ a 190,98/295,63/348,10°): non è parm99, con buona probabilità parmbsc1. Conferma da Giulia (leaprc usato).
- **Protocollo**: una sola temperatura, **400 K**. NPT (v-rescale 0,1 ps; Parrinello–Rahman 1 bar), dt 2 fs, vincoli su tutti i legami, PME 1,0 nm. Equilibratura 20 ns NPT Berendsen.
- **Produzione**: prod-1/2/3, **~746 ns** (obiettivo 1 µs). ~250 ns/day con 1 GPU + 32 core e PME su GPU; l'equilibratura con 8 core e PME su CPU faceva 78,7 ns/day.
- **Uscite**:
  - xtc ogni 1 ps del sistema intero (~78 GB per parte);
  - trr ogni 10 ps con posizioni e velocità, **niente forze**: force matching impossibile, RE sì;
  - solo DNA ogni 10 ps (`trj_tel26_dna_pbc.xtc`).
- **RMSD** (relativo all'inizio della produzione, quando la tetrade 3 era già aperta): guanine del nucleo 0,11 nm costante (massimo 0,24); tutto il DNA 0,35 nm (loop).

**Storia di tetradi e K⁺** (7/10). `01_extract_dna_k.sh` estrae DNA + K⁺ ogni 10 ps in `AA_unfold/dnak`; `02_tetrads_k.py` produce `tetrads_k_400K.{npz,png,log}`. Q = frazione degli 8 legami di Hoogsteen (N1–H···O6, N2–H···N7) fra guanine adiacenti, soglia 0,35 nm.

- La struttura di partenza (2JPZ senza ioni) ha Q = 1/1/1 e nessun K⁺ nel canale: gli ioni li ha messi `genion` a caso.
- Dopo 0,5 ns di equilibratura a 400 K un K⁺ (resid 28120) entra nel sito fra le tetradi 2 e 3. Tutte e tre le tetradi restano formate.
- A t ≈ −13,8 ns (6,2 ns di equilibratura) lo stesso K⁺ salta nel sito fra le tetradi 1 e 2. **Nello stesso istante la tetrade 3 (G6, G10, G18, G24) si rompe** in due coppie:
  - G18–G24 resta appaiata (N1–O6 0,86, N2–N7 0,95);
  - G6–G10 conserva solo N2–H···N7 (0,97; N1–O6 0,23);
  - G10–G18 e G24–G6 sono rotte (≤ 0,02);
  - Q_t3 = 0,38, raggio O6 0,33 nm contro 0,24.
- Lo stato resta **invariato per 746 ns**: Q_t3 0,37–0,39 in ogni blocco da 50 ns, mai sopra 0,75. Le tetradi 1 e 2 sono intatte (Q = 1,00) con il K⁺ fra loro sempre presente; nessun altro K⁺ entra nel canale.
- Lettura: senza K⁺ adiacente la tetrade 3 si apre in pochi ns a 400 K; le altre due, con il K⁺ fra loro, sono stabili su µs. **Il K⁺ del canale controlla la stabilità.**

## Pilota AA senza K⁺ nel canale (lanciato il 7/10)

- `03_make_pilot_inputs.py`:
  - dalla fine dell'equilibratura di Giulia sposta il K⁺ 28120 nel bulk, scambiandolo con un'acqua a 6,6 nm dal DNA e a 3,5 nm dagli altri K⁺;
  - scrive index (System, DNA_K), `em.mdp` e `md_rep1..4.mdp`: 400 K, seed 4002–4005, C-rescale, 200 ns, xtc DNA + K⁺ ogni 10 ps.
- `04_run_pilot.sh`: un nodo Booster, 4 copie (1 GPU + 8 core ciascuna), `module purge` prima di GROMACS 2023.3 (il venv PaiNN carica CUDA 12.2/12.6). Cartella `AA_unfold/pilot_noK400`. Job 59589552 + continuazione 59589555.
- Cosa darà:
  - se e come si apre il quadruplex senza K⁺ (ordine delle tetradi, intermedi);
  - l'insieme dello stato aperto a 400 K, per i termini locali (diedri, impilamento lungo il filamento, DH);
  - la forza di una tetrade senza K⁺ adiacente.
- Cosa non darà: popolazioni di equilibrio (unfolding forzato e probabilmente irreversibile), dipendenza da T, statistica (≤ 4 eventi).
- Budget IscrB_G4MES (6/10): 10 173 ore usate su 800 000; limite mensile ~65 750.

## Dataset CG a 400 K (7/10)

- `05_slice_copy.py`: ritaglia una copia da topologia e prior a 10 copie (lp2 → lp2_1c: 79 legami, 24 angoli, 31 diedri, esclusioni WCA; termini identici alla copia 0).
- `06_build_t400_dataset.sh` (DCGP, job 59589945): trjcat delle 3 parti DNA + K⁺, ritaglio, `build_cg_dataset.py` con i prior lp2_1c e `--allow-guard-violations`, frame ogni 10 ps → `AA_unfold/cg/tel26_lp2_1c_t400_dataset.bin`.
- **Da verificare**: il mapping del sito S delle G elenca O1P/O2P, mentre il sistema a 400 K usa OP1/OP2. Il builder salta i nomi assenti, e il job stampa i nomi dei due sistemi per il confronto.
- `system_t400.json`: configurazione a una copia per gli script 44–53 (`MLCG_SYSTEM`).

## Fase 1 — Separare i prior locali dai nativi non locali

- **Locali**, validi anche nello stato srotolato: legami, angoli, diedri di backbone (con smorzamento CBT), WCA. I diedri sono stati fittati sul nativo: andranno ricontrollati sui dati ad alta T e sullo stato aperto del pilota.
- **Nativi non locali**, che devono potersi rompere:
  - Hoogsteen B3–B3 e B2–B4;
  - impilamento B5–B5;
  - cappucci dei loop (Morse, D 50 kJ/mol ≈ 20 kT ciascuno);
  - **torsione B3–B5–B5–B3** (K 164–301 kT). La torsione è un termine legato: non si rompe mai e continua a orientare guanine non più impilate. Va modulata con una funzione di switching della distanza di impilamento.
- Verificare che tutti i Morse siano non legati (marker) e quindi rompibili senza "bond broken".
- Osservabili: Q per tetrade (come in `02_tetrads_k.py`, sui siti CG), Q dei loop, Rg, end-to-end.

## Fase 1b — Stato a 400 K

- Simulare il modello CG (lp2_1c) a kT di 400 K e confrontare con il dataset a 400 K: tetrade 3 aperta, tetradi 1–2 chiuse.
- Misura quanto sbagliano i prior a 300 K (energie libere) a un'altra T: è la prima informazione sulla dipendenza da T.

## Fase 2 — Unfolding con i soli prior (Gō-like)

- Scansione in T e/o fattore di scala λ sui termini nativi, per vedere quando e come la struttura si apre. Atteso: prima la tetrade 3 (esterna e senza K⁺), poi le altre.

## Fase 3 — Calibrazione

- Pochi parametri: scale delle famiglie di contatti nativi (Hoogsteen, impilamento, cappucci), più Debye–Hückel e il termine del K⁺ legato.
- Struttura: entropia relativa sui dati AA (nativo 300 K, stato parziale a 400 K, stato aperto del pilota). Termodinamica: Tm e ΔH sperimentali.
- La coordinazione del K⁺ nel canale è implicita ma decisiva: la forza di una tetrade dipende dall'occupazione dei siti adiacenti. Serve un termine per tetrade (o per coppia di tetradi) con dipendenza da T e da [K⁺].
- In solvente implicito i parametri sono energie libere: D(T) = ΔH − TΔS per famiglia.

## Fase 4 — Elettrostatica

- Debye–Hückel fra i fosfati (siti S), con κ dalla forza ionica (solo controioni, ~45 mM, nelle corse a 400 K). Conta nello stato srotolato: rigidità del filamento e repulsione.
- Trappole note: le esclusioni non valgono per il DH; `check_neutrality=False`; cutoff nel `cutoff_regular`.

## Fase 5 — Residuo ML

- Il residuo re1_it30 è allenato solo sul nativo e fuori dal nativo è incontrollato.
- Opzioni:
  - (a) spegnerlo con uno switching su Q;
  - (b) RE sull'insieme multi-T, sopra i prior della fase 3;
  - (c) escluderlo dal modello per la Tm.
- Criterio: il residuo non deve spostare la ΔG di unfolding. Confrontare gli istogrammi di Q con e senza residuo.

## Fase 6 — Tm nel CG

- Replica exchange in T (o Hamiltoniano) nel CG, ripesatura MBAR con U dipendente da T.
- Tm da frazione nativa = 0,5 e dal picco di C_p. Confronto con l'esperimento in funzione di [K⁺].

## Fase 7 — Ligandi

- Modello CG del ligando; parametri da AA ligando–G4 e/o costanti di legame.
- ΔTm contro FRET-melting per ligandi noti; confronto con la formula a due stati.

## Limiti da tenere presenti

- Un solo TEL26 per corsa AA: servono più copie indipendenti per avere statistica degli eventi.
- Contatti nativi sito-specifici ⇒ niente misfolding né appaiamenti non nativi. Per la Tm del TEL26 può bastare; la versione trasferibile (G–G per tipo) verrà dopo.
