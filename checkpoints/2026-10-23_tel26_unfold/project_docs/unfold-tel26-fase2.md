# TEL26 unfoldable — risultati fasi 1b–2 (7–23/10/2026)

Seguito di `claude/unfold-tel26.md`. Script in `MLCG_Framework_v2/tutorials/tel26_unfold`, dati in `/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold` (spostati il 16/10 da IscrB_G4MES per la quota; il vecchio percorso `/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold` è un symlink).

## Infrastruttura

- **Dataset t400**: `build_cg_dataset.py --positions-only` (il trr non ha forze; richiede `--priors`, implica `--allow-guard-violations`). → `cg/tel26_lp2_1c_t400_dataset.bin`, 74 580 frame.
- **WCA a 400 K**: l'AA scende sotto il guard dei prior a 300 K in 12 coppie di tipi (es. 0_4: 0,258 contro 0,302 nm). Per il modello unfoldable il WCA va rifittato su 300 + 400 K.
- **Plugin ricompilato con le teste termodinamiche.** `bootstrap_leonardo.sh` con STEP=build riallinea i sorgenti del plugin. Parity: E_ML legacy identico al bit, forze entro 2e-4; 34 ns/giorno a 10 copie.
- **Prior dipendenti da T** (8/10): `espresso_interactions.resolve_thermal_priors`, chiamata da `run_cg_md.py` ed `equilibrate.py`. Morse con `D_H`/`D_S` → D = max(D_H − T·D_S, 0) alla T del termostato; diedri con `k_H`/`k_S` idem. Nei file termici D = null. Nell'entropia relativa i prior li legge solo `run_cg_md` (06_relent.sh, kT 2,49 → 299,5 K).
- **`train_relent --thermo-scale κ`** (16/10): copia readout → readout_dT e moltiplica l'ultimo Linear per −κ, quindi U_ML(T) = U_ML(T0)·[1 − κ(T−T0)/T0] (zero a T0(1+1/κ)). Richiede config thermo, `--in`, `--steps 0`. I modelli salvati su GPU vanno creati in un job GPU (sul login: "no NVIDIA driver").
- **Compilazione su Leonardo**: `source MLCG_Framework_v2/hpc/env_leonardo.sh && cmake --build MLCG_Framework_v2/training/build --target train_relent -j 8`. Senza l'ambiente, `python3` è quello di sistema (3.6) e gli script con `from __future__ import annotations` falliscono.
- **Script**:
  - `02_tetrads_k.py`: Q per tetrade, cationi nel canale; distanza minima catione–O6 del core (`k_o6_min`), stati F/I/U con Q mediato su 1 ns, riga `[riassunto]`;
  - `03_make_pilot_inputs.py`: input del pilota AA; `--cation Li`; `--exclude-r R` = restraint di tipo 10 (`[ intermolecular_interactions ]`, armonico per r < R, k 10 000 kJ/mol/nm²) fra ogni catione e ciascuno dei 12 O6 del core (300 termini). Il pull code non va: grompp rifiuta distanze "distance" > 0,49 × scatola;
  - `04_run_pilot.sh`: 4 copie su un nodo Booster;
  - `08_tetrad_melt.py`: q per tetrade, P(n tetradi intatte); `--smooth-ns` media q nel tempo prima della soglia;
  - `09_scale_contacts.py`: per classe (tet, oth, twist) `--lam-*` oppure `--h-* --t0-*` con D(T) = h·D0·(1 − T/T0); torsioni CBT;
  - `10_contact_sweep.sh`, `11_fit_thermal.py`: griglia (λ_tet, λ_oth, T) e fit interpolato;
  - `12_pilot_tetrads.sh`: analisi del pilota AA. Con i restraint intermolecolari `trjconv -pbc mol` fallisce anche dal tpr senza restraint (causa non chiarita); rimedio: `dnak.tpr` di `pilot_noK400` (stessi 868 atomi), poi 02 a mano;
  - `13_validate_thermal.sh` + `14_thermal_check.py`: validazione diretta di D(T), `SMOOTH_NS`, ΔH di van 't Hoff da ln(P_U/P_F); 14 accetta mappa f o f1 (17: variabile `MAP`);
  - `15_thermal_prior_set.sh`: insieme `lp2th` a 10 copie (cg_priors.lp2th.json, parametri f; link a topologia/rb_info/dataset di lp2, dataset valido solo per le posizioni);
  - `16_check_relent.sh`: produzione a 300 K (10 copie, PaiNN + soli prior) e confronto con l'AA (53, 46, 08);
  - `17_melt_painn.sh`: curva di fusione con PaiNN attivo, una copia per corsa, corse distribuite sulle GPU; `CONTINUE=1` prosegue (T<T>_b);
  - `18_kappa_scan.sh`: crea `tel26_lp2th_th0_it60_k<κ>.pt` (in un job GPU) e con SUBMIT=1 lancia 17 per ogni κ;
  - `19_pop_labels.py`, `19_relent_pop.sh`: entropia relativa + vincolo sulle popolazioni (sezione sotto);
  - `20_intermediates.py`: stati per numero di tetradi con core set, permanenze, frazione pulita, mappa −ln P(q2, (q1+q3)/2);
  - `21_gtracts.py`: parametri per tratto G (c_a, Qr_a), regioni F / tripla elica / U, mappa −ln P(c*, Qr*).

## Risultati

- **CG lp2 soli prior**: nessuna apertura fino a 600 K (Gō-like: ΣD = 2700 kJ/mol per copia contro ΔH di unfolding 352).
- **sweep2 / sweep2u**: a λ costante transizione larga 60–80 K → D(T).
- **Fit interpolato** (11): f1 e f degeneri e al bordo della griglia.
- **Validazione diretta** (13): con q per frame la divisione F/I a bassa T è sfrangiatura di 5 ps; con q mediato su 0,5–1 ns il CG è **a due stati** (F ≈ 1 fino a 315–320 K, U ≈ 1 da 335 K). U(T) con 1 ns: f 0,06 / 0,26 / 0,79 / 0,99 a 320 / 325 / 330 / 335 K, esperimento 0,11 / 0,32 / 0,58 / 0,79; Tm 327,3 (f), 329,0 (f1) contro 328,4 K; ΔH_vH ~430 kJ/mol contro 352. Nessun intermedio stabile (atteso).
- **Decisione (14/10)**: parametri **f** (h_tet 0,093, T0_tet 2560 K, h_oth 1,094, T0_oth 373,7 K) come base termica.
- **Piloti AA a 400 K**: senza K⁺ nel canale un K⁺ della soluzione rientra; con Li⁺ lo ione convertito nel canale non esce.
- **Pilota `pilot_excl400`** (canale vietato, R = 0,4 nm): **primo unfolding AA**. Il crollo della tetrade 1, ultima a cedere, porta a U a ~23 / 35 / 83 ns in tre repliche; la quarta resta I (tetrade 1 intatta, la 2 si riforma). Stato U compatto (Rg 1,3–1,4 nm contro 1,1 del nativo, atomi pesanti; Hoogsteen residui Q 0,1–0,35): **non è un gomitolo esteso** (segnalato da Giulia il 23/10; "U" = nessuna tetrade con Q ≥ 0,5). Ordine di perdita AA 3 → 2 → 1. Togliere lo ione allarga subito le tetradi (raggio O6 0,28 / 0,33–0,40 contro 0,234 / 0,239 nm con K⁺). Non è equilibrio: riferimento strutturale dello stato U, non popolazioni. Note per Giulia (16/10): `tutorials/tel26_unfold/results/TEL26_AA_unfold_pilot_runs.pdf` (cartelle) e `TEL26_pilot_excl400_unfolding.pdf` (dettaglio).
- **Entropia relativa `th0` sopra `lp2th`** (14/10, 60 iterazioni da re1_it30): it1–30 con 25 ps e un gruppo di pesi (passi limitati dall'ESS, lr ~1e-5); it31–60 con `RE_RESIDUES_PER_COPY=26`, `MD_PS=50` (30–120 passi per iterazione a it31–41, poi guadagni 0–2 nat: vicino alla convergenza). ~2–3,5 min per iterazione.
- **Controllo a 300 K** (16, 2 ns × 10 copie, scarto 200 ps). Sovrapposizioni P(r) intra (tutti / B3–B3 / B5–B5 / S–S):
  - th0_it60: 0,9941 / 0,9828 / 0,9752 / 0,9572; th0_it30: 0,9915 / 0,966 / 0,932 / 0,925;
  - soli prior lp2th: 0,929 / 0,646 / 0,805 / 0,846 (molto peggio di lp2: 0,980 / 0,841 / 0,916 / 0,924);
  - tetto AA: 0,9955 / 0,979 / 0,971 / 0,961; vecchio lp2 + re1_it30: 0,9945 / 0,976 / 0,967 / 0,958;
  - tetradi (q mediato su 1 ns): AA 0,899 / 0,911 / 0,909, it60 0,891 / 0,912 / 0,906, soli prior 0,650 / 0,676 / 0,655; Rg 0,798 in entrambi.
  - **it60 sopra i prior termici riproduce la struttura nativa al livello del tetto AA**, come il modello di produzione su lp2, ma su prior che possono fondere.
- **Fusione con PaiNN it60** (17, 15/10, 25 ns): F a 300–365 K, a 380 K F 0,25 / I 0,75, nessun U. PaiNN allenato a 300 K sovrastabilizza F (l'entropia relativa a 300 K non vincola ΔG_FU).
- **Scansione κ** (18 + 17, 16–19/10; T 315 / 330 / 345 K, rami chiuso/aperto, 25 ns):
  - κ2: F ovunque; κ4: F ovunque (a 345 K q 0,62–0,73); κ6: F a 315 e 330 K, U a 345 K in entrambi i rami (Rg 1,57 contro 0,80 nativo, CG), nessuna isteresi → Tm fra 330 e 345 K.
  - E_ml/s = U_ML(T0) (media sulla seconda metà): F stretto ~−490 (315 K) fino a F allentato −275 (κ4, 345 K); U −40 (κ6, 345 K). Gap di energia PaiNN F→U 300–450 kJ/mol; dentro F l'allentamento costa già ~200.
  - Dagli esiti (prior f: ΔH_p ~430, Tm_p 327 K) lo spostamento in energia libera dovuto a PaiNN è ~60–230 kJ/mol, contro ΔG_FU(300 K) sperimentale ~30 (prior soli ~36).
  - **Conclusione**: con un solo κ non si ottengono Tm e larghezza insieme. PaiNN nella regione U è un'estrapolazione: l'entropia relativa a 300 K non vede mai U.

## Entropia relativa + vincolo sulle popolazioni (opzione A, 19–23/10)

- **`train_relent --pop CG.bin,ETICHETTE,T,L*[,λ]`** (ripetibile): L = ln(P_stato/P_resto) per ripesatura dei campioni di θ0 con p per frame (0/1 o morbida); penalità λ·h(L − L*), h quadratica fino a |d| = `--pop-huber` (2) e poi lineare; gradiente −β(⟨∂U/∂θ⟩_wU − ⟨∂U/∂θ⟩_wN); ESS per stato relativa a θ0; modello scelto su dS_holdout + Σ(P − P0). `--check-gradient` include il termine (errore ~1e-11 su dati sintetici e su quelli veri). Senza `--pop` nulla cambia. Con AdamW conta solo il peso relativo fra entropia relativa e vincolo.
- **`19_pop_labels.py`**: campioni a una T → dataset binario + etichette e bersaglio sperimentale (Buscaglia). `--state U|F` (F nella mappa `--map f1` = F + I1, default); `--order tetrad` (q per tetrade, soglia `--qcut` 0,5) oppure `--order tract` (U = tutti i c_a < `--tract-cut` 0,2, solo U); `--delta` larghezza delle sigmoidi (0,05 troppo larga vicino alla soglia: ⟨p_F⟩ 0,36 contro P_F netta 0,60; usare 0,02).
- **`19_relent_pop.sh`**: catena su un nodo Booster (4 GPU). Per iterazione: MD a 300 K (lp2th, 10 copie, 50 ps), a ogni T di POP_TS ∪ POP_F_TS due corse a una copia (lignaggi F e U, insieme f), etichette, train_relent --pop. Variabili: POP_TS, POP_F_TS, POP_F_MAP, POP_ORDER, POP_TRACT_CUT, POP_NS, POP_SKIP_PS, POP_DELTA, POP_LAMBDA, RE_ESS_MIN, START_MODEL, START_RE, START_F, START_U, TAG. Riprende da sola; T aggiunte a catena avviata partono da START_F / START_U. Uscite in `$A/relent_pop/<TAG>/`.
- **Catena pop1** (da k6; it01–it45, job 61360433, 61390905, 61512785, 61512786): da it38 le P_U (soglia per tetrade) oscillano attorno al bersaglio.
- **Validazione di pop1_it45** (23/10):
  - **300 K (16)**: intra 0,9957 / 0,9774 / 0,9769 / 0,9620, al tetto AA: il vincolo non ha rovinato la struttura nativa.
  - **Curva di fusione (17; 316–345 K, rami chiuso/aperto, 25 ns)**: Tm (U = 1/2, soglia per tetrade) 329,4 K contro 328,4; T_F (mappa f1) 319,6 contro 318,8 K; transizioni strette (4–8 K contro 20–30). Livelli "2 tetradi" (~320 K) e "solo tetrade 2" (324–328 K); U compatto (Rg ×1,2) a 332–336 K, esteso solo a 340–345 K.
- **Intermedi di pop1_it45 (20_intermediates.py)**: con q mediato su 0,1 ns e core set 0,4/0,6 le distribuzioni di q per tetrade sono unimodali e scivolano con T (0,7 a 316 K → 0,55/0,45 a 324–328 K → 0,4 a 332 K); i livelli intermedi vivono nella zona ambigua (frazione pulita 0–0,15), permanenze 0,2–0,5 ns, decine di passaggi avanti e indietro: **non sono stati popolati** ma tagli di soglia su un unico bacino.
- **Analisi per tratto G (21_gtracts.py)**: c_a = coppie B3-B3 native del tratto a con gli altri, Qr_a = coppie fra gli altri tre; mappa −ln P(c*, Qr*).
  - **Soli prior f** (13, 320–335 K, 100 ns × 2 rami × 2 repliche): **due stati**, bacino F (c* ≈ 0,55–0,6) e bacino U nell'origine (nessun tratto appaiato), cammino diagonale (i quattro tratti si staccano insieme), 30–90 passaggi F⇄U per T, permanenze di ns; P_U 0,05 / 0,37 / 0,85 / 0,99.
  - **k6**: bacino F netto a 330 K (frazione pulita 0,98), U nell'origine a 345 K.
  - **pop1_it45**: un solo bacino che scivola lungo la diagonale ((0,65, 0,75) a 316 K → (0,35, 0,45) a 328–332 K → origine solo a 340–345 K): **allentamento uniforme**, introdotto dal vincolo con la soglia per tetrade.
  - **Nessun modello ha la tripla elica I2 di Buscaglia** (angolo c* < 0,3, Qr* > 0,6 vuoto; solo un lobo secondario a 320 K in pop1, tratto A o B). Con questa rappresentazione (contatti B3 isotropi, ioni impliciti) la tripla elica non è un minimo: ottenerla richiede prior nuovi. In una tripla elica ogni tetrade conserva 3 coppie su 6 (q = 0,5): il parametro per tetrade non la distingue dall'allentamento.
- **pop2** (P_F + P_U con la soglia per tetrade, job 61929739/61929743): cancellato il 23/10 prima di produrre risultati.
- **Decisione (23/10)**: bersaglio = curva **U(T) di Buscaglia** (Tm 328 K, larghezza ~25 K fra 10 e 90 %), robusta all'interpretazione di I1/I2; **U definito come bacino** (tutti i c_a < 0,2, `POP_ORDER=tract`); gli stati allentati contano come "non U". Catena **pop3** da k6 (ancora cooperativo), P_U a 320 / 326 / 332 / 338 / 345 K, POP_NS 3, δ 0,02, λ 4. Poi validazione con 17 + 21 + 16 (Tm, larghezza, mappe bimodali, struttura a 300 K). Nel report: il CG riproduce U(T) con uno stato U cooperativo; gli stati non U del CG raggruppano F, I1 e I2 (niente tripla elica).
- Più avanti: WCA rifittato su 300 + 400 K; MBAR/Tm; ligandi; prior che permettano la tripla elica.

## Riferimenti

- Buscaglia R., Gray R. D., Chaires J. B., *Biopolymers* **99**(12), 1006–1018 (2013), doi:10.1002/bip.22247, PMID 23536479 (supplementari NIHMS648440). Dettagli in `claude/termodinamica-sperimentale-tel26.md`.

## Report (8/10)

`tutorials/tel26/report`: sezione 2 sul modello, figura PyMOL, confronto con i soli prior. Versioni italiana e inglese.
