# Checkpoint PaiNN-LT / TEL26 unfoldable — 23/10/2026

Punto di ripartenza completo della chat Cowork del 6–23/10/2026 sul modello TEL26 "unfoldable" (CG con prior termici + residuo PaiNN). Se si riparte da zero: leggere questo file, poi `project_docs/unfold-tel26-fase2.md` (stato tecnico dettagliato) e `project_docs/termodinamica-sperimentale-tel26.md` (bersaglio sperimentale).

## Contenuto della cartella

| file / cartella | cosa contiene |
|---|---|
| `CHECKPOINT.md` | questo file: contesto, regole di lavoro, percorsi, stato, prossimi passi |
| `stato_repo.txt` | git log, git status, md5 dei file chiave, elenco dei risultati, messaggi di commit pronti (al 23/10) |
| `project_docs/` | copia dei 10 documenti del Project claude.ai "PaiNN-LT" (versione più recente) |
| `memoria/` | file di memoria di Claude relativi al progetto (obiettivi, decisioni, Leonardo, Giulia, preferenze) |
| `trascrizione/chat_testo.md` | trascrizione testuale della chat (messaggi e comandi, senza gli output degli strumenti) |
| `trascrizione/riassunto_contesto_precedente.md` | riassunto automatico della parte di chat fino al 16/10 |
| `trascrizione/chat_completa.jsonl.gz` | trascrizione completa (JSONL, con output degli strumenti) |

I PDF preparati per Giulia sono già nel repo (`tutorials/tel26_unfold/results/TEL26_AA_unfold_pilot_runs.pdf`, `TEL26_pilot_excl400_unfolding.pdf`), insieme ai grafici dei risultati.

I documenti del Project restano anche su claude.ai (Project "PaiNN-LT"); qui sono una copia per sicurezza.

## Prompt per ripartire in una chat nuova

> Riprendiamo il lavoro sul TEL26 unfoldable (PaiNN-LT). Leggi `checkpoints/2026-10-23_tel26_unfold/CHECKPOINT.md` nella cartella PAINNLT e poi `project_docs/unfold-tel26-fase2.md`. Il prossimo passo è la catena pop3 (vedi "Stato e prossimi passi").

## Obiettivo

- Modello CG del TEL26 ibrido-2 (2JPZ, (TTAGGG)4TT), ioni impliciti, che si **srotola** e dà la **Tm** giusta, poi con ligandi.
- Architettura: prior fisici (Morse Hoogsteen/impilamento/cappucci, diedri CBT, WCA) dipendenti da T (D(T) = max(D_H − T·D_S, 0), parametri "f") + residuo **PaiNN** con teste termodinamiche U(T) = A + (T − T0)/T0·B (readout e readout_dT, T0 = 300 K).
- Riferimento sperimentale: **Buscaglia, Gray, Chaires, Biopolymers 99(12), 1006–1018 (2013)**, doi 10.1002/bip.22247 (2JPZ, 25 mM KCl): F ⇌ I1 ⇌ I2 ⇌ U, Tm (U = 50 %) 328,4 K, ΔH_tot 352 kJ/mol. I2 = tripla elica G-rica (Gray et al., JACS 134, 16834, 2012, su Tel22).
- Decisione del 23/10: si vincola solo la curva **U(T)** di Buscaglia, con U definito come **bacino** (nessun tratto G appaiato); gli stati non U del CG raggruppano F, I1 e I2 (il CG non ha la tripla elica).

## Regole di lavoro (come si lavora con Claude su questo progetto)

- Italiano, registro tecnico, utente esperto. Termini tecnici inglesi consolidati lasciati in inglese.
- **Commit li fa Cristiano dal Mac**: Claude scrive il messaggio in `PAINNLT/.git/MLCG_COMMIT_MSG_<nome>` e dà i comandi `git add …; git commit -F .git/MLCG_COMMIT_MSG_<nome>; git push`. Claude usa git solo in lettura (`git --no-optional-locks`). Firma dei commit: righe `Co-Authored-By: Claude …` e `Claude-Session: …` in fondo.
- Comandi sempre etichettati **Mac** / **Leonardo**; dopo `git pull` su Leonardo, controllo md5; copie con `scp -O leonardo:… …`.
- Su Leonardo prima di python: `source MLCG_Framework_v2/hpc/env_leonardo.sh` (senza, python3 è il 3.6 di sistema). Una shell nuova non ha le variabili (`A`, `U`, `MP`…): ridefinirle.
- Cancellazioni irreversibili: prima l'elenco all'utente.
- Il Mac è collegato a Cowork (cartella PAINNLT): Claude legge/modifica i file del repo direttamente lì; Leonardo lo raggiunge solo l'utente.

## Percorsi e account

| cosa | percorso |
|---|---|
| repo sul Mac | `/Users/demichel/WORK/NEURAL_NETWORKS/PAINNLT` |
| repo su Leonardo | `/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT` (log in `PAINNLT/logs/`) |
| script unfoldable | `MLCG_Framework_v2/tutorials/tel26_unfold` (`U`) |
| tutorial TEL26 (modelli, config, stati th0) | `MLCG_Framework_v2/tutorials/tel26` (`T26`) |
| dati | `A=/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold` (spostati dal 16/10 per la quota; `/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold` è un symlink) |
| AA di Giulia (400 K) | `/leonardo_work/IscrB_G4MES/GiuliaT/AllAtom` |

- Account Booster `IscrB_G4MES` (partizione boost_usr_prod, 4 A100 per nodo, 24 h). DCGP `IscrB_G4MES_0` ha fairshare ~0: evitare.
- Quote: per area di progetto e condivise fra i membri; HOME 50 GB personale.
- Compilazione di train_relent: `source MLCG_Framework_v2/hpc/env_leonardo.sh && cmake --build MLCG_Framework_v2/training/build --target train_relent -j 8` (verifica: `grep -ac -- "--pop-huber" MLCG_Framework_v2/training/build/train_relent` ≥ 1).

## Modelli e dati principali (su Leonardo)

| nome | dove | cosa |
|---|---|---|
| `tel26_lp2th_th0_it60.pt` | `$T26` | PaiNN allenato a entropia relativa a 300 K sopra i prior termici lp2th: struttura nativa al tetto AA, ma non fonde (Tm > 380 K) |
| `tel26_lp2th_th0_it60_k{2,4,6}.pt` | `$T26` | it60 con testa entropica proporzionale (`--thermo-scale κ`); k6 fonde fra 330 e 345 K a due stati |
| `state_tel26_lp2th_th0_it60.npz` | `$T26` | stato a 300 K (10 copie) della catena th0 |
| `pop1_it45.pt` | `$A/relent_pop/pop1/` | k6 + vincolo P_U con soglia per tetrade: Tm 329,4 K ma fusione per allentamento uniforme (non cooperativa). **Non usare come base** |
| melt runs | `$A/melt_painn/<modello>/{chiuso,aperto}/T<T>.*` | curve di fusione di 17 (it60, k2/k4/k6, pop1_it45) |
| soli prior f | `$A/thermo_val/f/{chiuso,aperto}/T<T>_r<r>.*` | validazione 13 (100 ns, 2 repliche per ramo) |
| pilota AA | `$A/pilot_excl400` (anche `pilot_noK400`, `pilot_li400`) | unfolding AA a 400 K con canale vietato ai cationi |

## Risultati chiave (dettagli in `project_docs/unfold-tel26-fase2.md`)

1. **Soli prior termici f**: due stati, Tm 327,3 K, transizione troppo stretta (ΔH_vH ~430 contro 352). Bacini F e U distinti nella mappa per tratto G (21).
2. **th0_it60** (PaiNN a 300 K): struttura a 300 K al tetto AA (P(r) intra 0,994 / B3 0,983 / B5 0,975 / S 0,957), ma sovrastabilizza F di 60–230 kJ/mol: nessuna fusione fino a 380 K.
3. **Scansione κ**: con un solo κ non si ottengono insieme Tm e larghezza; k6 è il punto di partenza cooperativo (F a 330 K, U esteso a 345 K).
4. **train_relent --pop** (vincolo sulle popolazioni sperimentali con ripesatura, gradiente verificato a 1e-11): funziona meccanicamente.
5. **pop1** (vincolo P_U con soglia q = 0,5 per tetrade): Tm 329,4 K e struttura a 300 K intatta, **ma** la fusione è diventata un allentamento continuo (20_intermediates, 21_gtracts): gli "intermedi" erano tagli di soglia e lo "U" a 328–332 K è un G4 allentato. Lezione: lo stato bersaglio va definito come bacino, non come soglia su un parametro d'ordine che scivola.
6. **Nessun modello ha la tripla elica I2** (angolo c* < 0,3, Qr* > 0,6 vuoto). Indizio debole: il tratto che si stacca più spesso è A (G4–6), cioè il 5′, come in Tel22.
7. **AA pilot_excl400**: core G4 perso in 3 repliche su 4 (23, 35, 83 ns a 400 K), stato U compatto (Rg 1,3–1,4 nm contro 1,1) con Hoogsteen residui: non gomitoli estesi (osservazione di Giulia del 23/10, corretta). PDF per Giulia in `tutorials/tel26_unfold/results/`.

## Stato e prossimi passi (al 23/10/2026)

**Aggiornamento 23/10 sera**: commit `--order tract` (b1fc0e8) e checkpoint (02b695f) fatti e pullati su Leonardo; prova delle etichette per tratto riuscita (k6 330 K: P_U 0, c 0,73–0,81; 345 K: P_U 0,97, c 0,04–0,06). **pop3 lanciata: job 61944829 + 61944832** (in dipendenza). Il passo successivo è quindi il 4 (controllo dei log, poi validazione).

**Fatto ma non ancora committato**: `--order tract` in `19_pop_labels.py` e `POP_ORDER`/`POP_TRACT_CUT` in `19_relent_pop.sh` (md5 `fad38faa…`, `9ed93fec…`; messaggio pronto `.git/MLCG_COMMIT_MSG_tract`). pop2 (job 61929739/61929743) cancellato.

**Prossimi passi, in ordine:**

1. **Mac**: `git add MLCG_Framework_v2/tutorials/tel26_unfold/{19_pop_labels.py,19_relent_pop.sh}; git commit -F .git/MLCG_COMMIT_MSG_tract; git push`
2. **Leonardo**: `git pull`, md5, prova delle etichette per tratto:
   ```
   source MLCG_Framework_v2/hpc/env_leonardo.sh
   A=/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold; U=$PWD/MLCG_Framework_v2/tutorials/tel26_unfold; K6=$A/melt_painn/tel26_lp2th_th0_it60_k6
   python3 $U/19_pop_labels.py --T 330 --order tract --delta 0.02 --template $A/cg/tel26_lp2_1c_t400_dataset.bin \
       --samples $K6/chiuso/T330.samples.npz $K6/aperto/T345.samples.npz --skip-ps 500 --out-prefix $HOME/poptestT
   rm -f $HOME/poptestT.*
   ```
   Atteso: P_U ≈ 0 a 330 K (c ≈ 0,7–0,8), ≈ 1 nella corsa a 345 K (c ≈ 0,05).
3. **Catena pop3** (da k6; default di 19 = k6, stato th0_it60 a 300 K, stati k6 330 chiuso / 345 aperto):
   ```
   E="ALL,TAG=pop3,POP_ORDER=tract,POP_TRACT_CUT=0.2,POP_TS=320 326 332 338 345,POP_NS=3,POP_SKIP_PS=300,POP_DELTA=0.02,RE_ESS_MIN=0.3,POP_LAMBDA=4,NITER=30"
   j=$(sbatch --parsable --export="$E" $U/19_relent_pop.sh); echo $j
   sbatch --dependency=afterany:$j --export="$E" $U/19_relent_pop.sh
   ```
   Bersagli P_U (Buscaglia): 0,11 / 0,37 / 0,68 / 0,86 / 0,96 a 320 / 326 / 332 / 338 / 345 K. Controllo: `grep -E "^\[OK\]|FAIL|ERROR|POP-ITER|^  T " logs/slurm-relpop-<id>.out | tail -30`.
4. **Validazione del modello pop3 scelto**:
   - 17 (curva di fusione, `TS="316 320 324 328 332 336 340 345"`, `CG_STEPS=6250000`, `MAP=f1`);
   - 21_gtracts.py sulle stesse corse: le mappe devono restare **bimodali** (bacino F + bacino U nell'origine) vicino a Tm;
   - 16 (struttura a 300 K, `CHECKPOINT` = ultimo `md_itNN/re.state.npz` della catena).
5. Più avanti: WCA rifittato su 300 + 400 K; MBAR/Tm con replica exchange; ligandi; prior capaci di tripla elica (Hoogsteen direzionali, K⁺ fra gli strati); estensione dell'analisi AA del pilota (legami H non nativi, impilamento, end-to-end) se Giulia la vuole.

## Script dell'unfoldable (tutorials/tel26_unfold), in breve

02 tetradi e K⁺ AA · 03/04 piloti AA (canale vietato `--exclude-r`) · 08 tetradi CG e stati · 09–11 scala dei contatti e fit termico · 12 analisi pilota · 13/14 validazione termica (mappe f/f1) · 15 insieme lp2th · 16 controllo a 300 K · 17 curva di fusione con PaiNN (`MAP`, `CONTINUE`) · 18 scansione κ · 19 entropia relativa + popolazioni (`19_pop_labels.py`, `19_relent_pop.sh`) · 20 intermedi per tetrade · 21 stati per tratto G. Dettagli e opzioni in `project_docs/unfold-tel26-fase2.md` e nelle docstring.

## Job storici utili (Leonardo)

- κ: 60697412 / 60697448 / 60697450. pop1: 61360433, 61390905, 61512785, 61512786. Validazione pop1_it45: 17 → 61741503, 16 → 61741504. pop2 (cancellato): 61929739, 61929743.
