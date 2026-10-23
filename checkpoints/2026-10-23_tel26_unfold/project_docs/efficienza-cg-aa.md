# TEL26 — efficienza del CG con ML rispetto all'AA

Aggiornato al 2026-10-03 (pomeriggio). Priorità indicata da Cristiano: il CG ha senso solo se è almeno 10–100× più efficiente dell'AA; altrimenti non vale la pena continuare. Vincolo aggiunto il 2/10: ottimizzare senza compromettere la precisione ottenuta (stessa Hamiltoniana, FP32 pieno, niente TF32/FP16, niente modifiche a modello o cutoff).

## Sintesi

**Il criterio è raggiunto e confermato su 2 × 20 ns di produzione (4/10): 54,2 ns/day aggregati, efficienza per GPU 28× (rmsd_core, minimo) – 140× (traslazione, nucleo); struttura stazionaria. Vedi "Produzione lunga".** Storia: Con i nuovi default (grafo PaiNN sul device, liste di Verlet, dt 4 fs, γ 2, energia completa ogni 10 registrazioni) 1 ns ha richiesto 1 h 16 min (contro 3 h 33 prima): **~19 ns/day effettivi** con log ogni 0,5 ps e avvio, ~30 ns/day di sola integrazione (11–12 ms per step). Struttura e tempi di rilassamento identici alla corsa di riferimento con stesso dt e γ.

**Aggiornamento 3/10 pomeriggio:**
- fusione della MLP a livello di nodo: step 11,9 → 10,3 ms (~34 ns/day di sola integrazione), parità 1,6·10⁻⁷, NVE identica;
- due replicas per GPU validate: struttura identica alla corsa singola, 45,9 ns/day aggregati (binario precedente alla fusione);
- profilo MD fast per la catena RE validato: 9,3× più veloce, struttura invariata;
- la corsa singola da 1 h 16 era rallentata da fattori esterni: con i suoi parametri il conto dà ~55–57 min (~25 ns/day effettivi; con la fusione ~28–29). Le efficienze qui sotto, calcolate a 18,9 ns/day, sono quindi conservative di ~1,5×.

Efficienza per GPU rispetto all'AA (137 ns/day) a ~19 ns/day: traslazione ~40–70× (α misurato ~300, teorico 514), siti del nucleo ~50×, rotazione ~16×, rmsd_core ~10×. Con 2 replicas per GPU (45,9 ns/day aggregati misurati) si sale a ~2,4× questi valori. Per budget fino a ~4× di più (vedi "per GPU e per budget").

## Impostazioni di produzione

- Default di `05_run_espresso.sh` dal 2/10 (`MLCG_COMMIT_MSG_prod_defaults`): `CG_DT=0.004`, `GAMMA=2`, `NEIGHBOR_SEARCH=verlet`, `MLCG_PAINN_GRAPH=device`, `ENERGY_INTERVAL=10`, `LOG_INTERVAL=125` (0,5 ps). Per riprodurre le corse precedenti: `CG_DT=0.001 GAMMA=1 NEIGHBOR_SEARCH=link-cell MLCG_PAINN_GRAPH=legacy ENERGY_INTERVAL=1 LOG_INTERVAL=20`.
- **Due replicas indipendenti per GPU** (standard di produzione): `run_replicas_per_gpu.sh` (`MLCG_COMMIT_MSG_replicas`).
  - Prova 2 × 1 ns re1 it30 (job 59250740): 1 h 02 min 47 s, **45,9 ns/day aggregati** (22,9 per replica), binario senza fusione MLP.
  - Struttura (job 59257323) identica alla corsa singola: tetradi 0,97/0,98/1,00× in entrambe, loop/code 1,20× contro 1,19×, scarto medio 0,0398 contro 0,0394 nm, per residuo entro ~0,01.
  - Script 46: repliche concatenate con `+`.
- **Produzione lunga** (`chain_replicas.sh`, `MLCG_COMMIT_MSG_chain_replicas`): catena `prod_re1_s01..s04`, 4 segmenti × 5 ns × 2 replicas (2 × 20 ns), da `rep1ns_re1_b`, LOG_INTERVAL 250 (1 ps), 7 h per job; job 59273995 → 59273996 → 59273999 → 59274000 (sottomessi il 3/10, ~22 h in tutto). Scopo: statistiche confrontabili con l'AA su loop e Rg.
- `MLCG_PAINN_CUDA_GRAPH` resta a 0: corretto ma più lento (vedi sotto).
- `06_relent.sh` (catena RE): `MD_PROFILE=fast` (default: dt 4 fs, verlet, grafo sul device, energia ogni 10 registrazioni, γ 20) oppure `legacy`. Validato con `compare_relent_md.sh` (job 59251432, 100 ps a γ 20, campione ogni 0,1 ps):
  - tempo 4364 s → 471 s (**9,3×**); integrazione 31,7 → 12,2 ms/step; registrazioni 1,16 → 0,14 s l'una; "resto del ciclo" 0 s (nessun overhead nascosto);
  - struttura: RMSF delle tetradi 0,95–1,04× in entrambe, scarto sui loop 0,094 contro 0,091 nm, per residuo entro qualche centesimo;
  - E_kin media 1396,6 ± 59 (legacy) contro 1412,9 ± 61 (fast): +1,2 %, ~2σ. Probabile effetto della discretizzazione di Langevin a 4 fs sulla temperatura cinetica; il gradiente RE usa medie configurazionali, invariate. Conferma opzionale: gradiente di un'iterazione RE con i due profili.
- `06_relent.sh` ha perso il bit eseguibile (commit "upd" 30e40ef dopo il ripristino 57ee7a8); irrilevante, si lancia con `bash`.

## Produzione lunga 2 × 20 ns (catena prod_re1_s01..s04, analisi 59325551)

- **Throughput**: 40 ns in 63 747 s (17 h 42) → **54,2 ns/day aggregati** (27,1 per replica); per segmento 51,0 / 60,5 / 53,3 / 53,0. Integrazione 10,6–12,9 ms/step per replica: ~20 % di variabilità fra nodi a parità di codice (s02 il più veloce). Registrazioni ~150 ms × 5000 = 4,5 %; resto del ciclo 0.
- **Traslazione**: D 32,4 nm²/ns (retta e fit coincidono su 20 ns; per replica 34,1 e 30,6) → **α_trasl 330–366**, 0,68 del teorico. Su 1 ns la retta (21–25) sottostimava per il tratto balistico. MSD per ora di GPU: **137×** l'AA.
- **Rotazione**: D_rot 38–39 rad²/ns, τ_P2 29,8 ps contro 1/(6D_rot) 4,3 ps (inerziale, fattore ~7). Decorrelazione **43–45×**; MSD angolare 303–314× (sovrastima).
- **Script 47** (α / efficienza per GPU a 54,2 ns/day): rotazione 108–114 / 43–45×; rmsd_core 71,5 / **28×**; traslazione 330–366 / 133–142×; siti del nucleo 255 / 101× ma **sottostimato**: con un frame ogni ps il τ CG (~0,5 ps) non è risolto e l'interpolazione lo gonfia a 0,69 ps; vale α 363 (campionamento 0,5 ps) → ~140×.
- **Rg e loop**: τ AA 5,6 e 7,0 ns contro 2,4 e 3,0 ps nel CG → α nominale ~2 400 (efficienza ~930×), **non interpretabile come guadagno**: il tempo a 1/e misura processi diversi. Nell'AA domina un rilassamento lento (riarrangiamenti dei loop, ns); nel CG solo fluttuazioni di ps in un unico bacino. Coerente con i difetti strutturali sui loop. Da verificare con la coda lunga dell'ACF CG (fino a ~5 ns) o un τ integrato.
- **Struttura**: primi 10 ns e ultimi 10 ns identici (tutte le sovrapposizioni entro 0,005; FES (rmsd_core,Q) JSD 0,0787 in entrambi, pari al tetto AA contro sé stesso 0,0786; il 0,12 su 1 ns era rumore). Stazionaria, nessuna deriva in 20 ns.
  - Nucleo: rmsd_core sovrapposizione 0,94–0,95 (tetto 0,93), tetradi 0,97/0,98/1,00×, Q e Rg medi come AA.
  - Loop (limite del modello, invariato): RMSF 1,18×, rmsd_loops 0,271 contro 0,210 nm (sovrapposizione 0,57), scarto medio 0,095 contro tetto 0,062 nm; T2 1,86×, T13/T14 1,7×, A3 1,45× troppo mobili, T19 0,78× troppo rigido.
- **Efficienza per GPU finale (2 replicas, 54,2 ns/day)**: traslazione ~137×, siti del nucleo ~140×, rotazione ~44×, rmsd_core ~28× (minimo fra gli osservabili confrontabili). Per budget fino a ~4× di più.
- Conclusione: la velocità non è più il collo di bottiglia; lo è la qualità del modello sui loop.

## Diagnosi dei loop: coda lunga delle ACF (script 49, job 59337900)

`49_tel26_loop_acf.py` (`MLCG_COMMIT_MSG_loop_acf`) su AA (136 ns × 10 copie, frame 20 ps) e prod (2 × 20 ns × 10 copie, frame 1 ps). ACF per copia (media di copia sottratta), A_slow = ACF(20 ps), τ della parte lenta, frazione della varianza fra copie.

| osservabile | AA: ACF 20 ps / 5 ns | AA: τ lento | AA: varianza fra copie | CG: ACF 20 ps / 50 ps–2 ns / 5 ns | CG: varianza fra copie |
|---|---|---|---|---|---|
| Rg | 0,80 / 0,38 | 8,0 ns | 0,50 | 0,02 / ~0 / 0,02 | 0,000 |
| tutti i loop | 0,90 / 0,43 | 8,3 ns | 0,50 | 0,02 / ~0 / 0,02 | 0,000 |
| coda 5' T1-A3 | 0,94 / 0,52 | 9,5 ns | 0,49 | −0,09 / ~0 / 0,01 | 0,000 |
| loop 1 T7-A9 | 0,88 / 0,36 | 6,8 ns | 0,21 | 0,04 / ~0 / 0,02 | 0,000 |
| loop 2 T13-A15 | 0,68 / 0,24 | 5,2 ns | 0,52 | 0,07 / ~0 / 0,02 | 0,000 |
| loop 3 T19-A21 | 0,90 / 0,42 | 9,1 ns | 0,61 | 0,19 / ~0 / 0,03 | 0,000 |
| coda 3' T25-T26 | 0,81 / 0,22 | 3,8 ns | 0,17 | 0,04 / ~0 / 0,04 | 0,000 |

- **AA**: 70–95 % della varianza dei loop (entro copia) è in un rilassamento lento di 4–10 ns; in più metà della varianza totale (17–61 % per loop) è fra copie: le copie restano in stati dei loop diversi per tutti i 136 ns. Il riferimento AA sui loop **non è convergente** (coerente col tetto half2 basso su rmsd_loops, 0,86).
- **CG**: >95 % della varianza decade entro ~5–10 ps (con oscillazioni smorzate a 2–10 ps, γ 2); nessuna componente lenta fra 50 ps e 2 ns; varianza fra copie zero (copie e replicas con la stessa media): **un solo bacino liscio, nessuno stato metastabile dei loop**. A 5 ns l'ACF CG è 0,01–0,04 in tutti i gruppi: forse una componente lenta debolissima, al livello del rumore.
- Quindi per Rg e loop α ~2 400 non è un'accelerazione: il CG non ha il processo. Il difetto strutturale dei loop (RMSF 1,18×, scarto medio 0,095 nm, sovrapposizione 0,57) è probabilmente un bacino unico "medio" al posto di più stati AA.
- **Stati dei loop (script 50, 5/10)**: AA 6788 frame × 10 copie; CG prod 2 × 20 ns (frame 5 ps). Permanenze con filtro a maggioranza su 100 ps; un CG con permanenza pari a tutta la corsa (20 005 ps) e "nan" sull'altro stato non lascia mai lo stato principale (le assegnazioni all'altro stato sono solo escursioni < 100 ps).

| loop | k (silhouette) | pop AA | permanenze AA | pop CG (grezze) | CG fuori dagli stati AA | dispersione CG / AA entro stato | lettura |
|---|---|---|---|---|---|---|---|
| coda 5' T1-A3 | 5 (0,49) | 0,34/0,21/0,20/0,20/0,04 | 0,9–3,2 ns | 0,26/0,22/0,26/0,20/0,07 | 42 % | 0,65 / 0,23 nm | nessuno stato: distribuzione diffusa, troppo mobile |
| loop 1 T7-A9 | 5 (0,34, debole) | 0,43/0,23/0,14/0,14/0,07 | 0,5–2,2 ns | 0,35/0,39/0,18/0,07/0,01 | 11 % | 0,19 / 0,14 nm | accettabile |
| loop 2 T13-A15 | 2 (0,51) | 0,90/0,10 | 111 / 45 ns | 0,93/0,07 (sempre nello stato 0) | 42 % | 0,26 / 0,16 nm | manca lo stato minore; attorno al principale troppo mobile |
| loop 3 T19-A21 | 2 (0,39) | 0,74/0,26 | 5,8 / 2,1 ns | 0,86/0,14 (sempre nello stato 0) | 18 % | 0,47 / 0,42 nm | **manca lo stato 1 (26 %)**: coerente con T19 troppo rigido (0,78×) |
| coda 3' T25-T26 | 2 (0,39) | 0,74/0,26 | 0,87 / 0,29 ns | 0,77/0,23 | 2 % | 0,14 / 0,13 nm | buono (stato 1 solo per escursioni brevi) |

  - Le popolazioni AA per copia non sono state stampate (filtrate dal grep): servono per valutare l'incertezza del riferimento (loop 2 stato 1 con permanenza 45 ns è probabilmente una o due copie).
  - Prossimo passo: popolazioni per copia + strutture rappresentative degli stati (loop 3, loop 2, coda 5') per capire fisicamente cosa manca (stacking sulle tetradi, contatti), poi target RE o termini nei prior.
- **Contatti dei loop (script 52, job 59376003; `MLCG_COMMIT_MSG_loop_contacts`)**, contatto = distanza fra baricentri di base < 0,70 nm (T e A sono un solo sito), errore AA = std fra copie / √10:

| coppia | f AA (per copia) | f CG | lettura |
|---|---|---|---|
| T2-A15 | 0,24 ± 0,09 | 0,56 | **CG troppo spesso** (3,7σ) |
| A9-G18 | 0,94 ± 0,05 (9 copie) | 0,79 | CG troppo poco (2,7σ): stacking di A9 sulla tetrade 3 indebolito |
| T1-A21 | 0,31 ± 0,12 (3 copie) | 0,10 | CG troppo poco (1,8σ); picco AA 0,65, CG 0,97 nm |
| T19-A21 | 0,00 (nessuna copia) | 0,09 | contatto spurio nel CG: loop 3 ripiegato su sé stesso |
| T1-A15 | 0,43 ± 0,13 (0–1 per copia) | 0,34 | **compatibile** (correzione: dal solo valore medio della distanza nello script 51 sembrava assente) |
| T20-G18 / T20-G22 | 0 / 0 (p5 0,77 / 0,90) | 0 / 0 (p5 1,10) | non sono contatti: nell'AA T20 si avvicina al solco (coda < 1 nm, stato 1 del loop 3), nel CG mai sotto 1,1 nm |
| A21-G16 / A21-G17 | 0,06 (solo copia 3) / 0 | 0,03 / 0,02 | non significativi |

  - Quadro rivisto: nella zona coda 5'–loop 2–loop 3 il CG sbaglia il registro degli appaiamenti (A15 si appaia con T2 invece che con T1 una parte del tempo, T1 raggiunge A21 troppo poco), A9 si impila meno su G18, il loop 3 si ripiega su sé stesso invece di avvicinarsi al solco. Con un solo sito per T e A la specificità di registro è difficile da ottenere con interazioni generiche.
  - Proposta: termini Morse pair-specific con profondità (con segno) allenata dall'RE per T1-A15, T1-A21, T2-A15, A9-G18, T19-A21 (e T20-G18 come avvicinamento): il gradiente RE ⟨∂U/∂D⟩_AA − ⟨∂U/∂D⟩_CG fa coincidere da solo le medie dei termini di contatto. r0 dai picchi AA (servono quelli di T2-A15, A9-G18, T19-A21: rifare lo script 52 con queste coppie). Riferimento AA incerto (copie non ergodiche): chiedere a Giulia più AA sui loop.
- **Contatti di stato (5/10)**: i 16 contatti di loop di lp2 sono tutti base di loop–nucleo, D = 50 kJ/mol, scelti da `find_loop_contacts.py` solo se persistenti in tutte le copie (`--copy-tol`): per costruzione mancano quelli dipendenti dallo stato (T1, loop 3). `add_state_contacts.py` (`MLCG_COMMIT_MSG_state_contacts`) aggiunge Morse stretti di ruolo "state" (r0 e larghezza dal picco di contatto AA, D iniziale 5 kJ/mol, popolazione AA in `g4_topology.state_contacts`); `_prior_set.sh` accetta il residuo ML su un insieme derivato dalla stessa base con `ALLOW_ML_ON_DERIVED_PRIORS=1`. `tune_state_contacts.py` (`MLCG_COMMIT_MSG_tune_state`): D ← D + kT [logit f_AA − logit f_CG].
  - lp2c (job 59379472): T1:DT–A15:DA r0 0,439, bacino < 0,515, f_AA 0,23 ± 0,11; T1–A21:DA r0 0,452, f_AA 0,14 ± 0,10 (copie 0 e 4); T20:DT–G18:S r0 0,484, f_AA 0,09 ± 0,09 (solo copia 5: da togliere); A9:DA–G18:B4 r0 0,540, f_AA 0,95 ± 0,05 (persistente). cg_priors.lp2c = lp2 + 40 contatti di stato, tutto il resto identico (verificato).
  - Iterazione 0: lp2c0 = lp2c senza T20–G18 (D = 5 kJ/mol), MD 2 × 1 ns da prod_re1_s04 con re1_it30 (job 59381807): 13,0 ms/passo, nessun costo; ripresa da checkpoint con più marker corretta in run_cg_md.py (`MLCG_COMMIT_MSG_ckpt_markers`).
  - Popolazioni nei bacini (f_AA / lp2 senza contatti di stato / lp2c0 con D = 5): T1–A15 0,23 ± 0,11 / 0,16 / 0,40; T1–A21 0,14 ± 0,10 / 0,04 / 0,16; A9–G18 0,95 ± 0,05 / 0,92 / 0,97. **Con lp2 tutte e tre erano già entro l'errore AA**: i contatti di stato spostano le popolazioni dentro le barre d'errore del riferimento (T1–A15 addirittura oltre). Con queste incertezze il riferimento AA non giustifica i contatti; iterazione 1 sospesa. tune_state_contacts.py ora lascia fermo D entro l'errore AA (`MLCG_COMMIT_MSG_tune_freeze`).
  - **Controllo lp2c0 contro lp2** (job 59389775, script 52 con baricentri e soglia 0,7 nm, e 46): T2–A15 0,554 → 0,608 (AA 0,24: **peggiora**, T2 segue T1 verso A15); T1–A15 0,345 → 0,590 (AA 0,43 ± 0,13: oltre); T1–A21 0,10 → 0,25 (AA 0,31: meglio); A9–G18 0,80 → 0,88 (AA 0,94: meglio); T19–A21 0,09 → 0,07; nuovo contatto spurio T1–G16 0,05 → 0,12 (AA 0). Struttura: nucleo invariato (rmsd_core, tetradi 0,97–1,00×); rmsd_loops sovrapposizione 0,58 → 0,62, RMSF loop 1,18 → 1,13×; scarto medio dei loop 0,095 → 0,106 nm (T1 0,08 → 0,21); Rg sovrapposizione 0,85 → 0,82. FES JSD 0,082 → 0,118 non confrontabile (lp2c0 su 2 × 1 ns contro 2 × 5 ns). **Esito misto, lo scarto robusto (T2–A15) non si corregge: lp2c0 non adottato; modello di produzione resta lp2 + re1_it30.** Con un sito solo per T e A la specificità di registro non si ottiene con contatti fra coppie.
  - Scarti robusti che restano (script 52, baricentri, soglia 0,7 nm): T2–A15 in eccesso nel CG (0,56 contro 0,24 ± 0,09, 3,7σ); forma delle distribuzioni (AA a picchi stretti, CG larghe), in parte dovuta alla non ergodicità delle copie AA.
- (4/10 sera) `50_tel26_loop_states.py` (`MLCG_COMMIT_MSG_loop_states`; PCA + k-means per loop sull'AA, k dalla silhouette, popolazioni per copia, permanenze, CG proiettato: popolazioni, frazione fuori dagli stati AA, distanza della media CG dai centri; verificato su dati sintetici). Job lanciato su prod (2 × 20 ns, --cg-stride 5); output in `slurm-tel26_loopst-*.out`, JSON `loop_states_prod.json`. Da leggere alla ripresa.
- Prossimo passo proposto: caratterizzare gli stati dei loop nell'AA (clustering per loop a nucleo allineato: stati, popolazioni, tempi di vita, differenze fra copie) e confrontare la distribuzione CG nello stesso spazio. Poi decidere fra target RE sugli stati dei loop, termini dedicati nei prior, o più dati AA sui loop (chiedere a Giulia).

## Conferma su 1 ns (job 59228514, analisi 59228515)

- Confronto `opt` (nuovi default) contro `g2_dt4` (stesso modello, dt 4 fs, γ 2, link-cell, percorso legacy, log per particella):
  - rmsd_core sovrapposizione 0,932 contro 0,929 (tetto 0,922); FES JSD 0,124 contro 0,123; RMSF delle tetradi 0,97–1,00× in entrambe; loop/code 1,19× contro 1,17×; scarto medio 0,0394 contro 0,0396 nm; per residuo identiche entro 0,03×.
  - τ: rotazione 28,9 contro 30,3 ps; nucleo 0,483 contro 0,482 ps; rmsd_core 1,08 contro 0,98 ps; Rg e loop 2,4/3,0 contro 2,3/2,9 ps.
  - **Fisica invariata**: le ottimizzazioni cambiano solo la velocità.
- Log al passo 0: per particella 2,0 s, vettoriale 1,57 s; `energy()` 1 403 ms, min_dist/max_f 32 ms; E_kin dalle velocità contro ESPResSo 3·10⁻¹⁶.
- Tempo: 250 000 step in 1 h 16 min. Integrazione ~48–53 min, log ~6 min; ~10–15 min non attribuiti. Le righe `[TIMING]` di `compare_relent_md` mostrano "resto del ciclo" 0 s e le repliche fanno 1 ns in 1 h 03 condividendo la GPU: il ritardo era esterno (nodo o filesystem), non del codice.

## Traslazione: misurata sotto la teoria

- D (nm²/ns) a γ 2: retta 21–25, fit Langevin 27–34 (τ_v 89–101 ps, estrapolato), teorico kT/(26γ) 47,9. A γ 20: 4,2–4,4 contro 4,79. AA: 0,093–0,100.
- Due corse indipendenti a γ 2 danno entrambe 0,57–0,72 del valore teorico: probabilmente non solo statistica. Ipotesi: interazioni fra le 10 copie (la formula vale per una molecola isolata) o attrito effettivo maggiore. **α_trasl da usare: ~300 (misurato); 514 è un limite superiore.**

## Indicatore MSD per ora di GPU (script 48)

- Proposta di Cristiano: MSD traslazionale e rotazionale per tempo di calcolo. Nel regime diffusivo MSD per ora = 6D × ns per ora, quindi per la traslazione coincide con α × rapporto di throughput.
- Limiti:
  - misura la velocità, non il campionamento: per l'orientazione (spazio compatto) conta la decorrelazione; a γ 2 la rotazione CG è inerziale (rilassamento della velocità angolare ~10² ps > τ_P2 30 ps) e l'MSD angolare sovrastima il guadagno;
  - la traslazione CG è in buona parte fissata dal termostato (D ≈ kT/(Nγ));
  - non vede i moti interni lenti (rmsd_core ~10× resta il collo di bottiglia).
  - Proposta: riportare efficienza di trasporto (MSD) ed efficienza di campionamento (τ), con il minimo fra gli osservabili rilevanti come numero principale.
- Traslazione dai D misurati: AA 3,2–3,4 nm² per ora di GPU; CG a 18,9 ns/day 99–161 → **29–50×**; solo integrazione (~30 ns/day) 46–80×; 2 replicas (45,9 ns/day) ~70–120×.
- `48_tel26_msd_wall.py` (`MLCG_COMMIT_MSG_msd_wall`): MSD del centro di massa e MSD angolare (orientazione del nucleo per Kabsch, rotazioni fra frame nel sistema del laboratorio sommate come vettori di rotazione), D e D_rot da retta e fit di Langevin, τ_P2 per confronto; tassi per ora di GPU e MSD a pari ore di GPU (tabella e grafico). Replicas separate da virgola, ns/day aggregati. Verificato su traiettorie sintetiche (D e D_rot entro il 3 %).
- **Risultati (job 59258422)**, rapporto CG/AA per ora di GPU (retta – fit Langevin):

| | g2_dt4 (6,8 ns/day) | opt (18,9) | 2 replicas (45,9 aggregati) |
|---|---|---|---|
| traslazione, MSD | 13,5–18,6× | 31,5–42× | 82–114× |
| rotazione, MSD angolare | 30–36× | 71–81× | 190–226× |
| rotazione, decorrelazione τ_P2 | 5,5× | 15,7× | 34× |

  - AA: D 0,093 (retta) / 0,100 (fit) nm²/ns; D_rot 0,049–0,051 rad²/ns; τ_P2 3,31 ns ≈ 1/(6D_rot) = 3,38 ns: rotazione AA diffusiva, conferma del metodo.
  - CG: D 21–23 (retta) / 29–35 (fit, τ_v estrapolato); D_rot 25–30 rad²/ns, quindi D_rot,CG/D_rot,AA ≈ 515, come il rapporto teorico della traslazione (514): entrambi seguono il rapporto degli attriti. Ma τ_P2 ≈ 30 ps contro 1/(6D_rot) ≈ 6 ps e τ_ω (fit) 67–80 ps > τ_P2: la rotazione CG si decorrela nel regime inerziale. L'MSD angolare sovrastima il guadagno sulle orientazioni di ~5×; per il campionamento vale τ_P2.
  - Le repliche danno gli stessi D della corsa singola (22,9 contro 21,3 retta).
  - A pari ore di GPU brevi (1–5 min) il CG è ancora balistico (τ_v ~10² ps ≈ 5–10 min di calcolo a 19 ns/day) e il rapporto è più basso (~2× a 1 min, ~10× a 5 min); per produzioni di ore vale il rapporto asintotico.
- **2 × 2 ns (job 59258663 + analisi 59259069)**, binario con fusione MLP per il secondo ns:
  - `[TIMING]` delle due replicas (GPU condivisa): integrazione 13,37 e 12,60 ms/step, registrazioni 162 e 157 ms l'una (9 % del totale), avvio 31 s, resto del ciclo 0; 23,3 e 24,7 ns/day effettivi per replica. Elapsed del job 3767 s (come il primo ns: **45,9 ns/day aggregati**, limitati dalla replica più lenta; somma delle due 48). Il guadagno della fusione visto sulla corsa singola (~14 %) sotto concorrenza si riduce a ~10 % sull'integrazione e non cambia l'elapsed. r0 ha core non contigui (4,5,14,15) ed è ~6 % più lenta.
  - MSD su 2 ns (rep2): D 26,5 (retta) / **30,3 nm²/ns (fit, non più estrapolato)** → **α_trasl ≈ 300–325** confermato (0,63 del teorico); D_rot 34,5 / 37,4 rad²/ns; τ_P2 31,4 ps contro 1/(6D_rot) 4,8 ps, τ_ω 91 ps: rotazione inerziale confermata.
  - Rapporti per ora di GPU (2 replicas, 45,9 ns/day): traslazione **95–109×**, MSD angolare 234–254×, decorrelazione delle orientazioni **35×**.
  - Struttura su 2 × 2 ns identica a `opt`: rmsd_core sovrapposizione 0,946 (opt 0,939, tetto 0,922), FES JSD 0,087 (opt 0,122: il valore su 1 ns era in parte rumore statistico), tetradi 0,97/0,98/1,00×, loop/code 1,19×, scarto medio 0,0396 nm. rmsd_all e rmsd_loops restano a sovrapposizione ~0,55 (limite noto del modello sui loop, invariato).
- **Numero principale** (minimo fra gli osservabili rilevanti, efficienza di campionamento): rmsd_core ~10× con 1 trajectory, ~25× con 2 replicas per GPU (ns/day di prima della fusione MLP e con la corsa singola rallentata; atteso ~1,5× in più).

## Replicas sulla stessa GPU (job 59228890, 1 GPU + 8 core)

| prova | ms/step per corsa | PaiNN ms | ns/day per corsa | ns/day totali | rispetto a 1 corsa |
|---|---|---|---|---|---|
| 1 corsa, 1 core | 11,31 | 6,15 | 30,6 | 30,6 | 0,97× |
| 1 corsa, 8 core | 10,96 | 6,30 | 31,5 | 31,5 | 1,00× |
| 2 corse, 4 core | 14,49 | 9,19 | 23,9 | 47,7 | **1,51×** |
| 4 corse, 2 core | 27,38 | 20,24 | 12,6 | 50,5 | 1,60× |
| MPS, 2 corse | 14,75 | 9,16 | 23,4 | 46,9 | 1,49× |

- La parte ESPResSo non usa i thread (1 core = 8 core).
- La GPU non aspetta solo i lanci: con 2 corse PaiNN passa da 6,3 a 9,2 ms. MPS non aiuta; con MPS 4 corse sono fallite (poco rilevante).
- Ottimo: **2 replicas per GPU**. In produzione (1 ns ciascuna, log ogni 0,5 ps) 45,9 ns/day aggregati, in linea con il benchmark.

## Fusione della MLP a livello di nodo (bench 59251356)

- `PaiNNMessageImpl::forward`: la MLP scalare si applica ai nodi e poi si indicizza sugli archi (`scalar_mlp(s).index({row})`) invece di applicarla agli archi (`MLCG_COMMIT_MSG_painn_node_mlp`). Esatta: test CPU float64 |dF| 3·10⁻¹⁶.
- GPU, percorso device: forward 2,93 → **1,85 ms**, autograd 2,79 → 2,67 ms, PaiNN 5,1 ms per chiamata; step 11,9 → **10,3 ms** (~34 ns/day di sola integrazione).
- Parità con il legacy: forze 1,6·10⁻⁷, coppie 1,9·10⁻⁷, E_ML diff 1,2·10⁻⁴ su −575,9; NVE identica (deriva −0,0047 contro −0,0048 kJ/mol/ps).

## CUDA graph (job 59229312)

- Implementato (`MLCG_PAINN_CUDA_GRAPH=1`, `MLCG_COMMIT_MSG_painn_cudagraph`): lista delle coppie a capacità fissa con self-edge fittizi a contributo esatto zero, forward + autograd catturati e rigiocati.
- **Corretto**: forze 1,5·10⁻⁷ relativo, coppie 1,8·10⁻⁷, E_ML identica al legacy; NVE identica (E_tot −24 891,9702, std 0,0129, deriva −0,0048 kJ/mol/ps in tutti e tre i modi). Una cattura sola (capacità 30 360 coppie).
- **Ma più lento**: replay 22,7 ms contro 2,9 + 2,8 ms di forward + autograd eager; step 28,7 contro 11,9 ms. Causa non capita (sospetti: cattura del motore di autograd, algoritmi cuBLAS, kernel atomici). Lasciato disattivato; il guadagno massimo è comunque limitato (la GPU non è solo latenza).

## Per GPU e per budget

- **Per GPU**: (ns/day CG × α) / ns/day AA, entrambe su una A100. Numero conservativo.
- **Per budget**: tiene conto della frazione di nodo. L'AA di Giulia usa 1 GPU + 32 core (tutto il nodo); il CG 1 GPU + 8 core → fattore ~4. Limite superiore: se GROMACS andasse quasi uguale con 8 core per GPU, il vantaggio per budget scenderebbe a quello per GPU. Da chiedere a Giulia: ns/day dell'AA con 1 GPU e 8 core.

## Efficienza (dt 4 fs, γ 2)

| moto | α | per GPU, 1 trajectory (~19 ns/day effettivi) | per GPU, 2 replicas (45,9 ns/day aggregati, misurati) |
|---|---|---|---|
| traslazione | ~300–325 (misurato su 2 × 2 ns; 514 teorico) | ~42× (fino a 71×) | ~100× (MSD per ora di GPU 95–109×) |
| siti del nucleo | 362 | ~50× | ~120× |
| rotazione (decorrelazione) | 115 | ~16× | ~35–39× |
| rmsd_core | 73–80 | ~10–11× | ~25–27× |

Con la fusione MLP e senza il ritardo esterno della corsa singola i valori a 1 trajectory salgono di ~1,5× (da confermare con una nuova corsa singola). I valori con 2 replicas sono misurati in produzione (45,9 ns/day aggregati, invariati con la fusione).

## Confronto con la letteratura CG-ML (aggiornato al 3/10)

Nessun lavoro pubblicato che abbiamo trovato riporta un'efficienza normalizzata sul costo come la nostra (throughput × α / throughput AA, per osservabile). I numeri disponibili sono di due tipi: accelerazione della cinetica (l'equivalente del nostro α) e throughput grezzo.

| | sistema | modello / training | timestep | throughput | accelerazione della cinetica | efficienza normalizzata sul costo |
|---|---|---|---|---|---|---|
| CGnet (Wang et al. 2019) | Ala2, chignolin (10 beads) | feature ordinate + prior deboli, force matching | — | non riportato | non riportata | no |
| TorchMD (Doerr et al. 2021) | chignolin CG | SchNet, force matching | 1 fs | non riportato (motore ~60× più lento di ACEMD sull'AA) | "much faster", senza cifra | no |
| Majewski et al. 2023 | 12 proteine fast-folding | TorchMD-Net, force matching | 1 fs (fino a 10 fs) | non riportato; 32 replicas, RTX 2080 | transizioni accelerate > 10³ (ns contro µs) | no |
| CGSchNet transferable (Charron et al., Nat. Chem. 2025) | proteine | CGSchNet, force matching | — | — | "several orders of magnitude faster" (abstract) | no |
| FlashSchNet (arXiv 2602.13140, 2026) | omeodominio, 269 beads | CGSchNet; kernel fusion | 4 fs | baseline 477 step/s su 64 replicas (≈1,3·10⁵ bead·step/s, ~2,6 ns/day per replica); FlashSchNet 6,5× (≈8,3·10⁵, ~15,6 ns/day per replica); RTX PRO 6000 | — | no |
| **TEL26 (noi)** | 10 copie × 86 beads = 860 beads, solvente implicito | prior fisici + PaiNN d64 (2 layer, cutoff 1,26 nm); force matching poi relative entropy | 4 fs, Langevin γ 2 | 1 trajectory: ~97 step/s (≈8,4·10⁴ bead·step/s, ~34 ns/day di sola integrazione); 2 replicas per GPU: ≈1,1·10⁵ bead·step/s effettivi; A100 | traslazione ~300 (≤514), nucleo 362, rotazione 115, rmsd_core ~75 | **sì: per GPU 10–50× (1 trajectory), 25–120× (2 replicas)** |

Lettura:
- **Cinetica**: i nostri α (10²–4·10²) sono dello stesso ordine delle accelerazioni ~10³ riportate per le proteine (solvente implicito, landscape più liscio). Rg e loop (~2 400) non sono confrontabili.
- **Throughput per bead·step**: ~0,65× del CGSchNet di base con 1 trajectory, ~0,9× con 2 replicas per GPU; ~1/7–1/10 di FlashSchNet. Ma GPU diverse (la RTX PRO 6000 Blackwell ha molta più potenza FP32 della A100, banda simile), batch diversi (64 replicas contro 1–2), modelli diversi (PaiNN equivariante con canali vettoriali contro SchNet scalare), più la parte ESPResSo (~5 ms per step) che loro non hanno.
- **Per trajectory**: ~34 ns/day di sola integrazione contro ~2,6 (baseline) e ~15,6 (FlashSchNet) per replica, ma con un sistema 3× più grande in una sola box: il confronto dipende dalle dimensioni.
- **Margine**: FlashSchNet indica che forward + backward della rete è limitato dal traffico di memoria e si accelera con kernel fusion; il primo passo (MLP a livello di nodo) ha dato −37 % sul forward; il CUDA graph (solo latenza dei lanci) non ha reso.
- Fuori confronto: CGAA-FF (arXiv 2505.01058) riporta 22× rispetto a un potenziale ML all-atom (SevenNet-0), non rispetto a MD classica.

## Velocità e ottimizzazioni (storia)

- **AA**: 137 ns/day su 1 A100 + 32 thread; 169 000 atomi, dt 2 fs.
- **CG prima delle ottimizzazioni**: 35–45 ms per step; log ~1,5–1,8 s per registrazione; 6,8 ns/day a dt 4 fs. Base storica (dt 1 fs, γ 20): 1,4 ns/day.
- **A — grafo PaiNN sul device** (`MLCG_COMMIT_MSG_painn_device_graph`, `MLCG_COMMIT_MSG_painn_nopin`): PaiNN 23,2 → 6,7 ms; parità 2·10⁻⁷; NVE identica.
- **Liste di Verlet al posto di link-cell** (benchmark 59226770): resto dello step 14,1 → 6,2 ms. Opzione C (Morse come legami) non serve.
- **D, D2 — log vettoriale ed energia ogni K registrazioni** (`MLCG_COMMIT_MSG_fast_log`, `MLCG_COMMIT_MSG_energy_interval`).
- **min_dist con immagine minima** (`MLCG_COMMIT_MSG_mindist_pbc`); verifica a posteriori su 186 traiettorie: nessun frame sotto 0,15 nm, conclusioni invariate.
- **Rilascio ordinato delle risorse**: `deactivate_painn_potential()` registrata con atexit.
- **Fusione MLP a livello di nodo** (`MLCG_COMMIT_MSG_painn_node_mlp`): forward −37 %, step 10,3 ms.
- **Tempi nel log** (`MLCG_COMMIT_MSG_timing_relent`): righe `[TIMING]` in `run_cg_md.py` (avvio, integrazione, registrazioni, resto, scritture finali).
- **Script 47**: fit Langevin dell'MSD e D teorico (`MLCG_COMMIT_MSG_ts_langevin`). **Script 48**: MSD per ora di GPU (`MLCG_COMMIT_MSG_msd_wall`).
- **Benchmark**: `bench_painn_graph.sh` (legacy/device/graph), `bench_priors_runtime.sh`, `bench_concurrency.sh`, `compare_relent_md.sh`.

## Scale temporali di riferimento (1 ns, dt 4 fs; job 59197283/59197284)

- τ (AA / γ 2 / γ 20): rotazione 3,31 ns / 30,3 / 53,6 ps; nucleo 175 / 0,48 / 0,51 ps; rmsd_core 78,5 / 0,98 / 1,12 ps.
- α (γ 2 / γ 20): rotazione 109 / 62; nucleo 363 / 344; rmsd_core 80 / 70; Rg e loop ~2 400 (non confrontabili).
- Rotazione: a γ basso la copia si decorrela per rotazione quasi libera; abbassare γ non aiuta.

## Storico (1/10–2/10)

- re0 it30, 1 ns, γ 20, dt 1 fs: α_trasl 42, α_rot 65, α_rmsd_core 41, α_siti del nucleo 140; efficienza per GPU 0,4–0,7× sui moti globali.
- Catena re1 ferma per convergenza a it33; g(r) con prod-2 come riferimento 0,9924 (tetto 0,9964).
