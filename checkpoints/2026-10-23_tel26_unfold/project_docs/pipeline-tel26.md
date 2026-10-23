# Pipeline TEL26 (2JPZ, ibrido 3+1) — stato e numeri

> **Obiettivo del progetto: riprodurre la g(r) delle simulazioni atomistiche.**
> Ogni scelta — prior, architettura, checkpoint, protocollo — si giudica su
> quanto la g(r) CG (intra e inter, per canale) si avvicina a quella all-atom
> mappata. Stabilità e skill sulle forze sono condizioni necessarie, non il
> bersaglio.

Aggiornato al 2026-09-28 (miglior modello: **lp1 + residuo PaiNN a entropia relativa, it30, P(r) intra 0,985** (tetto 0,9955, confermato da prod-2) contro 0,976 dei soli prior; il force matching peggiorava la struttura: vedi in fondo allo studio di fattibilità). Al 2026-09-23: Su 100 ps termostatati il D=64 ep1 tiene compatto il
quadruplex (B3–B3 intra 0,49 contro 0,13 dei soli prior) ma **perde la
struttura fine** delle tetradi. È in corso uno **studio di fattibilità su prior
analitici non convenzionali** (Morse sito-sito, FENE, LJ pair-specific). Primo
insieme: Morse sito-sito sulle B3 (`PRIOR_SET=b3morse`). Tutorial in
`MLCG_Framework_v2/tutorials/tel26/`.

> **Il criterio operativo:** prima la *sopravvivenza* (`E_kin`, `max_f`,
> `min_dist` lungo la corsa), poi la sovrapposizione della g(r) per canale —
> **intra B3–B3** in primo luogo — contro i soli prior **nelle stesse
> condizioni** (stesso γ, stessa durata).

## Da fare, in ordine

1. **Commit sul Mac**: `.git/MLCG_COMMIT_MSG_FTOP` (AA_FORCES_TOPOLOGY passato
   allo stadio dataset). Verificare che il push di `f2d0908` (PRIOR_SET) sia
   arrivato: il primo tentativo ha preso un 500 da GitHub.
2. **Analisi b3morse** a produzione solo-prior finita (job 58468720):
   `RUNS="priors=samples_priors_g20_100ps.npz
   b3morse=samples_priors_b3morse_100ps.npz
   ep1=samples_tel26_d64_model.ep1_g20_100ps.npz"`, `RDF_ARGS="--types
   --skip-ps 50 --run-stride 10 --plot rdf_tel26_b3morse"`. Decide se allenare
   (`train PRIOR_SET=b3morse RUN=d64`) o correggere prima i prior.
3. Se i picchi B3–B3 escono troppo stretti: rifare il fit con `--width std` o
   con una σ corretta per il doppio conteggio (vedi sotto).
4. **Build di trainer ed ESPResSo** per il TF32 quando non ci sono produzioni;
   poi i benchmark `bench_b4_tf32`, `bench_b32`, `bench_b32_tf32` contro `d64`.
5. Ottenere dalla collega la definizione del modello unfoldable (mapping,
   potenziali, parametri): è il bersaglio finale del residuo ML.

Nota operativa: la copia di lavoro su Leonardo è
`/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT`; ne esiste un'altra sotto
`$HOME/PAINNLT` che i job non usano. Log in `PAINNLT/logs/`.

## I dati di partenza

`/leonardo_work/IscrB_G4MES/GiuliaC/machine_learning/Tel26_Hybrid_2jpz_10_monomeri/box_12nm_spce/run/`

| file | contenuto |
|---|---|
| `md_tel26_hybrid_2jpz_box12nm.tpr` | sistema completo, 168 979 atomi |
| `prod-1.part0001.xtc` | posizioni, 8 680 atomi, 230 MB |
| `prod-1.part0001.trr` | forze, sistema completo, 13,8 GB |
| `prod-2.part0002.*` | continuazione, ~3 200 frame |
| `prod-3.part0003.*` | troncone, un frame: da ignorare |

`dt = 0.002`, `nstxout = 0`, `nstfout = nstxout-compressed = 10000`,
`compressed-x-grps = non-Water`. Forze e posizioni agli stessi passi: 6 788
frame, ~136 ns. 260 nucleotidi (10 × 26) più 250 K⁺. Lo stadio dataset vuole
`AA_FORCES_TOPOLOGY` = il `.tpr` completo.

## Topologia CG

Scatola 11,91 nm, sequenza `(TTAGGG)₄TT`. Tetradi `(4,12,16,22)`,
`(5,11,17,23)`, `(6,10,18,24)`; piega ibrida 3+1, tre verifiche concordi. 250
bond di backbone, 180 Morse **COM–COM** (D = 50 kJ/mol, a = 0,3 nm⁻¹), 240
angoli. Il tranello PBC (`8947717`) rovinava le distanze ma non il registro.

## Pavimento di rumore

| | TEL22 | TEL26 canonico | TEL26 b3morse |
|---|---|---|---|
| RMS forza residua istantanea (kJ/mol/nm) | 887,1 | 721,2 | 717,6 |
| R² del canale radiale di coppia | 0,0033 | 0,0050 | 0,0101 |
| \|z\| massimo dati reali / shuffled | 11,0 / 2,4 | 21,9 / 2,1 | 31,2 / 2,1 |

Allineamento `.xtc`/`.trr` confermato. Quel R² è un **pavimento** (`7e5331f`):
la skill del trainer, `100 × R²`, arriva all'8–9%.

## La selezione del checkpoint

Dal commento in `training/train_painn.cpp`, verifica sul TEL22:

| modello | g(r) B3–B3 misurata | accordo dalle forze | dalla g(r) su riferimento |
|---|---|---|---|
| D=32 | 0,587 | 0,000 | 0,035 |
| D=64 | **0,746** (migliore) | 0,357 | 0,358 |
| D=128 | **0,286** (peggiore) | **0,480** (max) | **0,568** (max) |

Le metriche sull'ensemble di riferimento **ordinano al contrario**. Si usano i
checkpoint periodici, non il `best` dell'early stopping, e si sceglie sulla
g(r) prodotta dalla dinamica.

## Primo training, D=128 — instabile

Config partita da quella TEL22, **hidden 128** — la larghezza peggiore secondo
quel commento, mai riportata nella config. 40 epoche, skill piatta all'8%.

| checkpoint | `E_kin` / equip. | `max_f` | esito |
|---|---|---|---|
| ep5 | 1,62 | 600 | completato, ma deriva verso B3–B3 a ~0,22 nm |
| ep10–ep40, `best` | 3,7–39 | 13 000–630 000 | abort |

**Perché**: non overfitting sul validation set, ma estrapolazione fuori dalla
varietà campionata. Una rete più larga impara più ripida l'attrazione B3–B3, e
sotto la distanza di contatto — dove il riferimento non ha dati e la WCA è
troppo morbida — continua. Con il 99,5% di rumore nel target la loss tocca il
pavimento subito e le epoche successive inseguono rumore.

## Secondo training, D=64

`tel26_training_config.d64.json`: hidden 64, 15 epoche, snapshot con manifest a
ogni epoca. Skill 8,6–9,0%. Sweep a 2 ps: sopravvivono 14 su 15; la firma
B3–B3 sparisce. Solo ep1–ep3 superano i prior sul B3–B3 intra.

## Il termostato: γ è un attrito, non un tasso

In ESPResSo il rilassamento dura **m/γ**: con masse di 250–330 amu il `γ = 1`
storico dà ~300 ps, e su corse brevi il termostato non agisce. Gli stati ML
uscivano a 1,7–2,2 volte l'equipartizione. Opzioni aggiunte: `GAMMA`,
`EQ_GAMMA`, `EQ_ML_STEPS`, `RUN_TAG`. Con γ = 20 (τ ≈ 15 ps) il calore esce.

## g(r): 20 ps e 100 ps termostatati (γ = 20), prior canonici

Sovrapposizione intra:

| canale | prior 20 ps | ep1 20 ps | prior 100 ps | ep1 100 ps | prior 50–100 ps | ep1 50–100 ps |
|---|---|---|---|---|---|---|
| tutti i siti | 0,695 | 0,907 | 0,674 | 0,925 | 0,667 | **0,929** |
| **B3–B3 (Hoogsteen)** | 0,207 | 0,466 | 0,151 | 0,485 | **0,131** | **0,493** |
| B5–B5 (stacking) | 0,528 | 0,687 | 0,513 | 0,697 | 0,511 | 0,692 |
| S–S (backbone) | 0,794 | 0,771 | 0,803 | 0,759 | **0,804** | 0,754 |

100 ps, ep1: `E_kin` 0,99–1,00 × equip. da 60 ps; `E_ML` da −31 213 a un
plateau ~−33 636; `min_dist` 0,25–0,27 nm. Prior stabili a 1,00–1,07.

**Cosa mostrano i grafici** (`rdf_tel26_100ps_*`):

- i soli prior **si srotolano** e continuano a farlo nella seconda metà: massa
  del B3–B3 a 1,5–2 nm; con `rmax` 5 intra 0,55, picco spostato di 0,72 nm;
- ep1 è **stazionario** (seconda metà = corsa intera), compatto, ma **impasta
  la struttura fine**: il riferimento ha due picchi B3–B3 netti a 0,45 e 0,65
  nm — lato e diagonale della tetrade, rapporto √2 — e ep1 tiene solo il primo
  (~0,42); i picchi B5–B5 a 0,42/0,75/1,0/1,27 nm sono sfumati;
- inter a `rmax` 2: ep1 L1 relativo 1,56 contro 0,07 dei prior, S–S inter sopra
  il riferimento da 0,7 nm — **possibile aggregazione indotta dal ML**;
- inter a `rmax` 5: dominato dalla coda ideale (0,88–0,97 per tutti), poco
  informativo. `rmax` 2 per la struttura, 5 solo per lo srotolamento.

Lettura: il residuo ML, su un segnale vicino al pavimento di rumore, non
ricostruisce una geometria che i prior non suggeriscono. I Morse COM–COM
(curvatura al minimo 2Da² = 9 kJ/mol/nm², fluttuazione ~0,5 nm) tengono i
centri, non l'orientamento delle basi.

## Studio di fattibilità: prior analitici non convenzionali

Obiettivo: la g(r) all-atom con prior analitici + residuo ML, tetradi per ora
formate (il modello unfoldable della collega, con LJ per legami
H/elettrostatica e ioni impliciti, è il bersaglio successivo; i suoi dettagli
non ci sono ancora). Stesso protocollo per ogni insieme: solo-prior e prior+ML
(D=64, sweep, 100 ps a γ = 20, analisi per canale con `--skip-ps 50
--run-stride 10`, inter anche con `--rmax 5`).

**Plumbing — `PRIOR_SET`** (`tutorials/tel26/_prior_set.sh`, `f2d0908`). Il
dataset dipende dai prior (le forze allenate sono F_AA − F_prior), quindi ogni
insieme ha topologia, dataset, `cg_priors` e modelli suoi:
`tel26_topology.<set>.json`, `tel26_<set>_dataset.bin`, `cg_priors.<set>.json`,
`tel26_<set>_<run>_model.pt`. `04`/`05` ricavano l'insieme dal manifest del
modello e rifiutano un residuo ML sopra prior diversi. Solo-prior:
`PRIOR_SET` esplicito, output `samples_priors_<set>[_<tag>].npz`.

**Passo 1 — Morse sito-sito B3** (`fit_tetrad_site_morse.py`). Tetradi dai K4
dei Morse esistenti; estremi spostati sul sito B3; per ciascuna delle 18
classi (tetrade × coppia, mediate sulle copie) r0 = mediana,
a = √(kT/2Dσ²) con σ = 1,4826·MAD, r_cut = r0 + 7/a, D = 50, `exclude_wca:
false`, reversibili. Sul TEL22: diagonale/lato 1,42, 1,42, 1,38.

Risultati sul TEL26:

- **fit** (1 358 frame): diagonale/lato 1,397, 1,411, 1,412; lati 0,39–0,46
  nm, diagonali 0,56–0,64 nm, σ 0,014–0,040 nm, a 3,9–11,2 nm⁻¹ (curvatura
  massima ~1,3·10⁴ kJ/mol/nm², periodo ~0,5 ps, ben risolto a dt 0,001),
  r_cut 1,1–2,2 nm. Tetradi del riferimento molto più strette che nel TEL22;
- **dataset** `tel26_b3morse_dataset.bin`: 6 788 frame, 24 min su DCGP;
- **pavimento di rumore**: R² radiale 0,0101 (doppio del canonico), |z| 31,2.
  Il residuo medio **cresce** invece di calare. Interpretazione probabile:
  **doppio conteggio** — ogni σ è attribuita a un solo Morse, ma la
  distribuzione osservata contiene già gli altri cinque contatti, gli angoli e
  la WCA, quindi i sei Morse sommati sono più rigidi del necessario e il
  residuo spinge in senso opposto. Segnale più grande e strutturato, quindi più
  apprendibile; rischio: picchi B3–B3 troppo stretti nei soli prior;
- **solo-prior 100 ps** (job 58468720): `E_kin` 0,99–1,07 × equip.,
  `min_dist` 0,27–0,30 nm, `max_f` 430–790, nessun abort;
- **g(r), seconda metà (50–100 ps)** — il solo-prior b3morse **batte il
  canonico + ML** sui tre canali strutturali:

  | intra | prior canonici | **b3morse, soli prior** | canonico + ML ep1 |
  |---|---|---|---|
  | tutti i siti | 0,667 | 0,873 | **0,929** |
  | B3–B3 | 0,131 | **0,585** | 0,493 |
  | B5–B5 | 0,511 | **0,720** | 0,692 |
  | S–S | 0,804 | **0,806** | 0,754 |

  | inter | prior canonici | b3morse | ep1 |
  |---|---|---|---|
  | tutti (L1 rel.) | 0,965 (0,07) | 0,936 (**0,10**) | 0,920 (1,56) |
  | B3–B3 | 0,288 | 0,396 | 0,484 |
  | S–S | 0,720 | 0,771 | 0,805 |

  Nessun eccesso di contatti fra copie (L1 0,10 contro 1,56 di ep1).

  **Dai grafici** (`rdf_tel26_b3morse_*.png`): i picchi B3–B3 non sono troppo
  stretti, anzi più bassi del riferimento (0,45 nm: 2,85 contro 4,0; 0,63 nm:
  1,6 contro 3,3), con una coda fino a 2 nm dove il riferimento è nullo oltre
  1 nm; manca la spalla a 0,35 nm (B3 di tetradi adiacenti). Nel B5–B5 manca
  il picco a 0,42 nm delle guanine sovrapposte e i picchi 0,75/0,9/1,0/1,27
  diventano una gobba larga: **le tetradi si formano ma non restano
  impilate**. Il doppio conteggio della rigidità, se c'è, è sovrastato da
  questo. S–S: nessuno riproduce i minimi a 1,25 e 1,55 nm (geometria dei
  loop). Inter: ep1 mostra aggregazione netta (S–S inter 0,2–0,4 dove il
  riferimento è 0,05–0,13); b3morse segue il riferimento.

- **Passo 1b — impilamento** (`--stacking CG_DG_B5 --keep-tetrads`,
  `PRIOR_SET=b3stack`): Morse B5–B5 fra guanine sovrapposte (stesso tratto,
  residui consecutivi, tetradi adiacenti; 8 per copia), `"role": "stacking"`.
  Sul TEL22 alcune coppie hanno r0 ≈ 0,9 nm (syn/anti nell'antiparallela):
  vincolo reversibile alla geometria del riferimento, non contatto fisico.
  Commit in sospeso: `.git/MLCG_COMMIT_MSG_STACK`.
- **Confronto completo** (analisi 58503870, seconda metà di 100 ps a γ = 20):

  | intra | prior | b3morse | **b3stack** | ep1 can. | b3m ep1 | b3m ep2 | b3m ep3 | b3m ep4 | b3m ep5 |
  |---|---|---|---|---|---|---|---|---|---|
  | tutti | 0,667 | 0,873 | 0,938 | 0,929 | 0,951 | **0,952** | 0,949 | 0,945 | 0,946 |
  | B3–B3 | 0,131 | 0,585 | **0,722** | 0,493 | 0,699 | 0,682 | 0,682 | 0,672 | 0,615 |
  | B5–B5 | 0,511 | 0,720 | **0,842** | 0,692 | 0,743 | 0,744 | 0,747 | 0,735 | 0,757 |
  | S–S | 0,804 | 0,806 | **0,838** | 0,754 | 0,762 | 0,756 | 0,776 | 0,758 | 0,743 |

  Inter, L1 relativo (aggregazione): prior 0,07, b3morse 0,10, b3stack 0,41,
  b3m ep1 2,29, ep2 0,81, ep3 1,10, ep4 0,22, ep5 0,74; non monotono con le
  epoche, statistica povera (10 copie, 50 ps).

  **Letture.** (1) Il **solo-prior b3stack** è il migliore sui tre canali
  strutturali, meglio di ogni corsa con ML. (2) Il ML sopra b3morse porta il
  totale a 0,95 (i canali non analizzati) e recupera parte dell'impilamento,
  ma meno del Morse B5 esplicito. (3) **Il ML peggiora il backbone** in modo
  sistematico con entrambi gli insiemi di prior (S–S 0,80 → 0,74–0,78).
  (4) ep1–ep3 equivalenti; ep4–ep5 peggiorano B3–B3 e crescono le forze
  massime (5 000–6 800 contro ~1 500 dei soli prior); `min_dist` minimo
  0,215–0,226 nm per tutti, soli prior compresi (fisiologico).
  Costo: soli prior 36 min (b3morse), 54 min (b3stack, +120 marker); ML 69–78.
- **Dai grafici** (`rdf_tel26_all_*`): b3stack mette il picco B5–B5 a 0,42 nm
  esattamente all'altezza del riferimento; le corse ML sul b3morse non lo
  formano mai (~0,4). ep5 ha un picco B3–B3 spurio a ~0,30 nm (inizio del
  collasso). **Il ML attacca le copie**: S–S inter 0,2–0,48 fra 0,7 e 1,1 nm
  (riferimento 0,04) in ep1 canonico, b3m_ep1, b3m_ep2; mai nei soli prior.
  S–S intra a 1,2–2 nm (minimi 1,28/1,55, picchi 1,42/1,9): nessuno lo
  riproduce.
- In corso: training D=64 su `PRIOR_SET=b3stack` (58505059) e produzioni
  ep1–ep3 (58505060–62).
- **b3stack + ML** (training 58505059, produzioni 58505060–62; analisi
  58516550): stabili (`min_dist` 0,223–0,226 come i soli prior, `max_f`
  2 100–3 800 contro 1 400).

  | intra | b3stack prior | b3m ep1 | b3s ep1 | b3s ep2 | b3s ep3 |
  |---|---|---|---|---|---|
  | tutti | 0,938 | 0,951 | 0,950 | 0,947 | **0,954** |
  | B3–B3 | 0,722 | 0,699 | **0,756** | 0,752 | 0,730 |
  | B5–B5 | **0,842** | 0,742 | 0,791 | 0,784 | 0,775 |
  | S–S | **0,838** | 0,762 | 0,779 | 0,741 | 0,754 |

  Il ML sopra b3stack: recupera in parte la spalla B3–B3 a 0,35 nm e il
  totale (canali non analizzati), ma **abbassa** il secondo picco B3–B3 (0,65
  nm: 1,5 contro 1,95 dei prior e 3,3 del riferimento), il picco B5–B5 a 0,42
  (1,0 contro 1,27) e peggiora l'S–S. Inter: **ep1 aggrega di nuovo** (S–S e
  B3–B3 inter in eccesso da 1,0 nm), come ep1 canonico e b3m_ep1; ep2 ed ep3 no
  (seguono il riferimento o stanno sotto). Il primo checkpoint aggrega in tutti
  e tre gli insiemi di prior.
- **Passo 2 — Debye–Hückel fra i backbone** (implementato, commit
  `.git/MLCG_COMMIT_MSG_DH`): carica −1 sul sito unico di DA/DT e sul sito S di
  DG, λ_D = 0,87 nm (250 K⁺ in 11,91 nm, I = 0,123 M), ε_r = 78, r_cut 4 λ_D.
  **ESPResSo non applica le esclusioni all'elettrostatica**: il DH agisce su
  tutte le coppie cariche, legate comprese, e il builder sottrae la stessa
  somma. Verificato sul TEL22 (dataset con/senza DH, differenza = DH entro
  2·10⁻⁴). Da verificare su Leonardo: `diagnose_debye_huckel.py` (runtime).
  Nota: il WCA "auto" non è deterministico fra due build (differenze ~10⁻⁴ nm
  nei σ): per confronti esatti fra dataset usare `--priors`.
- **Passo 3 — LJ 12-6 al posto dei Morse** (implementato, stesso commit
  `.git/MLCG_COMMIT_MSG_DH_LJ`): `fit_tetrad_site_morse.py --form lj`,
  σ = r0/2^{1/6}, ε = kT r0²/(72 σ_r²), r_cut = 3,5 σ, shift a zero. Profondità
  legata alla larghezza osservata (sul TEL22 ε = 0,5–4,4 kT; sul TEL26, più
  stretto, attesi ~2–12 kT). Runtime sugli stessi marker dei Morse. Verificato
  sul TEL22 nel builder (3·10⁻⁶ relativo); runtime da verificare con
  `diagnose_pair_specific_lj.py`. Rischio: muro r⁻¹² ripido dove il
  riferimento scende sotto σ (sul TEL22 forze LJ fino a ~7 000 kJ/mol/nm).
  Varianti da lanciare: `b3dh` (Morse + DH) e `b3dh_lj` (LJ + DH), soli prior
  poi prior + ML.
- **Verifiche runtime su Leonardo** superate: `diagnose_debye_huckel.py` e
  `diagnose_pair_specific_lj.py` coincidono col kernel del dataset a ~10⁻¹⁶;
  le esclusioni non agiscono sull'elettrostatica (confermato). Sul login serve
  `LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6)` davanti a `pypresso`
  (l'RPATH del python di spack porta la libstdc++ del gcc 8); nei job no.
  Fix: `check_neutrality=False` (controioni impliciti, carica netta voluta).
- **Soli prior b3dh / b3dh_lj** (100 ps, analisi 58525546). Pavimento |z|:
  b3dh 33,2, b3dh_lj 44,1 (b3stack 31,0). Stabili (`min_dist` 0,226–0,227).

  | intra | b3stack | b3dh | b3dh_lj |
  |---|---|---|---|
  | tutti | 0,938 | 0,933 | 0,905 |
  | B3–B3 | 0,722 | 0,721 | 0,659 |
  | B5–B5 | 0,842 | 0,841 | 0,798 |
  | S–S | 0,838 | 0,828 | 0,826 |

  Il DH **non tocca l'intra** (curve sovrapposte al b3stack): la struttura S–S
  a 1,2–2 nm non viene dalla repulsione fra fosfati. I **LJ tengono le
  tetradi** (ε 2–14 kT), con picchi più bassi (B5 0,42 nm: 0,95 contro 1,27) e
  code più lunghe. **Inter: il DH a carica piena (q = −1) svuota l'inter fino
  a 2 nm** (B3, B5, S–S ~0, totale 0,02 contro 0,13 a 2 nm). Anche il b3stack
  senza DH sta un po' sotto il riferimento nell'S–S inter. Scansione della
  carica efficace (condensazione dei K⁺; U ∝ q²): `b3dh05` (q = −0,5, dataset
  58526642, produzione 58526644) e `b3dh03` (q = −0,3, 58526646/58526649), sui
  Morse. La carica si fissa sull'inter senza toccare l'intra.
- **Svolta (decisione di Cristiano): separare struttura della copia e
  interazioni fra copie.** Scansione della carica sospesa; DH e inter messi da
  parte. Bersaglio ora: **P(r) intra per canale**. Ciclo veloce per i soli
  prior: `derive_prior_set.py --base b3stack --set <nuovo>` (niente dataset;
  04/05 usano `tel26_dataset.bin` come configurazione iniziale se il ML è
  spento). Primo candidato: rete elastica B5 sul nucleo di guanine
  (`--stacking-mode core`, 66 coppie per copia) per i picchi B5–B5 a
  0,75–1,27 nm; poi geometria della tetrade (secondo picco B3–B3) e loop (S–S a
  1,2–2 nm, angoli/diedri).
- **b3core** (Hoogsteen + 66 Morse B5 sul nucleo, soli prior): B3 0,749, B5
  **0,916**, S–S 0,862, totale 0,937 (invariato). Picchi B5 tutti al posto
  giusto ma troppo alti (rete sovravincolata: 84 vincoli > 66 gdl interni del
  nucleo); B3 con coda fino a 1,3 nm e senza spalla a 0,35.
- **Decisione di Cristiano: solo prior che corrispondono a interazioni
  fisiche** (Hoogsteen B3 nel piano; impilamento fra guanine sovrapposte, modo
  `tract`). La rete `core` resta solo come confronto; b3core25/b3core15 da
  annullare.
- **Metodo convergente** (preoccupazione: "di ora in ora rischiamo di non
  convergere"): `iterate_contacts.py`, aggiustamento iterativo di r0 e a per
  classe sulle distribuzioni del riferimento, da una corsa coi soli prior;
  arresto a |Δmed| < 0,01 nm e |σ ratio − 1| < 10 %, massimo ~5 iterazioni.
  Partenza: b3stack (corsa già disponibile). Insiemi b3it1, b3it2, …, derivati
  sempre da `--base b3stack`.
- **Iterazione 1 sul b3stack**: i 26 contatti hanno già |Δmed| ≤ 0,015 nm e
  σ_CG/σ_rif 0,80–1,07: le distanze fra basi a contatto sono giuste; lo scarto
  della g(r) sta nelle coppie fra basi non a contatto → manca l'orientazione
  relativa delle tetradi impilate.
- **Torsione** (fisica: dipendenza angolare dell'impilamento): diedro a coseno
  B3–B5–B5–B3 fra guanine sovrapposte, 8 per copia (`--twist`). TEL26:
  φ0 ≈ −35° fra tetradi 1-2, ≈ +20° fra 2-3 (firma della piega ibrida), dev.
  circolare 3–4,5°, K 164–301 kT. Convenzione verificata
  (`diagnose_dihedral.py`: energia 10⁻¹⁵, forza 10⁻⁹).
- **Stadio `iterate`** (catena automatica iterate→derive→04→05 in un job GPU,
  stop a `[CONVERGIUTO]`). Catene in parallelo: `cA` (contatti, da b3it1,
  job 58534673), `twA` (contatti + torsione, da b3tw, 58534676), `twB` (idem,
  DAMP 0,5). Corse di partenza b3it1 58533947, b3tw 58534377.
  Nota: `AFTER` va messo davanti al comando (`AFTER=<id> bash hpc/...`); dal
  commit AFTER funziona anche come argomento.
- **Esito delle catene** (analisi 58559971): twA convergiuta in 2 iterazioni
  (twA2), twB in 3 (twB3), cA quasi (25–26/26 per 4 iterazioni; una σ di
  tetrade oscilla 0,89–1,19, rumore).

  | intra | b3stack | cA4 | twA2 | twB3 | b3core (non fisico) |
  |---|---|---|---|---|---|
  | tutti | 0,938 | 0,941 | **0,944** | 0,944 | 0,937 |
  | B3–B3 | 0,722 | 0,722 | **0,751** | 0,749 | 0,749 |
  | B5–B5 | 0,842 | 0,850 | 0,867 | 0,869 | **0,916** |
  | S–S | 0,838 | 0,837 | 0,837 | 0,842 | 0,862 |

  Termini locali a convergenza ≠ g(r): guadagno piccolo. La torsione vale
  ~+0,03 su B3–B3 e B5–B5. Curve fisiche quasi sovrapposte: secondo picco
  B3–B3 a 0,65 nm ~1,9 (rif 3,3) con coda fino a 1,4 nm; picchi B5–B5 a
  0,75/1,0/1,27 impastati; S–S a 1,2–2 nm piatto; P(r) totale in difetto a
  0,8–1,0 nm e in eccesso oltre 1,4 nm (copia troppo estesa). `--all-pairs`:
  scarto distribuito (0,004–0,011 per coppia); peggiori le coppie con DA
  (sovrapp. 0,58–0,72) e DT (0,73–0,85) → **loop e code**.
- **Notte 24→25/9, due rami in parallelo sopra twA2**:
  1. **Diedri di backbone** (`fit_backbone_dihedrals.py`, 23 per copia sul
     sito 0, K ≤ 100 kT): corsa `bbd0` coi soli prior, poi catena `iterate`
     PREFIX=bbd NITER=4.
  2. **ML**: dataset `PRIOR_SET=twA2`, pavimento, training D=64,
     produzioni ep1–ep3 da 100 ps (`tel26_twA2_d64_model.ep<N>.pt`,
     RUN_TAG=g20_100ps). Verifica l'ipotesi "ML dopo i prior fisici".

  Job: bbd0 58560832 → iterate bbd 58560833; dataset twA2 58560834 →
  pavimento 58560835, training 58560836 → produzioni ep1–ep3
  58560837–58560839. Fit dei diedri di backbone: rigidi nei tratti di
  guanine (8–11: 30 kT, 11°; 14–17, 15–18: 11 kT, 18°), morbidi nella coda 5'
  (1–4: 0,9 kT, 77°) e nel tratto 17–24 col loop TTA 19–21 (1,8–2,4 kT,
  45–52°).

- **25–26/9: esiti e nuovi insiemi (tutti soli prior, seconda metà di 100 ps)**.
  Crash `bond broken` nella catena bbd: diedro di backbone indefinito (tre siti
  allineati, coda 3' residui 23–26); corretto con
  `install_dihedral_undefined_zero.py` + build (58607174), verificato.

  | intra | twA2 | bbd0 | **hb0** | hbit2 | hb1 (solo N2–N7 corti) |
  |---|---|---|---|---|---|
  | tutti | 0,944 | 0,958 | **0,971** | 0,972 | — |
  | B3–B3 | 0,751 | 0,783 | **0,843** | 0,835 | 0,828 |
  | B5–B5 | 0,867 | 0,885 | **0,905** | 0,900 | 0,891 |
  | S–S | 0,837 | 0,853 | **0,902** | 0,891 | 0,872 |

  - **bbd0** = twA2 + 23 diedri di backbone per copia: loop a posto (DA 0,59–0,72
    → 0,73–0,80, DT → 0,80–0,90). Iterare non serve (bbd2 = bbd0).
  - ML sopra twA2 (ep1–3): totale 0,956–0,960 ma B5–B5 0,77–0,82, S–S
    0,74–0,77, DT–DT 0,68 → ipotesi "ML dopo prior fisici" smentita.
  - **hb0** = bbd0 + secondo legame di Hoogsteen N2-H···N7, Morse B2(i)–B4(j),
    120 contatti (`fit_hoogsteen_hbond.py`; tetradi {4,12,16,22},
    {5,11,17,23}, {6,10,18,24}; 5 classi a 0,31–0,39 nm, 7 a 0,47–0,57 nm =
    vincoli di appaiamento più che legami). Coppie fra siti di guanina
    0,82–0,87 → 0,89–0,92. Corsa ~2 h (marker ~480). hb1 (solo i 50 corti,
    `--max-r0 0,42`) tiene ~metà del guadagno. **Scelta di Cristiano: hb0.**
  - Inter-copia rumorosa (0,84–0,93 fra corse quasi identiche): non giudicata.
- **In corso (26/9)**:
  1. **ML sopra hb0**: dataset 58697074 → pavimento 58697095, training D=64
     58697096 → produzioni ep1–3 58697098–58697100
     (`samples_tel26_hb0_d64_model.ep<N>_g20_100ps.npz`).
  2. **lp0** = hb0 + 3 contatti persistenti delle basi dei loop
     (`find_loop_contacts.py --s-max 0.07 --copy-tol 0.12`): A3–G22(B4)
     0,54 nm, A9–G18(B4) 0,54 nm, T8–T26 0,68 nm; 30 contatti. Le altre 11 basi
     di loop non hanno partner persistenti (mobilità vera). Corsa 58697360,
     analisi bbd0/hb0/lp0 58697361.

- **26/9, tetto e ML regolarizzato**.
  - Tetto di sovrapposizione (`44_tel22_rdf.py --ref-range 0:0.5` con corsa
    `@ref:0.5:1`): intra **0,9955** (B3–B3 0,979, B5–B5 0,971, S–S 0,961);
    14 ns danno lo stesso. Rumore della corsa CG (finestre `#50:75`,
    `#75:100`): ±0,0005 sul totale. hb0 0,971 → scarto 0,024 reale.
    Senza le coppie interne ai residui (`--no-same-residue`) lo scarto resta:
    non è il corpo rigido, è il campo di forza, distribuito su tutte le coppie.
    Inter: il riferimento contro sé stesso dà 0,85, sotto hb0 (0,93): con 10
    copie e 50 ps l'inter non si giudica.
  - lp0 (hb0 + cappucci A3–G22, A9–G18, T8–T26): coppie DT–base +0,03–0,04,
    DA–base +0,03–0,05 → **nuova base lp0** (~0,974).
  - ML d64 sopra hb0: ep1, ep2, ep3 tutte fermate dal guardrail (`max_f >
    10000` in 20 passi, `min_dist` ed `E_kin` normali) a 18,7, 42,9, 3,5 ps.
  - **Criterio di convergenza** (proposto): 100 ps senza guardrail, poi 1 ns;
    intra ≥ 0,985 e nessun canale peggiore di lp0.
  - In corso: dataset lp0 58710195; d64reg (λ_forze 0,3, wd 1e-3, 5 epoche)
    training 58710196, produzioni ep1–5 58710197–58710202; d64reg2 (λ 1,0,
    wd 3e-3) training 58710734, produzioni 58710735–58710740; blocco `prod-2`
    con gli stessi prior (`PART=prod2`) 58712038 → `concat_datasets.py --set
    lp0 --parts prod2 --out-set lp0x` se serve più dato.
  - Esiti previsti: stabile e sopra lp0 → validazione 1 ns; stabile senza
    guadagno → lp0x (più dati); esplode ancora → attenuazione del residuo ML
    fuori dall'intervallo del riferimento (plugin).

- **26/9 sera: la causa delle esplosioni ML e la CBT.**
  - d64reg su lp0: 5/5 epoche fermate dal guardrail (ep1 40,7 ps … ep4 già in
    equilibrazione). `diagnose_crash.py`: forze 14 000/8 000/6 000 su 15A, 14T,
    16G (copia 9), angolo 14T–15A–16G a sin θ = 0,0087; due diedri di backbone
    (K ≈ 10 kT) con forza ~K/(b sin θ). **Non la rete: il diedro a coseno
    singolare**, che il ML raggiunge raddrizzando i loop (stessa patologia del
    "bond broken").
  - Rimedio: diedro con **smorzamento angolare** (idea CBT di Bulacu 2013):
    V = K g(θ1) g(θ2)(1 − cos), g = 1 per sin θ ≥ 0,3, g = x(2 − x) sotto,
    x = sin²θ/0,09. Il sin³ originale di Bulacu scartato: sul backbone (angoli
    140–150°) ⟨sin³ sin³⟩ = 0,008–0,05, K ricalibrati 30–100 kT. Patch
    `install_dihedral_cbt.py` (mult < 0), builder `cbt: true`, guard
    `require_cbt_dihedral`, `apply_cbt.py` (K/⟨g1g2⟩, ⟨g1g2⟩ 0,86–1,00, K +0–16 %).
    Verificato fuori da ESPResSo: forze = −∇E a 4·10⁻⁶, energia = builder a
    2·10⁻¹³.
  - **lp1** = lp0 con diedri di backbone smorzati. Job: build 58734430; soli
    prior lp1 58734439; dataset lp1 58734443 → training d64 58734450 (prod.
    ep1–3 58734453, 58734460, 58734461) e d64reg 58734451 (ep1–5 58734492,
    58734496, 58734497, 58734503, 58734506).
  - Altri difetti corretti: training paralleli nella stessa cartella (flock in
    03), controllo WCA con `--priors` (avviso), `--no-same-residue` che gonfiava
    l'inter.

- **27/9: il ML stabile peggiora la struttura.** Analisi 58760433 (scala del
  tetto): lp1 soli prior 0,9762 (= lp0 0,9755); d64 ep1–3 0,934–0,940,
  d64reg ep2–3 0,934–0,939; B5–B5 0,90 → 0,77, S–S 0,91 → 0,76–0,79, B3–B3
  0,84 → 0,78–0,81. d64 ep1–3 tutte intere a 100 ps; d64reg ep1, ep4, ep5
  fermate (ep5: due timine consecutive 13T–14T collassate a 0,02 nm, coppia
  legata senza WCA). Blocco `prod-2` fermato dal controllo di guardia WCA
  (S–B2 0,376 < 0,395; B2–B3 0,2524 < 0,2551): serve filtrare i frame.
  Esperimenti diagnostici sul dataset lp1: d64t0 (torque_weight 0) training
  58761341, produzioni 58761343–58761345; d64half (validation_fraction 0,6)
  training 58761342, produzioni 58761346, 58761348, 58761350.

- **27/9 pomeriggio.** Diagnostici su lp1: d64t0 (senza coppie) 0,915–0,918,
  d64half (40 % dei dati) 0,932–0,936, d64 0,940 → né coppie né quantità di
  dati. Parità dei prior runtime ↔ dataset sul sistema completo (frame 0,
  `compare_prior_parity.py`): forze 9·10⁻⁵, coppie 10⁻⁴ relativo, pendenza
  1,000000 → nessuna differenza. Ipotesi rimasta: base incompleta del force
  matching. In corso: d64l3 (3 strati) training 58831328, produzioni
  58831330–58831332; d64l3c16 (3 strati, cutoff 1,6 nm) training 58831329,
  produzioni 58831333–58831335. Se non migliorano: allenamento sulla struttura.

- **28/9: rete più ampia, nessun salto.** Analisi 58882141: d64l3 e d64l3c16
  ep1–2 stabili (100 ps), ep3 fermate dal guardrail (pozzi spuri fra siti di
  base, come sul TEL22). B5–B5 0,78–0,80 e S–S 0,76–0,80 contro 0,77 del d64
  e 0,90–0,91 di lp1; coppie per tipo 0,78–0,87 (lp1 0,89–0,93). Solo le
  coppie DT–base salgono con l3c16 ep2 (0,87–0,90, livello lp1). Conclusione:
  il limite non è il raggio della rete. Prossimo passo proposto: allenamento
  a entropia relativa (solo posizioni, partendo dal d64 sulle forze).

- **28/9: allenamento a entropia relativa (implementato, commit
  `.git/MLCG_COMMIT_MSG_relent`).** Perché il force matching funziona in
  CGnet/CGSchNet e non qui: ~10⁶ frame contro ~6 800 con rumore al 99 %; prior
  deboli (la rete impara quasi tutto il PMF) contro prior già tarati sulla
  struttura (il residuo è la parte che FM vede peggio: partendo da lp1 FM può
  solo allontanarsi); metrica su 1–2 CV contro g(r) sito-sito. La patologia
  delle epoche tardive (pozzi spuri) è documentata anche là (Hessian matching,
  arXiv 2605.12823); Thaler et al., JCP 157, 244103 (2022): RE più efficiente
  sui dati e meno sensibile ai prior.
  - `training/train_relent.cpp`: gradiente β[⟨∂U/∂θ⟩_AA − ⟨∂U/∂θ⟩_CG] sulla sola
    energia della rete (i prior si cancellano); campioni CG ripesati finché
    ESS/N ≥ 0,5; ΔS_rel esatto rispetto a θ0 (β⟨ΔU⟩_AA + ln⟨e^{−βΔU}⟩_CG) su
    un sottoinsieme AA e sul 20 % finale tenuto fuori; salva lo stato col ΔS
    di holdout più basso. `--zero-init` (ultimo strato del readout a zero:
    U_ML ≡ 0, la prima corsa è lp1). `--check-gradient`: surrogato contro
    differenze finite in doppia precisione, errore 10⁻¹⁰ con ESS/N 0,7.
  - `preprocessing/samples_to_cg_bin.py`: `--sample_npz` → formato dataset,
    forze a zero, ordine dei siti verificato sul template.
  - `tutorials/tel26/06_relent.sh` e stadio `relent`: MD (25 ps, campioni
    dopo 5 ps ogni 0,1 ps, ripartendo dallo stato precedente con
    `--allow_checkpoint_mismatch`) → conversione → train_relent → manifest;
    ripresa automatica; stop dopo 2 iterazioni senza guadagno.
  - Scelta: partenza da **lp1 a correzione nulla**, non dal d64 (che parte già
    degradato). Prova su sistema giocattolo: ΔS −0,53 con distribuzioni
    diverse, −0,015 con la stessa distribuzione.
  - Da fare: build su Leonardo, `--check-gradient` sul TEL26 reale, catena
    NITER=1 per misurare il costo, poi la catena lunga e la g(r) per
    iterazione.
  - **Primi numeri (28/9).** `--check-gradient` sul TEL26 reale: convergenza
    eps², Richardson 10⁻¹²–10⁻¹³ (dopo aver reso relativa la perturbazione del
    controllo: quella assoluta faceva esplodere le feature). Job 58894711 ucciso
    dal gestore PMIx di Slurm al primo programma dopo la MD nello stesso srun →
    un passo srun per programma (`STEP_SRUN`). Job 58902567: MD 25 ps in 21 min
    (1,2 ps/min), train_relent 31 s; ESS/N sotto 0,5 al passo 9, miglior passo 7
    con ΔS holdout −7,37 (train −6,84): S_rel scende e l'holdout lo conferma.
    Pochi passi per iterazione perché ΔU è estensivo (10 copie): possibile
    ripesatura per copia se le iterazioni risultano troppe.
  - **g(r) dopo la prima iterazione** (analisi 58903272, 20 ps per corsa):
    intra it00 (lp1 a correzione nulla, stesso protocollo) 0,9743 → it01 0,9752;
    B3–B3 0,836 → 0,844, B5–B5 0,907 → 0,908, S–S 0,895 → 0,902. Tutti i canali
    nella direzione giusta, nessun peggioramento (col force matching: 0,976 →
    0,94); guadagno ancora al livello del rumore sul totale (±0,001 a 20 ps).
    lp1 a 50–100 ps dà 0,9763: il confronto giusto è it00 contro itN, a parità
    di finestra. it02: 2 passi, ΔS holdout −9,88; β·gap predetto −6,52 per
    it01, misurato −6,16 sui campioni nuovi (ripesatura coerente).
  - **Catena re0, it02–it08** (job 58903277, 58927703). it03 1 passo, it04–it05
    0 passi: **falsa convergenza** (β·gap sui campioni nuovi −6,2, −13,9,
    −18,8, −19,4: il gradiente non era nullo; il primo passo di Adam a lr 1e-3
    usciva già dalla regione ESS ≥ 0,5). Corretto con regione di fiducia
    adattiva (dimezzamento del lr fino a 6 volte, ripartenza da 2·lr accettato,
    guadagno solo se ΔS < −0,5; commit `MLCG_COMMIT_MSG_relent_trust`). it06
    −6,5, it07 −4,4, it08 −5,2 (lr ~1–2,5·10⁻⁴, 2–3 passi per iterazione).
    ΔS cumulato ~−42 sull'intero sistema, senza segni di rallentamento.
  - **g(r) it00–it05** (analisi 58926414, 20 ps ciascuna). it03, it04, it05
    sono lo **stesso modello** (pesi identici) su tratti consecutivi: il
    totale sale lo stesso (0,9765 → 0,9772 → 0,9778) → c'è deriva nel tempo
    (anche lp1: 0,9743 a 5–25 ps, 0,9763 a 50–100 ps), il totale non si giudica
    così. Canali stabili sui tre tratti: **B3–B3 0,836 → 0,8485 (+0,012,
    reale)**, **B5–B5 0,907 → 0,901 (−0,006, reale)**, S–S nel rumore.
    Inter it05: compare il B3–B3 inter e L1 1,13 (it03–it04 0,3–0,4) →
    possibile inizio di contatti fra copie, da seguire.
  - **Validazione alla pari** (in corso): it08, 100 ps da
    `equilibrated_priors_lp1_100ps.npz`, γ 20, log ogni 20 passi (job
    58934766), analisi 50–100 ps contro lp1 (58934768). La ripesatura per copia
    (10× la regione di fiducia) resta in sospeso fino a questo esito.
  - **28/9 sera: primo modello ML che batte i soli prior sulla struttura.**
    Validazione alla pari (job 58934766, analisi 58934768, finestra 50–100 ps,
    stesso stato iniziale e protocollo di lp1): intra **it08 0,9792 contro lp1
    0,9763** (+0,0029, sei volte il rumore di ±0,0005); scarto dal tetto
    (0,9955) da 0,0192 a 0,0163. B3–B3 0,837 → **0,860**, B5–B5 0,904 → 0,897,
    S–S 0,906 → 0,904 (rumore). Inter 0,854 → 0,921, L1 0,63 → 0,42: nessuna
    aggregazione (col force matching L1 1,5–2,3). Catena it09–it12: ΔS −8,9,
    −10,8, −6,6, −7,8 (cumulato ~−76), nessun rallentamento → continuare.
  - **29/9: it12 migliora ancora, su tutti i canali** (validazione 58946462,
    analisi 58946463, alla pari): intra lp1 0,9763 → it08 0,9792 → **it12
    0,9819**; B3–B3 0,837 → 0,860 → **0,881**, B5–B5 0,904 → 0,897 → **0,907**
    (recuperato), S–S 0,906 → 0,904 → **0,916**. Scarto dal tetto 0,0192 →
    0,0136 (−29 %). Inter nel rumore (0,85–0,92, tetto 0,85). Catena it13–it30
    (58946458): ΔS da −1,8 a −15,8 per iterazione, media ~−7,5 in it13–21 e ~−5
    in it22–30 (leggero rallentamento), cumulato ~−188. In corso: validazioni
    it20 (58964121), it30 (58964123), analisi 58964124, catena fino a it50
    (58964127).
  - **Ripesatura per copia** (`--residues-per-copy 26`, `RE_RESIDUES_PER_COPY`):
    energie della rete per (frame, copia) via batch_indices = frame·G + copia,
    pesi e ESS per copia, ΔS fattorizzato. `--check-gradient` passa con e senza
    gruppi; giocattolo con 10 copie: ΔS per iterazione −1,7 → −3,6. Commit
    `MLCG_COMMIT_MSG_relent_percopia` (file consegnati in chat: il Mac non era
    raggiungibile).
  - **29/9 mattina: criterio intra raggiunto con it30.** Validazioni alla pari
    (58964121/23, analisi 58964124), intra lp1 0,9763 → it12 0,9819 → it20
    0,9839 → **it30 0,9852**; B3–B3 0,837 → 0,881 → 0,885 → **0,911**, B5–B5
    0,904 → 0,907 → 0,915 → **0,922**, S–S 0,906 → 0,916 → 0,923 → **0,926**.
    Scarto dal tetto 0,0192 → 0,0103 (−46 %); incrementi in calo (+0,0020 in 8
    iterazioni, +0,0013 in 10).
  - **Tetto confermato** (dataset prod-2 solo posizioni 58969744, analisi
    58969762, riferimento prima metà di prod-1): half2 0,9955, **prod2 0,9954**
    (B3–B3 0,981, B5–B5 0,972, S–S 0,956 contro 0,961): il riferimento è
    convergente anche su scala più lunga, solo l'S–S cala di 0,005 (lieve
    deriva dei loop). Inter: tetto 0,84–0,85, le corse CG ci stanno sopra →
    l'inter con 10 copie non si giudica.
  - **Ripesatura per copia** (catena 58968567, da it35): it35 −9,8 (6 passi),
    it36 −11,6 (10), it37 −2,8, it38 −0,2 (lr crollato a ~3–6·10⁻⁵): i guadagni
    calano, primo segno di avvicinamento all'ottimo raggiungibile con questa
    rete e questo campionamento (ΔS fattorizzato, non confrontabile 1:1).
  - Prossimo: validazione da 1 ns di it30 in 4 tratti da 250 ps (i primi due
    prima della manutenzione del Booster, 30/9 08:00 – 1/10 08:00).
  - **Struttura della copia (script 46, job 58984847)**, riferimento prima metà
    di prod-1: rmsd_core AA 0,043 nm (half2 0,044, prod2 0,043) contro lp1
    0,102, it12 0,087, it20 0,085, **it30 0,077**; rmsd_loops AA 0,21 contro CG
    0,46–0,48; Rg medio giusto (AA 0,799, it30 0,797, lp1 0,824) ma
    distribuzione 1,7× più larga; Q 0,906 AA, 0,911–0,912 CG (contatti H
    persino un poco più stretti, manca la coda a Q basso). Lettura: **la
    struttura media è giusta, il CG è troppo morbido nei modi collettivi**
    (nucleo ~2× in RMSD, loop ~2,2×), coerente con i minimi B5–B5 riempiti e
    con la direzione del KL (copre le code). L'entropia relativa lo corregge
    lentamente (0,102 → 0,077). Ipotesi fisica da verificare: gli ioni K⁺ nel
    canale centrale (assenti nel CG) irrigidiscono la colonna; la spalla B3–B3
    a 0,35 nm che il CG non forma è la distanza O6···O6 della gabbia del K⁺.

Ordine: prima il **solo-prior**. Se tiene le tetradi e i due picchi B3–B3, il
residuo ML ha un punto di partenza sensato; se no, il problema è nei prior.

**Passo 2 — FENE** sul backbone (tipo già supportato: `k`, `r0`, `r_max`).

**Passo 3 — LJ attrattivo pair-specific** per i legami H: oggi il runtime ha
solo WCA per coppie di tipi; serve un'estensione sul meccanismo dei marker dei
Morse. Ogni nuovo contatto non-Morse deve portare `exclude_wca: false`.

## Prestazioni

| | Mac M3 Max (TEL22) | Leonardo A100 (TEL26) |
|---|---|---|
| training D=64, per frame per epoca | 0,16 s | **0,019 s** (~8×) |
| MD, per passo | ~82 ms | 20–30 ms stimati (~3–4×) |

Produzione solo-prior da 100 ps su A100: ~35 min in tutto (~3 ps/min).
Training limitato dall'overhead: D=64 e D=128 costano uguale. Leve: batch più
grande, TF32 nel solo training (plugin in FP32: in MD le forze sono −∇E), più
simulazioni per GPU. **Benchmark da fare** (vedi sopra).

## Difetti trovati e corretti

| difetto | commit | perché non era emerso |
|---|---|---|
| `Universe(tpr, xtc)` con conteggi diversi | `94f11df` | il caso d'uso per cui lo script esiste |
| apostrofo in `${VAR:?...}` | `09ee3dc` | errore riportato sull'ultima riga |
| PBC nel registro delle tetradi | `8947717` | rovinava distanze, non ordinamento |
| terminali come template duplicati | `b0b1517` | TEL22 usa nomi semplici |
| `MORSE` assente nella build | `fba11ae`, `8ea01c4` | `myconfig.hpp` in `build/`, rigenerata |
| log Slurm sparsi | `3d45341` | `--output` relativo alla cwd |
| `--types` mai collegato | `5569ada` | documentato e mai chiamato |
| equilibrato condiviso fra job paralleli | `7a9f5e4` | nessun errore, stati mescolati |
| manifest solo per il modello finale | `253d264` | checkpoint rifiutati in produzione |
| `import torch` a ogni `source` | `d112126` | solo lentezza |
| config TEL22 a D=128 malgrado il commento | guida TEL22 | lezione nel C++, non nella config |
| termostato con γ = 1 (τ ≈ 300 ps) | commit gamma | sembrava instabilità del modello |
| `AA_FORCES_TOPOLOGY` non passato dallo stadio dataset | commit FTOP | funzionava se esportata nella shell |

Il filo conduttore: **ogni difetto emergeva uno stadio dopo quello in cui era
stato introdotto.** I commit si fanno dal terminale del Mac: dalla sandbox
anche `git status` lascia `index.lock` in `.git`.

## Poi

- Guardrail in fondo a `equilibrate.py`.
- Correlazione angolare fra assi dei quadruplex.
- CV strutturali per 26 residui (`43`, `45`), per sbloccare `select`.
- Rifare il B3–B3 del TEL22 con γ efficace e traiettoria piena.
- `prod-2` come secondo blocco di dati; seconda realizzazione (altro seed).

## Promemoria per Giulia

Le produzioni salvano le forze ogni 20 ps. Per le prossime, `nstxout =
nstfout = 500`: si possono mediare le forze su finestre. Va chiesto **prima**
che riparta una produzione.
