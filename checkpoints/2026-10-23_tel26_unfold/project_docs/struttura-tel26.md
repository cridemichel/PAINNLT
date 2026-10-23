# TEL26 — struttura della copia: dove il CG sbaglia (script 46)

Aggiornato al 2026-10-01 sera. Complementa `claude/pipeline-tel26.md`; questo documento ne aggiorna lo stato, perché lì il miglior modello è ancora re0 it30.
Riepilogo con grafici per Cristiano: doc "TEL26 CG-ML: a che punto siamo" (https://claude.ai/code/artifact/4609daf8-9aa7-4b3a-b361-b74b0fefeac0).
Riferimento: prima metà di prod-1 (`--ref-range 0:0.5`); tetto: `half2=@ref:0.5:1` e `prod2=@bin:tel26_lp1_dataset.prod2.bin`.
Corse CG: 100 ps a γ 20, finestra 50–100 ps, stesso stato iniziale per modello (`equilibrated_priors_lp2_100ps.npz` per lp2/re1).

## MODELLO DI RIFERIMENTO: `tel26_lp2_re1_it30.pt` (lp2 + residuo PaiNN d64 a entropia relativa)

**g(r) intra** (analisi 59031771 e 59088728):

| canale | tetto (half2) | lp1 | re0 it30 (lp1 + ML) | lp2 | re1 it08 | **re1 it30** |
|---|---|---|---|---|---|---|
| tutti i siti | 0,9955 | 0,9763 | 0,9852 | 0,9808 | 0,9934 | **0,9939** |
| B3–B3 | 0,979 | 0,837 | 0,911 | 0,842 | 0,971 | **0,9725** |
| B5–B5 | 0,971 | 0,904 | 0,922 | 0,914 | 0,946 | **0,9654** |
| S–S | 0,956–0,961 | 0,906 | 0,926 | 0,927 | 0,949 | **0,955** |

- Scarto dal tetto sul totale: 0,0192 (lp1) → 0,0103 (re0 it30) → **0,0016** (−92 %).
- Ogni canale è entro 0,006–0,007 dal tetto.

**Struttura** (validazioni 59031770 e 59088723), media con sovrapposizione degli istogrammi:

| | AA | tetto (half2) | lp1 | re0 it30 | lp2 | re1 it08 | **re1 it30** |
|---|---|---|---|---|---|---|---|
| rmsd_core (nm) | 0,043 | 0,044 (0,94) | 0,102 (0,01) | 0,077 (0,14) | 0,086 (0,04) | 0,053 (0,64) | **0,043 (0,93)** |
| rmsd_all | 0,091 | 0,098 (0,84) | 0,207 (0,00) | 0,191 (0,04) | 0,136 (0,16) | 0,115 (0,45) | **0,111 (0,52)** |
| rmsd_loops | 0,210 | 0,228 (0,86) | 0,477 (0,03) | 0,456 (0,08) | 0,284 (0,44) | 0,267 (0,56) | **0,266 (0,54)** |
| Q | 0,906 | 0,910 (0,88) | 0,911 (0,83) | 0,911 (0,81) | 0,913 (0,77) | 0,908 (0,85) | **0,909 (0,81)** |
| Rg | 0,799 | 0,797 (0,89) | 0,824 (0,33) | 0,796 (0,68) | 0,806 (0,67) | 0,798 (0,82) | **0,798 (0,83)** |
| JSD della FES (rmsd_core, Q) | — | 0,054 | 0,983 | 0,789 | 0,938 | 0,230 | **0,088** |
| scarto tetradi 1/2/3 | — | 0,009/0,013/0,014 | 0,064/0,036/0,059 | 0,036/0,034/0,037 | 0,049/0,027/0,039 | 0,022/0,021/0,018 | **0,011/0,009/0,012** |
| scarto loop/code | — | 0,062 | 0,300 | 0,262 | 0,105 | 0,101 | **0,105** |
| RMSF delle G (CG/AA) | 1,0× | ~1,0× | 1,9–2,2× | 1,5–1,8× | 1,6–1,9× | 1,0–1,24× | **0,92–1,05×** |

- **Nucleo al tetto AA/AA**: rmsd_core e scarti delle tetradi sono entro il rumore del riferimento. I grandi passi di it29–it30 hanno migliorato il modello.
- **Cappucci**: A3 0,030, T8 0,011, A9 0,034, A15 0,051 (1,04×), T25 0,028, T26 0,037.
- **Restano** (tutti senza un cappuccio efficace):
  - T13–T14: 1,7×, scarti 0,09–0,12;
  - T2: 1,6×, 0,136;
  - loop a elica T19–A21: RMSF giuste, posizione spostata di 0,08–0,19 nm.
  - Si vede in rmsd_all (0,52) e rmsd_loops (0,54).
- **Inter** (non giudicabile con 10 copie, tetto 0,84–0,85): re1 0,81. B3–B3 inter = 0: nel CG le copie non avvicinano mai i B3.
- **Test indipendente su prod-2** (mai usato nell'allenamento), script 46 con riferimento `tel26_lp1_dataset.prod2.bin` e `prod1=@bin:tel26_lp1_dataset.bin` come tetto:

  | | prod-1 contro prod-2 | lp1 | lp2 | re1 it30 |
  |---|---|---|---|---|
  | scarto tetradi 1/2/3 (nm) | 0,009/0,010/0,009 | 0,060/0,032/0,064 | 0,046/0,021/0,040 | 0,015/0,012/0,011 |
  | loop/code | 0,044 | 0,313 | 0,110 | 0,096 |
  | RMSF delle G | 1,00–1,14× | | | 0,95–1,15× |

  Il modello generalizza. La g(r) con prod-2 come riferimento è il job 59121332 (`44_tel22_rdf.py` diretto via `sbatch --wrap`).
- **Stabilità**:
  - re0 it30 su lp1 è stabile per 1 ns (quattro tratti identici);
  - in corso 1 ns di re1 it08 (59089001 → …03 → …05 → …06) e di re1 it30 (59100910 → …12 → …14 → …15), circa 4 h 20 min per tratto.
- **Prossima decisione** (chiesta a Cristiano nel doc): lp3 per i loop (T2, T13, T14, posizione del loop a elica) oppure modello unfoldable. Proposta: prima continuare il RE (nessuna interazione nuova); lp3 solo come piano B. Un lp3 costerebbe circa +10–15 % di contatti, ma aggiunge vincoli specifici della piega nativa.

## Lezioni

- La morbidezza del nucleo **non richiedeva i K⁺ espliciti**. Con i cappucci giusti nei prior (lp2), il RE porta il nucleo al livello AA. Con lp1 si fermava a 1,58×.
- **Il RE non crea bacini che il CG non visita.** Con lp1 i cappucci si scioglievano in meno di 5 ps e il RE non li ha mai recuperati. Servivano prior che li rendessero metastabili; poi il RE affina.
- Un solo Morse per base fissa un guscio, non una posizione: servono tre partner non allineati sulla faccia della tetrade.

## Catena re1 (job 59005910 per it01–it30; continuazione 59121341 da it31, NITER 50)

- **Impostazione**: `PRIOR_SET=lp2`, `START_MODEL=tel26_lp1_re0_it30.pt`, `START_CHECKPOINT=equilibrated_priors_lp2_100ps.npz`, `RE_RESIDUES_PER_COPY=26`, dataset lp2 59005909.
- **ΔS holdout per iterazione**, a ondate:
  - it01–it07: −11,3, −5,7, −9,6, −11,0, −11,4, −10,2, −8,6;
  - it08 −1,3, it09 −2,3;
  - it10–it18: −5,7, −8,0, −4,5, −4,4, 0, −1,3, −8,2, −7,3, −4,6;
  - it19–it28: −1,4, −0,9, −1,5, −2,1, 0, −0,8, −2,7, 0, −2,8, −1,0;
  - it29 −11,4 (62 passi), it30 −9,8 (89 passi).
  - Le iterazioni a guadagno nullo (it14, it23, it26) non sono mai consecutive, quindi la catena non si è fermata.
- **Costo**: 12 h 07 min.
- **Validazione alla pari**: comando `sbatch --wrap` con `run_cg_md.py` (`--steps 100000 --log_interval 20 --gamma 20 --allow_checkpoint_mismatch`); circa 3 h 20 min per 100 ps. Con `--log_interval 100` 250 ps richiedono circa 4 h 20 min.

## lp2 (cappucci sulla faccia della tetrade)

- **Costruzione**:
  - `find_loop_contacts.py --r-max 0.9 --s-max 0.07 --copy-tol 0.10 --min-sep 2 --site-choice sigma --max-per-base 3 --face-tetrads '4,12,16,22;6,10,18,24'` su hb0: 16 contatti per copia;
  - poi `apply_cbt.py` e `derive_prior_set.py --base lp1 --set lp2`;
  - poi il dataset lp2 (stadio dataset).
- **Soli prior**: scarto totale 0,131 (lp1) → 0,056, loop 0,300 → 0,105; i cappucci restano stabili.
- lp2b (partner anche nella tetrade centrale) è peggiore su A3. Scelto lp2.
