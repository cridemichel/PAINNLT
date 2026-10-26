# TEL26 unfoldable — fase 3: barriera F/U, catena di perturbazioni, criterio di chiusura (23–26/10/2026)

Seguito di `claude/unfold-tel26-fase2.md` (fino alla decisione del 23/10: bersaglio U(T) di Buscaglia, U per tratto). Script in `MLCG_Framework_v2/tutorials/tel26_unfold`, dati in `/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold/relent_pop/`. Stato operativo e comandi in `checkpoints/2026-10-23_tel26_unfold/CHECKPOINT.md` (repo PAINNLT).

## Catene pop3–pop5 e barriera F/U (23–26/10)

- **pop3** (da k6, etichette per tratto): allentamento senza frame U nei campioni. Il gradiente delle code delle etichette morbide sposta il bacino campionato. → `POP_MIN_FRAMES` (una T entra nel vincolo solo con ≥ 25 frame netti in ciascuno stato).
- **pop4** (da k6, con POP_MIN_FRAMES): U dominante da 332–336 K, ma le mappe di 21 a it09 mostrano un solo minimo che trasla lungo la diagonale, cioè fusione graduale. Con le teste termiche il residuo resta stretto a 300 K e allenta F ad alta T.
- **pop5** (piano B: `CONFIG=d64`, PaiNN indipendente da T, `START_MODEL=zero`): la struttura a 300 K migliora (dS_hold −18…−28 per iterazione a it14–19), ma la Tm del modello scende sotto 324 K.
  - **Il P_U = 0,5 misurato a 324 K era un artefatto.** I due lignaggi sono rimasti fermi ciascuno in uno stato, e scambiati: quello partito da F è finito in U, quello partito da U in F. `POP_COEXIST=lineage` (coesistenza in un singolo lignaggio) ha escluso tutte le T.
  - **Con il modello fermo da it20 a it36** (~50 ns per lignaggio e per T a 315–327 K) si sono viste solo due transizioni in ~500 ns. Nelle prime iterazioni (it00–05) le transizioni a 324/327 K erano frequenti: **l'allenamento a 300 K alza la barriera F/U da pochi ns a > 50 ns**, e la frazione di frame U non stima più P_U.
  - **Errore trovato**: lo START_U di pop5 (`thermo_val/f/aperto/T330_r0`) finiva in F (P_U del ramo aperto a 330 K = 0,23).
- **Scartati**:
  - T-REMD: il collo di bottiglia è il ripiegamento a bassa T, e gli scambi sarebbero possibili solo fra processi `run_cg_md` riavviati;
  - un'ancora dalle sole transizioni: troppo poche.

## Catena di perturbazioni (POP_CHAIN, 26/10)

- **Idea**: la ripesatura dentro ciascun bacino resta esatta anche senza transizioni; manca solo il valore assoluto di L = ln(P_U/P_N). Lo si porta lungo l'allenamento come L(θ_n, T) = L(θ_0, T) + Σ_j ΔL_j(T). Ogni ΔL_j è la ripesatura da θ_{j−1} a θ_j sui campioni dei due bacini dell'iterazione j, e L(θ_0, T) viene dal modello dei soli prior, dove le transizioni sono frequenti.
- **Ancora** (`22_anchor_L0.py` sulle corse di 13, `thermo_val/f`, 305–350 K, 4 × 95 ns per T, U per tratto): **Tm(θ_0) = 331,9 ± 0,4 K, ΔH = 492 ± 76 kJ/mol**.
  - I rami chiuso e aperto concordano: 0,017/0,028 a 325 K, 0,29/0,23 a 330 K, 0,85/0,82 a 335 K, con 30/114/38 transizioni.
  - Larghezza fra 10 e 90 % di U ≈ 8 K, contro ~20 K sperimentali.
  - La Tm di 327,3 K riportata sopra usa la definizione di U per tetrade, meno severa.
- **Codice**:
  - `train_relent --pop CG,ETICHETTE,T,L*,λ,L0_abs`: costante L0_abs − ln(Σp/Σ(1−p)) sommata a ln p_U, con gradiente, pesi ed ESS invariati; verificato su dati sintetici, errore relativo 2e-11;
  - `19_pop_chain.py` (init/get/update/show): griglia 300–360 K; dove manca un bacino, ΔL è interpolato in 1/T dalle T misurate;
  - `19_relent_pop.sh`: variabili `POP_CHAIN`, `POP_ANCHOR`, `POP_RESEED` (un lignaggio che chiude l'iterazione fuori dal proprio bacino riparte da START_F / START_U) e `RE_ONLY_IF_NO_POP` (fuori dalla catena: allenamento solo a 300 K se nessuna T qualifica);
  - `19_pop_labels.py`: campo `last_U_hard`.
- **pop6** (26/10, job 62332808 + 62332809): da zero, d64, START_F `chiuso/T320_r0`, START_U `aperto/T345_r0`, POP_TS 318–338 ogni 4 K, λ 2, POP_NS 3, **NITER 30**.

## Criterio di chiusura del modello unfoldable (26/10)

**A cosa serve il modello.** Serve a confrontare ligandi sul TEL26 tramite la stabilizzazione termica. Il riferimento è il lavoro del gruppo: Bertini, Libera *et al.*, *Nucleic Acids Research* **54**, gkag403 (2026), doi:10.1093/nar/gkag403. Quello studio usa:

- Tel26 e il dimero Tel50, in 20 mM KCl con 5 mM di tampone fosfato, pH 7;
- BRACO-19, PhenDC3, RHPS4 e SYUIQ-5;
- CD con analisi SVD (tre stati), SAXS e Monte Carlo a grana grossissima.

Le indicazioni che ne derivano per il criterio:

- **La Tm assoluta dipende dalle condizioni.** Tel26 libero ha Tm ≈ 62 °C (≈ 335 K) nel NAR 2026, contro 328,4 K in Buscaglia (25 mM KCl): circa 7 K di differenza fra due condizioni sperimentali vicine. Chiedere al CG la Tm entro ±3 K da uno solo dei due valori è più stretto di quanto i dati giustifichino.
- **Il confronto fra ligandi è una classificazione per ordine e classi**, non un ΔTm al kelvin:
  - BRACO-19 e PhenDC3 tengono strutture ripiegate oltre 90 °C ai rapporti più alti (ΔTm > ~28 K);
  - RHPS4 e SYUIQ-5 arrivano allo srotolamento completo.
  - I valori numerici sono nelle tabelle supplementari S3 e S4, ancora da recuperare.
- **In un modello a due stati ΔTm scala con la larghezza della transizione.** Per un ligando che lega solo F, ΔTm ≈ (R Tm²/ΔH_eff) ln(1 + K[L]), con ΔH_eff = 4,39 R Tm²/(T90 − T10). Un CG con transizione larga 8 K invece di 20 K (soli prior: ΔH_eff ≈ 490 contro ~200 kJ/mol) sottostima ogni ΔTm di un fattore ~2,5. **Per i ligandi la larghezza conta più della Tm assoluta.**
- **La stabilità segue il cambio di topologia**, non la compattazione: PhenDC3 porta da ibrida ad antiparallela, e BRACO-19 ha il cambio di CD più grande. In più l'intermedio (simile a una tripla elica o a un G4 parallelo) cambia da ligando a ligando. Il CG ha una sola topologia (ibrida, 2JPZ) e nessuna tripla elica, quindi **non può rappresentare i ligandi che agiscono cambiando topologia**. I primi bersagli sono quindi i ligandi che si impilano sulle tetradi terminali: **SYUIQ-5** (il più piccolo cambio di CD su Tel26) e **RHPS4**. BRACO-19 e PhenDC3 restano solo come classe "forte", da confrontare in modo qualitativo.

**Requisiti.** Vanno verificati con una validazione indipendente dall'allenamento:

- 17 lungo con rami da F e da U, o MBAR;
- 21 per le mappe;
- 16 per la struttura a 300 K.

| | requisito | soglia |
|---|---|---|
| R1 | struttura a 300 K | ogni sovrapposizione intra di 16 ≥ 0,95 × il valore AA (0,9955 / 0,979 / 0,971 / 0,961) |
| R2 | Tm (U per tratto) | 326–337 K, cioè fra i due esperimenti ± 2 K; rami F/U concordi (\|ΔP_U\| < 0,1 a ogni T); errore statistico ≤ 1 K |
| R3 | larghezza | T90 − T10 della curva U(T) fra 15 e 25 K (sperimentale ~20 K) |
| R4 | meccanismo | due stati: U = bacino con i tratti separati; mappe di 21 bimodali vicino alla Tm; nessun allentamento progressivo |
| R5 | stabilità | due checkpoint validati consecutivi (it15 e it30) che soddisfano R1–R4, con Tm entro 1,5 K fra loro |

**Limite di tempo e regola di decisione.**

- pop6 ha **al massimo 30 iterazioni**, con validazione a it15 e a it30.
- **R1–R5 tutti soddisfatti**: il modello è chiuso e si passa ai ligandi (prima SYUIQ-5 e RHPS4).
- **Solo R2 fallisce**, di ≤ 8 K, con R3 e R4 soddisfatti: il modello si accetta. Il confronto fra ligandi usa ΔTm relativi e lo scarto sulla Tm si documenta.
- **R3 o R4 falliscono**: **niente pop7**. Si sceglie il miglior modello esistente secondo la tabella (pop6 migliore, k6; k6 va misurato su R2 e R3 con 17 + 22), si dichiarano i limiti e si passa ai ligandi.
- **Fuori portata in ogni caso**: intermedi I1/I2 (tripla elica) e cambi di topologia indotti dal ligando. Richiedono prior nuovi, in una fase successiva.

## Riferimenti

- Buscaglia R., Gray R. D., Chaires J. B., *Biopolymers* **99**(12), 1006–1018 (2013), doi:10.1002/bip.22247: Tm 328,4 K, ΔH totale 352 kJ/mol, modello F ⇌ I1 ⇌ I2 ⇌ U (2JPZ, 25 mM KCl).
- Gray R. D., Buscaglia R., Chaires J. B., *JACS* **134**(40), 16834–16844 (2012), doi:10.1021/ja307543z: I2 = tripla elica G.
- Bertini L., Libera V. *et al.*, *Nucleic Acids Research* **54**, gkag403 (2026), doi:10.1093/nar/gkag403 ([pdf](https://cristiano-de-michele.netlify.app/publication/nar-2026/nar-2026.pdf)): Tel26/Tel50 con BRACO-19, PhenDC3, RHPS4, SYUIQ-5; Tm del Tel26 libero ≈ 62 °C in 20 mM KCl con 5 mM di tampone fosfato; ΔTm numerici nelle tabelle supplementari S3 e S4.
