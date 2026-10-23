# Diagnosi apprendimento PaiNN — TEL22 e Ala2

Stato al 2026-09-05. **Indagine sulla metrica chiusa; sweep di checkpoint
completato; prossimo passo: regolarizzazione Lipschitz.** Sei ipotesi
refutate lungo il percorso sono conservate in fondo.

## CONCLUSIONE

**La rete apprende. Era la metrica assoluta a essere inutilizzabile.
Addestrare oltre il minimo della val loss fitta il rumore, degrada la
struttura e infine rende il campo di forza dinamicamente instabile.**

Cinque risultati:

1. La MSE skill sulle forze istantanee ha un tetto teorico di **~0.3%**:
   il target è rumore al 99.7%. Nessun intervento può spostarla.
2. Con metriche **di distanza** la rete migliora molto: a 10 ps la deriva
   strutturale passa da **77.7%** (soli prior) a **23.2%** (ep6), JS
   +0.141 nats, CI non sovrapposti.
3. Con la metrica **topologica** il quadro si ridimensiona: l'appaiamento
   di Hoogsteen degrada da 57% a 35.8% in 30 ps (casuale 33%, riferimento
   71.3%), con recupero parziale.
4. **Più training peggiora**: ep10 e ep20 sono peggio di ep6 su
   integrità, riscaldamento e profondità di E_ML.
5. **ep40 esplode**: guardrail a 3.6 ps (`max_f > 10 000`). Il modello ha
   sviluppato pozzi attrattivi spurii fra siti base della guanina
   (B2–B3, B3–B5, B4–B5) che pompano energia nel sistema.

**Sintesi fisica**: il quadruplex resta compatto ma le guanine non sono
appaiate con i partner corretti. Lo stacking sopravvive, la specificità
in-piano no. E il campo di forza appreso senza vincoli sviluppa, con il
training, **forze non limitate** fuori dal manifold di training.

## Sweep di checkpoint (C1: 64/2, full Morse, LR costante, no regolariz.)

Snapshot ogni 5 epoche (`checkpoint_every_epochs`, opzione aggiunta al
trainer). Val force loss: 0.9811 (ep6, minimo) → 0.9825 (ep10) → 1.0014
(ep20) → 1.0647 (ep40). Train fit: 3.0% → 15.2%.

Simulazione 10 ps, stesso `equilibrated.npz` per tutti:

| | deriva 5 ps | deriva 10 ps | integrità 0–5 | E_ML 1° ps | E_kin 1° ps | esito |
|---|---|---|---|---|---|---|
| priors | 51.6% | 77.7% | 47.4% | — | — | stabile |
| **ep6** | **10.9%** | **23.2%** | **57.0%** | −875 | **1777** | stabile |
| ep10 | 18.4% | 42.0% | 51.8% | −63 | 1963 | stabile |
| ep20 | 17.2% | 28.4% | 49.3% | −1712 | 2390 | stabile, si scalda |
| ep40 | — | — | — | −4148 | 2976 | **esplode a 3.6 ps** |

Rumore run-to-run ~4 punti. La deriva non è monotona (ep20 < ep10 a
10 ps), ma integrità, riscaldamento e profondità di E_ML degradano tutti
in modo monotono. Sulla **stessa** configurazione iniziale i quattro
modelli valutano E_ML a −875, ~0, −1700, −2600 kJ/mol: la funzione appresa
vaga in regioni che non pesano sulla loss.

Traiettoria ep40 fino all'abort:

| t (ps) | E_ML | E_kin | f_max | min_dist |
|---|---|---|---|---|
| 0.0 | −2631 | 1797 | 1203 | 0.238 |
| 1.0 | −4471 | 3017 | 2086 | 0.231 |
| 2.5 | −5450 | 3672 | 6775 | 0.218 |
| 3.6 | −6123 | 4141 | **10 223** | 0.211 |

**ep6 è il miglior checkpoint su ogni metrica.** Per superarlo va cambiato
il *modello*, non allungato il training.

## CORREZIONE: la selezione dei checkpoint NON sceglie su rumore

Affermazione precedente (sbagliata): "l'early stopping sulla val force
loss sceglie su rumore; l'epoca 6 è un'estrazione casuale".

```
val_loss(θ) = E[(f_θ − f*)²]  +  E[ε²]
              errore sulla         noise floor:
              forza media          COSTANTE fra le epoche
```

Il floor domina il valore assoluto ma è **costante fra le epoche della
stessa run** e si cancella nei confronti. L'early stopping ha trovato il
minimo a ep6, ed ep6 è il migliore. **Il framework faceva la cosa giusta**
— e lo sweep mostra che era *essenziale*, non solo utile.

Limiti: non confronta setup con noise floor diverso; risoluzione ~0.004
(distingue ep6 da ep20, non ep5 da ep6). E **non anticipa l'instabilità**:
da 0.98 a 1.06 nulla suggerisce "esplode in 3.6 ps". Solo il test dinamico
lo rivela.

## Il regolarizzatore giusto: vincolo di Lipschitz

Il fallimento osservato è *forze non limitate*. Il weight decay le riduce
indirettamente; la **proiezione spettrale** limita direttamente la
costante di Lipschitz per layer, cioè l'ampiezza massima delle forze.

`spectral_projection_strength: 4.0` è usato in 22 config del repo — tutti
Ala2. **Nessun config TEL22 l'ha mai attivata.** È un vincolo hard: dopo
ogni step ogni layer denso viene riproiettato a norma spettrale ≤ 4.

Config pronti (una sola variabile rispetto a C1):
- **C3** `tel22_training_config_C3_spectral4_snapshots.json`:
  `spectral_projection_strength 4.0` — **da fare per primo**,
  mechanism-matched al fallimento.
- **C2** `tel22_training_config_C2_wd1e-3_snapshots.json`:
  `weight_decay 1e-3` — secondo.

Lettura: se in C3 la val loss continua a scendere oltre ep6 *e* ep20/ep40
sono stabili e strutturalmente ≥ ep6-C1, il tetto si alza restando nel
message passing. Se C3 si comporta come C1, il vincolo a 4.0 è troppo
lasso (provare 1–2) oppure il rimedio è il segnale (sotto).

## Perché tenere il message passing (e non la shared head)

Obiezione dell'utente, corretta: la shared head **sostituisce** il message
passing e perde le correlazioni a più corpi che motivano PaiNN. Inoltre le
sue 131 feature sono tutte **intra-copia**, il registro è cablato, non è
trasferibile. Per un force field CG generale è un vicolo cieco; resta
utile solo come *upper bound* su TEL22.

L'evidenza dice che il message passing **funziona**: con 2 layer e cutoff
1.26 nm il campo recettivo a 2 hop copre la tetrade e i vicini impilati, i
canali vettoriali portano l'orientazione, e B1 (128/4) fitta all'85% in
memorizzazione. Il difetto di ep6 è di **segnale di training**, non di
rappresentazione. Si corregge il segnale, non si toglie l'espressività.

Se un giorno servisse il descrittore geometrico, la forma giusta è
l'**ibrido** (feature geometriche accanto al message passing:
`ordered_geometry` con `head_only=false`), non la sostituzione.

## Ridurre la varianza del target — richiede nuova MD all-atom

Il target corretto è la forza *media* condizionata, non l'istantanea.
GROMACS ha scritto forze ogni 500 step = **1 ps** (`nstfout=500`,
`dt=0.002`) su 500 000 step = 1 ns: i 1001 frame sono *tutti* quelli
esistenti. Nessuna risoluzione più fine da cui mediare.

Fix principiato: nuova produzione con `nstfout` 10–20 step, media delle
forze proiettate su finestre ~1 ps → target denoised; e più lunga → più
frame. È l'esperimento che distingue "PaiNN non ce la fa su TEL22" da
"PaiNN non ce la fa con 1 ns campionato a 1 ps".

## Risultato termodinamico ep6 (dettaglio)

Protocollo: `equilibrated.npz` comune, dt 0.001, campione ogni 20 step,
full Morse, Langevin kT=2.49 γ=1.0.

Run 50 ps: deriva 9% (0–5 ps) → plateau ~43% dopo 30 ps (dentro il rumore
nelle ultime 4 finestre). `rise3_4` (stacking) stabile 0.44–0.52 su 50 ps;
`tet4_d4` (in-piano) picca a 30–35 ps e rientra. Integrità: 57% → 35.8%
(30–35 ps) → 44% (45–50 ps): disordine dinamico.

Sanità: riproducibilità fino a ~7.5 ps poi divergenza caotica (rumore ~4
punti); `E_tot` −8% e `E_kin` −29% su 50 ps sono il termostato che
raffredda un checkpoint caldo, non un problema.

## Il noise floor sulle forze

RMS residua 887.1 kJ/mol/nm per componente; segnale di forza media ~51 →
**R²_max ≈ 0.0033** (limite inferiore). Coppie con ambiente locale
identico entro 0.03 nm hanno forze scorrelate. Allineamento verificato
(|z|=10.9 vs 2.4 shuffled). Profilo ⟨F·u⟩(r) oscillatorio a corto raggio.

## CONSEGUENZE OPERATIVE

1. **Riportare sempre distanze, topologia E stabilità dinamica**
   (36/37 + 38 + energia/f_max). Ogni metrica da sola inganna.
2. **La val force loss è valida entro una run** e l'early stopping al suo
   minimo è essenziale. Non anticipa l'instabilità.
3. **Mai addestrare oltre il minimo senza vincolo di Lipschitz.**
4. **Sconsigliato** 128/4 senza regolarizzazione.

## Trappola: `diagnostic_overfit_frames`

Train e validation sono lo stesso insieme: la "val skill" misura
memorizzazione; `Val_Zero_F_Norm` vale esattamente 1.0 (spia).

## Topologia e struttura TEL22

210 harmonic + **180 Morse**, 200 angoli, WCA ε=50.
`prepare_variant_a_topology.py` rimuove i Morse. Rigid body DA, DG, DT.
1001 frame, 220 molecole, 820 siti (guanina 6 siti, A/T 1). Split 801/200,
`split_seed=42`. Registro tetradi (1-based): `{2,10,14,22} {3,9,15,21}
{4,8,16,20}`, antiparallelo, coerente con PDB 143D.

## Cosa sta usando TEL22 oggi

`painn_canonical_context_silu_v2`, 2 layer message passing, hidden 64,
num_rbf 32, cutoff 1.2616 nm. Nessuna testa geometrica.

## Ipotesi refutate

1. Noise floor su Ala2 A1 (train=val). 2. `grad_clip_norm` (inattivo).
3. Underfitting da capacità (`overfit8` 88%). 4. Scheduler LR (LR fisso
identico). 5. Prior che catturano tutto (|z|=10.9). 6. Selezione su
rumore (floor costante; ep6 è davvero il migliore).

Previsione **confermata**: ep40 peggio di ep10 — catastroficamente.

## Costi (M3 Max, MPS)

Training 64/2: 0.64 s/step, 128 s/epoca, 40 ep ≈ 86 min. 128/4: 3.05
s/step. CG MD: 82 ms/step, 10 ps ≈ 14 min.

## Script (`tutorials/tel22/diagnostics/scripts/`)

31 noise floor per lag (inconcludente) · 32 noise floor locale · 33
profilo ⟨F·u⟩(r) · 34 riferimento G4 · 35 CG MD (input da `BASE`, solo
`MODEL` varia; permette il solo mismatch di provenienza del modello;
auto-rinomina `.FAILED` le run morte senza campioni) · 36 JS+CI ·
37 deriva · 38 integrità topologica · 39 training con snapshot.

## Modifiche al framework

- `train_painn.cpp`: `checkpoint_every_epochs` (default 0). Backup
  `.bak_snapshots`.
- `run_cg_md.py`: shim per kwargs non supportati dal plugin. Backup
  `.bak_preshim`.

## Fragilità dell'ambiente

- `pypresso` → interprete dell'albero v1. `Makefile` → Homebrew cmake.
- `_init.so` linkato a `libprotobuf.36.0.0` (protobuf **36.0**); un
  `brew upgrade` a 36.1 con cleanup l'ha rimossa fra due run. Fix:
  symlink temporaneo o `make -j` in `espresso/build`; poi
  `brew pin protobuf abseil` (abseil ABI richiesto: 2608).
- bash 3.2: `${ARRAY[@]+"${ARRAY[@]}"}`.

## Prossimi passi

1. **Training C3** (spectral 4.0, ~86 min) + valutazione ep6/ep20/ep40
   con 37/38/energia. Domanda: il vincolo di Lipschitz alza il tetto?
2. Se C3 insufficiente: strength 1–2, poi C2 (weight decay).
3. Se la regolarizzazione non basta: **nuova MD all-atom** con `nstfout`
   piccolo e più lunga → target denoised, più frame.
4. Criterio di integrità più fino (soglie su B3–B3, B2–B4).
