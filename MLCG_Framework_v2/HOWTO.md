# MLCG Framework v2 — Guida tecnica completa

> Stato documentato: **13 agosto 2026**.
> Questa guida descrive il framework nella configurazione corrente, inclusi i fix recenti per:
> Morse bonded analitico in ESPResSo, validazione tollerante ai round-trip `float32` del manifest,
> dummy neighbor-list limitata ai soli tipi ML, e certificazione NVE basata su `sigma_E = std(E)`.

---

## Indice

1. [Scopo del framework e filosofia del modello](#1-scopo-del-framework-e-filosofia-del-modello)
2. [Convenzioni, unità e notazione](#2-convenzioni-unità-e-notazione)
3. [Decomposizione dell'Hamiltoniana CG](#3-decomposizione-dellhamiltoniana-cg)
4. [Mapping atomistico → coarse grained](#4-mapping-atomistico--coarse-grained)
5. [Forze e torque generalizzati di riferimento](#5-forze-e-torque-generalizzati-di-riferimento)
6. [Rigid body: masse, inerzie, Kabsch e virtual sites](#6-rigid-body-masse-inerzie-kabsch-e-virtual-sites)
7. [Prior analitici e Direct Boltzmann Inversion](#7-prior-analitici-e-direct-boltzmann-inversion)
8. [WCA pair-specific: costruzione, guardrail e sottrazione](#8-wca-pair-specific-costruzione-guardrail-e-sottrazione)
9. [Target residuo del force matching](#9-target-residuo-del-force-matching)
10. [Architettura PaiNN implementata](#10-architettura-painn-implementata)
11. [Forze e torque predetti dalla rete](#11-forze-e-torque-predetti-dalla-rete)
12. [Funzione di loss e normalizzazione](#12-funzione-di-loss-e-normalizzazione)
13. [Energy gauge e scala energetica](#13-energy-gauge-e-scala-energetica)
14. [Formato del dataset binario](#14-formato-del-dataset-binario)
15. [Manifest del modello e provenance](#15-manifest-del-modello-e-provenance)
16. [Checkpoint di ESPResSo](#16-checkpoint-di-espresso)
17. [Installazione Python e build del trainer](#17-installazione-python-e-build-del-trainer)
18. [Integrazione del plugin PaiNN/Morse in ESPResSo](#18-integrazione-del-plugin-painnmorse-in-espresso)
19. [Configurazione `topology_config.json`](#19-configurazione-topology_configjson)
20. [Configurazione del training](#20-configurazione-del-training)
21. [Pipeline completa: script e parametri](#21-pipeline-completa-script-e-parametri)
22. [Equilibrazione CG in dettaglio](#22-equilibrazione-cg-in-dettaglio)
23. [Produzione CG in dettaglio](#23-produzione-cg-in-dettaglio)
24. [Certificazione NVE](#24-certificazione-nve)
25. [Interpretazione quantitativa dello scaling NVE](#25-interpretazione-quantitativa-dello-scaling-nve)
26. [Test e controlli di consistenza](#26-test-e-controlli-di-consistenza)
27. [Tutorial TEL22: configurazione corrente](#27-tutorial-tel22-configurazione-corrente)
28. [Troubleshooting](#28-troubleshooting)
29. [Checklist per adattare il framework a un nuovo sistema](#29-checklist-per-adattare-il-framework-a-un-nuovo-sistema)

---

# 1. Scopo del framework e filosofia del modello

`MLCG_Framework_v2` costruisce e simula un modello coarse-grained (CG) nel quale la forza totale è separata in due contributi:

1. **prior fisici/strutturali espliciti**, semplici e interpretabili;
2. **potenziale residuo PaiNN**, appreso dai dati mediante force matching.

Il core è chemistry-agnostic. TEL22 è soltanto un'applicazione/tutorial: il core non contiene nomi di residui, site type o topologie specifiche del DNA.

L'idea centrale è

\[
U_{\mathrm{CG}}(\mathbf R,\mathbf Q)
=
U_{\mathrm{prior}}(\mathbf R,\mathbf Q)
+
U_{\mathrm{ML}}(\mathbf R,\mathbf Q;\theta),
\]

dove:

- \(\mathbf R\) sono le coordinate traslazionali dei corpi CG;
- \(\mathbf Q\) rappresenta l'orientazione dei corpi rigidi multi-site;
- \(U_{\mathrm{prior}}\) contiene WCA, bond, angle e dihedral;
- \(U_{\mathrm{ML}}\) è l'energia residua PaiNN;
- \(\theta\) sono i parametri della rete.

Il training non cerca quindi di riprodurre direttamente la forza atomistica totale. Il preprocessing sottrae prima le forze e i torque generati dai prior:

\[
\mathbf F^{\mathrm{target}}_m
=
\mathbf F^{\mathrm{ref}}_m
-
\mathbf F^{\mathrm{prior}}_m,
\]

\[
\boldsymbol\tau^{\mathrm{target}}_m
=
\boldsymbol\tau^{\mathrm{ref}}_m
-
\boldsymbol\tau^{\mathrm{prior}}_m.
\]

PaiNN impara questi target residui.

Un punto fondamentale del modello corrente è che **gli edge PaiNN intra-molecolari sono esclusi**. La rete descrive quindi il residuo intermolecolare tra siti appartenenti a corpi CG differenti; la struttura intramolecolare è delegata ai prior bonded e alla rigidificazione.

---

# 2. Convenzioni, unità e notazione

Le convenzioni correnti sono:

| Grandezza | Unità |
|---|---|
| posizione | nm |
| tempo | ps |
| massa | amu |
| momento d'inerzia | amu nm² |
| energia | kJ/mol |
| forza | kJ/(mol nm) |
| torque | kJ/mol |
| temperatura di preprocessing | K |
| temperatura dinamica | `kT` in kJ/mol |

A 300 K:

\[
RT \simeq 0.008314462618 \times 300
\simeq 2.494\ \mathrm{kJ/mol}.
\]

Gli script TEL22 usano tipicamente `kT=2.49`.

Il builder legge le coordinate tramite MDAnalysis e converte Å → nm dividendo per 10. Per le forze il codice corrente applica un fattore ×10 prima di costruire i target in kJ/(mol nm). Se si usa un reader MDAnalysis diverso dal workflow GROMACS/TRR verificato, la convenzione di unità delle forze deve essere ricontrollata esplicitamente.

Le PBC sono ortorombiche e vengono applicate con Minimum Image Convention:

\[
\Delta\mathbf r_{\mathrm{MIC}}
=
\Delta\mathbf r
-
\mathbf L\,\mathrm{round}
\left(
\frac{\Delta\mathbf r}{\mathbf L}
\right).
\]

---

# 3. Decomposizione dell'Hamiltoniana CG

In produzione NVE l'Hamiltoniana rilevante è

\[
H =
K_{\mathrm{trans}}
+
K_{\mathrm{rot}}
+
U_{\mathrm{WCA}}
+
U_{\mathrm{bond}}
+
U_{\mathrm{angle}}
+
U_{\mathrm{dihedral}}
+
U_{\mathrm{ML}}.
\]

Per i corpi reali:

\[
K_{\mathrm{trans}}
=
\sum_m
\frac12 M_m |\mathbf v_m|^2,
\]

\[
K_{\mathrm{rot}}
=
\sum_m
\frac12
\sum_{\alpha=1}^{3}
I_{m,\alpha}\omega_{m,\alpha}^2.
\]

I virtual sites hanno massa e inerzia numericamente molto piccole (`1e-5`) soltanto perché ESPResSo richiede massa positiva; non rappresentano DOF dinamici indipendenti.

In `energy.csv`, la colonna storicamente chiamata `E_class` è un nome legacy: il valore deriva da `system.analysis.energy()["total"]` e contiene **l'energia totale gestita da ESPResSo**, quindi cinetica + potenziali classici. Il totale usato per la certificazione è

\[
E_{\mathrm{tot}}
=
E_{\mathrm{ESPResSo}}
+
E_{\mathrm{ML}}.
\]

---

# 4. Mapping atomistico → coarse grained

Il mapping è definito nel JSON di topologia.

Per ogni residuo/molecola \(m\), il framework mantiene:

- un centro dinamico reale, posto al COM atomistico;
- uno o più siti virtuali PaiNN;
- una forza totale di riferimento;
- un torque totale di riferimento.

## 4.1 Centro del corpo

Il centro è sempre il centro di massa atomistico:

\[
\mathbf R_m
=
\frac{\sum_{a\in m} m_a\mathbf r_a}
{\sum_{a\in m}m_a}.
\]

Prima del calcolo il residuo viene ricostruito localmente attraverso PBC rispetto al primo atomo, in modo da non calcolare COM artificiali attraverso il bordo periodico.

## 4.2 Posizione dei siti CG

Per ogni site definition:

### `mapping_method = "COM"`

\[
\mathbf r_s
=
\frac{\sum_{a\in s}m_a\mathbf r_a}
{\sum_{a\in s}m_a}.
\]

### `mapping_method = "COG"`

\[
\mathbf r_s
=
\frac{1}{N_s}
\sum_{a\in s}\mathbf r_a.
\]

### `mapping_method = "ATOM"`

\[
\mathbf r_s = \mathbf r_{a_0},
\]

dove \(a_0\) è il primo atomo selezionato.

La wildcard

```json
"CG_SITE": ["*"]
```

seleziona tutti gli atomi del residuo. È particolarmente importante per un corpo single-site: nel runtime corrente un corpo con un solo sito non possiede rotazione; il suo unico sito deve quindi coincidere con il COM entro `1e-6 nm`.

## 4.3 Site type

Ogni nome di sito è associato a un intero non negativo:

```json
"site_types": {
  "CG_A": 0,
  "CG_B": 1
}
```

Questi interi sono gli indici dell'embedding PaiNN. Deve valere:

\[
0 \le t_s < \texttt{num_species}.
\]

L'ordine dei site type all'interno di una molecola è parte del formato del dataset e viene usato per associare il record alla corretta rigid-body template.

---

# 5. Forze e torque generalizzati di riferimento

Per ogni molecola/residuo:

\[
\mathbf F^{\mathrm{ref}}_m
=
\sum_{a\in m}\mathbf f_a.
\]

Il torque è calcolato rispetto al COM:

\[
\boldsymbol\tau^{\mathrm{ref}}_m
=
\sum_{a\in m}
(\mathbf r_a-\mathbf R_m)
\times
\mathbf f_a.
\]

Questi sono i target generalizzati naturali per il corpo rigido.

Il preprocessing rifiuta una traiettoria in cui tutte le forze mappate risultino nulle: è un guardrail contro TRR senza record di forza o contro conversioni che abbiano eliminato le forze.

Per il force matching è quindi obbligatorio usare una traiettoria che contenga forze atomistiche reali. Nel workflow GROMACS:

- `nstfout > 0`;
- formato TRR;
- `gmx trjconv ... -force` quando si genera la traiettoria PBC-corrected.

---

# 6. Rigid body: masse, inerzie, Kabsch e virtual sites

## 6.1 Massa e tensore d'inerzia

Per il corpo \(m\):

\[
M_m = \sum_a m_a.
\]

Il tensore d'inerzia rispetto al COM è

\[
\mathbf I
=
\sum_a
m_a
\left[
|\mathbf r_a-\mathbf R|^2\mathbf 1
-
(\mathbf r_a-\mathbf R)
(\mathbf r_a-\mathbf R)^T
\right].
\]

Il tensore viene diagonalizzato:

\[
\mathbf A^T\mathbf I\mathbf A
=
\mathrm{diag}(I_1,I_2,I_3),
\]

con autovalori ordinati e base resa destrorsa. `rigid_bodies_info.json` memorizza:

- `mass_amu`;
- `inertia_amu_nm2`;
- `body_frame = "principal_axes"`;
- site type;
- offset del sito nel body frame.

## 6.2 Geometria media multi-site

Per un residuo multi-site le configurazioni istantanee contengono deformazioni atomistiche che non possono esistere in un corpo rigido.

Il framework costruisce quindi una geometria ideale:

1. raccoglie gli offset istantanei dei siti rispetto al COM;
2. allinea gli snapshot mediante Kabsch;
3. ripete tre iterazioni allineamento → media;
4. trasforma la geometria media nel frame degli assi principali;
5. usa **la stessa geometria** per:
   - fitting WCA;
   - fitting dei bonded priors;
   - sottrazione dei prior;
   - ricostruzione nel runtime.

Per Kabsch viene minimizzato il RMSD tra due insiemi di punti mediante SVD della matrice di covarianza:

\[
\mathbf H = \mathbf P^T\mathbf Q,
\]

con correzione del segno per imporre

\[
\det\mathbf R = +1.
\]

## 6.3 Runtime ESPResSo

Per ogni molecola il runtime crea:

- una particella reale al COM;
- `type = num_species + 1`, quindi invisibile a PaiNN;
- massa e principal moments reali;
- rotazione abilitata solo se `num_sites > 1`;
- un virtual site per ogni sito CG, con il site type PaiNN.

I virtual sites vengono legati al COM tramite `vs_auto_relate_to()`.

La forza sui virtual sites viene trasferita dal meccanismo di virtual sites al corpo centrale, inclusa la componente di torque.

---

# 7. Prior analitici e Direct Boltzmann Inversion

Il builder esegue un primo pass sulla traiettoria per costruire statistiche geometriche e un secondo pass per sottrarre i prior.

## 7.1 Harmonic bond

\[
U(r)
=
\frac12 k(r-r_0)^2,
\]

\[
F_r
=
-k(r-r_0).
\]

Quando `r0="auto"`:

\[
r_0 = \langle r\rangle.
\]

Quando anche `k="auto"` il framework usa l'approssimazione armonica della Boltzmann inversion:

\[
k
=
\frac{1}{\beta\,\mathrm{Var}(r)},
\qquad
\beta=\frac{1}{RT}.
\]

Bond con lo stesso campo `name` condividono le statistiche.

## 7.2 FENE

Il runtime/preprocessing usa:

\[
F_r
=
-\frac{k(r-r_0)}
{1-\left[(r-r_0)/r_{\max}\right]^2}.
\]

La forma energetica corrispondente è

\[
U(r)
=
-\frac12 k r_{\max}^2
\ln\left[
1-\left(\frac{r-r_0}{r_{\max}}\right)^2
\right].
\]

Il dominio richiede:

\[
|r-r_0| < r_{\max}.
\]

## 7.3 Morse analitico

La versione corrente usa un vero bonded `MorseBond` nel core ESPResSo:

\[
U(r)
=
D
\left[
1-e^{-a(r-r_0)}
\right]^2,
\]

\[
F_r
=
-2aD
\left[
1-e^{-a(r-r_0)}
\right]
e^{-a(r-r_0)}.
\]

Parametri:

- `D`: profondità energetica, kJ/mol;
- `a`: inverse length, nm\(^{-1}\);
- `r0`: distanza di equilibrio, nm;
- `r_cut`: cutoff bonded opzionale; l'integrazione corrente usa un default ampio se non specificato.

Il Morse non viene più approssimato mediante interpolazioni indipendenti di energia e forza. Questa modifica è essenziale per una certificazione NVE rigorosa.

### Dissociazione, rebinding e semantica topologica

Il `MorseBond` è **dissociativo dal punto di vista energetico**, ma non viene rimosso dinamicamente dalla lista dei bonded interactions. Per `r > r0` la forza attrattiva tende gradualmente a zero e l'energia tende a `D`; se i due corpi tornano vicini, il contatto può riformarsi automaticamente. Questa è la semantica adatta a un contatto reversibile di folding/unfolding, non a una rottura irreversibile della connettività.

Per default un bond Morse ha `exclude_wca=false`: la sola presenza del restraint Morse **non** crea una topological 1-2 WCA exclusion. Il core repulsivo tra i relativi virtual sites resta quindi attivo. Bond harmonic/FENE usati per la connettività covalente hanno invece default `exclude_wca=true`; con la policy WCA v3 viene esclusa soltanto la site-pair esplicitamente bonded.

`D` è una scala energetica del singolo contatto, non direttamente una free-energy di unfolding dell'intera struttura. Se più Morse devono dissociarsi cooperativamente, i loro contributi si sommano e competono con entropia, prior bonded, WCA e residuo ML. Anche `a` è cruciale: la lunghezza caratteristica è circa `1/a`, quindi valori piccoli producono attrazioni molto larghe.

Il cutoff non deve essere usato come meccanismo di bond breaking. Nell'implementazione corrente energia e forza vengono omesse per `r >= r_cut`; un attraversamento effettivo del cutoff introdurrebbe una discontinuità energetica. Per NVE e per unfolding reversibile usare quindi un `r_cut` molto oltre tutte le distanze raggiungibili, oppure una futura forma di cutoff esplicitamente shiftata/smoothed.

## 7.4 Harmonic angle

\[
U(\theta)
=
\frac12 k(\theta-\theta_0)^2.
\]

Per `theta0="auto"`:

\[
\theta_0=\langle\theta\rangle.
\]

Per `k="auto"`:

\[
k
=
\frac{1}{\beta\,\mathrm{Var}(\theta)}.
\]

Gli angoli sono espressi in radianti.

`prior_geometry.default_angle_site` consente di applicare automaticamente lo stesso indice di virtual site ai tre vertici quando la topologia non specifica `site_i`, `site_j`, `site_k`.

## 7.5 Cosine dihedral

La convenzione del preprocessing è

\[
U(\phi)
=
K
\left[
1-\cos(n\phi-\phi_0)
\right].
\]

`phi0="auto"` è calcolato mediante media circolare:

\[
\phi_0
=
\mathrm{atan2}
\left(
\langle\sin\phi\rangle,
\langle\cos\phi\rangle
\right).
\]

`k="auto"` usa, nella versione corrente, l'approssimazione

\[
k \approx \frac{1}{\beta\,\mathrm{Var}(\phi)}.
\]

Nel preprocessing la forza del cosine dihedral è valutata tramite differenza centrale con passo `1e-6 nm`, per evitare mismatch silenziosi di convenzione segno/indice. Se un nuovo sistema usa dihedrals, è raccomandato un test di parity esplicito con ESPResSo.

---

# 8. WCA pair-specific: costruzione, guardrail e sottrazione

Il WCA è il prior repulsivo nonbonded.

Per ogni coppia di site type \((i,j)\):

\[
U_{\mathrm{WCA}}(r)
=
\begin{cases}
4\epsilon_{ij}
\left[
(\sigma_{ij}/r)^{12}
-
(\sigma_{ij}/r)^6
\right]
+\epsilon_{ij},
&
r<r_{c,ij},
\\
0,
&
r\ge r_{c,ij}.
\end{cases}
\]

Il builder pone

\[
\sigma_{ij}
=
\frac{r_{c,ij}}{2^{1/6}},
\]

quindi il cutoff coincide con il minimo del Lennard-Jones e sia energia sia forza vanno a zero al cutoff.

La forza radiale ha modulo

\[
F(r)
=
\frac{24\epsilon}{r}
\left[
2(\sigma/r)^{12}
-
(\sigma/r)^6
\right].
\]

## 8.1 Esclusioni topologiche — policy v3

La policy corrente distingue la **relazione topologica tra molecole** dalla **site-pair che deve realmente perdere il core WCA**:

- tutte le coppie di virtual sites interne allo stesso rigid body sono escluse;
- per una coppia topologica 1–2 con `exclude_wca=true`, vengono escluse **solo le virtual-site pair esplicitamente bonded**; tutte le altre cross-pair tra i due rigid bodies mantengono WCA;
- una relazione 1–2 definita da un bond COM-COM non esclude automaticamente alcuna virtual-site pair;
- le coppie 1–3 definite dagli endpoint di angle con `exclude_wca=true` mantengono, nella policy v3, l'esclusione **all-sites**;
- un Morse ha default `exclude_wca=false`, perché rappresenta tipicamente un contatto/restraint dissociativo e non la connettività covalente.

La policy viene serializzata in `cg_priors.json` come `wca_exclusions` e il runtime richiede esplicitamente:

```text
policy_version = 3
direct_scope = bonded_site_pairs_only
one_three_scope = molecule_pair_all_sites
pair_source = explicit_topology_pairs_v3
```

I metadata contengono sia `direct_pairs` sia `direct_site_pairs`. Il runtime li cross-checka con i bonded priors e rifiuta priors legacy/incoerenti.

### Perché l'esclusione 1–2 deve essere site-aware

Un'esclusione 1–2 applicata a **tutte** le cross-site pair di due rigid bodies è in generale troppo ampia: il fatto che, per esempio, `site0-site0` sia legata non implica che `site2-site3` debba poter interpenetrare senza repulsione. Nel caso TEL22 questa policy legacy ha prodotto un collasso riproducibile di siti non bonded appartenenti a residui backbone-adjacent. Un test A/B che riattivava WCA su tutte le cross-pair 1–2 salvo la site-pair realmente bonded ha eliminato il failure mode.

La correzione deve essere **simmetrica tra preprocessing e runtime**. Cambiare soltanto le esclusioni durante MD modifica l'Hamiltoniana rispetto a quella usata per sottrarre i prior dai target di training. Dopo un cambio di policy WCA bisogna quindi rigenerare dataset/priors, riaddestrare, riequilibrare e ricertificare NVE.

## 8.2 Fit automatico

Il percorso corrente raccomandato è:

```json
"wca_sigma": "auto"
```

Il builder raccoglie distanze sulla **geometria rigidificata effettiva**, non sui siti mapped deformabili.

Per ogni pair type calcola:

- quantile basso \(q_{\mathrm{low}}\);
- minimo fisico esatto \(r_{\min}\);
- conteggio campioni.

Poi introduce raggi di tipo \(R_i\) minimizzando:

\[
\mathcal L_R
=
\sum_{ij}
w_{ij}
\left[
R_i+R_j-q_{\mathrm{low},ij}
\right]^2,
\]

con

\[
w_{ij}=\frac{N_{ij}}{N_{ij}+1000}.
\]

Il cutoff preliminare è una shrinkage estimate:

\[
r_c
=
\alpha q_{\mathrm{low}}
+
(1-\alpha)(R_i+R_j),
\]

\[
\alpha=\frac{N}{N+1000},
\]

e viene limitato a

\[
r_c\le q_{\mathrm{low}}.
\]

## 8.3 Physical-support guard

Viene imposto:

\[
r_{\mathrm{guard}}
=
f_{\mathrm{guard}}r_c,
\]

e

\[
r_{\mathrm{guard}}
\le
m_{\mathrm{phys}}\,r_{\min},
\]

dove:

- `wca_guard_fraction = f_guard`;
- `wca_physical_guard_margin = m_phys`.

Equivalentemente il cutoff è limitato da

\[
r_c
\le
\frac{m_{\mathrm{phys}}r_{\min}}
{f_{\mathrm{guard}}}.
\]

Questo impedisce che il deep core WCA invada configurazioni fisiche realmente viste nel training set.

## 8.4 Calibrazione di epsilon

Il parametro `wca_guard_kbt` fissa l'energia al guard radius:

\[
U_{\mathrm{WCA}}(r_{\mathrm{guard}})
=
g\,k_BT,
\]

dove \(g=\texttt{wca_guard_kbt}\).

Da questa condizione viene risolto \(\epsilon\).

### Nota su `wca_epsilon` e `wca_overrides`

Nel percorso pair-specific `wca_sigma="auto"`, l'epsilon effettivo di ogni coppia viene ricalibrato dalla condizione precedente. I campi legacy `wca_epsilon` e `wca_overrides` vengono ancora letti/serializzati in metadata, ma **non devono essere usati come meccanismo principale per controllare i `wca_pairs` automatici**.

Il runtime attuale si aspetta `wca_pairs`; per una nuova pipeline è quindi preferibile `wca_sigma="auto"` oppure riuso di un `cg_priors.json` già valido tramite `--priors`.

---

# 9. Target residuo del force matching

Dopo la costruzione dei prior, per ogni frame il builder ricostruisce i siti rigidi e sottrae i contributi classici:

\[
\mathbf F_m^{\mathrm{res}}
=
\mathbf F_m^{\mathrm{ref}}
-
\left(
\mathbf F_m^{\mathrm{WCA}}
+
\mathbf F_m^{\mathrm{bond}}
+
\mathbf F_m^{\mathrm{angle}}
+
\mathbf F_m^{\mathrm{dihedral}}
\right),
\]

\[
\boldsymbol\tau_m^{\mathrm{res}}
=
\boldsymbol\tau_m^{\mathrm{ref}}
-
\boldsymbol\tau_m^{\mathrm{prior}}.
\]

Il torque di un contributo applicato su un virtual site è

\[
\boldsymbol\tau_m
=
(\mathbf r_s-\mathbf R_m)\times\mathbf f_s.
\]

Questi valori residui vengono scritti nel dataset come target di training.

`--clip_forces` applica un clipping componente-per-componente sia alle forze sia ai torque. Il default corrente è **nessun clipping**. Per prior analitici ben condizionati questo è il comportamento raccomandato, perché il clipping modifica il problema statistico appreso.

---

# 10. Architettura PaiNN implementata

La variante ammessa dal trainer/runtime è:

```text
painn_canonical_context_silu_v2
```

La rete mantiene per ogni sito:

- feature scalari \(s_i\in\mathbb R^D\);
- feature vettoriali \(v_i\in\mathbb R^{3\times D}\).

## 10.1 Embedding

Il site type \(z_i\) viene trasformato in

\[
s_i^{(0)}
=
\mathrm{Embedding}(z_i),
\]

mentre

\[
v_i^{(0)}=0.
\]

## 10.2 RBF

Per distanza \(d_{ij}\), i centri Gaussiani coprono \([0,r_c]\).

Indicando con \(\mu_k\) i centri:

\[
\mathrm{RBF}_k(d)
=
\exp
\left[
-\frac{(d-\mu_k)^2}{\sigma_{\mathrm{RBF}}^2}
\right]
c(d),
\]

con

\[
\sigma_{\mathrm{RBF}}
=
\frac{r_c}{N_{\mathrm{RBF}}}.
\]

## 10.3 Toxvaerd cutoff

Il cutoff implementato è

\[
x=\frac{r_c-d}{r_c},
\]

\[
c(d)
=
\frac{x^4}
{x^4+\alpha^4},
\qquad d\le r_c,
\]

e

\[
c(d)=0,\qquad d>r_c.
\]

`toxvaerd_alpha` controlla quanto rapidamente il filtro decade avvicinandosi al cutoff.

## 10.4 Message block

Il contesto scalare viene trasformato da:

```text
Linear(D,D) → SiLU → Linear(D,3D)
```

e moltiplicato elemento per elemento per un filtro radiale

```text
Linear(num_rbf,3D, bias=False).
```

I tre blocchi risultanti controllano:

- messaggio scalare;
- scaling della feature vettoriale del vicino;
- contributo nella direzione \(\hat{\mathbf r}_{ij}\).

L'aggregazione è una **somma** sui vicini (`sum_v1`).

## 10.5 Update block

Si formano due trasformazioni lineari delle feature vettoriali:

\[
v_v=W_vv,\qquad
v_u=W_uv.
\]

La norma stabilizzata è

\[
\|v_v\|_\epsilon
=
\sqrt{\sum_{\alpha}v_{v,\alpha}^2+\epsilon},
\]

con \(\epsilon=10^{-8}\).

La MLP scalare usa

\[
[s,\|v_v\|_\epsilon]
\]

e produce tre blocchi che aggiornano scalari e vettori secondo la struttura equivariant PaiNN.

## 10.6 Readout

```text
Linear(D,D/2) → SiLU → Linear(D/2,1)
```

produce un contributo energetico per sito.

---

# 11. Forze e torque predetti dalla rete

Il modello è energy-based.

Per ogni frame:

\[
U_{\mathrm{ML}}
=
\sum_i u_i.
\]

Gli edge vengono costruiti solo tra siti di molecole differenti e a distanza MIC

\[
r_{ij}\le r_c.
\]

Ogni coppia fisica è rappresentata in entrambe le direzioni.

Le forze vengono ottenute mediante autograd dal medesimo scalare energetico:

\[
\mathbf f_{ij}
=
-\frac{\partial U_{\mathrm{ML}}}
{\partial\mathbf r_{ij}}.
\]

La forza molecolare è aggregata sui due estremi degli edge:

\[
\mathbf F_m^{\mathrm{ML}}
=
\sum_{i\in m}\mathbf f_i.
\]

Il torque molecolare previsto è:

\[
\boldsymbol\tau_m^{\mathrm{ML}}
=
\sum_{i\in m}
(\mathbf r_i-\mathbf R_m)
\times
\mathbf f_i.
\]

Durante il training `create_graph=true`, perché la loss dipende dalle forze e quindi la derivata della loss rispetto ai parametri della rete richiede derivate di secondo ordine dell'energia.

La correzione PBC contiene un `round(...).detach()`: la scelta dell'immagine periodica è trattata come discreta e non viene differenziata.

---

# 12. Funzione di loss e normalizzazione

Le scale sono calcolate **sul training set**.

Per le forze:

\[
F_{\mathrm{RMS}}
=
\sqrt{
\frac{1}{N_F}
\sum F_{\mathrm{target}}^2
}.
\]

Per i torque, soltanto sui corpi multi-site:

\[
\tau_{\mathrm{RMS}}
=
\sqrt{
\frac{1}{N_\tau}
\sum \tau_{\mathrm{target}}^2
}.
\]

La loss normalizzata è:

\[
L_F
=
\frac{
\mathrm{MSE}
(
\mathbf F^{\mathrm{pred}},
\mathbf F^{\mathrm{target}}
)
}{
F_{\mathrm{RMS}}^2
},
\]

\[
L_\tau
=
\frac{
\mathrm{MSE}_{\mathrm{multi-site}}
(
\boldsymbol\tau^{\mathrm{pred}},
\boldsymbol\tau^{\mathrm{target}}
)
}{
\tau_{\mathrm{RMS}}^2
}.
\]

La loss totale:

\[
L
=
L_F
+
\lambda_\tau L_\tau
+
\lambda_{\mathrm{Lip}}L_{\mathrm{Lip}}.
\]

Nel file JSON:

- `torque_weight = λ_tau`;
- `lipschitz_lambda = λ_Lip`.

Il termine opzionale Lipschitz implementato è

\[
L_{\mathrm{Lip}}
=
\frac{
\left\langle
\|\mathbf f_i\|^2
\right\rangle_i
}{
F_{\mathrm{RMS}}^2
}.
\]

È quindi una penalizzazione diretta della norma delle forze di sito, non una stima rigorosa della costante di Lipschitz globale.

Il trainer stampa anche il baseline zero predictor sulla validation. Se

\[
L_F^{\mathrm{val}}\approx L_{F,0}^{\mathrm{val}},
\]

il modello non sta ottenendo un vantaggio significativo rispetto alla predizione di forza nulla.

---

# 13. Energy gauge e scala energetica

Un training solo su forze lascia indeterminata una costante additiva nell'energia.

Il framework usa:

```text
isolated_species_zero_v1
```

Per ogni site type \(z\) viene calcolata l'energia del sito isolato e sottratta dal readout:

\[
u_i
=
\left[
u_i^{\mathrm{raw}}
-
u_{\mathrm{isolated}}(z_i)
\right]
s_E.
\]

Poiché il termine sottratto dipende dal tipo e dai parametri ma non dalle coordinate, non cambia le forze.

La scala energetica `energy_scale` non è un iperparametro indipendente del JSON: durante il training viene impostata a

\[
s_E = F_{\mathrm{RMS,train}}.
\]

È una convenzione di conditioning numerico del modello. Il valore viene salvato come registered buffer nei pesi e riportato nell'effective config/manifest delle versioni correnti.

---

# 14. Formato del dataset binario

Il trainer legge un formato binario custom.

Header globale:

```text
int32 num_frames
```

Per ogni frame:

```text
int32 num_molecules
int32 num_total_sites
float32 box[3]
```

Per ogni molecola:

```text
int32 molecule_id
int32 num_sites
float32 center[3]
float32 target_force[3]
float32 target_torque[3]
```

Per ogni sito:

```text
int32 site_type
float32 position[3]
```

Guardrail del reader:

- `num_frames > 0`;
- ID molecola sequenziali;
- `site_type < num_species`;
- valori finiti;
- box positivo;
- `num_total_sites` coerente;
- nessun byte extra a fine file.

Le neighbor list del training vengono precomputate una sola volta per l'intero dataset, perché coordinate e cutoff non cambiano tra epoche.

---

# 15. Manifest del modello e provenance

Il trainer produce:

```text
model.pt
model.pt.manifest.json
```

Schema corrente:

```text
schema_version = 3
framework = MLCG_Framework_v2
energy_gauge = isolated_species_zero_v1
```

Il manifest registra almeno:

- variante architetturale;
- `num_species`;
- `hidden_channels`;
- `n_layers`;
- `num_rbf`;
- `cutoff`;
- `toxvaerd_alpha`;
- effective training config;
- hash SHA256 e dimensioni di modello/dataset/config;
- split seed e configurazione validation;
- unità di force/torque.

Il runtime valida il manifest prima di caricare il modello.

Per i float architetturali viene tollerato il normale round-trip float32, ad esempio:

```text
1.2616 → 1.2616000175476074
```

senza accettare differenze fisicamente significative. La tolleranza corrente è dell'ordine di `rel_tol=1e-6`, `abs_tol=1e-8`.

`--allow_missing_model_manifest` è esclusivamente una via legacy esplicita.

---

# 16. Checkpoint di ESPResSo

L'equilibrazione produce `equilibrated.npz`.

Il checkpoint schema corrente contiene:

- posizione;
- velocità;
- quaternion;
- `omega_body`;
- box;
- ID particelle;
- tipi;
- `mol_id`;
- hash degli input;
- architettura;
- `dt`;
- `kT`;
- metadata di provenance.

Quando `run_cg_md.py` carica un checkpoint, confronta:

- dataset;
- config;
- priors;
- rigid-body info;
- modello;
- manifest;
- box;
- identità e ordine delle particelle.

Le opzioni:

- `--allow_legacy_checkpoint`;
- `--allow_checkpoint_mismatch`;

devono essere usate soltanto per diagnostica/recupero consapevole, non per produzione.

---

# 17. Installazione Python e build del trainer

## 17.1 Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Dipendenze Python principali:

```text
MDAnalysis
numpy
scipy
matplotlib
```

## 17.2 LibTorch

Il trainer è C++17/LibTorch.

```bash
cd training
mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH=/percorso/a/libtorch ..
cmake --build . -j
```

Vengono costruiti:

```text
train_painn
eval_parity
```

Il trainer sceglie automaticamente:

1. CUDA se disponibile;
2. MPS su macOS se disponibile;
3. CPU altrimenti.

Per NVE il device del runtime ML deve invece essere preferibilmente **CPU**, per ridurre il floor numerico dell'energia.

---

# 18. Integrazione del plugin PaiNN/Morse in ESPResSo

La cartella:

```text
simulation/espresso_plugin/
```

contiene il bridge C++/Cython.

La versione corrente include anche il bonded Morse analitico.

Usare lo script:

```bash
cd simulation/espresso_plugin
bash copy_plugin_files.sh /path/to/espresso
```

oppure la sintassi prevista dalla revisione locale dello script. Su macOS la versione corrente controlla i file con `cmp -s` e stampa `[SKIP]` se sorgente e destinazione sono già identici, evitando l'errore di `cp` sullo stesso file.

Dopo la copia, ricompilare ESPResSo con Torch:

```bash
cd /path/to/espresso
mkdir -p build
cd build

cmake .. -DCMAKE_PREFIX_PATH=/percorso/a/libtorch
cmake --build . -j
```

## 18.1 Proprietà del plugin PaiNN

Il plugin:

- usa solo particelle con `type < num_species`;
- usa i displacement PBC già generati dal Verlet traversal ESPResSo;
- deduplica alias/ghost periodic della stessa coppia fisica;
- costruisce due edge diretti per coppia;
- calcola energia e forza dallo **stesso scalare Hamiltoniano**;
- forza il single-rank come percorso certificato;
- mantiene un accumulatore energetico più accurato su CPU.

La condizione geometrica fondamentale è:

\[
L_{\min}
>
2(r_c+\mathrm{skin}).
\]

## 18.2 Dummy interaction per la neighbor list

ESPResSo deve conoscere un'interazione nonbonded con cutoff almeno pari al cutoff PaiNN affinché la neighbor list includa gli edge ML.

Il runtime registra quindi una `SoftSphere(a=0)` fittizia.

**Correzione corrente importante:** la dummy interaction deve essere installata soltanto per

```text
type = 0 ... num_species-1
```

e **non** per il tipo COM dummy.

I COM dei corpi single-site coincidono esattamente con il loro virtual site. Applicare una SoftSphere singolare anche alla coppia site/COM a \(r=0\) può produrre `NaN` nell'energia nonbonded nonostante `a=0`.

---

# 19. Configurazione `topology_config.json`

Il template neutro è:

```text
preprocessing/topology_config.json
```

## 19.1 `temperature`

Temperatura in K usata per:

\[
\beta=\frac1{RT},
\]

DBI e calibrazione WCA.

Deve essere coerente con la distribuzione atomistica usata per costruire il dataset.

## 19.2 `mapping.mapping_method`

Valori:

- `COM`;
- `COG`;
- `ATOM`.

Controlla la geometria dei siti, non il COM dinamico del corpo, che resta mass-weighted.

## 19.3 `mapping.residues`

Definisce quali residui vengono inclusi e quali atomi appartengono a ogni sito.

Esempio:

```json
"residues": {
  "MOL": {
    "CG_A": ["A1", "A2"],
    "CG_B": ["B1", "B2"]
  }
}
```

Residui non presenti vengono ignorati dal mapping.

## 19.4 `mapping.site_types`

Indici di specie PaiNN. Devono essere coerenti con il training config.

## 19.5 `prior_geometry.default_angle_site`

Indice di virtual site da applicare come default ai tre vertici degli angle prior.

`-1` indica il COM; valori `>=0` sono **indici di sito nella molecola**, non site type.

## 19.6 `bonds`

Campi comuni:

```text
mol_i, mol_j
site_i, site_j
type
name
exclude_wca
```

Per harmonic:

```text
k, r0
```

con `k="auto"`, `r0="auto"` supportati.

Per FENE:

```text
k, r0, r_max
```

Per Morse:

```text
D, a, r0
```

e opzionalmente `r_cut`. Il default di `exclude_wca` per Morse è `false`; per harmonic/FENE è `true`. Con WCA policy v3, un harmonic/FENE site-site con `exclude_wca=true` sopprime WCA soltanto sulla specifica coppia `(site_i, site_j)`. Un bond COM-COM può definire semantica bonded ma non crea una virtual-site exclusion.

Per un Morse dissociativo, `r_cut` deve restare molto oltre le distanze fisicamente esplorate: non usarlo come soglia di rottura.

## 19.7 `angles`

```text
mol_i, mol_j, mol_k
site_i, site_j, site_k
type = harmonic
k
theta0
name
exclude_wca
```

`theta0` è in radianti.

## 19.8 `dihedrals`

```text
mol_i, mol_j, mol_k, mol_l
site_i, site_j, site_k, site_l
type = cosine
k
n
phi0
name
```

## 19.9 `wca_sigma`

Valore raccomandato:

```json
"wca_sigma": "auto"
```

Nel framework pair-specific attuale questo attiva il fit completo e genera `wca_pairs`.

## 19.10 `wca_quantile_percent`

Quantile basso delle distanze fisiche usato per l'onset WCA.

Valori più piccoli:

- rendono il core meno invasivo;
- aumentano sensibilità statistica alle code rare.

Deve stare in `(0,50)`.

## 19.11 `wca_guard_fraction`

\[
r_{\mathrm{guard}}
=
f_{\mathrm{guard}}r_c.
\]

Default template: `0.8`.

Più piccolo significa calibrare l'altezza della barriera più in profondità nel core.

## 19.12 `wca_guard_kbt`

Fissa

\[
U(r_{\mathrm{guard}})
=
\texttt{wca_guard_kbt}\,k_BT.
\]

Valori grandi aumentano \(\epsilon\) e la rigidità del core.

## 19.13 `wca_physical_guard_margin`

Impone il margine rispetto al minimo fisico osservato.

Default: `0.98`.

## 19.14 `decoy_target_fraction`

Default raccomandato:

```text
0.0
```

I decoy legacy hanno target residuo zero per l'intero frame senza una loss mask per molecola. Per questo sono disabilitati.

## 19.15 `allow_unmasked_zero_target_decoys`

Deve restare `false` salvo riproduzione intenzionale di vecchie ablation.

## 19.16 `decoy_random_seed`

Seed della generazione decoy, se abilitata.

## 19.17 `rigid_bodies`

Opzionale.

Per ogni resname può controllare:

```text
auto_align_sites
sites.<name>.relative_pos_nm
```

Con `auto_align_sites=true` la geometria viene inferita/mediata dai dati. Con `false` la geometria deve essere completamente specificata.

---

# 20. Configurazione del training

File template:

```text
training/cg_model_config.json
```

Per TEL22:

```text
tutorials/tel22/tel22_training_config.json
```

## 20.1 Parametri architetturali

### `architecture_variant`

Deve essere esattamente:

```text
painn_canonical_context_silu_v2
```

### `num_species`

Numero di embeddings.

Deve valere:

\[
\texttt{num_species}
>
\max(\texttt{site_type}).
\]

### `hidden_channels`

Dimensione \(D\) delle feature.

Aumentarlo aumenta:

- capacità;
- memoria;
- costo dei message/update blocks.

### `n_layers`

Numero di blocchi message + update.

Aumenta la profondità del message passing e il receptive field grafico effettivo.

### `num_rbf`

Numero di funzioni radiali Gaussiane.

Aumentarlo aumenta la risoluzione della dipendenza radiale.

### `cutoff`

Cutoff PaiNN in nm.

Influenza:

- quantità di edge;
- costo;
- informazione ambientale accessibile;
- requisito sulla dimensione del box;
- dummy neighbor-list runtime.

### `toxvaerd_alpha`

Controlla la forma del cutoff smooth.

Non deve essere cambiato tra training e runtime.

## 20.2 Ottimizzazione

### `epochs`

Numero massimo di epoche.

Può terminare prima per early stopping.

### `learning_rate`

Learning rate iniziale AdamW.

### `weight_decay`

Weight decay AdamW.

### `batch_size`

Numero di frame per minibatch.

Poiché un frame può contenere centinaia di siti, questo parametro influenza fortemente memoria e numero di optimizer update per epoca.

### `grad_clip_norm`

Clipping della norma globale dei gradienti.

- `0`: disabilitato;
- `>0`: `clip_grad_norm_`.

Il trainer riporta mean, P50, P95, max e frazione di batch clipped.

Se la frazione clipped è vicina a 1, il valore può essere un forte collo di bottiglia dell'ottimizzazione.

### `reduce_lr_patience`

Numero di epoche senza miglioramento prima di dimezzare il learning rate.

Il learning rate non scende sotto circa `1e-6`.

### `early_stopping_patience`

Numero di epoche senza miglioramento prima dello stop.

Il modello migliore viene salvato quando la validation loss migliora.

## 20.3 Loss

### `torque_weight`

\[
L=L_F+\texttt{torque_weight}\,L_\tau+\dots
\]

`0` attiva il fast path force-only nel training.

Il torque resta comunque calcolabile come metrica diagnostica in validation.

### `lipschitz_lambda`

Peso della penalizzazione sulla norma quadratica delle forze di sito.

Default consigliato finché non esiste una motivazione specifica: `0`.

## 20.4 Split e diagnostica

### `diagnostic_overfit_frames`

- `0`: training normale;
- `N>0`: tiny-set diagnostic, gli stessi N frame sono train e validation.

La validation in questo modo **non** misura generalizzazione.

### `physical_validation_only`

Esclude i decoy legacy dalla validation.

Default sicuro: `true`.

### `include_decoys_in_train`

Default sicuro: `false`.

### `validation_fraction`

Frazione validation quando lo split non usa un tail size esplicito.

Deve stare in `(0,1)`.

### `validation_split_mode`

- `random`;
- `tail`.

`tail` è utile quando un builder esterno preordina il dataset per ottenere un holdout controllato.

### `validation_tail_frames`

Se `>0` e `validation_split_mode="tail"`, usa esattamente quel numero di frame finali come validation.

### `split_seed`

Seed dello split.

### `shuffle_each_epoch`

Riordina il training set a ogni epoca con seed deterministico dipendente dall'epoca.

## 20.5 Diagnostica/MPS

### `report_grad_norms`

Abilita statistiche sui gradienti pre-clipping.

### `mps_empty_cache_every_batches`

Solo macOS/MPS.

- `0`: nessun `emptyCache` periodico;
- `N>0`: svuota cache ogni N training batch.

Può ridurre picchi di memoria al costo di overhead.

---

# 21. Pipeline completa: script e parametri

La pipeline TEL22 è:

```text
01_run_gromacs.sh
      ↓
02_build_dataset.sh
      ↓
03_train_model.sh
      ↓
04_equilibrate.sh
      ↓
06_certify_nve.sh
      ↓
05_run_espresso.sh
```

La certificazione NVE è opportuno eseguirla prima di considerare il modello/runtime pronto per produzione conservativa.

---

## 21.1 `01_run_gromacs.sh` — traiettoria atomistica

Scopo:

- ottenere coordinate;
- box;
- velocità se richieste;
- soprattutto **forze atomistiche**.

Workflow TEL22 di riferimento:

1. scarica PDB 143D;
2. seleziona il primo modello NMR;
3. `pdb2gmx` con AMBER99SB-ILDN/TIP3P;
4. inserisce 10 copie in box 8 nm;
5. solvata;
6. aggiunge 0.15 M KCl e neutralizza;
7. minimizzazione;
8. NVT;
9. NPT;
10. produzione;
11. `trjconv -pbc whole -force`;
12. `gmx check`.

La produzione di riferimento è circa 1 ns con frame/forze ogni 1 ps.

### Parametri fisici importanti

Non sono tutti CLI dello script: molti sono nei file `mdp/`.

- `dt`: timestep atomistico;
- `nsteps`: durata;
- `nstxout`: frequenza coordinate;
- `nstvout`: frequenza velocità;
- `nstfout`: frequenza forze;
- termostato/barostato;
- cutoff elettrostatici;
- PME;
- constraints.

Per il force matching:

```text
nstfout > 0
```

è non negoziabile.

La traiettoria finale deve risultare in `gmx check` con:

```text
Coords
Forces
Box
```

e, se desiderato, `Velocities`.

### Smoke test corto

Una versione corta può essere usata solo per verificare la pipeline, non per produrre un modello scientificamente convergente. Nel test TEL22 recente sono stati usati 20 ps NVT + 20 ps NPT + 50 ps production, con 51 frame ogni 1 ps.

---

## 21.2 `02_build_dataset.sh`

Wrapper di:

```text
preprocessing/build_cg_dataset.py
```

Variabili environment:

```bash
AA_TOPOLOGY=md.gro
AA_TRAJECTORY=md_whole.trr
PYTHON_BIN=python3
```

Uso:

```bash
cd tutorials/tel22
bash 02_build_dataset.sh
```

Output:

```text
tel22_dataset.bin
cg_priors.json
rigid_bodies_info.json
```

### CLI del builder

```bash
python3 preprocessing/build_cg_dataset.py \
  --topology ... \
  --trajectory ... \
  --config ... \
  --output ... \
  --priors-output ... \
  --rb-info-output ...
```

Parametri:

| parametro | significato |
|---|---|
| `--topology`, `-c` | topologia letta da MDAnalysis |
| `--trajectory`, `-f` | traiettoria con forze |
| `--config`, `-j` | mapping/topologia CG |
| `--priors`, `-p` | riusa priors esistenti e salta la nuova inferenza statistica |
| `--output`, `-o` | dataset binario |
| `--priors-output` | path di `cg_priors.json` |
| `--rb-info-output` | path di `rigid_bodies_info.json` |
| `--clip_forces` | clipping opzionale componente-per-componente |

### Quando usare `--priors`

Serve quando si vuole rigenerare il dataset con **esattamente gli stessi prior**. Il file deve già contenere la policy WCA esplicita corrente.

---

## 21.3 `03_train_model.sh`

Wrapper del trainer C++.

Variabile:

```bash
TRAINER=/path/to/train_painn
```

Default:

```text
../../training/build/train_painn
```

Uso:

```bash
bash 03_train_model.sh
```

che equivale concettualmente a:

```bash
training/build/train_painn \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config.json
```

### `--resume`

Il trainer supporta:

```bash
train_painn dataset.bin model.pt config.json --resume
```

Il resume è deliberatamente esplicito.

Se il modello esiste e `--resume` manca, il trainer rifiuta di sovrascriverlo implicitamente.

Prima del resume vengono verificati manifest, configurazione e artifact.

### Output

```text
tel22_model.pt
tel22_model.pt.manifest.json
cg_training_log.csv
```

---

## 21.4 `training/create_model_manifest.py`

Utility per finalizzare/rigenerare il sidecar manifest:

```bash
python3 training/create_model_manifest.py \
  --model model.pt \
  --config config.json \
  --dataset dataset.bin
```

Non modifica i pesi; aggiorna metadata, hash e architecture provenance.

Usarlo soprattutto quando si sta migrando un modello prodotto da una revisione compatibile del trainer.

---

## 21.5 `training/eval_parity`

Executable diagnostico per confronti del modello C++.

Va compilato insieme al trainer.

È utile quando si cambia:

- architettura;
- PBC;
- RBF;
- gauge;
- serializzazione;
- plugin.

Non sostituisce un test NVE.

---

## 21.6 `simulation/espresso_plugin/copy_plugin_files.sh`

Copia i file custom nel tree ESPResSo.

Dopo ogni modifica a:

```text
PaiNN_Architecture.hpp
PaiNN_ML_Potential.*
painn.pyx
morse_bond.hpp / binding associati
```

occorre ricopiare e ricompilare ESPResSo.

Una compilazione precedente non incorpora automaticamente i nuovi sorgenti del framework.

---

## 21.7 `04_equilibrate.sh`

Wrapper di `simulation/equilibrate.py`.

Variabili:

```bash
PYRESSO=../../espresso/build/pypresso
DEVICE=auto
```

Uso:

```bash
PYRESSO=../../espresso/build/pypresso \
bash 04_equilibrate.sh
```

Output:

```text
equilibrated.npz
```

### CLI `equilibrate.py`

| parametro | default | effetto |
|---|---:|---|
| `--model` | required | pesi PaiNN |
| `--config` | required | config NN |
| `--priors` | required | prior |
| `--rb_info` | required | masse/inerzie/geometrie |
| `--dataset` | required | frame iniziale |
| `--dt` | `0.002` ps | timestep warmup |
| `--out_checkpoint` | `equilibrated.npz` | output |
| `--device` | `auto` | `cpu/mps/cuda/auto` |
| `--kT` | `2.49` | temperatura dinamica |
| `--steps_sd` | `5000` | steepest-descent steps |
| `--steps_md` | `2000` | classical NVT warmup |
| `--steps_ml_capped` | `2000` | ML NVT con force cap |
| `--steps_ml_uncapped` | `2000` | ML NVT finale uncapped |
| `--warmup_chunk` | `100` | chunk per progress/cap ramp |
| `--toxvaerd_alpha` | config | override runtime |
| `--allow_missing_model_manifest` | false | legacy |
| `--allow_unsafe_mpi` | false | multi-rank non certificato |

Gli override architetturali, incluso `toxvaerd_alpha`, devono restare coerenti con il manifest.

---

## 21.8 `05_run_espresso.sh`

Wrapper di produzione.

Variabili:

```bash
PYRESSO=../../espresso/build/pypresso
DEVICE=auto
CG_STEPS=20000
CG_DT=0.001
```

Uso:

```bash
PYRESSO=../../espresso/build/pypresso \
CG_DT=0.001 \
CG_STEPS=20000 \
bash 05_run_espresso.sh
```

Tempo simulato:

\[
t_{\mathrm{sim}}
=
\texttt{CG_STEPS}\times\texttt{CG_DT}.
\]

Con i default:

\[
20000\times0.001 = 20\ \mathrm{ps}.
\]

---

## 21.9 `simulation/run_cg_md.py`

Parametri runtime correnti:

| parametro | significato |
|---|---|
| `--model` | modello PaiNN; può essere omesso per una diagnostica classical-only |
| `--config` | configurazione architettura |
| `--priors` | prior runtime |
| `--rb_info` | rigid-body metadata |
| `--dataset` | dataset usato per topologia/frame iniziale |
| `--checkpoint` | checkpoint da caricare |
| `--dt` | timestep |
| `--steps` | numero di step |
| `--log_interval` | frequenza output standard |
| `--device` | `cpu/mps/cuda/auto` |
| `--kT` | temperatura Langevin |
| `--init_kT` | reinizializza velocità Maxwell-Boltzmann |
| `--nve` | disattiva termostato |
| `--no_log` | disabilita logging opzionale/non essenziale |
| `--no_vtf` | disabilita output VTF |
| `--energy_file` | path CSV energia |
| `--trajectory_file` | path traiettoria VTF |
| `--toxvaerd_alpha` | override con controllo manifest |
| `--allow_missing_model_manifest` | legacy |
| `--allow_legacy_checkpoint` | legacy |
| `--allow_checkpoint_mismatch` | bypass provenance |
| `--allow_unsafe_mpi` | multi-rank non certificato |
| `--allow_nonconservative_tables` | consente esplicitamente prior tabulati in NVE |

Nel runtime analitico corrente Morse non è più classificato come tabella non conservativa; il guard NVE riguarda i prior realmente tabulati.

### `--init_kT`

Per la traslazione:

\[
v_\alpha
\sim
\mathcal N
\left(
0,\frac{kT}{M}
\right).
\]

Per la rotazione:

\[
\omega_\alpha
\sim
\mathcal N
\left(
0,\frac{kT}{I_\alpha}
\right).
\]

Per una produzione derivata da equilibrazione è normalmente preferibile caricare le velocità dal checkpoint anziché reinizializzarle.

---

## 21.10 `06_certify_nve.sh`

Certifica la conservatività numerica del sistema completo.

Variabili principali del wrapper:

```bash
PYRESSO=../../espresso/build/pypresso
NVE_DTS="0.001 0.002 0.005 0.01"
NVE_DURATION_PS=5.0
```

Uso raccomandato:

```bash
NVE_DTS="0.001 0.002 0.005 0.01" \
NVE_DURATION_PS=5.0 \
PYRESSO=../../espresso/build/pypresso \
bash 06_certify_nve.sh --overwrite
```

Il wrapper chiama `simulation/certify_nve.py`.

L'opzione `--overwrite` permette di rigenerare directory già esistenti.

Il certifier richiede almeno tre `dt` positivi.

---

## 21.11 `simulation/certify_nve.py`

È l'orchestratore della certificazione NVE. In sequenza:

1. valida artifact, lista dei timestep e durata fisica;
2. calcola gli hash degli input rilevanti;
3. rifiuta per default prior esplicitamente tabulati/non conservativi;
4. usa CPU come backend di riferimento per la certificazione;
5. converte la stessa durata fisica nel numero di step appropriato per ogni `dt`;
6. esegue `run_cg_md.py --nve` per ciascun timestep;
7. forza il campionamento dell'energia a ogni step nel protocollo corrente;
8. legge le serie energetiche;
9. calcola le metriche per-run tramite `nve_analysis.py`;
10. esegue il fit power-law rispetto a `dt`;
11. scrive report JSON e tabella CSV;
12. restituisce exit code non zero in caso di fallimento numerico o del subprocess.

Il certifier verifica **conservatività e comportamento numerico dell'integratore**; non misura la qualità fisica del force matching.

---

## 21.12 `simulation/nve_analysis.py`

Contiene le metriche numeriche NVE.

Metriche principali:

- `sigma_E = std(E_tot)` come osservabile primaria;
- `rms_dE`, mantenuto come diagnostica secondaria;
- drift relativo tra blocco iniziale e finale;
- fit power-law vs timestep.

---

## 21.13 `tutorials/plot_metrics_cg.py`

Utility grafica per log del training/metriche.

Non entra nella definizione del modello e può essere usata dopo il training per ispezionare:

- train/validation loss;
- componenti force/torque;
- MAE;
- trend di overfitting.

---

## 21.14 `preprocessing/geometry_utils.py` — modulo interno, non CLI

Non viene invocato direttamente dalla shell, ma centralizza primitive geometriche che devono restare testabili e coerenti con il preprocessing:

- diagonalizzazione del tensore d'inerzia con ordinamento dei momenti principali;
- correzione della base per avere determinante positivo e quindi assi principali destrorsi;
- displacement minimum-image per box ortorombico;
- matrice delle distanze MIC.

La convenzione della base è tale che, se le colonne di `principal_axes` sono gli assi del body frame espressi nello space frame,

\[
\mathbf r_{\mathrm{body}}
=
\mathbf A^T\mathbf r_{\mathrm{space}}.
\]

Modificare queste convenzioni richiede di verificare rigid geometry, quaternion/orientation runtime e torque.

---

## 21.15 `simulation/framework_utils.py` — validazione e provenance condivise

Anche questo è un modulo interno e non uno script da lanciare direttamente. È importato da equilibrazione/produzione e contiene guardrail condivisi, tra cui:

- schema e validazione del model manifest;
- identificatore dell'architettura PaiNN e dell'energy gauge;
- SHA256 degli artifact;
- validazione della policy di esclusione WCA 1–2/1–3;
- costruzione delle liste di coppie WCA escluse;
- salvataggio/validazione dei checkpoint con particle signature;
- controllo di config, hash, box e stato prima del restart.

I parametri architetturali float del manifest devono usare la tolleranza corrente compatibile con round-trip `float32` (`rel_tol` circa `1e-6`, `abs_tol` circa `1e-8`), mentre stringhe/interi/booleani restano confrontati esattamente.

Non conviene duplicare questi controlli nei wrapper: la loro centralizzazione serve a impedire che `equilibrate.py` e `run_cg_md.py` accettino artifact differenti con regole differenti.

---

# 22. Equilibrazione CG in dettaglio

L'equilibrazione corrente ha quattro fasi.

## Phase 1 — Steepest descent classico

PaiNN non è ancora attivo.

Default:

```text
steps_sd = 5000
f_max = 10000
gamma = 50
max_displacement = 0.001
```

Scopo: rimuovere overlap/strain iniziali usando soltanto i prior.

## Phase 2 — Classical NVT con force cap

Velocity-Verlet + Langevin.

Inizio:

```text
force_cap = 500
gamma = gamma_rot = 50
```

Il cap cresce gradualmente fino a circa 1000.

Scopo: riscaldare/assestare il sistema senza esporlo immediatamente a forze estreme.

## Attivazione PaiNN

Dopo la rilassazione classica il plugin viene attivato con:

```text
num_species
hidden_channels
n_layers
num_rbf
cutoff
toxvaerd_alpha
device
```

## Phase 3 — ML NVT capped

Force cap iniziale 500, ramp fino a circa 1500.

È una fase di introduzione controllata del residuo ML.

## Phase 4 — ML NVT uncapped

```text
force_cap = 0
gamma = gamma_rot = 1
```

Questa fase usa l'Hamiltoniana di produzione.

Solo dopo questa fase viene scritto il checkpoint.

### Quando aumentare le fasi

Aumentare gli step se:

- geometria iniziale lontana dal supporto;
- WCA molto stiff;
- PaiNN inizialmente produce forze elevate;
- il checkpoint finale mostra forti tensioni.

Ridurre soltanto per smoke test.

---

# 23. Produzione CG in dettaglio

In produzione normale:

- integratore VV;
- WCA e bonded prior attivi;
- PaiNN attivo;
- `force_cap = 0`;
- Langevin attivo se `--nve` non è specificato.

In NVE:

```text
force_cap = 0
thermostat = OFF
integrator = velocity-Verlet
```

Il runtime esegue un force recalculation a step 0, importante per:

- energia ML iniziale;
- diagnosi NaN prima dell'integrazione;
- consistenza del CSV.

## 23.1 Energy logging

Il CSV diagnostico corrente contiene campi del tipo:

```text
Step
Time_ps
E_tot
E_kin
E_kin_trans
E_kin_rot
E_class
E_ml
min_dist
min_pair
min_pids
f_max
torque_max
```

Interpretazione:

- `E_tot`: ESPResSo + PaiNN;
- `E_kin`: cinetica riportata da ESPResSo;
- `E_kin_trans`: ricostruita esplicitamente;
- `E_kin_rot`: ricostruita esplicitamente;
- `E_class`: nome legacy per totale ESPResSo, non solo potenziale;
- `E_ml`: energia PaiNN;
- `min_dist`: minima distanza diagnostica;
- `f_max`: massimo modulo forza;
- `torque_max`: massimo torque.

Il runtime corrente deve abortire immediatamente se trova energia non finita, stampando i termini di `system.analysis.energy()` responsabili.

---

# 24. Certificazione NVE

Lo scopo della certificazione non è verificare che il modello ML sia accurato rispetto all'AA. Verifica invece che:

1. forze e energia appartengano allo stesso Hamiltoniano;
2. i prior siano conservativi;
3. il runtime/integratore mostri lo scaling atteso;
4. non esista un drift secolare incompatibile con VV.

## 24.1 Protocollo corrente

Durata fisica fissa:

```text
5 ps
```

Timestep default:

```text
0.001
0.002
0.005
0.01 ps
```

Energia salvata:

```text
ogni singolo integration step
```

Numero di campioni, incluso step 0:

| dt (ps) | steps a 5 ps | campioni |
|---:|---:|---:|
| 0.001 | 5000 | 5001 |
| 0.002 | 2500 | 2501 |
| 0.005 | 1000 | 1001 |
| 0.010 | 500 | 501 |

Per la certificazione usare CPU:

```bash
DEVICE=cpu
```

o il comportamento CPU di default del certifier.

## 24.2 Metrica primaria

\[
\sigma_E(\Delta t)
=
\sqrt{
\left\langle
\left(E-\langle E\rangle\right)^2
\right\rangle
}.
\]

Viene usata la deviazione standard di popolazione (`ddof=0`).

Per un integratore simplettico del secondo ordine in regime asintotico:

\[
\sigma_E
\propto
(\Delta t)^p,
\qquad
p\approx 2.
\]

Si esegue il fit:

\[
\log\sigma_E
=
\log C
+
p\log\Delta t.
\]

## 24.3 Drift separato

Il drift è valutato separatamente confrontando il valore medio nei blocchi iniziale/finale, tipicamente 20% + 20%.

Una forma normalizzata è:

\[
D_{\mathrm{rel}}
=
\frac{
|\langle E\rangle_{\mathrm{final}}
-
\langle E\rangle_{\mathrm{iniziale}}|
}{
E_{\mathrm{scale}}
}.
\]

Threshold corrente di riferimento:

```text
relative block drift <= 1e-4
```

## 24.4 Threshold sul fit

Valori di riferimento usati dal certificatore:

```text
1.7 <= p <= 2.3
R² >= 0.97
```

Il risultato non deve essere interpretato meccanicamente se il punto a `dt=0.01 ps` è fuori dal regime stabile/asintotico. Un timestep molto grande può essere un **stress point** e non un timestep produttivo.

Analogamente, andando molto sotto `dt=0.001 ps`, specialmente con parti del calcolo in float32, lo scaling può essere dominato dal roundoff invece che dall'errore di discretizzazione.

---

# 25. Interpretazione quantitativa dello scaling NVE

Un risultato ideale mostra:

\[
\frac{\sigma_E(\Delta t)}
{\sigma_E(\Delta t/2)}
\approx 4.
\]

L'esponente locale è:

\[
p_{\mathrm{local}}
=
\frac{
\log[\sigma_E(dt_2)/\sigma_E(dt_1)]
}{
\log(dt_2/dt_1)
}.
\]

Possibili casi:

### `p ≈ 2`, R² alto, drift basso

Comportamento coerente con VV conservativo.

### `p < 2` solo ai dt più piccoli

Probabile floor numerico/roundoff.

### punto più grande fortemente fuori fit

Possibile timestep troppo grande o uscita dal regime asintotico.

### NaN a step 0

Non è un problema di timestep: il sistema iniziale o un termine energetico è già non finito.

### drift cresce ma sigma scala correttamente

Possibile bias lento, inizializzazione non equilibrata o termine non perfettamente conservativo.

### forze esplodono/min distance collassa

Stabilità fisica del modello/prior, non semplicemente precisione dell'integratore.

---

# 26. Test e controlli di consistenza

## 26.1 Test Python

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Coprono:

- geometry helpers;
- manifest/checkpoint;
- preprocessing guardrails;
- source invariants;
- NVE analysis;
- assenza di dipendenze TEL22 nel core.

## 26.2 `git diff --check`

```bash
git diff --check
```

Deve restituire output vuoto.

## 26.3 Plugin PBC regression

Eseguire con ESPResSo patchato:

```bash
PYRESSO=/path/to/pypresso \
python3 tests/run_painn_plugin_pbc_regression.py
```

Il percorso esatto dipende dalla build locale.

## 26.4 Morse smoke test

Dopo rebuild di ESPResSo verificare che il bonded Morse restituisca energia e forze finite e opposte sui due estremi.

## 26.5 Audit priors

Controllare almeno:

```python
import json

p = json.load(open("cg_priors.json"))
print(len(p.get("bonds", [])))
print(len(p.get("angles", [])))
print(len(p.get("dihedrals", [])))
print(len(p.get("wca_pairs", {})))
```

e verificare assenza di NaN/Inf nel JSON.

---

# 27. Tutorial TEL22: configurazione corrente

TEL22 è un esempio, non una dipendenza del framework.

Configurazione recente:

```text
10 copie TEL22
22 residui/copia
~82 siti CG/copia
~820 siti/frame
8 site types PaiNN
cutoff = 1.2616 nm
hidden_channels = 64
n_layers = 2
num_rbf = 32
learning_rate = 0.001
batch_size = 4
torque_weight = 0.5
grad_clip_norm = 1.0
```

Priors prodotti dal tutorial corrente:

```text
bonds = 390
  harmonic = 210
  Morse = 180
angles = 200
dihedrals = 0
tabulated priors = 0
```

Il run AA corto da 50 ps produce 51 frame a 1 ps ed è adatto a smoke test, non a un training definitivo.

Per un dataset scientifico si deve usare una traiettoria sufficientemente lunga e rappresentativa.

---

# 28. Troubleshooting

## 28.1 `Model manifest mismatch: cutoff`

Caso tipico:

```text
manifest=1.2616000175476074
runtime=1.2616
```

È un round-trip float32. La versione corrente del validator usa una tolleranza adeguata.

Se la differenza è, per esempio, `1.27` vs `1.2616`, il mismatch è reale e deve essere rifiutato.

---

## 28.2 `E_tot = NaN` a step 0

Stampare i singoli termini di:

```python
system.analysis.energy()
```

Il caso TEL22 diagnosticato era:

```text
('non_bonded', 0, 9): nan
('non_bonded', 1, 9): nan
```

Il tipo 9 era il COM dummy, coincidente con siti single-site. La causa era la dummy SoftSphere registrata anche sui COM.

Fix corrente:

```python
for i in range(nn_config["num_species"]):
    for j in range(i, nn_config["num_species"]):
        ...
```

Non usare `num_species + 2`.

---

## 28.3 NVE fallisce per priors tabulati

Una tabella in cui energia e forza sono interpolate separatamente non garantisce automaticamente

\[
\mathbf F=-\nabla U.
\]

Per certificazione rigorosa usare prior analitici. Il Morse corrente è analitico.

`--allow_nonconservative_tables` è soltanto una escape hatch diagnostica.

---

## 28.4 `At least three energy samples are required`

Il run è troppo corto rispetto al logging.

Nel protocollo corrente il problema è evitato salvando energia a **ogni step** e usando 5 ps.

---

## 28.5 `p` NVE basso su run molto corto

Non concludere non-conservatività da poche osservazioni.

Una stima di `std(E)` richiede una serie temporale sufficiente e la stessa durata fisica per tutti i timestep.

---

## 28.6 Box troppo piccolo

Deve valere:

\[
L_{\min}>2(r_c+\mathrm{skin}).
\]

Con `skin=0.4 nm` e `cutoff=1.2616 nm`:

\[
L_{\min}>3.3232\ \mathrm{nm}.
\]

---

## 28.7 Forze residue enormi

Controllare in ordine:

1. unità delle forze AA;
2. PBC/unwrapping;
3. tabella diagnostica WCA;
4. exact physical minimum vs `r_guard`;
5. geometria raw vs rigid;
6. bonded parameters;
7. eventuale clipping;
8. outlier atomistici.

Il builder stampa percentili di `|F_reference|`, `|F_WCA|` e `|F_residual|`.

---

## 28.8 Torque single-site

Un corpo single-site non ha orientazione dinamica nel runtime corrente.

Il suo unico sito deve essere al COM e il torque non entra nella torque loss.

---

## 28.9 Multi-rank MPI

Il plugin PaiNN multi-rank non è certificato.

Usare una sola rank salvo parity experiment esplicito con:

```text
--allow_unsafe_mpi
```

---

## 28.10 MPS vs CPU

MPS può accelerare training/runtime, ma parti del calcolo restano float32.

Per certificazione energetica usare CPU, dove il plugin può usare accumulo più accurato.

---

# 29. Checklist per adattare il framework a un nuovo sistema

1. **AA trajectory**
   - coordinate;
   - box;
   - forze reali;
   - sampling appropriato.

2. **Mapping**
   - definire `residues`;
   - site names;
   - site type consecutivi/validi;
   - scegliere COM/COG/ATOM.

3. **Rigid bodies**
   - verificare se il mapping multi-site deve essere rigidificato;
   - single-site esattamente al COM;
   - controllare inertia/principal axes.

4. **Bonded topology**
   - definire bond/angle/dihedral;
   - separare connettività covalente e contatti dissociativi: harmonic/FENE tipicamente `exclude_wca=true`, Morse tipicamente `exclude_wca=false`;
   - per un contatto Morse verificare che `D`, `a` e il numero di contatti cooperativi consentano davvero la scala di unfolding desiderata.

5. **WCA**
   - usare `wca_sigma="auto"`;
   - per 1–2 usare la policy v3 `bonded_site_pairs_only`, non un'esclusione all-sites;
   - mantenere identica la policy tra preprocessing, equilibration e produzione;
   - ispezionare la tabella diagnostica WCA;
   - verificare `r_guard < r_min`;
   - dopo ogni cambio di policy rigenerare priors/dataset, retrain, re-equilibrate e rifare NVE.

6. **Dataset**
   - nessun NaN;
   - force reference non zero;
   - target residuali con scale plausibili.

7. **Training config**
   - `num_species` corretto;
   - cutoff fisicamente ragionevole;
   - capacità adeguata;
   - split riproducibile;
   - torque weight solo se esistono corpi multi-site.

8. **Training**
   - confrontare validation con zero predictor;
   - monitorare gradient clipping;
   - non interpretare tiny-overfit come generalizzazione.

9. **Manifest**
   - presente e coerente.

10. **ESPResSo**
    - plugin aggiornato;
    - Morse analitico;
    - dummy neighbor interaction solo sui tipi ML;
    - single rank.

11. **Equilibration**
    - completare fase uncapped;
    - salvare checkpoint con provenance.

12. **NVE**
    - CPU;
    - 5 ps;
    - energia ogni step;
    - range dt 0.001–0.01 ps;
    - `sigma_E ~ dt^2`;
    - drift basso.

13. **Production**
    - scegliere `dt` nel regime stabile;
    - non usare automaticamente il massimo dt testato;
    - controllare periodicamente min distance, force max, torque max ed energia.

---

## Comandi minimi TEL22

```bash
cd tutorials/tel22

# 1. AA
bash 01_run_gromacs.sh

# 2. Dataset
bash 02_build_dataset.sh

# 3. Training
bash 03_train_model.sh

# 4. Equilibration
PYRESSO=../../espresso/build/pypresso \
bash 04_equilibrate.sh

# 5. NVE certification
NVE_DTS="0.001 0.002 0.005 0.01" \
NVE_DURATION_PS=5.0 \
PYRESSO=../../espresso/build/pypresso \
bash 06_certify_nve.sh --overwrite

# 6. Production
CG_DT=0.001 \
CG_STEPS=20000 \
PYRESSO=../../espresso/build/pypresso \
bash 05_run_espresso.sh
```

La sequenza precedente è un esempio TEL22. Per un sistema generico sostituire i file di mapping/config e passare esplicitamente gli artifact a `build_cg_dataset.py`, `train_painn`, `equilibrate.py` e `run_cg_md.py`.
