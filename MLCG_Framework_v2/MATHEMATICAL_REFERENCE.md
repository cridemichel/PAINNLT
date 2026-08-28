# MLCG Framework v2 — Riferimento matematico

Questo documento descrive la matematica implementata dal framework: mapping
atomistico-CG, rigid body, prior, target residuali, rete PaiNN, training,
dinamica e certificazione NVE. L'obiettivo è permettere di interpretare i
parametri senza dover ricostruire le convenzioni dai sorgenti.

La guida operativa completa resta [`HOWTO.md`](HOWTO.md); la versione inglese
di questo riferimento è [`MATHEMATICAL_REFERENCE_EN.md`](MATHEMATICAL_REFERENCE_EN.md).
Le sezioni 1–14 forniscono la mappa compatta; la Parte II, dalle sezioni 15–21,
espone le derivazioni implementative complete di PaiNN, prior, DBI/IBI e
interpolazione conservativa.

## 1. Convenzioni e unità

| Grandezza | Simbolo | Unità interna |
|---|---:|---:|
| posizione/distanza | $\mathbf r,r$ | nm |
| tempo | $t$ | ps |
| energia | $U,E$ | kJ mol$^{-1}$ |
| forza | $\mathbf F$ | kJ mol$^{-1}$ nm$^{-1}$ |
| torque | $\boldsymbol\tau$ | kJ mol$^{-1}$ |
| massa | $m$ | u |
| inerzia | $I$ | u nm$^2$ |
| angolo | $\theta,\phi$ | rad |
| temperatura energetica | $k_BT=RT$ | kJ mol$^{-1}$ |

Per ogni energia conservativa vale

$$
\mathbf F_i=-\nabla_{\mathbf r_i}U.
$$

Con box ortorombico, la minimum image convention (MIC) è applicata componente
per componente:

$$
\Delta\mathbf r\leftarrow\Delta\mathbf r-\mathbf L\,
\operatorname{round}(\Delta\mathbf r/\mathbf L).
$$

I vettori di edge PaiNN sono orientati come
$\mathbf r_{ij}=\mathbf r_i-\mathbf r_j$. Le formule dei prior tabulati hanno
convenzioni specifiche descritte nella sezione 6.

## 2. Stato coarse grained e rigid body

### 2.1 Mapping atomistico

Per un corpo $m$ composto dagli atomi $a\in m$, il centro di massa è

$$
\mathbf R_m=\frac{\sum_{a\in m}m_a\mathbf r_a}{\sum_{a\in m}m_a}.
$$

Il mapping può usare anche il centro geometrico o un atomo selezionato, ma il
centro dinamico del rigid body e i torque sono riferiti al COM. La forza e il
torque generalizzati atomistici sono

$$
\mathbf F_m^{AA}=\sum_{a\in m}\mathbf f_a^{AA},\qquad
\boldsymbol\tau_m^{AA}=\sum_{a\in m}
(\mathbf r_a-\mathbf R_m)\times\mathbf f_a^{AA}.
$$

Un corpo single-site deve avere il sito sul COM: una forza fuori centro
richiederebbe un grado di libertà rotazionale che il corpo non possiede.

### 2.2 Massa, inerzia e assi principali

Indicando con $\boldsymbol\rho_a=\mathbf r_a-\mathbf R_m$,

$$
\mathbf I_m=\sum_{a\in m}m_a
\left[(\boldsymbol\rho_a\cdot\boldsymbol\rho_a)\mathbf 1
-\boldsymbol\rho_a\boldsymbol\rho_a^T\right].
$$

Il tensore simmetrico viene diagonalizzato; gli autovettori sono ordinati per
autovalore e corretti affinché la terna sia destrorsa. Masse, momenti
principali e geometria body-frame vengono salvati in `rigid_bodies_info.json`.

### 2.3 Geometria media e Kabsch

Le geometrie multi-site dei frame vengono allineate iterativamente con Kabsch.
Date due configurazioni centrate $P,Q$,

$$
H=P^TQ=U\Sigma V^T,\qquad
R=V\,\operatorname{diag}(1,1,\operatorname{sign}\det(VU^T))\,U^T.
$$

La correzione del determinante evita riflessioni. Al runtime, la posizione di
un sito virtuale $s$ è $\mathbf r_{ms}=\mathbf R_m+Q_m\mathbf d_{ms}$, dove
$Q_m$ è la rotazione del corpo e $\mathbf d_{ms}$ l'offset body-frame.

## 3. Hamiltoniana e force matching residuale

La decomposizione fondamentale è

$$
U_{CG}(\mathbf R,Q)=U_{prior}(\mathbf R,Q)+U_{ML}(\mathbf R,Q).
$$

Il dataset non contiene le forze AA totali, ma i target residuali:

$$
\mathbf F_m^{res}=\mathbf F_m^{AA}-\mathbf F_m^{prior},\qquad
\boldsymbol\tau_m^{res}=\boldsymbol\tau_m^{AA}-\boldsymbol\tau_m^{prior}.
$$

Questa identità è una condizione di coerenza, non un dettaglio di training.
Cambiare un prior cambia il target; occorre ricostruire il dataset, riaddestrare
PaiNN e rigenerare il checkpoint. Sottrarre e simulare geometrie o parametri
diversi produce una Hamiltoniana incoerente.

Per un sito a offset $\mathbf d$ su cui agisce $\mathbf f$,

$$
\mathbf F_m\mathrel{+}=\mathbf f,\qquad
\boldsymbol\tau_m\mathrel{+}=(\mathbf r_s-\mathbf R_m)\times\mathbf f.
$$

## 4. Prior analitici

### 4.1 Bond armonico

$$
U(r)=\frac12k(r-r_0)^2,\qquad F_r=-k(r-r_0).
$$

Se automatici, $r_0=\langle r\rangle$ e
$k=[\beta\operatorname{Var}(r)]^{-1}$, con $\beta=(RT)^{-1}$.
`r0` sposta il minimo; aumentare `k` restringe le fluttuazioni e aumenta la
frequenza più rapida, imponendo in genere un timestep minore.

### 4.2 FENE

Con $x=r-r_0$ e estensione massima $r_{max}$,

$$
U(r)=-\frac12kr_{max}^2\ln\left[1-(x/r_{max})^2\right],\qquad
F_r=-\frac{kx}{1-(x/r_{max})^2}.
$$

Il dominio è $|x|<r_{max}$; la forza diverge al bordo. `k` regola la curvatura
locale, `r_max` il limite geometrico.

### 4.3 Angolo armonico

$$
U(\theta)=\frac12k(\theta-\theta_0)^2,
\qquad \frac{dU}{d\theta}=k(\theta-\theta_0).
$$

Se automatici, $\theta_0=\langle\theta\rangle$ e
$k=[\beta\operatorname{Var}(\theta)]^{-1}$. La conversione in forze cartesiane
contiene $1/\sin\theta$; geometrie vicine a $0$ o $\pi$ sono singolari.

### 4.4 Dihedro cosine

$$
U(\phi)=K[1-\cos(n\phi-\phi_0)],\qquad
\frac{dU}{d\phi}=Kn\sin(n\phi-\phi_0).
$$

`n` è la molteplicità, `K` l'altezza della modulazione e `phi0` la fase. La
fase automatica usa la media circolare
$\operatorname{atan2}(\langle\sin\phi\rangle,\langle\cos\phi\rangle)$.

### 4.5 Morse switched

Posto $y=e^{-a(r-r_0)}$,

$$
U_0(r)=D(y^2-2y),\qquad F_0(r)=2D\,a\,y(y-1).
$$

Il minimo è $U_0(r_0)=-D$ e la curvatura locale è

$$
U_0''(r_0)=2Da^2.
$$

Quindi `D` controlla profondità e curvatura, `a` controlla larghezza e
stiffness quadraticamente, `r0` la distanza di equilibrio. Nel tratto
$r_s<r<r_c$, con $t=(r-r_s)/(r_c-r_s)$,

$$
S(t)=1-10t^3+15t^4-6t^5,
$$

$$
U=SU_0,\qquad F=SF_0-U_0\frac{dS}{dr},\qquad
\frac{dS}{dr}=-\frac{30t^2(1-t)^2}{r_c-r_s}.
$$

Per $r\ge r_c$, $U=F=0$. Lo switch rende continui energia, forza e curvatura
ai bordi. Ridurre troppo $r_c-r_s$ concentra la curvatura nella coda. I Morse
pair-specific e type-pair condividono il kernel e, se sovrapposti, si sommano.

### 4.6 WCA

Con $r_c=2^{1/6}\sigma$,

$$
U_{WCA}(r)=
\begin{cases}
4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]+\epsilon,&r<r_c,\\
0,&r\ge r_c,
\end{cases}
$$

$$
F_r=\frac{24\epsilon}{r}
\left[2(\sigma/r)^{12}-(\sigma/r)^6\right].
$$

`sigma` fissa la scala del core, `epsilon` la sua altezza. Nel fit automatico,
il cutoff di coppia deriva da un quantile basso regolarizzato e rispetta

$$
r_c\le\frac{m\,r_{min}^{phys}}{g},
$$

dove $g=$ `wca_guard_fraction` e $m=$ `wca_physical_guard_margin`. Infine
`epsilon` è scelta imponendo
$U(g r_c)=$ `wca_guard_kbt` $\,k_BT$. Un `guard_kbt` maggiore irrigidisce il
core; un margine minore lo allontana dai dati fisici.

## 5. DBI e IBI

La direct Boltzmann inversion usa la densità corretta per il Jacobiano:

$$
U_0(q)=-k_BT\ln P_{corr}(q)+C,
$$

con $P_{corr}(r)=P(r)/r^2$, $P_{corr}(\theta)=P(\theta)/\sin\theta$ e nessuna
correzione Jacobiana per il dihedro periodico. Il codice identifica supporto
statistico, leviga il profilo e costruisce code confining fuori supporto.

L'aggiornamento IBI sul supporto comune target/simulazione è

$$
U_{i+1}(q)=U_i(q)+\alpha k_BT
\ln\frac{P_i(q)}{P_{target}(q)}.
$$

`alpha` smorza l'update; `max_update_kT` lo limita; smoothing e taper riducono
rumore e discontinuità. Dopo ogni modifica dei prior IBI si ricostruiscono i
target residuali.

## 6. Tabelle e spline conservative

Le tabelle legacy interpolano linearmente energia e terza colonna. Le
convenzioni sono: bond = forza radiale $-dU/dr$; angle = $dU/d\theta$;
dihedral = fattore geometrico ESPResSo, non semplicemente $-dU/d\phi$.
Interpolare energia e forza separatamente non garantisce
$\mathbf F=-\nabla U$ e non è adatto a una certificazione NVE stretta.

La rappresentazione `pchip_hermite_v1` memorizza nodi $(q_i,U_i,m_i)$ con
$m_i=dU/dq$. Nell'intervallo $q=q_i+th$, $t\in[0,1]$:

$$
U(t)=h_{00}U_i+h_{10}hm_i+h_{01}U_{i+1}+h_{11}hm_{i+1},
$$

con $h_{00}=2t^3-3t^2+1$, $h_{10}=t^3-2t^2+t$,
$h_{01}=-2t^3+3t^2$, $h_{11}=t^3-t^2$. Energia e derivata provengono dallo
stesso polinomio. Gli angoli coprono $[0,\pi]$; i dihedri $[0,2\pi]$ con
energia e derivata periodiche.

## 7. Grafo e architettura PaiNN

Ogni sito fisico ha tipo $z_i$, scalari $\mathbf s_i\in\mathbb R^D$ e vettori
$\mathbf v_i\in\mathbb R^{3\times D}$:

$$
\mathbf s_i^{(0)}=\operatorname{Embedding}(z_i),\qquad \mathbf v_i^{(0)}=0.
$$

Gli edge con MIC collegano siti di molecole diverse entro `cutoff`; ogni coppia
compare in entrambe le direzioni.

### 7.1 Base radiale e cutoff

Per centri $\mu_k$ uniformi in $[0,r_c]$ e
$\sigma_{RBF}=r_c/N_{RBF}$,

$$
R_k(d)=\exp[-(d-\mu_k)^2/\sigma_{RBF}^2]c(d),
$$

$$
c(d)=\frac{x^4}{x^4+\alpha^4},\quad
x=(r_c-d)/r_c,\quad d\le r_c,
$$

e $c=0$ oltre il cutoff. Aumentare `num_rbf` aumenta la risoluzione radiale;
aumentare `cutoff` aumenta contesto, edge, costo e memoria. Aumentare
`toxvaerd_alpha` sopprime il filtro più all'interno del cutoff; ridurlo mantiene
$c\simeq1$ più a lungo ma rende la transizione più concentrata.

### 7.2 Message block

Una MLP `D -> D -> 3D` con SiLU produce il contesto scalare; una trasformazione
lineare `num_rbf -> 3D` produce il filtro radiale. Il prodotto è diviso in
$(q_0,q_1,q_2)$. Per l'edge $j\to i$:

$$
\Delta\mathbf s_i\mathrel{+}=q_0,\qquad
\Delta\mathbf v_i\mathrel{+}=\mathbf v_j\odot q_1+
\hat{\mathbf r}_{ij}\,q_2.
$$

L'aggregazione è una somma (`sum_v1`).

### 7.3 Update e readout

Con $\mathbf v_v=W_v\mathbf v$, $\mathbf v_u=W_u\mathbf v$ e

$$
\|\mathbf v_v\|_\varepsilon=
\sqrt{\sum_{a=1}^3v_{v,a}^2+10^{-8}},
$$

una MLP su $[\mathbf s,\|\mathbf v_v\|_\varepsilon]$ produce $(a,b,c)$:

$$
\Delta\mathbf s=a+(\mathbf v_v\cdot\mathbf v_u)\odot b,
\qquad \Delta\mathbf v=\mathbf v_u\odot c.
$$

Dopo `n_layers`, il readout `D -> D/2 -> 1` con SiLU produce $u_i$.
`hidden_channels` controlla la larghezza; `n_layers` la profondità di message
passing. Entrambi aumentano capacità, costo e memoria.

## 8. Gauge energetico, forze e torque PaiNN

Il modello sottrae, per specie, l'energia dello stesso sito isolato propagato
solo attraverso gli update block:

$$
u_i=s_E[\tilde u_i-u_{iso}(z_i)],\qquad U_{ML}=\sum_i u_i.
$$

Questo fissa il gauge di un training basato sulle forze. Nel trainer
$s_E=$ `energy_scale` viene posto al Force RMS del training; scala insieme
energie e loro derivate.

Le forze di edge sono calcolate con autograd dalla stessa energia:

$$
\mathbf f_{ij}=-\frac{\partial U_{ML}}{\partial\mathbf r_{ij}}.
$$

Le forze sui corpi si ottengono sommando gli edge con segno opposto sui due
estremi; i torque sono $\sum_s(\mathbf r_s-\mathbf R_m)\times\mathbf f_s$.
Durante il training `create_graph=true` perché ottimizzare una loss sulle forze
richiede derivate seconde dell'energia rispetto a coordinate e pesi.

## 9. Loss e ottimizzazione

Le scale, calcolate soltanto sul training split, sono

$$
F_{RMS}=\sqrt{N_F^{-1}\sum F_{target}^2},\qquad
\tau_{RMS}=\sqrt{N_\tau^{-1}\sum\tau_{target}^2}.
$$

Il torque include soltanto corpi multi-site. La loss è

$$
L=\frac{\operatorname{MSE}(\mathbf F,\mathbf F^*)}{F_{RMS}^2}
+\lambda_\tau
\frac{\operatorname{MSE}(\boldsymbol\tau,\boldsymbol\tau^*)}{\tau_{RMS}^2}
+\lambda_L\frac{\langle\|\mathbf f_s\|^2\rangle}{F_{RMS}^2}.
$$

`torque_weight` è $\lambda_\tau$. `lipschitz_lambda` è $\lambda_L$, ma il
termine implementato è una penalità sulla norma delle forze di sito, non una
stima rigorosa della costante di Lipschitz globale.

L'ottimizzatore è AdamW. `learning_rate` regola il passo; `weight_decay` la
penalità sui pesi; `grad_clip_norm` limita la norma globale del gradiente;
`batch_size` scambia rumore statistico con memoria; `epochs` è un limite
superiore. `reduce_lr_patience` riduce il learning rate dopo plateau;
`early_stopping_patience` termina dopo mancati miglioramenti. Split e
normalizzazione devono restare train-only per evitare leakage.

## 10. Dinamica e termostato

La Hamiltoniana meccanica è

$$
H=K_{trans}+K_{rot}+U_{prior}+U_{ML}.
$$

ESPResSo integra traslazioni e rotazioni dei rigid body. In NVT il Langevin
aggiunge attrito e rumore con relazione fluttuazione-dissipazione; in forma
continua traslazionale,

$$
m\dot{\mathbf v}=\mathbf F-\gamma\mathbf v+
\sqrt{2\gamma k_BT}\,\boldsymbol\eta(t),
$$

con analogo termine rotazionale. In NVE il termostato è disattivato.
`dt` deve risolvere la frequenza più alta del potenziale; aumentarlo accelera il
run ma aumenta l'errore di discretizzazione e può destabilizzare Morse, WCA,
FENE o angoli rigidi. `kT` imposta la distribuzione termica, non la stiffness
dei prior. Force cap e steepest descent sono strumenti di warm-up e non devono
restare nell'Hamiltoniana produttiva.

`device` e `ml_precision` cambiano il rumore numerico, non le equazioni.
CPU/float64 è una closure diagnostica; CPU/float32 è il riferimento ordinario
per la certificazione. Su MPS, `MLCG_MPS_EMPTY_CACHE_INTERVAL=100` è la policy
runtime predefinita di rilascio cache; `0` la disabilita. Il parametro training
`mps_empty_cache_every_batches` è separato e non modifica la loss.

## 11. Certificazione NVE

Per ogni timestep, dalla serie $E_n$ si calcola

$$
\sigma_E=\sqrt{N^{-1}\sum_n(E_n-\bar E)^2}.
$$

Su run di uguale durata fisica si adatta

$$
\sigma_E=C\Delta t^p,\qquad
\log\sigma_E=\log C+p\log\Delta t.
$$

Velocity Verlet è di ordine due: nel regime asintotico conservativo ci si
aspetta $p\simeq2$. Il fit riporta $R^2$ in spazio log-log. Il drift usa le
medie del primo e ultimo 20%:

$$
D_{rel}=\frac{|\bar E_{last}-\bar E_{first}|}
{\max(|E_0|,\langle|E|\rangle,1)}.
$$

I default del certificatore sono $1.7\le p\le2.3$, $R^2\ge0.97$ e
$D_{rel}\le10^{-4}$. Un $p$ basso ai timestep minimi può indicare floor FP32;
un punto grande fuori fit può essere fuori dal regime asintotico; un drift
alto è un criterio distinto dallo scaling.

## 12. Riferimento rapido dei parametri

### Topologia e prior

| Parametro | Significato matematico | Se aumenta |
|---|---|---|
| `temperature` | $T$ usata in $\beta=(RT)^{-1}$ | prior auto/DBI più morbidi a varianza fissa |
| `k`, `D`, `epsilon` | scale energetiche | forze e curvature maggiori |
| `r0`, `theta0`, `phi0` | posizione/fase del minimo | trasla il riferimento geometrico |
| Morse `a` | inverso della larghezza | curvatura locale $2Da^2$ maggiore |
| `r_switch` | inizio smooth switch | più vicino a `r_cut`: coda più concentrata |
| `r_cut` | fine esatta dell'interazione | più coppie e costo maggiore |
| `wca_quantile_percent` | quantile basso per il core | onset tendenzialmente più esterno |
| `wca_guard_fraction` | $r_{guard}/r_c$ | calibrazione più vicina al cutoff |
| `wca_guard_kbt` | $U(r_{guard})/k_BT$ | core più alto/rigido |
| `wca_physical_guard_margin` | margine sul minimo fisico | core ammesso più vicino ai dati |

### PaiNN e training

| Parametro | Ruolo | Trade-off principale |
|---|---|---|
| `num_species` | cardinalità dei tipi | deve coprire esattamente i site type |
| `hidden_channels` | $D$ | capacità contro memoria/costo |
| `n_layers` | blocchi message/update | contesto/capacità contro costo |
| `num_rbf` | risoluzione radiale | dettaglio contro costo |
| `cutoff` | $r_c$ del grafo | contesto contro numero di edge |
| `toxvaerd_alpha` | forma del cutoff | transizione più larga se maggiore |
| `torque_weight` | $\lambda_\tau$ | accuratezza rotazionale contro forza COM |
| `lipschitz_lambda` | $\lambda_L$ | forze di sito più piccole, possibile underfit |
| `learning_rate` | passo AdamW | velocità contro instabilità |
| `weight_decay` | regolarizzazione pesi | controllo capacità contro underfit |
| `grad_clip_norm` | soglia norma gradiente | stabilità contro update attenuati |
| `batch_size` | frame per update | varianza contro memoria |
| `validation_fraction` | quota validation | stima più stabile contro meno training data |

### Runtime e certificazione

| Parametro | Significato | Effetto principale |
|---|---|---|
| `dt` | $\Delta t$ | costo $\propto1/dt$ a durata fissa; errore VV $O(dt^2)$ |
| `steps` | numero di passi | durata $=steps\,dt$ |
| `kT` | temperatura energetica NVT | ampiezza delle fluttuazioni termiche |
| `log_interval` | passo di campionamento | volume I/O e risoluzione temporale |
| `ml_precision` | precisione PaiNN | floor numerico e costo |
| `neighbor_search` | algoritmo di coppia | prestazioni; non deve cambiare la fisica |
| `duration_ps` | durata NVE per dt | stabilità statistica del fit |
| `slope_min/max` | intervallo ammesso per $p$ | severità del gate sull'ordine |
| `min_r2` | linearità log-log minima | severità del power-law gate |
| `max_relative_drift` | drift relativo massimo | severità del gate secolare |

## 13. Regole di interpretazione

1. Un prior più rigido riduce il residuo che PaiNN deve apprendere, ma aumenta
   le frequenze dell'Hamiltoniana e può peggiorare stabilità e floor FP32.
2. Una validation loss bassa non certifica conservazione energetica: servono
   parity energia-forza e sweep NVE multi-$dt$.
3. $p\simeq2$ certifica l'ordine numerico nel dominio testato, non l'accuratezza
   scientifica rispetto alla distribuzione atomistica.
4. Un modello più grande non è automaticamente più fisico: capacità, dataset,
   decomposizione prior/residuo e timestep vanno valutati insieme.
5. Manifest e hash fanno parte della definizione matematica del candidato:
   impediscono di combinare modello, prior, dataset e checkpoint incompatibili.

## 14. Sorgenti normativi

Le formule qui riportate seguono l'implementazione corrente in:

- `preprocessing/geometry_utils.py`, `build_cg_dataset.py`, `prior_kernels.py`,
  `conservative_spline.py`;
- `ibi/ibi_core.py`, `ibi/build_dbi_priors.py`;
- `training/PaiNN_Architecture.hpp`, `training/train_painn.cpp`;
- `simulation/run_cg_md.py`, `equilibrate.py`, `nve_analysis.py`,
  `certify_nve.py`;
- `simulation/espresso_plugin/` per il bridge ESPResSo.

In caso di divergenza, test e sorgenti della revisione effettivamente costruita
sono autoritativi; questo documento deve essere aggiornato insieme a ogni
modifica di equazioni, convenzioni o parametri.

---

# Parte II — Derivazioni implementative

## 15. PaiNN: formulazione completa della variante implementata

La variante ammessa da trainer e runtime è
`painn_canonical_context_silu_v2`. Questa sezione descrive quella variante,
non una generica famiglia PaiNN. Indichiamo con $N$ il numero di siti fisici,
$D=$ `hidden_channels`, $K=$ `num_rbf` e $L=$ `n_layers`.

### 15.1 Rappresentazioni e simmetrie

Per ogni sito $i$ il layer $\ell$ mantiene:

$$
\mathbf s_i^{(\ell)}\in\mathbb R^D,
\qquad
\mathbf v_i^{(\ell)}\in\mathbb R^{3\times D}.
$$

La componente $s_{ic}$ è uno scalare invariante; la colonna
$\mathbf v_{ic}\in\mathbb R^3$ è un vettore equivarante. Per una rotazione
ortogonale $R$ e una traslazione $\mathbf a$:

$$
\mathbf r_i\mapsto R\mathbf r_i+\mathbf a,\qquad
\mathbf s_i\mapsto\mathbf s_i,\qquad
\mathbf v_{ic}\mapsto R\mathbf v_{ic}.
$$

L'energia finale deve essere invariante. Il modello ottiene:

- invarianza traslazionale usando solo spostamenti relativi;
- equivarianza rotazionale usando vettori, norme, prodotti scalare e
  direzioni radiali;
- invarianza rispetto all'ordine dei vicini mediante aggregazione per somma;
- invarianza rispetto alla permutazione di siti dello stesso tipo mediante
  embedding condivisi e readout additivo.

Le feature iniziali sono

$$
\mathbf s_i^{(0)}=E_{z_i},\qquad
\mathbf v_i^{(0)}=0,
$$

dove $E\in\mathbb R^{N_{species}\times D}$ è la matrice di embedding e $z_i$
è il site type. `num_species` deve quindi essere maggiore del massimo indice di
tipo e deve coincidere tra dataset, config, manifest e runtime.

### 15.2 Grafo periodico diretto

Per ogni coppia fisica entro il cutoff vengono costruiti i due edge $i\leftarrow
j$ e $j\leftarrow i$. Con `row=i`, `col=j`:

$$
\mathbf r_{ij}=\operatorname{MIC}(\mathbf r_i-\mathbf r_j),\qquad
d_{ij}=\sqrt{\mathbf r_{ij}\cdot\mathbf r_{ij}+\varepsilon_d},\qquad
\widehat{\mathbf r}_{ij}=\frac{\mathbf r_{ij}}{d_{ij}},
$$

con $\varepsilon_d=10^{-8}$ nella norma implementata. Questo termine evita
divisioni esattamente per zero; a distanze fisiche ordinarie è trascurabile.
Gli edge di training sono precomputati per ogni frame; al runtime la neighbor
list ESPResSo deve produrre lo stesso insieme fisico entro `cutoff`.

La MIC contiene un `round` non differenziabile ai bordi dell'immagine. Il
trainer applica `detach` alla scelta dell'immagine: all'interno di ciascuna
regione MIC la derivata è esatta, mentre il salto discreto non entra nel grafo
autograd. Questa è la convenzione corretta purché nessuna coppia rilevante sia
ambigua sul mezzo box.

### 15.3 Base radiale gaussiana

I centri sono uniformi:

$$
\mu_k=\frac{k}{K-1}r_c,\qquad k=0,\ldots,K-1,
$$

e il codice usa

$$
\sigma_{RBF}=\frac{r_c}{K},\qquad
g_k(d)=\exp\left[-\frac{(d-\mu_k)^2}{\sigma_{RBF}^2}\right].
$$

La derivata non finestrata è

$$
\frac{dg_k}{dd}=-\frac{2(d-\mu_k)}{\sigma_{RBF}^2}g_k(d).
$$

Aumentare $K$ riduce $\sigma_{RBF}$ e consente variazioni radiali più fini, ma
aumenta parametri e moltiplicazioni. Un $K$ elevato non crea informazione se la
distribuzione delle distanze o il numero di frame non la supportano.

### 15.4 Cutoff Toxvaerd e regolarità

Per $d\le r_c$:

$$
x=1-\frac d{r_c},\qquad
c(d)=\frac{x^4}{x^4+\alpha^4},
$$

mentre $c(d)=0$ per $d>r_c$. La base effettiva è
$R_k(d)=g_k(d)c(d)$. Le derivate utili sono

$$
\frac{dc}{dx}=\frac{4\alpha^4x^3}{(x^4+\alpha^4)^2},
\qquad
\frac{dc}{dd}=-\frac{4\alpha^4x^3}
{r_c(x^4+\alpha^4)^2},
$$

$$
\frac{dR_k}{dd}=c\frac{dg_k}{dd}+g_k\frac{dc}{dd}.
$$

Poiché vicino al cutoff $c\sim x^4/\alpha^4$, valore e prime tre derivate
rispetto alla distanza vanno a zero dal lato interno. Ciò evita un impulso di
forza quando un edge attraversa $r_c$. `toxvaerd_alpha` non è un cutoff
aggiuntivo: controlla la larghezza della regione attenuata. A $x=\alpha$ vale
$c=1/2$; quindi un $\alpha$ maggiore sposta il mezzo cutoff più all'interno.

### 15.5 Message block con indici espliciti

Per ogni edge $j\to i$, una MLP sul contesto del mittente produce

$$
\mathbf h_j=W_2\operatorname{SiLU}(W_1\mathbf s_j+\mathbf b_1)+\mathbf b_2
\in\mathbb R^{3D},
$$

mentre una trasformazione lineare senza bias della base radiale produce

$$
\mathbf w_{ij}=W_R\mathbf R(d_{ij})\in\mathbb R^{3D}.
$$

Il prodotto canale per canale
$\mathbf q_{ij}=\mathbf h_j\odot\mathbf w_{ij}$ viene diviso in tre vettori
$\mathbf q^{s},\mathbf q^{v},\mathbf q^{r}\in\mathbb R^D$. Per canale $c$:

$$
\mathbf m^{s}_{ij,c}=q^{s}_{ij,c},
$$

$$
\mathbf m^{v}_{ij,c}=
q^{v}_{ij,c}\mathbf v_{j,c}+
q^{r}_{ij,c}\widehat{\mathbf r}_{ij}.
$$

Il destinatario riceve la somma

$$
\Delta s_{i,c}^{msg}=\sum_{j\in\mathcal N(i)}m^{s}_{ij,c},
\qquad
\Delta\mathbf v_{i,c}^{msg}=
\sum_{j\in\mathcal N(i)}\mathbf m^{v}_{ij,c},
$$

seguita dal residuo

$$
\mathbf s_i\leftarrow\mathbf s_i+\Delta\mathbf s_i^{msg},\qquad
\mathbf v_i\leftarrow\mathbf v_i+\Delta\mathbf v_i^{msg}.
$$

$q^s$ è scalare; $q^v$ moltiplica un vettore equivarante; $q^r$ moltiplica la
direzione radiale equivarante. Ogni termine ha quindi la trasformazione
corretta. La somma, invece della media, rende l'ampiezza sensibile al numero di
vicini: questo è intenzionale e il manifest registra `sum_v1`.

### 15.6 Update block intra-sito

Due mappe lineari agiscono soltanto sui canali, identicamente per le tre
componenti spaziali:

$$
\mathbf v^v_{i,c}=\sum_d(W_v)_{cd}\mathbf v_{i,d},
\qquad
\mathbf v^u_{i,c}=\sum_d(W_u)_{cd}\mathbf v_{i,d}.
$$

Si costruiscono gli invarianti

$$
n_{i,c}=\sqrt{\mathbf v^v_{i,c}\cdot\mathbf v^v_{i,c}+10^{-8}},
\qquad
p_{i,c}=\mathbf v^v_{i,c}\cdot\mathbf v^u_{i,c}.
$$

Una MLP `2D -> D -> 3D` con SiLU riceve la concatenazione
$[\mathbf s_i,\mathbf n_i]$ e produce tre blocchi
$\mathbf a_i,\mathbf b_i,\mathbf c_i\in\mathbb R^D$. L'update è

$$
\Delta s_{i,c}^{upd}=a_{i,c}+b_{i,c}p_{i,c},
\qquad
\Delta\mathbf v_{i,c}^{upd}=c_{i,c}\mathbf v^u_{i,c},
$$

$$
\mathbf s_i\leftarrow\mathbf s_i+\Delta\mathbf s_i^{upd},\qquad
\mathbf v_i\leftarrow\mathbf v_i+\Delta\mathbf v_i^{upd}.
$$

Norme e prodotti scalari sono invarianti; il coefficiente $c_{i,c}$ è
invariante e moltiplica un vettore equivarante. L'update non richiede nuovi
edge e mescola l'informazione vettoriale accumulata nel sito.

### 15.7 Readout additivo e gauge per specie

Dopo $L$ blocchi, il readout

$$
\widetilde u_i=W_o^{(2)}
\operatorname{SiLU}(W_o^{(1)}\mathbf s_i+\mathbf b_o^{(1)})+b_o^{(2)}
$$

produce uno scalare per sito. Un training sulle sole derivate non determina le
costanti additive dell'energia. Il framework calcola quindi, per ciascuna
specie $z$, un riferimento isolato: embedding del tipo, vettori nulli, nessun
message block e soli update block. Se tale readout è $u_{iso}(z)$,

$$
u_i=s_E[\widetilde u_i-u_{iso}(z_i)],
\qquad
U_{ML}=\sum_i u_i.
$$

Un sito isolato ha così energia ML nulla per costruzione. Il buffer $s_E$ è
inizializzato dal trainer a $F_{RMS}$ del training set. Il fattore scala energia
e forze simultaneamente e rende più naturale l'ordine di grandezza dei pesi
iniziali; non è un parametro da cambiare a posteriori nel runtime.

### 15.8 Forze da edge e conservatività

Il forward di training usa gli spostamenti MIC come variabili foglia. Per ogni
edge diretto:

$$
\mathbf g_{ij}=\frac{\partial U_{ML}}{\partial\mathbf r_{ij}},
\qquad
\mathbf f_{ij}=-\mathbf g_{ij}.
$$

Il contributo viene aggregato con segni opposti:

$$
\mathbf f_i\mathrel{+}=\mathbf f_{ij},
\qquad
\mathbf f_j\mathrel{-}=\mathbf f_{ij}.
$$

Poiché entrambi gli edge diretti fanno parte del forward, non si inserisce un
fattore $1/2$ manuale: autograd deriva esattamente l'energia realmente
calcolata. L'antisimettria dell'aggregazione garantisce forza totale interna
nulla a precisione numerica, mentre l'energia scalare garantisce
conservatività all'interno della regione MIC e della neighbor list.

### 15.9 Costo e significato dei parametri PaiNN

Indicando con $E$ il numero di edge diretti, il costo dominante di un blocco è
approssimativamente $O(ED+ND^2+EKD)$, con memoria autograd molto superiore al
solo forward perché il force matching richiede `create_graph=true`.

| Parametro | Intervento matematico | Rischio se troppo grande |
|---|---|---|
| `hidden_channels=D` | numero di canali scalari/vettoriali | memoria, overfitting, costo $D^2$ |
| `n_layers=L` | raggio informativo in hop | grafi profondi e derivate seconde costose |
| `num_rbf=K` | risoluzione di $d_{ij}$ | basi ridondanti rispetto ai dati |
| `cutoff=r_c` | supporto del grafo | crescita del numero di edge e vincoli sul box |
| `toxvaerd_alpha` | posizione/larghezza dell'attenuazione | perdita di contesto se troppo grande |
| `num_species` | righe dell'embedding/gauge | mismatch semantico se non coincide coi tipi |

## 16. Prior: costruzione, derivate e geometria delle forze

### 16.1 Regola generale per un prior di distanza

Sia $\boldsymbol\delta=\operatorname{MIC}(\mathbf r_j-\mathbf r_i)$,
$r=\|\boldsymbol\delta\|$ e
$\widehat{\boldsymbol\delta}=\boldsymbol\delta/r$. Per un'energia $U(r)$:

$$
\nabla_{\mathbf r_i}r=-\widehat{\boldsymbol\delta},
\qquad
\mathbf F_i=\frac{dU}{dr}\widehat{\boldsymbol\delta},
\qquad
\mathbf F_j=-\mathbf F_i.
$$

Se si definisce la forza radiale $F_r=-dU/dr$ orientata da $j$ verso $i$,
la stessa relazione si scrive
$\mathbf F_i=-F_r\widehat{\boldsymbol\delta}$. Dichiarare la convenzione è
essenziale quando la terza colonna di una tabella è chiamata genericamente
`force`.

### 16.2 Stima armonica automatica

Vicino a un minimo, $U(q)\simeq U(q_0)+k(q-q_0)^2/2$. Se il Jacobiano della
coordinata è trascurato localmente, la distribuzione è gaussiana:

$$
P(q)\propto\exp[-\beta k(q-q_0)^2/2],
\qquad
\operatorname{Var}(q)=\frac1{\beta k}.
$$

Da qui derivano

$$
q_0=\langle q\rangle,
\qquad
k=\frac1{\beta\operatorname{Var}(q)}.
$$

Questa stima `auto` è un'approssimazione armonica locale, distinta dalla DBI
Jacobiano-corretta. Elementi con lo stesso `name` possono condividere le
statistiche, aumentando il campione ma imponendo che siano fisicamente
equivalenti. Varianza quasi nulla implica $k$ enorme: occorre un floor o una
scelta manuale, non un timestep arbitrariamente piccolo come rimedio.

### 16.3 Harmonic e FENE

Per il bond armonico $U''(r)=k$ ovunque. In prima approssimazione la frequenza
relativa più rapida è $\omega\sim\sqrt{k/\mu}$, con massa ridotta $\mu$;
stabilità e accuratezza richiedono $\omega\Delta t\ll1$.

Per FENE, posto $x=r-r_0$ e $R=r_{max}$:

$$
U'(r)=\frac{kx}{1-x^2/R^2},
$$

$$
U''(r)=k\frac{1+x^2/R^2}{(1-x^2/R^2)^2}.
$$

La curvatura è $k$ al minimo e diverge più rapidamente della forza al bordo.
Un frame con $|x|\ge R$ non è semplicemente improbabile: è fuori dal dominio
matematico del prior.

### 16.4 Morse: forza, curvatura e switch

Con $y=e^{-a(r-r_0)}$:

$$
U_0'=2D\,a\,y(1-y),\qquad
U_0''=2Da^2y(2y-1),\qquad
F_0=-U_0'=2D\,a\,y(y-1).
$$

La scala di decadimento è $a^{-1}$ e la stiffness al minimo è $2Da^2$.
Per lo switch quintico di larghezza $w=r_c-r_s$:

$$
S'=-\frac{30t^2(1-t)^2}{w},
\qquad
S''=-\frac{60t(1-t)(1-2t)}{w^2}.
$$

La curvatura switched è

$$
U''=S''U_0+2S'U_0'+SU_0''.
$$

Ai due bordi $S'$ e $S''$ sono nulli; al cutoff anche $S=0$. Energia, forza e
curvatura si raccordano quindi senza salto. Tuttavia i termini $1/w$ e $1/w^2$
mostrano perché una finestra troppo stretta può diventare numericamente rigida.

I contatti pair-specific selezionano endpoint espliciti COM/site e usano marker
tecnici coincidenti al runtime. I type-pair si applicano invece a tutte le
coppie di tipi fisici ammesse dalle exclusions. Se entrambi selezionano la
stessa coppia, le energie si sommano: non esiste deduplicazione fisica
automatica.

### 16.5 WCA e calibrazione automatica completa

Posto $s_6=(\sigma/r)^6$, per $r<r_c$:

$$
U=4\epsilon(s_6^2-s_6)+\epsilon,
$$

$$
U'=\frac{24\epsilon}{r}(s_6-2s_6^2),
\qquad
U''=\frac{24\epsilon}{r^2}(26s_6^2-7s_6).
$$

Il fit automatico non assegna semplicemente un `sigma` per tipo. Per ogni
coppia $(a,b)$ raccoglie le distanze fisiche nonbonded e calcola un quantile
basso $q_{ab}$ e il minimo streaming esatto $r^{min}_{ab}$. Introduce raggi di
tipo $R_a$ e minimizza

$$
\mathcal J(\{R\})=\sum_{ab}w_{ab}(R_a+R_b-q_{ab})^2,
\qquad
w_{ab}=\frac{N_{ab}}{N_{ab}+N_0},
$$

con $N_0=1000$ nell'implementazione e limiti numerici sui raggi. La
regolarizzazione gerarchica fa sì che coppie poco campionate si appoggino alla
somma dei raggi di tipo. Definendo

$$
\lambda_{ab}=\frac{N_{ab}}{N_{ab}+N_0},
$$

il cutoff preliminare è

$$
r_{c,ab}^{(0)}=\lambda_{ab}q_{ab}+
(1-\lambda_{ab})(R_a+R_b),
$$

poi viene limitato da

$$
r_{c,ab}=\min\left[
r_{c,ab}^{(0)},q_{ab},
\frac{m\,r^{min}_{ab}}{g}
\right].
$$

Qui $g=$ `wca_guard_fraction` e
$m=$ `wca_physical_guard_margin`. Si pone

$$
\sigma_{ab}=\frac{r_{c,ab}}{2^{1/6}},
\qquad r_{guard}=g r_{c,ab}.
$$

Se $z=(\sigma/r_{guard})^6$, la calibrazione energetica
$U(r_{guard})=Gk_BT$, con $G=$ `wca_guard_kbt`, dà

$$
\epsilon_{ab}=\frac{Gk_BT}{4(z^2-z)+1}.
$$

Il minimo stimato da istogramma resta diagnostico; il guard di supporto usa il
minimo streaming esatto, perché un istogramma può nascondere l'osservazione più
corta. Percentuale sotto $r_c$, percentuale sotto $r_{guard}$ e rapporto
$r_{guard}/r_{min}$ misurano quanto il core invade la distribuzione fisica.

### 16.6 Angoli: gradiente cartesiano

Definiamo

$$
\mathbf a=\operatorname{MIC}(\mathbf r_i-\mathbf r_j),\quad
\mathbf b=\operatorname{MIC}(\mathbf r_k-\mathbf r_j),\quad
A=\|\mathbf a\|,\quad B=\|\mathbf b\|,
$$

$$
c=\cos\theta=\frac{\mathbf a\cdot\mathbf b}{AB}.
$$

I gradienti del coseno sono

$$
\nabla_i c=\frac{\mathbf b}{AB}-c\frac{\mathbf a}{A^2},
\qquad
\nabla_k c=\frac{\mathbf a}{AB}-c\frac{\mathbf b}{B^2}.
$$

Poiché $d\theta/dc=-1/\sin\theta$, per
$G_\theta=dU/d\theta$:

$$
\mathbf F_i=\frac{G_\theta}{\sin\theta}\nabla_i c,
\quad
\mathbf F_k=\frac{G_\theta}{\sin\theta}\nabla_k c,
\quad
\mathbf F_j=-(\mathbf F_i+\mathbf F_k).
$$

Il clamp numerico di $c$ in $[-1,1]$ corregge roundoff, ma non elimina la
singolarità fisico-geometrica a $\sin\theta=0$.

### 16.7 Dihedri: periodicità e geometria

Il dihedro è costruito da tre vettori MIC e dai normali ai due piani. La fase
viene portata in $[0,2\pi)$. Per una rappresentazione conservativa, definite
le derivate geometriche $\nabla_x\phi$, le forze sono sempre

$$
\mathbf F_x=-\frac{dU}{d\phi}\nabla_x\phi,
\qquad x\in\{i,j,k,l\}.
$$

Con $\mathbf v_{12},\mathbf v_{23},\mathbf v_{34}$, normali unitari
$\mathbf n_{12},\mathbf n_{23}$ e norme non normalizzate $l_{12},l_{23}$:

$$
\nabla_i\phi=-\frac{\|\mathbf v_{23}\|}{l_{12}}\mathbf n_{12},
\qquad
\nabla_l\phi=\frac{\|\mathbf v_{23}\|}{l_{23}}\mathbf n_{23},
$$

$$
A=\frac{\mathbf v_{12}\cdot\mathbf v_{23}}{\|\mathbf v_{23}\|^2},
\qquad
C=\frac{\mathbf v_{34}\cdot\mathbf v_{23}}{\|\mathbf v_{23}\|^2},
$$

$$
\nabla_j\phi=-(1+A)\nabla_i\phi+C\nabla_l\phi,
\qquad
\nabla_k\phi=A\nabla_i\phi-(1+C)\nabla_l\phi.
$$

Normali quasi nulli rendono il dihedro indefinito. Nel percorso cosine legacy,
il preprocessing valuta la forza mediante differenza centrale cartesiana con
passo $10^{-6}$ nm e rimuove la piccola forza media residua da roundoff; una
nuova topologia con dihedri richiede comunque parity esplicita col runtime.

### 16.8 Exclusions topologiche e parità preprocessing/runtime

La policy corrente distingue connettività e forma energetica:

- le coppie di siti interne allo stesso rigid body sono escluse dal nonbonded;
- per una relazione 1–2, solo le site-pair esplicitamente bonded sono escluse;
- un bond COM–COM non elimina automaticamente tutte le coppie fra virtual site;
- gli endpoint 1–3 di un angle esplicitamente esclusivo applicano la policy
  all-sites prevista;
- un Morse ha `exclude_wca=false` per default perché non implica connettività
  covalente.

La stessa maschera deve essere applicata nella sottrazione dei prior e nel
runtime. Una singola coppia presente soltanto in uno dei due percorsi altera
sia il target residuale sia l'Hamiltoniana ricostruita.

## 17. Direct Boltzmann inversion in dettaglio

### 17.1 Dalla distribuzione canonica al potenziale

Per una coordinata interna $q$, la probabilità osservata contiene la misura
geometrica $J(q)$:

$$
P(q)=Z_q^{-1}J(q)e^{-\beta U(q)}.
$$

Quindi

$$
U(q)=-k_BT\ln\frac{P(q)}{J(q)}+C.
$$

Le misure usate sono

$$
J(r)=r^2,\qquad J(\theta)=\sin\theta,\qquad J(\phi)=1.
$$

Dividere per il Jacobiano evita di interpretare il volume di fase come forza
effettiva. Vicino a $r=0$ e agli estremi angolari il Jacobiano è regolarizzato
numericamente, ma regioni prive di supporto non diventano per questo dati
affidabili.

### 17.2 Istogramma e densità

Dato un insieme $q_n$, l'istogramma produce conteggi $C_b$ e una densità
$P_b$ sui centri $q_b$. La densità viene normalizzata numericamente:

$$
\sum_b P_b\Delta q\simeq1
$$

o con la corrispondente quadratura sulla griglia. Il logaritmo usa un floor
positivo per evitare $\log0$, ma il floor non deve generare code fisiche: per
questo il supporto è trattato separatamente.

### 17.3 Supporto statistico

Una bin è affidabile soltanto se soddisfa i criteri configurati su conteggio e
densità. Piccoli gap interni possono essere riempiti fino a un numero massimo
di bin; code esterne senza campioni restano non supportate. Se il supporto ha
meno di `min_support_points`, il gruppo non contiene informazione sufficiente
per un update stabile.

Definita la maschera $M_b$, la PMF grezza è

$$
U_b^{raw}=-k_BT\ln P_{corr,b},\qquad b\in M,
$$

e viene traslata imponendo $\min_{b\in M}U_b=0$. La costante non cambia le
forze ma mantiene confrontabili e numericamente ben scalate le tabelle.

### 17.4 Smoothing nel dominio supportato

Il profilo supportato è filtrato con una gaussiana discreta di ampiezza
`histogram_smoothing_sigma` in unità di bin. Per un kernel normalizzato $G$:

$$
\widetilde U_b=\sum_{b'}G_{b-b'}U_{b'}.
$$

Per coordinate periodiche il filtro usa wrapping; per bond/angle usa condizioni
non periodiche. Lo smoothing riduce la varianza della derivata, ma introduce
bias e può cancellare barriere strette: va scelto rispetto alla risoluzione
della griglia e al campionamento effettivo.

### 17.5 Interpolazione ed estrapolazione delle code

Nel supporto non periodico il profilo è interpolato con PCHIP, che evita molte
oscillazioni spurie dei cubic spline ordinari. Ai bordi, pendenza e valore sono
stimati su una finestra configurata; fuori supporto vengono costruite code
confining coerenti con tali condizioni. L'obiettivo non è indovinare la PMF
dove non esistono dati, ma impedire che il simulatore entri liberamente in una
regione non campionata.

Per un dihedro si usa una spline cubica periodica e si impone

$$
U(0)=U(2\pi),\qquad U'(0)=U'(2\pi).
$$

### 17.6 Pareti angolari

Per proteggere le singolarità a $0$ e $\pi$, il generatore può aggiungere
pareti quadratiche di larghezza $w$ e costante $k_w$. A sinistra:

$$
U_{wall}=\frac12k_w(w-\theta)^2,\qquad
\frac{dU_{wall}}{d\theta}=-k_w(w-\theta),\quad\theta<w.
$$

A destra, con $\theta_r=\pi-w$:

$$
U_{wall}=\frac12k_w(\theta-\theta_r)^2,\qquad
\frac{dU_{wall}}{d\theta}=k_w(\theta-\theta_r),\quad\theta>\theta_r.
$$

`wall_width` decide quanto dominio viene regolarizzato; `wall_k` decide la
rigidità e quindi può limitare il timestep.

### 17.7 Gruppi, pooling e modalità DBI/IBI

Le coordinate non vengono necessariamente invertite una per una. Gli elementi
con lo stesso gruppo logico possono condividere i campioni:

$$
\mathcal Q_g=\bigcup_{a\in g}\{q_{a,n}\}.
$$

Il pooling riduce il rumore solo se gli elementi sono realmente equivalenti;
in caso contrario produce una PMF media che non rappresenta nessuno di essi.
La modalità iniziale DBI costruisce $U_0$ dalla distribuzione AA. La modalità
IBI conserva invece lo stato $U_i$, confronta una traiettoria CG col target e
genera $U_{i+1}$. Prior fissi non appartenenti al gruppo vengono copiati senza
reinterpretazione.

Le griglie devono essere strettamente crescenti e uniformi. I bond usano un
dominio finito configurato o derivato dai dati; gli angle coprono esattamente
$[0,\pi]$ e i dihedral $[0,2\pi]$. Cambiare griglia fra iterazioni richiede una
interpolazione esplicita e rende meno diretto il confronto di $\Delta U$.

## 18. Iterative Boltzmann inversion in dettaglio

### 18.1 Segno dell'aggiornamento

Se $U_i$ produce $P_i$ e il target è $P_*$, l'update implementato è

$$
\Delta U_i(q)=\alpha k_BT\ln\frac{P_i(q)}{P_*(q)},
\qquad
U_{i+1}=U_i+\Delta U_i.
$$

Se $P_i>P_*$, $\Delta U>0$: la regione viene penalizzata. Se $P_i<P_*$,
$\Delta U<0$: viene favorita. `alpha` è un damping, non una temperatura; per
$\alpha=1$ si applica l'update teorico intero, mentre valori minori riducono
oscillazioni dovute a campioni finiti e accoppiamento fra coordinate.

### 18.2 Supporto comune

Target e simulazione hanno maschere $M_*$ e $M_i$. Si aggiorna soltanto

$$
M_{upd}=M_*\cap M_i.
$$

Se l'intersezione è insufficiente, il framework preserva il potenziale
precedente. Aggiornare una bin dove una delle distribuzioni è soltanto il floor
numerico produrrebbe un log-ratio enorme e privo di significato statistico.

### 18.3 Clipping e smoothing dell'update

Il log-ratio è limitato energeticamente:

$$
\Delta U\leftarrow
\operatorname{clip}(\Delta U,-U_{max},U_{max}),
\qquad
U_{max}=\texttt{max_update_kT}\,k_BT.
$$

Poi viene applicato un filtro con ampiezza `update_smoothing_sigma`. Clipping e
smoothing hanno ruoli diversi: il primo limita outlier, il secondo limita
curvatura/rumore ad alta frequenza.

### 18.4 Interpolazione, taper e code

Per variabili non periodiche l'update supportato viene interpolato con PCHIP.
Una finestra coseno lo porta gradualmente a zero presso i bordi su
`taper_bins`, così le code sicure del potenziale precedente non vengono
distrutte da dati marginali. Successivamente le sole regioni target non
supportate vengono ricostruite con la policy di extrapolazione.

Per variabili periodiche l'update viene interpolato sull'intero periodo con
wrapping e si reimpone l'uguaglianza degli endpoint. Dopo l'update si sottrae
sempre il minimo energetico, operazione di gauge che non cambia le forze.

### 18.5 Dalla nuova energia alla forza

L'oggetto fondamentale dell'IBI è $U_{i+1}$, non una forza aggiornata
indipendentemente. Il generatore deriva il profilo interpolato e lo converte
nella convenzione richiesta dal target:

| coordinata | terza colonna legacy |
|---|---|
| bond | $-dU/dr$ |
| angle | $+dU/d\theta$ |
| dihedral | fattore ESPResSo $-(dU/d\phi)/\sin\phi$ fuori dai punti singolari |

Il fattore dihedro viene esteso ai nodi con $|\sin\phi|$ piccolo usando il
valore regolare più vicino e poi limitato da `force_max`. Questa è una
convenzione del bonded tabulato ESPResSo, non una nuova legge fisica.

### 18.6 Convergenza e identifiabilità

IBI cerca di riprodurre distribuzioni marginali, non determina in generale
un'unica Hamiltoniana many-body. Coordinate bonded accoppiate, prior nonbonded
e PaiNN possono compensarsi. Una convergenza credibile richiede almeno:

1. sovrapposizione sufficiente fra supporti;
2. riduzione stabile della distanza fra istogrammi su più iterazioni;
3. update $\Delta U$ che diminuiscono senza saturare il clipping;
4. assenza di code visitate artificialmente;
5. stabilità dinamica e parity energia-forza;
6. ricostruzione del dataset residuale prima del retraining PaiNN.

Più iterazioni non correggono un sampling insufficiente. Se la simulazione non
visita una regione target, occorre migliorare overlap, inizializzazione o
potenziale, non aumentare soltanto `max_update_kT`.

### 18.7 Parametri IBI e loro effetto

| Parametro | Significato | Troppo piccolo | Troppo grande |
|---|---|---|---|
| `kT` | scala energetica dell'inversione | update deboli | update/barriere forti |
| `alpha` | damping dell'update | convergenza lenta | oscillazioni |
| `histogram_smoothing_sigma` | smoothing DBI | derivate rumorose | dettagli cancellati |
| `update_smoothing_sigma` | smoothing di $\Delta U$ | update ruvido | correzione troppo diffusa |
| `max_update_kT` | clipping in unità $k_BT$ | correzione saturata | sensibilità agli outlier |
| `min_support_points` | supporto minimo | profili fragili | gruppi validi respinti |
| `taper_bins` | raccordo a zero | kink ai bordi | supporto utile ridotto |
| `force_max` | limite tabella | prior troncato | instabilità numerica |
| `wall_width`, `wall_k` | pareti angolari | singolarità accessibili | angolo troppo rigido |

### 18.8 Cosa significa “IBI regolarizzata” nel framework

La regolarizzazione implementativa è la composizione di più operatori
trasparenti, non un unico termine astratto:

$$
U_{i+1}=\mathcal E\!\left[
U_i+\mathcal T\!\left(
\mathcal S\!\left(
\mathcal C\!\left[
\alpha k_BT\log\frac{P_i}{P_*}
\right]\right)\right)\right],
$$

dove $\mathcal C$ è il clipping, $\mathcal S$ lo smoothing,
$\mathcal T$ il taper sul supporto e $\mathcal E$ la ricostruzione delle code.
Per gli angle si aggiunge eventualmente l'operatore di parete
$U\mapsto U+U_{wall}$. Ogni operatore ha un parametro e un effetto verificabile.
Non si deve descrivere questo percorso come una penalità di Tikhonov o una
regolarizzazione sulla seconda derivata, a meno che tale termine venga
effettivamente aggiunto al codice.

La promozione “conservative” è un passaggio distinto: sostituisce la coppia di
colonne interpolate indipendentemente con un'unica energia Hermite e la sua
derivata analitica. Regolarità statistica e conservatività numerica risolvono
problemi diversi e vanno verificate entrambe.

## 19. Tabelle legacy e spline conservative

### 19.1 Perché energia e forza interpolate separatamente non bastano

Su una griglia uniforme $q_i$, l'interpolazione lineare di una colonna $T$ è

$$
T(q)=(1-t)T_i+tT_{i+1},\qquad t=\frac{q-q_i}{h}.
$$

Se si interpola $U$ linearmente, $dU/dq=(U_{i+1}-U_i)/h$ è costante
nell'intervallo. Se si interpola anche una colonna forza, essa varia linearmente.
Le due possono coincidere solo in casi speciali. Perciò una tabella legacy può
riprodurre bene la struttura ma violare localmente $F=-dU/dq$.

### 19.2 Polinomio Hermite cubico

La spline conservativa memorizza $U_i$ e $m_i=U'_i$. Con
$t=(q-q_i)/h$:

$$
U=h_{00}U_i+h_{10}hm_i+h_{01}U_{i+1}+h_{11}hm_{i+1},
$$

$$
h_{00}=2t^3-3t^2+1,\quad
h_{10}=t^3-2t^2+t,\quad
h_{01}=-2t^3+3t^2,\quad
h_{11}=t^3-t^2.
$$

La derivata analitica usata per le forze è

$$
U'(q)=\frac{(6t^2-6t)U_i+(-6t^2+6t)U_{i+1}}{h}
+(3t^2-4t+1)m_i+(3t^2-2t)m_{i+1}.
$$

Valore e derivata agli estremi coincidono esattamente coi dati nodali. Non
esiste una seconda interpolazione della forza: la conservatività è strutturale.

### 19.3 Domini

- bond: griglia uniforme; sotto il minimo usa continuazione tangente
  $U=U_0+m_0(q-q_0)$, mentre $r\ge q_{max}$ è fuori dominio;
- angle: coordinata limitata a $[0,\pi]$ solo per roundoff geometrico;
- dihedral: wrapping periodico in $[0,2\pi)$ e uguaglianza di $U,U'$ agli
  endpoint.

La continuazione tangente è conservativa ma non sostituisce una coda fisica
ben costruita. Un bond che visita spesso la continuazione segnala supporto
insufficiente.

### 19.4 Proiezione delle derivate

Una spline di distanza restituisce $U'(r)$ e usa la geometria della sezione
16.1. Una spline angolare restituisce $U'(\theta)$ e usa il fattore
$1/\sin\theta$ della sezione 16.6. Una spline dihedrale restituisce direttamente
$U'(\phi)$ e usa i gradienti geometrici della sezione 16.7. Questa separazione
fra derivata scalare e Jacobiano cartesiano rende testabile ogni livello.

### 19.5 Conversione di tabelle esistenti

Per convertire una tabella legacy si ricostruisce un'energia compatibile con la
convenzione forza mediante integrazione trapezoidale. Per esempio, per un bond:

$$
U_{i+1}=U_i-\frac h2(F_i+F_{i+1}).
$$

Per un angle la terza colonna è $+U'(\theta)$ e il segno cambia di conseguenza.
Per il dihedro legacy, se il fattore è $f_{ESP}$, la derivata scalare è

$$
U'(\phi)=-f_{ESP}(\phi)\sin\phi.
$$

La ricostruzione sceglie un anchor energetico, integra in entrambe le direzioni
e sottrae il minimo. La successiva spline deve essere ricertificata: la
conversione elimina l'incoerenza energia-forza, ma può modificare leggermente
la distribuzione rispetto alla tabella originaria.

## 20. Target, loss e derivate seconde

### 20.1 Dalle forze di sito alle grandezze di corpo

Per il corpo $m$ con siti $s\in m$:

$$
\mathbf F_m^{ML}=\sum_{s\in m}\mathbf f_s^{ML},
$$

$$
\boldsymbol\tau_m^{ML}=\sum_{s\in m}
\operatorname{MIC}(\mathbf r_s-\mathbf R_m)\times\mathbf f_s^{ML}.
$$

Il torque è definito in lab frame rispetto al COM. Traslare tutti i siti dello
stesso vettore non lo cambia; cambiare offset o orientazione del rigid body sì.

### 20.2 Normalizzazione train-only

Con $N_F$ componenti cartesiane di forza e $N_\tau$ componenti dei soli corpi
multi-site:

$$
F_{RMS}^2=\frac1{N_F}\sum_a(F_a^*)^2,
\qquad
\tau_{RMS}^2=\frac1{N_\tau}\sum_b(\tau_b^*)^2.
$$

Le MSE normalizzate sono adimensionali. La validation usa le scale del training,
non le proprie; così non può apparire artificialmente migliore per una
varianza target diversa.

Il baseline nullo è

$$
L_{0,F}^{val}=\frac{\langle|\mathbf F^*|^2\rangle_{val}}{F_{RMS}^2},
\qquad
L_{0,\tau}^{val}=\frac{\langle|\boldsymbol\tau^*|^2\rangle_{val}}{\tau_{RMS}^2}.
$$

Una validation loss vicina a tale valore significa che il modello non
generalizza meglio della forza residuale nulla, anche se la train loss scende.

### 20.3 Penalità sulle forze di sito

Il termine denominato storicamente `lipschitz` è

$$
L_L=\frac1{N_sF_{RMS}^2}\sum_s\|\mathbf f_s^{ML}\|^2.
$$

Esso penalizza forze di sito grandi, incluse coppie che possono cancellarsi
nella forza COM. Non calcola
$\sup_x\|\nabla f(x)\|$ e quindi non è una stima della costante di Lipschitz
globale. Può ridurre curvature apprese, ma un valore eccessivo porta a
underfitting.

### 20.4 Perché il force matching richiede derivate seconde

Se $\theta$ sono i pesi e
$\mathbf F_\theta=-\nabla_{\mathbf r}U_\theta$, allora

$$
\nabla_\theta L_F
=2(\mathbf F_\theta-\mathbf F^*)^T
\frac{\partial\mathbf F_\theta}{\partial\theta}
=-2(\mathbf F_\theta-\mathbf F^*)^T
\frac{\partial^2U_\theta}{\partial\theta\,\partial\mathbf r}.
$$

Per questo autograd deve conservare il grafo della prima derivata
(`create_graph=true`). È anche la ragione per cui memoria e costo di training
sono molto maggiori dell'inferenza energetica.

### 20.5 Gradient clipping e AdamW

Se il gradiente globale ha norma $\|g\|_2$ e la soglia è $G$:

$$
g\leftarrow g\min\left(1,\frac G{\|g\|_2}\right).
$$

La `GradClip_Fraction` è la frazione di batch in cui $\|g\|_2>G$. Un clipping
frequente all'inizio può essere fisiologico; persistente vicino al 100% indica
learning rate, scale o curvature problematiche.

AdamW mantiene momenti esponenziali e applica weight decay disaccoppiato:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}=(1-\eta\lambda_w)\theta_t
-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

`learning_rate` è $\eta$ e `weight_decay` è $\lambda_w$; gli altri parametri
seguono i default LibTorch se non esposti dal config.

## 21. Esempi quantitativi e checklist

### 21.1 Softening Morse TEL22

A $D$ e $r_0$ invariati, passare da $a_0=0.300$ a $a_1=0.255$ mantiene la
profondità $-D$, ma cambia la curvatura al minimo secondo

$$
\frac{k_1}{k_0}=\left(\frac{a_1}{a_0}\right)^2
=0.85^2=0.7225.
$$

La stiffness locale diminuisce del $27.75\%$. Il residuale target cambia però
in ogni frame, perché la forza Morse completa non è una semplice riscalatura
globale; il retraining resta obbligatorio.

### 21.2 Lettura di un update IBI

Se in una bin $P_i/P_*=2$, $k_BT=2.49$ kJ/mol e $\alpha=0.2$:

$$
\Delta U=0.2\times2.49\times\ln2\simeq0.345\ \text{kJ/mol}.
$$

La bin viene alzata perché è sovrapopolata. Se il rapporto fosse $1/2$, la
correzione avrebbe lo stesso modulo e segno opposto, prima di clipping,
smoothing e taper.

### 21.3 Diagnostica PaiNN

Per distinguere accuratezza e correttezza numerica:

1. confrontare validation loss con il baseline nullo;
2. controllare train/validation gap e più split temporali;
3. verificare parity energia-forza e forza/torque di rigid body;
4. eseguire NVE multi-$\Delta t$ in FP32;
5. ripetere in FP64 per localizzare il floor di precisione;
6. confrontare priors-only e full Hamiltonian dallo stesso checkpoint meccanico.

Una closure FP64 con $p\simeq2$ dimostra coerenza conservativa del percorso
numerico, ma non dimostra che il residuale appreso sia accurato rispetto
all'atomistico.

### 21.4 Checklist prima di cambiare un prior

- verificare unità, segno e dominio;
- controllare $U$, $U'$ e, per NVE, la regolarità di $U''$ ai raccordi;
- applicare le stesse exclusions offline e runtime;
- misurare l'invasione del supporto fisico;
- ricostruire dataset e manifest;
- riaddestrare da zero o riprendere soltanto con provenance compatibile;
- riequilibrare senza riusare un checkpoint incoerente;
- certificare NVT, NVE FP32 e closure FP64 quando necessaria.

### 21.5 Checklist per una nuova coordinata IBI

- scegliere il Jacobiano corretto;
- verificare numero di campioni ed effective sample size;
- scegliere binning e griglia senza sovrarisoluzione;
- ispezionare supporto e gap;
- regolare smoothing e code separatamente;
- limitare update e forze con motivazione fisica;
- usare rappresentazione conservativa per NVE;
- controllare distribuzione, energia-forza e stabilità dopo ogni iterazione;
- ricostruire il residuale PaiNN soltanto dopo aver fissato i prior promossi.
