# MLCG Framework v2 — Riferimento matematico

Questo documento descrive la matematica implementata dal framework: mapping
atomistico-CG, rigid body, prior, target residuali, rete PaiNN, training,
dinamica e certificazione NVE. L'obiettivo è permettere di interpretare i
parametri senza dover ricostruire le convenzioni dai sorgenti.

La guida operativa completa resta [`HOWTO.md`](HOWTO.md); la versione inglese
di questo riferimento è [`MATHEMATICAL_REFERENCE_EN.md`](MATHEMATICAL_REFERENCE_EN.md).

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
\Delta\mathbf r\leftarrow\Delta\mathbf r-mathbf L\,
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
U_0(r)=D(y^2-2y),\qquad F_0(r)=2Day(y-1).
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
\Delta\mathbf s_i\mathrel{+}=q_0,qquad
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
\sigma_E=C\Delta t^p,qquad
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
