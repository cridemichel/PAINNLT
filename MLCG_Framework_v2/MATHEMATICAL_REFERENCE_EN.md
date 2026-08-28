# MLCG Framework v2 — Mathematical reference

This document describes the mathematics implemented by the framework:
atomistic-to-CG mapping, rigid bodies, priors, residual targets, PaiNN,
training, dynamics, and NVE certification. Its purpose is to make every model
parameter interpretable without reconstructing conventions from the source.

The complete operational guide remains [`HOWTO_EN.md`](HOWTO_EN.md); the
Italian version is [`MATHEMATICAL_REFERENCE.md`](MATHEMATICAL_REFERENCE.md).
Sections 1–14 provide the compact map; Part II, sections 15–21, gives the full
implementation-level derivations of PaiNN, priors, DBI/IBI, and conservative
interpolation.

## 1. Conventions and units

| Quantity | Symbol | Internal unit |
|---|---:|---:|
| position/distance | $\mathbf r,r$ | nm |
| time | $t$ | ps |
| energy | $U,E$ | kJ mol$^{-1}$ |
| force | $\mathbf F$ | kJ mol$^{-1}$ nm$^{-1}$ |
| torque | $\boldsymbol\tau$ | kJ mol$^{-1}$ |
| mass | $m$ | u |
| inertia | $I$ | u nm$^2$ |
| angle | $\theta,\phi$ | rad |
| thermal energy | $k_BT=RT$ | kJ mol$^{-1}$ |

Every conservative energy obeys

$$
\mathbf F_i=-\nabla_{\mathbf r_i}U.
$$

For an orthorhombic box, the minimum-image convention (MIC) is applied
component by component:

$$
\Delta\mathbf r\leftarrow\Delta\mathbf r-\mathbf L\,
\operatorname{round}(\Delta\mathbf r/\mathbf L).
$$

PaiNN edge vectors use $\mathbf r_{ij}=\mathbf r_i-\mathbf r_j$. Tabulated
priors have representation-specific conventions documented in section 6.

## 2. Coarse-grained state and rigid bodies

### 2.1 Atomistic mapping

For body $m$ containing atoms $a\in m$, the center of mass is

$$
\mathbf R_m=\frac{\sum_{a\in m}m_a\mathbf r_a}{\sum_{a\in m}m_a}.
$$

Mapping may also use a geometric center or a selected atom, but rigid-body
dynamics and torque use the COM. Atomistic generalized force and torque are

$$
\mathbf F_m^{AA}=\sum_{a\in m}\mathbf f_a^{AA},\qquad
\boldsymbol\tau_m^{AA}=\sum_{a\in m}
(\mathbf r_a-\mathbf R_m)\times\mathbf f_a^{AA}.
$$

A one-site body must place its site at the COM: an off-center force would
require a rotational degree of freedom that the body does not have.

### 2.2 Mass, inertia, and principal axes

With $\boldsymbol\rho_a=\mathbf r_a-\mathbf R_m$,

$$
\mathbf I_m=\sum_{a\in m}m_a
\left[(\boldsymbol\rho_a\cdot\boldsymbol\rho_a)\mathbf 1
-\boldsymbol\rho_a\boldsymbol\rho_a^T\right].
$$

The symmetric tensor is diagonalized; eigenvectors are sorted by eigenvalue
and corrected to form a right-handed basis. Masses, principal moments, and
body-frame geometry are stored in `rigid_bodies_info.json`.

### 2.3 Mean geometry and Kabsch alignment

Multi-site frame geometries are iteratively aligned with Kabsch. For centered
configurations $P,Q$,

$$
H=P^TQ=U\Sigma V^T,\qquad
R=V\,\operatorname{diag}(1,1,\operatorname{sign}\det(VU^T))\,U^T.
$$

The determinant correction excludes reflections. At runtime, virtual site $s$
has $\mathbf r_{ms}=\mathbf R_m+Q_m\mathbf d_{ms}$, where $Q_m$ is the body
rotation and $\mathbf d_{ms}$ its body-frame offset.

## 3. Hamiltonian and residual force matching

The fundamental decomposition is

$$
U_{CG}(\mathbf R,Q)=U_{prior}(\mathbf R,Q)+U_{ML}(\mathbf R,Q).
$$

The dataset stores residual rather than total AA targets:

$$
\mathbf F_m^{res}=\mathbf F_m^{AA}-\mathbf F_m^{prior},\qquad
\boldsymbol\tau_m^{res}=\boldsymbol\tau_m^{AA}-\boldsymbol\tau_m^{prior}.
$$

This identity is a consistency condition, not a training detail. Changing a
prior changes the target, so the dataset, PaiNN model, and checkpoint must be
rebuilt. Subtracting and simulating different geometries or parameters yields
an inconsistent Hamiltonian.

For a site at offset $\mathbf d$ carrying force $\mathbf f$,

$$
\mathbf F_m\mathrel{+}=\mathbf f,\qquad
\boldsymbol\tau_m\mathrel{+}=(\mathbf r_s-\mathbf R_m)\times\mathbf f.
$$

## 4. Analytic priors

### 4.1 Harmonic bond

$$
U(r)=\frac12k(r-r_0)^2,\qquad F_r=-k(r-r_0).
$$

For automatic values, $r_0=\langle r\rangle$ and
$k=[\beta\operatorname{Var}(r)]^{-1}$, with $\beta=(RT)^{-1}$.
`r0` moves the minimum; increasing `k` narrows fluctuations and raises the
fastest frequency, generally requiring a smaller timestep.

### 4.2 FENE

With $x=r-r_0$ and maximum extension $r_{max}$,

$$
U(r)=-\frac12kr_{max}^2\ln\left[1-(x/r_{max})^2\right],\qquad
F_r=-\frac{kx}{1-(x/r_{max})^2}.
$$

The domain is $|x|<r_{max}$ and the force diverges at the boundary. `k`
controls local curvature; `r_max` controls the geometric limit.

### 4.3 Harmonic angle

$$
U(\theta)=\frac12k(\theta-\theta_0)^2,
\qquad \frac{dU}{d\theta}=k(\theta-\theta_0).
$$

For automatic values, $\theta_0=\langle\theta\rangle$ and
$k=[\beta\operatorname{Var}(\theta)]^{-1}$. Conversion to Cartesian forces
contains $1/\sin\theta$; geometries near $0$ or $\pi$ are singular.

### 4.4 Cosine dihedral

$$
U(\phi)=K[1-\cos(n\phi-\phi_0)],\qquad
\frac{dU}{d\phi}=Kn\sin(n\phi-\phi_0).
$$

`n` is the multiplicity, `K` the modulation height, and `phi0` the phase. The
automatic phase uses the circular mean
$\operatorname{atan2}(\langle\sin\phi\rangle,\langle\cos\phi\rangle)$.

### 4.5 Switched Morse

Let $y=e^{-a(r-r_0)}$:

$$
U_0(r)=D(y^2-2y),\qquad F_0(r)=2D\,a\,y(y-1).
$$

The minimum is $U_0(r_0)=-D$ and the local curvature is

$$
U_0''(r_0)=2Da^2.
$$

Thus `D` controls depth and curvature, `a` controls width and stiffness
quadratically, and `r0` is the equilibrium distance. For $r_s<r<r_c$, with
$t=(r-r_s)/(r_c-r_s)$,

$$
S(t)=1-10t^3+15t^4-6t^5,
$$

$$
U=SU_0,\qquad F=SF_0-U_0\frac{dS}{dr},\qquad
\frac{dS}{dr}=-\frac{30t^2(1-t)^2}{r_c-r_s}.
$$

For $r\ge r_c$, $U=F=0$. The switch makes energy, force, and curvature
continuous at the boundaries. Making $r_c-r_s$ too narrow concentrates tail
curvature. Pair-specific and type-pair Morse terms share this kernel and add
when both select the same physical pair.

### 4.6 WCA

With $r_c=2^{1/6}\sigma$,

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

`sigma` sets the core length scale and `epsilon` its height. In automatic
fitting, the pair cutoff comes from a regularized low quantile and satisfies

$$
r_c\le\frac{m\,r_{min}^{phys}}{g},
$$

where $g=$ `wca_guard_fraction` and $m=$ `wca_physical_guard_margin`.
`epsilon` is then selected by imposing
$U(g r_c)=$ `wca_guard_kbt` $\,k_BT$. A larger `guard_kbt` stiffens the core;
a smaller margin moves it away from physical data.

## 5. DBI and IBI

Direct Boltzmann inversion uses a Jacobian-corrected density:

$$
U_0(q)=-k_BT\ln P_{corr}(q)+C,
$$

with $P_{corr}(r)=P(r)/r^2$, $P_{corr}(\theta)=P(\theta)/\sin\theta$, and no
Jacobian correction for the periodic dihedral. The implementation identifies
statistical support, smooths the profile, and constructs confining tails
outside support.

The IBI update over common target/simulation support is

$$
U_{i+1}(q)=U_i(q)+\alpha k_BT
\ln\frac{P_i(q)}{P_{target}(q)}.
$$

`alpha` damps the update; `max_update_kT` clips it; smoothing and tapering
reduce noise and discontinuities. Residual targets must be rebuilt after every
prior update.

## 6. Tables and conservative splines

Legacy tables linearly interpolate energy and their third column. Conventions
are: bond = radial force $-dU/dr$; angle = $dU/d\theta$; dihedral = ESPResSo
geometry factor, not simply $-dU/d\phi$. Independently interpolating energy and
force does not guarantee $\mathbf F=-\nabla U$ and is unsuitable for strict
NVE certification.

The `pchip_hermite_v1` representation stores nodes $(q_i,U_i,m_i)$ with
$m_i=dU/dq$. In $q=q_i+th$, $t\in[0,1]$:

$$
U(t)=h_{00}U_i+h_{10}hm_i+h_{01}U_{i+1}+h_{11}hm_{i+1},
$$

where $h_{00}=2t^3-3t^2+1$, $h_{10}=t^3-2t^2+t$,
$h_{01}=-2t^3+3t^2$, and $h_{11}=t^3-t^2$. Energy and derivative come from
the same polynomial. Angles span $[0,\pi]$; dihedrals span $[0,2\pi]$ with
periodic energy and derivative.

## 7. Graph and PaiNN architecture

Each physical site has type $z_i$, scalar features
$\mathbf s_i\in\mathbb R^D$, and vector features
$\mathbf v_i\in\mathbb R^{3\times D}$:

$$
\mathbf s_i^{(0)}=\operatorname{Embedding}(z_i),\qquad \mathbf v_i^{(0)}=0.
$$

MIC edges connect sites belonging to different molecules within `cutoff`; each
physical pair occurs in both directions.

### 7.1 Radial basis and cutoff

For centers $\mu_k$ uniformly spaced in $[0,r_c]$ and
$\sigma_{RBF}=r_c/N_{RBF}$,

$$
R_k(d)=\exp[-(d-\mu_k)^2/\sigma_{RBF}^2]c(d),
$$

$$
c(d)=\frac{x^4}{x^4+\alpha^4},\quad
x=(r_c-d)/r_c,\quad d\le r_c,
$$

and $c=0$ beyond the cutoff. Increasing `num_rbf` increases radial resolution;
increasing `cutoff` increases context, edge count, cost, and memory. Increasing
`toxvaerd_alpha` suppresses the filter farther inside the cutoff; decreasing it
keeps $c\simeq1$ longer but concentrates the transition.

### 7.2 Message block

A `D -> D -> 3D` SiLU MLP produces scalar context and a bias-free
`num_rbf -> 3D` linear map produces the radial filter. Their elementwise
product is split into $(q_0,q_1,q_2)$. For edge $j\to i$:

$$
\Delta\mathbf s_i\mathrel{+}=q_0,\qquad
\Delta\mathbf v_i\mathrel{+}=\mathbf v_j\odot q_1+
\hat{\mathbf r}_{ij}\,q_2.
$$

Aggregation is a sum (`sum_v1`).

### 7.3 Update and readout

With $\mathbf v_v=W_v\mathbf v$, $\mathbf v_u=W_u\mathbf v$, and

$$
\|\mathbf v_v\|_\varepsilon=
\sqrt{\sum_{a=1}^3v_{v,a}^2+10^{-8}},
$$

an MLP over $[\mathbf s,\|\mathbf v_v\|_\varepsilon]$ produces $(a,b,c)$:

$$
\Delta\mathbf s=a+(\mathbf v_v\cdot\mathbf v_u)\odot b,
\qquad \Delta\mathbf v=\mathbf v_u\odot c.
$$

After `n_layers`, a `D -> D/2 -> 1` SiLU readout produces $u_i$.
`hidden_channels` controls width and `n_layers` message-passing depth. Both
increase capacity, compute, and memory.

## 8. Energy gauge, PaiNN forces, and torques

For each species, the model subtracts the energy of the same isolated site
propagated only through update blocks:

$$
u_i=s_E[\tilde u_i-u_{iso}(z_i)],\qquad U_{ML}=\sum_i u_i.
$$

This fixes the gauge left undetermined by force-based training. The trainer
sets $s_E=$ `energy_scale` to the training Force RMS; it scales both energies
and their derivatives.

Edge forces are obtained with autograd from that same energy:

$$
\mathbf f_{ij}=-\frac{\partial U_{ML}}{\partial\mathbf r_{ij}}.
$$

Body forces sum edge contributions with opposite signs at the two endpoints;
torques are $\sum_s(\mathbf r_s-\mathbf R_m)\times\mathbf f_s$. Training uses
`create_graph=true` because optimizing a force loss requires second
derivatives of energy with respect to coordinates and weights.

## 9. Loss and optimization

Scales computed only from the training split are

$$
F_{RMS}=\sqrt{N_F^{-1}\sum F_{target}^2},\qquad
\tau_{RMS}=\sqrt{N_\tau^{-1}\sum\tau_{target}^2}.
$$

Torque includes multi-site bodies only. The loss is

$$
L=\frac{\operatorname{MSE}(\mathbf F,\mathbf F^*)}{F_{RMS}^2}
+\lambda_\tau
\frac{\operatorname{MSE}(\boldsymbol\tau,\boldsymbol\tau^*)}{\tau_{RMS}^2}
+\lambda_L\frac{\langle\|\mathbf f_s\|^2\rangle}{F_{RMS}^2}.
$$

`torque_weight` is $\lambda_\tau$. `lipschitz_lambda` is $\lambda_L$, but its
implemented term penalizes site-force magnitude; it is not a rigorous estimate
of the global Lipschitz constant.

The optimizer is AdamW. `learning_rate` controls update size; `weight_decay`
penalizes weights; `grad_clip_norm` bounds the global gradient norm;
`batch_size` trades statistical noise for memory; `epochs` is an upper bound.
`reduce_lr_patience` reduces learning rate after a plateau and
`early_stopping_patience` stops after no improvement. Split statistics and
normalization must remain train-only to prevent leakage.

## 10. Dynamics and thermostat

The mechanical Hamiltonian is

$$
H=K_{trans}+K_{rot}+U_{prior}+U_{ML}.
$$

ESPResSo integrates rigid-body translation and rotation. In NVT, Langevin adds
friction and noise connected by fluctuation-dissipation; in continuous
translational form,

$$
m\dot{\mathbf v}=\mathbf F-\gamma\mathbf v+
\sqrt{2\gamma k_BT}\,\boldsymbol\eta(t),
$$

with an analogous rotational term. NVE disables the thermostat. `dt` must
resolve the highest potential frequency; increasing it accelerates simulation
but increases discretization error and can destabilize stiff Morse, WCA, FENE,
or angle terms. `kT` controls thermal sampling, not prior stiffness. Force caps
and steepest descent are warm-up tools and must not remain in the production
Hamiltonian.

`device` and `ml_precision` alter numerical noise, not the equations.
CPU/float64 is a closure diagnostic; CPU/float32 is the ordinary certification
reference. On MPS, `MLCG_MPS_EMPTY_CACHE_INTERVAL=100` is the default runtime
cache-release policy and `0` disables it. The training parameter
`mps_empty_cache_every_batches` is separate and does not alter the loss.

## 11. NVE certification

For each timestep, the energy series $E_n$ gives

$$
\sigma_E=\sqrt{N^{-1}\sum_n(E_n-\bar E)^2}.
$$

Fixed-physical-duration runs are fitted to

$$
\sigma_E=C\Delta t^p,\qquad
\log\sigma_E=\log C+p\log\Delta t.
$$

Velocity Verlet is second order, so a conservative asymptotic regime should
give $p\simeq2$. The fit reports log-log $R^2$. Drift compares the first and
last 20% block means:

$$
D_{rel}=\frac{|\bar E_{last}-\bar E_{first}|}
{\max(|E_0|,\langle|E|\rangle,1)}.
$$

Certifier defaults are $1.7\le p\le2.3$, $R^2\ge0.97$, and
$D_{rel}\le10^{-4}$. A low $p$ only at the smallest timesteps may indicate an
FP32 floor; a large-dt outlier may be outside the asymptotic regime; excessive
drift is a separate failure mode.

## 12. Quick parameter reference

### Topology and priors

| Parameter | Mathematical meaning | Effect when increased |
|---|---|---|
| `temperature` | $T$ used in $\beta=(RT)^{-1}$ | softer auto/DBI priors at fixed variance |
| `k`, `D`, `epsilon` | energy scales | larger forces and curvatures |
| `r0`, `theta0`, `phi0` | minimum position/phase | shifts geometric reference |
| Morse `a` | inverse width | local curvature $2Da^2$ increases |
| `r_switch` | start of smooth switch | closer to `r_cut`: more concentrated tail |
| `r_cut` | exact end of interaction | more pairs and higher cost |
| `wca_quantile_percent` | low core quantile | generally moves onset outward |
| `wca_guard_fraction` | $r_{guard}/r_c$ | calibrates closer to cutoff |
| `wca_guard_kbt` | $U(r_{guard})/k_BT$ | higher/stiffer core |
| `wca_physical_guard_margin` | margin on physical minimum | permits core closer to data |

### PaiNN and training

| Parameter | Role | Main trade-off |
|---|---|---|
| `num_species` | type cardinality | must exactly cover site types |
| `hidden_channels` | $D$ | capacity versus memory/compute |
| `n_layers` | message/update blocks | context/capacity versus compute |
| `num_rbf` | radial resolution | detail versus compute |
| `cutoff` | graph $r_c$ | context versus edge count |
| `toxvaerd_alpha` | cutoff shape | broader transition when larger |
| `torque_weight` | $\lambda_\tau$ | rotational versus COM-force accuracy |
| `lipschitz_lambda` | $\lambda_L$ | smaller site forces, possible underfit |
| `learning_rate` | AdamW step | speed versus instability |
| `weight_decay` | weight regularization | capacity control versus underfit |
| `grad_clip_norm` | gradient-norm threshold | stability versus attenuated updates |
| `batch_size` | frames per update | variance versus memory |
| `validation_fraction` | validation share | estimate stability versus training data |

### Runtime and certification

| Parameter | Meaning | Main effect |
|---|---|---|
| `dt` | $\Delta t$ | cost $\propto1/dt$ at fixed duration; VV error $O(dt^2)$ |
| `steps` | integration steps | duration $=steps\,dt$ |
| `kT` | NVT thermal energy | thermal fluctuation amplitude |
| `log_interval` | sampling stride | I/O volume and time resolution |
| `ml_precision` | PaiNN precision | numerical floor and cost |
| `neighbor_search` | pair algorithm | performance; physics must be unchanged |
| `duration_ps` | NVE duration per dt | statistical stability of fit |
| `slope_min/max` | accepted range for $p$ | strictness of order gate |
| `min_r2` | minimum log-log linearity | strictness of power-law gate |
| `max_relative_drift` | maximum relative drift | strictness of secular gate |

## 13. Interpretation rules

1. A stiffer prior reduces what PaiNN must learn, but raises Hamiltonian
   frequencies and may worsen stability and the FP32 floor.
2. Low validation loss does not certify energy conservation; energy-force
   parity and a multi-$dt$ NVE sweep are still required.
3. $p\simeq2$ certifies numerical order in the tested domain, not scientific
   accuracy relative to the atomistic distribution.
4. A larger model is not automatically more physical: capacity, data,
   prior/residual decomposition, and timestep must be assessed together.
5. Manifests and hashes are part of the candidate's mathematical definition:
   they prevent mixing incompatible models, priors, datasets, and checkpoints.

## 14. Normative source map

The equations above follow the current implementation in:

- `preprocessing/geometry_utils.py`, `build_cg_dataset.py`, `prior_kernels.py`,
  and `conservative_spline.py`;
- `ibi/ibi_core.py` and `ibi/build_dbi_priors.py`;
- `training/PaiNN_Architecture.hpp` and `training/train_painn.cpp`;
- `simulation/run_cg_md.py`, `equilibrate.py`, `nve_analysis.py`, and
  `certify_nve.py`;
- `simulation/espresso_plugin/` for the ESPResSo bridge.

If they diverge, tests and sources from the revision actually being built are
authoritative. This document must be updated whenever equations, conventions,
or parameters change.

---

# Part II — Implementation-level derivations

## 15. PaiNN: complete formulation of the implemented variant

The only variant accepted by trainer and runtime is
`painn_canonical_context_silu_v2`. This section describes that variant, not a
generic PaiNN family. Let $N$ be the number of physical sites,
$D=$ `hidden_channels`, $K=$ `num_rbf`, and $L=$ `n_layers`.

### 15.1 Representations and symmetries

For each site $i$, layer $\ell$ maintains

$$
\mathbf s_i^{(\ell)}\in\mathbb R^D,
\qquad
\mathbf v_i^{(\ell)}\in\mathbb R^{3\times D}.
$$

Component $s_{ic}$ is an invariant scalar, while column
$\mathbf v_{ic}\in\mathbb R^3$ is an equivariant vector. Under an orthogonal
rotation $R$ and translation $\mathbf a$:

$$
\mathbf r_i\mapsto R\mathbf r_i+\mathbf a,\qquad
\mathbf s_i\mapsto\mathbf s_i,\qquad
\mathbf v_{ic}\mapsto R\mathbf v_{ic}.
$$

The final energy must be invariant. The model obtains:

- translation invariance by using relative displacements only;
- rotational equivariance through vectors, norms, dot products, and radial
  directions;
- invariance to neighbor ordering through sum aggregation;
- invariance to permutations of same-type sites through shared embeddings and
  additive readout.

Initial features are

$$
\mathbf s_i^{(0)}=E_{z_i},\qquad
\mathbf v_i^{(0)}=0,
$$

where $E\in\mathbb R^{N_{species}\times D}$ is the embedding matrix and $z_i$
is the site type. `num_species` must exceed the largest type index and must
agree across dataset, configuration, manifest, and runtime.

### 15.2 Directed periodic graph

For every physical pair inside the cutoff, both edges $i\leftarrow j$ and
$j\leftarrow i$ are built. With `row=i`, `col=j`:

$$
\mathbf r_{ij}=\operatorname{MIC}(\mathbf r_i-\mathbf r_j),\qquad
d_{ij}=\sqrt{\mathbf r_{ij}\cdot\mathbf r_{ij}+\varepsilon_d},\qquad
\widehat{\mathbf r}_{ij}=\frac{\mathbf r_{ij}}{d_{ij}},
$$

with $\varepsilon_d=10^{-8}$ in the implemented norm. This prevents an exact
division by zero and is negligible at ordinary physical distances. Training
edges are cached per frame; at runtime the ESPResSo neighbor list must produce
the same physical set inside `cutoff`.

MIC contains a non-differentiable `round` at image boundaries. The trainer
detaches the image choice: the derivative is exact inside each MIC region,
while the discrete jump is excluded from autograd. This is the appropriate
convention provided no relevant pair is ambiguous at half the box length.

### 15.3 Gaussian radial basis

Centers are uniformly spaced:

$$
\mu_k=\frac{k}{K-1}r_c,\qquad k=0,\ldots,K-1,
$$

and the implementation uses

$$
\sigma_{RBF}=\frac{r_c}{K},\qquad
g_k(d)=\exp\left[-\frac{(d-\mu_k)^2}{\sigma_{RBF}^2}\right].
$$

The unswitched derivative is

$$
\frac{dg_k}{dd}=-\frac{2(d-\mu_k)}{\sigma_{RBF}^2}g_k(d).
$$

Increasing $K$ narrows $\sigma_{RBF}$ and permits finer radial variation, but
adds parameters and operations. A large $K$ creates no information if the
distance distribution or number of frames cannot support it.

### 15.4 Toxvaerd cutoff and smoothness

For $d\le r_c$:

$$
x=1-\frac d{r_c},\qquad
c(d)=\frac{x^4}{x^4+\alpha^4},
$$

and $c(d)=0$ for $d>r_c$. The effective basis is
$R_k(d)=g_k(d)c(d)$. Useful derivatives are

$$
\frac{dc}{dx}=\frac{4\alpha^4x^3}{(x^4+\alpha^4)^2},
\qquad
\frac{dc}{dd}=-\frac{4\alpha^4x^3}
{r_c(x^4+\alpha^4)^2},
$$

$$
\frac{dR_k}{dd}=c\frac{dg_k}{dd}+g_k\frac{dc}{dd}.
$$

Near the cutoff, $c\sim x^4/\alpha^4$, so its value and first three distance
derivatives vanish from the inner side. This avoids a force impulse as an edge
crosses $r_c$. `toxvaerd_alpha` is not another cutoff: it controls the width of
the attenuated region. At $x=\alpha$, $c=1/2$; a larger $\alpha$ therefore
moves the half-cutoff farther inward.

### 15.5 Message block with explicit indices

For edge $j\to i$, an MLP over sender context produces

$$
\mathbf h_j=W_2\operatorname{SiLU}(W_1\mathbf s_j+\mathbf b_1)+\mathbf b_2
\in\mathbb R^{3D},
$$

while a bias-free linear transform of the radial basis produces

$$
\mathbf w_{ij}=W_R\mathbf R(d_{ij})\in\mathbb R^{3D}.
$$

The channelwise product
$\mathbf q_{ij}=\mathbf h_j\odot\mathbf w_{ij}$ is split into
$\mathbf q^{s},\mathbf q^{v},\mathbf q^{r}\in\mathbb R^D$. For channel $c$:

$$
m^{s}_{ij,c}=q^{s}_{ij,c},
$$

$$
\mathbf m^{v}_{ij,c}=
q^{v}_{ij,c}\mathbf v_{j,c}+
q^{r}_{ij,c}\widehat{\mathbf r}_{ij}.
$$

The receiver obtains

$$
\Delta s_{i,c}^{msg}=\sum_{j\in\mathcal N(i)}m^{s}_{ij,c},
\qquad
\Delta\mathbf v_{i,c}^{msg}=
\sum_{j\in\mathcal N(i)}\mathbf m^{v}_{ij,c},
$$

followed by the residual update

$$
\mathbf s_i\leftarrow\mathbf s_i+\Delta\mathbf s_i^{msg},\qquad
\mathbf v_i\leftarrow\mathbf v_i+\Delta\mathbf v_i^{msg}.
$$

$q^s$ is scalar; $q^v$ scales an equivariant vector; $q^r$ scales the
equivariant radial direction. Every term therefore has the correct
transformation. Sum rather than mean makes amplitude sensitive to neighbor
count; this is intentional and recorded in the manifest as `sum_v1`.

### 15.6 Intra-site update block

Two linear maps act on channels only, identically for all three spatial
components:

$$
\mathbf v^v_{i,c}=\sum_d(W_v)_{cd}\mathbf v_{i,d},
\qquad
\mathbf v^u_{i,c}=\sum_d(W_u)_{cd}\mathbf v_{i,d}.
$$

The invariants are

$$
n_{i,c}=\sqrt{\mathbf v^v_{i,c}\cdot\mathbf v^v_{i,c}+10^{-8}},
\qquad
p_{i,c}=\mathbf v^v_{i,c}\cdot\mathbf v^u_{i,c}.
$$

A `2D -> D -> 3D` SiLU MLP receives
$[\mathbf s_i,\mathbf n_i]$ and returns
$\mathbf a_i,\mathbf b_i,\mathbf c_i\in\mathbb R^D$. The update is

$$
\Delta s_{i,c}^{upd}=a_{i,c}+b_{i,c}p_{i,c},
\qquad
\Delta\mathbf v_{i,c}^{upd}=c_{i,c}\mathbf v^u_{i,c},
$$

$$
\mathbf s_i\leftarrow\mathbf s_i+\Delta\mathbf s_i^{upd},\qquad
\mathbf v_i\leftarrow\mathbf v_i+\Delta\mathbf v_i^{upd}.
$$

Norms and dot products are invariant; $c_{i,c}$ is invariant and scales an
equivariant vector. The update requires no new edges and mixes vector
information accumulated at a site.

### 15.7 Additive readout and species gauge

After $L$ blocks, the readout

$$
\widetilde u_i=W_o^{(2)}
\operatorname{SiLU}(W_o^{(1)}\mathbf s_i+\mathbf b_o^{(1)})+b_o^{(2)}
$$

produces one scalar per site. Derivative-only training does not determine
additive energy constants. For each species $z$, the framework computes an
isolated reference: its embedding, zero vector features, no message blocks,
and update blocks only. If that readout is $u_{iso}(z)$,

$$
u_i=s_E[\widetilde u_i-u_{iso}(z_i)],
\qquad
U_{ML}=\sum_i u_i.
$$

An isolated site thus has zero ML energy by construction. The trainer sets the
$s_E$ buffer to the training-set $F_{RMS}$. It scales energies and forces
together and gives initial weights a natural magnitude; it is not a runtime
parameter to change after training.

### 15.8 Edge forces and conservation

Training treats MIC displacements as leaf variables. For each directed edge:

$$
\mathbf g_{ij}=\frac{\partial U_{ML}}{\partial\mathbf r_{ij}},
\qquad
\mathbf f_{ij}=-\mathbf g_{ij}.
$$

The contribution is aggregated with opposite signs:

$$
\mathbf f_i\mathrel{+}=\mathbf f_{ij},
\qquad
\mathbf f_j\mathrel{-}=\mathbf f_{ij}.
$$

Because both directed edges belong to the actual forward pass, no manual
$1/2$ is inserted: autograd differentiates the energy that was truly
computed. Antisymmetric aggregation gives zero total internal force to
numerical precision, while scalar energy gives conservation inside a fixed
MIC and neighbor-list region.

### 15.9 Cost and meaning of PaiNN parameters

For $E$ directed edges, the dominant cost of one block is approximately
$O(ED+ND^2+EKD)$. Autograd memory is much larger than forward-only inference
because force matching uses `create_graph=true`.

| Parameter | Mathematical action | Risk when too large |
|---|---|---|
| `hidden_channels=D` | scalar/vector channel count | memory, overfitting, $D^2$ compute |
| `n_layers=L` | message/update depth | deep graphs and costly second derivatives |
| `num_rbf=K` | resolution of $d_{ij}$ | bases redundant relative to data |
| `cutoff=r_c` | graph support | more edges and box constraints |
| `toxvaerd_alpha` | attenuation location/width | loss of context if too large |
| `num_species` | embedding/gauge rows | semantic mismatch if types disagree |

## 16. Priors: construction, derivatives, and force geometry

### 16.1 General rule for a distance prior

Let $\boldsymbol\delta=\operatorname{MIC}(\mathbf r_j-\mathbf r_i)$,
$r=\|\boldsymbol\delta\|$, and
$\widehat{\boldsymbol\delta}=\boldsymbol\delta/r$. For energy $U(r)$:

$$
\nabla_{\mathbf r_i}r=-\widehat{\boldsymbol\delta},
\qquad
\mathbf F_i=\frac{dU}{dr}\widehat{\boldsymbol\delta},
\qquad
\mathbf F_j=-\mathbf F_i.
$$

If radial force $F_r=-dU/dr$ is oriented from $j$ toward $i$, the same relation
is $\mathbf F_i=-F_r\widehat{\boldsymbol\delta}$. Stating this convention is
essential when a table's third column is generically named `force`.

### 16.2 Automatic harmonic inference

Near a minimum, $U(q)\simeq U(q_0)+k(q-q_0)^2/2$. When the coordinate Jacobian
is locally neglected, the distribution is Gaussian:

$$
P(q)\propto\exp[-\beta k(q-q_0)^2/2],
\qquad
\operatorname{Var}(q)=\frac1{\beta k}.
$$

Therefore

$$
q_0=\langle q\rangle,
\qquad
k=\frac1{\beta\operatorname{Var}(q)}.
$$

This `auto` estimate is a local harmonic approximation, distinct from
Jacobian-corrected DBI. Entries sharing a `name` may pool statistics, increasing
sample size while assuming physical equivalence. Near-zero variance yields a
huge $k$: it calls for a floor or manual choice, not an arbitrarily small
timestep as the only remedy.

### 16.3 Harmonic and FENE

For a harmonic bond, $U''(r)=k$ everywhere. To first order the fastest relative
frequency is $\omega\sim\sqrt{k/\mu}$ for reduced mass $\mu$, so stability and
accuracy require $\omega\Delta t\ll1$.

For FENE, with $x=r-r_0$ and $R=r_{max}$:

$$
U'(r)=\frac{kx}{1-x^2/R^2},
$$

$$
U''(r)=k\frac{1+x^2/R^2}{(1-x^2/R^2)^2}.
$$

Curvature equals $k$ at the minimum and diverges faster than force at the
boundary. A frame with $|x|\ge R$ is not merely unlikely: it is outside the
mathematical domain of the prior.

### 16.4 Morse force, curvature, and switch

With $y=e^{-a(r-r_0)}$:

$$
U_0'=2D\,a\,y(1-y),\qquad
U_0''=2Da^2y(2y-1),\qquad
F_0=-U_0'=2D\,a\,y(y-1).
$$

The decay length is $a^{-1}$ and minimum stiffness is $2Da^2$. For a quintic
switch of width $w=r_c-r_s$:

$$
S'=-\frac{30t^2(1-t)^2}{w},
\qquad
S''=-\frac{60t(1-t)(1-2t)}{w^2}.
$$

Switched curvature is

$$
U''=S''U_0+2S'U_0'+SU_0''.
$$

$S'$ and $S''$ vanish at both boundaries, and $S$ also vanishes at the cutoff;
energy, force, and curvature therefore join continuously. The $1/w$ and
$1/w^2$ terms nevertheless show why a narrow switch window can be stiff.

Pair-specific contacts select explicit COM/site endpoints and use coincident
technical markers at runtime. Type-pair Morse applies to all physical type
pairs allowed by exclusions. If both select the same pair, energies add; there
is no automatic physical deduplication.

### 16.5 WCA and the complete automatic calibration

Let $s_6=(\sigma/r)^6$. For $r<r_c$:

$$
U=4\epsilon(s_6^2-s_6)+\epsilon,
$$

$$
U'=\frac{24\epsilon}{r}(s_6-2s_6^2),
\qquad
U''=\frac{24\epsilon}{r^2}(26s_6^2-7s_6).
$$

Automatic fitting does not simply assign one `sigma` per type. For each pair
$(a,b)$ it collects physical nonbonded distances, a low quantile $q_{ab}$, and
the exact streaming minimum $r^{min}_{ab}$. It introduces type radii $R_a$ and
minimizes

$$
\mathcal J(\{R\})=\sum_{ab}w_{ab}(R_a+R_b-q_{ab})^2,
\qquad
w_{ab}=\frac{N_{ab}}{N_{ab}+N_0},
$$

with $N_0=1000$ in the implementation and numerical bounds on radii. This
hierarchical regularization lets poorly sampled pairs fall back toward a sum
of type radii. With

$$
\lambda_{ab}=\frac{N_{ab}}{N_{ab}+N_0},
$$

the preliminary cutoff is

$$
r_{c,ab}^{(0)}=\lambda_{ab}q_{ab}+
(1-\lambda_{ab})(R_a+R_b),
$$

then

$$
r_{c,ab}=\min\left[
r_{c,ab}^{(0)},q_{ab},
\frac{m\,r^{min}_{ab}}{g}
\right].
$$

Here $g=$ `wca_guard_fraction` and
$m=$ `wca_physical_guard_margin`. Next,

$$
\sigma_{ab}=\frac{r_{c,ab}}{2^{1/6}},
\qquad r_{guard}=g r_{c,ab}.
$$

If $z=(\sigma/r_{guard})^6$, imposing
$U(r_{guard})=Gk_BT$, where $G=$ `wca_guard_kbt`, gives

$$
\epsilon_{ab}=\frac{Gk_BT}{4(z^2-z)+1}.
$$

The histogram-derived minimum remains diagnostic; the support guard uses the
exact streaming minimum because a histogram can hide the shortest
observation. Fractions below $r_c$ and $r_{guard}$ and the ratio
$r_{guard}/r_{min}$ quantify core invasion into physical data.

### 16.6 Angles: Cartesian gradient

Define

$$
\mathbf a=\operatorname{MIC}(\mathbf r_i-\mathbf r_j),\quad
\mathbf b=\operatorname{MIC}(\mathbf r_k-\mathbf r_j),\quad
A=\|\mathbf a\|,\quad B=\|\mathbf b\|,
$$

$$
c=\cos\theta=\frac{\mathbf a\cdot\mathbf b}{AB}.
$$

Cosine gradients are

$$
\nabla_i c=\frac{\mathbf b}{AB}-c\frac{\mathbf a}{A^2},
\qquad
\nabla_k c=\frac{\mathbf a}{AB}-c\frac{\mathbf b}{B^2}.
$$

Because $d\theta/dc=-1/\sin\theta$, for
$G_\theta=dU/d\theta$:

$$
\mathbf F_i=\frac{G_\theta}{\sin\theta}\nabla_i c,
\quad
\mathbf F_k=\frac{G_\theta}{\sin\theta}\nabla_k c,
\quad
\mathbf F_j=-(\mathbf F_i+\mathbf F_k).
$$

Numerically clamping $c$ to $[-1,1]$ corrects roundoff but does not remove the
geometric singularity at $\sin\theta=0$.

### 16.7 Dihedrals: periodicity and geometry

The dihedral uses three MIC vectors and normals to the two planes; its phase is
mapped to $[0,2\pi)$. For a conservative representation, once geometric
derivatives $\nabla_x\phi$ are defined, forces always satisfy

$$
\mathbf F_x=-\frac{dU}{d\phi}\nabla_x\phi,
\qquad x\in\{i,j,k,l\}.
$$

With $\mathbf v_{12},\mathbf v_{23},\mathbf v_{34}$, unit normals
$\mathbf n_{12},\mathbf n_{23}$, and unnormalized normal lengths
$l_{12},l_{23}$:

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

Nearly zero normals make the dihedral undefined. In the legacy cosine path,
preprocessing evaluates force by a central Cartesian difference with
$10^{-6}$ nm step and removes the small mean-force remainder from roundoff; a
new dihedral topology still requires explicit parity with runtime.

### 16.8 Topological exclusions and preprocessing/runtime parity

The current policy distinguishes connectivity from energy form:

- site pairs inside one rigid body are excluded from nonbonded interactions;
- for a 1–2 relation, only explicitly bonded site pairs are excluded;
- a COM–COM bond does not automatically remove every virtual-site pair;
- 1–3 endpoints of an explicitly excluding angle use the specified all-sites
  policy;
- Morse defaults to `exclude_wca=false` because it does not imply covalent
  connectivity.

The same mask must be used during prior subtraction and runtime. A single pair
present in only one path changes both the residual target and reconstructed
Hamiltonian.

## 17. Direct Boltzmann inversion in detail

### 17.1 From the canonical distribution to a potential

For internal coordinate $q$, observed probability includes geometric measure
$J(q)$:

$$
P(q)=Z_q^{-1}J(q)e^{-\beta U(q)}.
$$

Therefore

$$
U(q)=-k_BT\ln\frac{P(q)}{J(q)}+C.
$$

The measures used are

$$
J(r)=r^2,\qquad J(\theta)=\sin\theta,\qquad J(\phi)=1.
$$

Dividing by the Jacobian prevents phase-space volume from being interpreted as
an effective force. Near $r=0$ and angle endpoints the Jacobian is numerically
regularized, but unsupported regions do not thereby become reliable data.

### 17.2 Histogram and density

Given samples $q_n$, a histogram returns counts $C_b$ and density $P_b$ at bin
centers $q_b$. Density is normalized numerically:

$$
\sum_bP_b\Delta q\simeq1
$$

or by the corresponding grid quadrature. A positive floor prevents $\log0$,
but that floor must not create physical tails; support is therefore handled
separately.

### 17.3 Statistical support

A bin is reliable only when it satisfies configured count and density
criteria. Small internal gaps may be filled up to a maximum number of bins;
unsupported outer tails remain unsupported. If support contains fewer than
`min_support_points`, the group lacks enough information for a stable update.

For mask $M_b$, raw PMF is

$$
U_b^{raw}=-k_BT\ln P_{corr,b},\qquad b\in M,
$$

and is shifted so $\min_{b\in M}U_b=0$. The constant does not affect forces,
but keeps tables comparable and numerically scaled.

### 17.4 Smoothing on supported data

The supported profile is filtered by a discrete Gaussian of width
`histogram_smoothing_sigma` in bin units. For normalized kernel $G$:

$$
\widetilde U_b=\sum_{b'}G_{b-b'}U_{b'}.
$$

Periodic coordinates use wrapping; bonds and angles use nonperiodic boundary
handling. Smoothing reduces derivative variance but adds bias and may erase a
narrow barrier; it must be chosen relative to grid resolution and effective
sampling.

### 17.5 Interpolation and tail extrapolation

Inside nonperiodic support, the profile is interpolated by PCHIP, avoiding many
spurious oscillations of an ordinary cubic spline. At each boundary, value and
slope are estimated over a configured window; outside support, confining tails
are constructed consistently with those conditions. Their purpose is not to
guess the PMF without data, but to prevent free entry into an unsampled region.

For a dihedral, a periodic cubic spline is used and

$$
U(0)=U(2\pi),\qquad U'(0)=U'(2\pi)
$$

is imposed.

### 17.6 Angular walls

Quadratic walls of width $w$ and constant $k_w$ may protect singularities at
$0$ and $\pi$. On the left:

$$
U_{wall}=\frac12k_w(w-\theta)^2,\qquad
\frac{dU_{wall}}{d\theta}=-k_w(w-\theta),\quad\theta<w.
$$

On the right, with $\theta_r=\pi-w$:

$$
U_{wall}=\frac12k_w(\theta-\theta_r)^2,\qquad
\frac{dU_{wall}}{d\theta}=k_w(\theta-\theta_r),\quad\theta>\theta_r.
$$

`wall_width` determines how much of the domain is regularized; `wall_k`
determines stiffness and can therefore constrain the timestep.

### 17.7 Groups, pooling, and DBI/IBI modes

Coordinates need not be inverted independently. Entries in the same logical
group may pool samples:

$$
\mathcal Q_g=\bigcup_{a\in g}\{q_{a,n}\}.
$$

Pooling reduces noise only when entries are physically equivalent; otherwise
it yields an average PMF representing none of them. Initial DBI mode constructs
$U_0$ from the AA distribution. IBI mode instead preserves state $U_i$,
compares a CG trajectory with the target, and generates $U_{i+1}$. Fixed priors
outside the group are copied without reinterpretation.

Grids must be strictly increasing and uniform. Bonds use a configured finite
domain or one derived from data; angles span exactly $[0,\pi]$ and dihedrals
$[0,2\pi]$. Changing the grid between iterations requires explicit
interpolation and makes comparison of $\Delta U$ less direct.

## 18. Iterative Boltzmann inversion in detail

### 18.1 Sign of the update

If $U_i$ produces $P_i$ and the target is $P_*$, the implemented update is

$$
\Delta U_i(q)=\alpha k_BT\ln\frac{P_i(q)}{P_*(q)},
\qquad
U_{i+1}=U_i+\Delta U_i.
$$

If $P_i>P_*$, $\Delta U>0$ and the region is penalized. If $P_i<P_*$,
$\Delta U<0$ and it is favored. `alpha` is damping, not a temperature:
$\alpha=1$ applies the full theoretical update, while smaller values reduce
oscillation from finite samples and coupled coordinates.

### 18.2 Common support

Target and simulation have masks $M_*$ and $M_i$. Updates occur only on

$$
M_{upd}=M_*\cap M_i.
$$

If the intersection is insufficient, the framework preserves the previous
potential. Updating where either distribution is merely a numerical floor
would produce a huge, statistically meaningless log ratio.

### 18.3 Update clipping and smoothing

The log ratio is energy-clipped:

$$
\Delta U\leftarrow
\operatorname{clip}(\Delta U,-U_{max},U_{max}),
\qquad
U_{max}=\texttt{max_update_kT}\,k_BT.
$$

It is then filtered with width `update_smoothing_sigma`. Clipping and smoothing
serve different purposes: the former limits outliers; the latter limits
high-frequency curvature and noise.

### 18.4 Interpolation, taper, and tails

For nonperiodic variables, supported updates are interpolated with PCHIP. A
cosine window brings the correction smoothly to zero near boundaries over
`taper_bins`, preserving safe tails of the previous potential. Only unsupported
target regions are then rebuilt by the extrapolation policy.

For periodic variables, the update is interpolated over the full period with
wrapping and endpoint equality is restored. The minimum energy is always
subtracted after an update, a gauge operation that leaves forces unchanged.

### 18.5 From updated energy to force

The fundamental IBI object is $U_{i+1}$, not an independently updated force.
The generator differentiates the interpolated profile and converts it to the
required target convention:

| coordinate | legacy third column |
|---|---|
| bond | $-dU/dr$ |
| angle | $+dU/d\theta$ |
| dihedral | ESPResSo factor $-(dU/d\phi)/\sin\phi$ away from singular points |

At nodes with small $|\sin\phi|$, the dihedral factor uses the nearest regular
value and is then clipped by `force_max`. This is an ESPResSo bonded-table
convention, not a new physical law.

### 18.6 Convergence and identifiability

IBI seeks matching marginal distributions; it does not generally determine a
unique many-body Hamiltonian. Coupled bonded coordinates, nonbonded priors,
and PaiNN may compensate for one another. Credible convergence requires at
least:

1. sufficient overlap of supports;
2. stable reduction of histogram discrepancy over several iterations;
3. shrinking $\Delta U$ without sustained clipping;
4. no artificial visits to extrapolated tails;
5. dynamical stability and energy-force parity;
6. rebuilding residual targets before retraining PaiNN.

More iterations do not repair inadequate sampling. If simulation never visits
a target region, overlap, initialization, or potential must improve; merely
raising `max_update_kT` is not a solution.

### 18.7 IBI parameters and effects

| Parameter | Meaning | Too small | Too large |
|---|---|---|---|
| `kT` | inversion energy scale | weak updates | strong updates/barriers |
| `alpha` | update damping | slow convergence | oscillation |
| `histogram_smoothing_sigma` | DBI smoothing | noisy derivatives | erased detail |
| `update_smoothing_sigma` | $\Delta U$ smoothing | rough update | overly diffuse correction |
| `max_update_kT` | clipping in $k_BT$ units | saturated correction | outlier sensitivity |
| `min_support_points` | minimum support | fragile profiles | valid groups rejected |
| `taper_bins` | transition to zero | boundary kink | useful support reduced |
| `force_max` | table limit | truncated prior | numerical instability |
| `wall_width`, `wall_k` | angular walls | accessible singularity | overly stiff angle |

### 18.8 What “regularized IBI” means in this framework

Implementation-level regularization is a composition of transparent operators,
not one abstract term:

$$
U_{i+1}=\mathcal E\!\left[
U_i+\mathcal T\!\left(
\mathcal S\!\left(
\mathcal C\!\left[
\alpha k_BT\log\frac{P_i}{P_*}
\right]\right)\right)\right],
$$

where $\mathcal C$ is clipping, $\mathcal S$ smoothing,
$\mathcal T$ support tapering, and $\mathcal E$ tail reconstruction. For
angles, the wall operator $U\mapsto U+U_{wall}$ may also be applied. Each
operator has an explicit parameter and testable effect. This path must not be
described as Tikhonov or second-derivative regularization unless such a term is
actually added to the code.

“Conservative” promotion is a separate operation: it replaces independently
interpolated columns with one Hermite energy and its analytical derivative.
Statistical regularity and numerical conservation solve different problems and
both must be verified.

## 19. Legacy tables and conservative splines

### 19.1 Why separate energy and force interpolation is insufficient

On uniform grid $q_i$, linear interpolation of column $T$ is

$$
T(q)=(1-t)T_i+tT_{i+1},\qquad t=\frac{q-q_i}{h}.
$$

If $U$ is linearly interpolated, $dU/dq=(U_{i+1}-U_i)/h$ is constant in the
interval. A separately interpolated force varies linearly. They coincide only
in special cases. A legacy table may therefore reproduce structure while
locally violating $F=-dU/dq$.

### 19.2 Cubic Hermite polynomial

The conservative spline stores $U_i$ and $m_i=U'_i$. With
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

The analytical derivative used for force is

$$
U'(q)=\frac{(6t^2-6t)U_i+(-6t^2+6t)U_{i+1}}{h}
+(3t^2-4t+1)m_i+(3t^2-2t)m_{i+1}.
$$

Endpoint value and derivative exactly match nodal data. There is no second
force interpolation: conservation is structural.

### 19.3 Domains

- bond: uniform grid; below minimum it uses tangent continuation
  $U=U_0+m_0(q-q_0)$, while $r\ge q_{max}$ is outside the domain;
- angle: coordinate is clamped to $[0,\pi]$ only for geometric roundoff;
- dihedral: periodic wrapping in $[0,2\pi)$ with equal $U,U'$ at endpoints.

Tangent continuation is conservative but does not replace a physically sound
tail. A bond that frequently visits the continuation indicates insufficient
support.

### 19.4 Projection of derivatives

A distance spline returns $U'(r)$ and uses section 16.1 geometry. An angle
spline returns $U'(\theta)$ and uses the $1/\sin\theta$ factor of section 16.6.
A dihedral spline returns $U'(\phi)$ directly and uses section 16.7 geometric
gradients. Separating scalar derivative from Cartesian Jacobian makes each
layer independently testable.

### 19.5 Conversion of existing tables

To convert a legacy table, an energy compatible with the force convention is
reconstructed by trapezoidal integration. For a bond:

$$
U_{i+1}=U_i-\frac h2(F_i+F_{i+1}).
$$

For an angle, the third column is $+U'(\theta)$ and the sign changes
accordingly. For a legacy dihedral factor $f_{ESP}$, scalar derivative is

$$
U'(\phi)=-f_{ESP}(\phi)\sin\phi.
$$

Reconstruction selects an energy anchor, integrates in both directions, and
subtracts the minimum. The resulting spline must be recertified: conversion
removes energy-force inconsistency but may slightly change the distribution
relative to the original table.

## 20. Targets, loss, and second derivatives

### 20.1 From site forces to body quantities

For body $m$ with sites $s\in m$:

$$
\mathbf F_m^{ML}=\sum_{s\in m}\mathbf f_s^{ML},
$$

$$
\boldsymbol\tau_m^{ML}=\sum_{s\in m}
\operatorname{MIC}(\mathbf r_s-\mathbf R_m)\times\mathbf f_s^{ML}.
$$

Torque is in the lab frame about the COM. Translating every site by the same
vector leaves it unchanged; changing rigid-body offsets or orientation does
not.

### 20.2 Train-only normalization

For $N_F$ Cartesian force components and $N_\tau$ components from multi-site
bodies only:

$$
F_{RMS}^2=\frac1{N_F}\sum_a(F_a^*)^2,
\qquad
\tau_{RMS}^2=\frac1{N_\tau}\sum_b(\tau_b^*)^2.
$$

Normalized MSE values are dimensionless. Validation uses training scales, not
its own, so a different target variance cannot make validation appear
artificially better.

The zero baseline is

$$
L_{0,F}^{val}=\frac{\langle|\mathbf F^*|^2\rangle_{val}}{F_{RMS}^2},
\qquad
L_{0,\tau}^{val}=\frac{\langle|\boldsymbol\tau^*|^2\rangle_{val}}{\tau_{RMS}^2}.
$$

A validation loss close to this value means the model does not generalize
better than a zero residual force even when training loss decreases.

### 20.3 Site-force penalty

The historically named `lipschitz` term is

$$
L_L=\frac1{N_sF_{RMS}^2}\sum_s\|\mathbf f_s^{ML}\|^2.
$$

It penalizes large site forces, including pairs that may cancel in COM force.
It does not compute $\sup_x\|\nabla f(x)\|$ and is therefore not an estimate of
the global Lipschitz constant. It may reduce learned curvature, but an
excessive value causes underfitting.

### 20.4 Why force matching requires second derivatives

If $\theta$ denotes weights and
$\mathbf F_\theta=-\nabla_{\mathbf r}U_\theta$, then

$$
\nabla_\theta L_F
=2(\mathbf F_\theta-\mathbf F^*)^T
\frac{\partial\mathbf F_\theta}{\partial\theta}
=-2(\mathbf F_\theta-\mathbf F^*)^T
\frac{\partial^2U_\theta}{\partial\theta\,\partial\mathbf r}.
$$

Autograd must therefore retain the first-derivative graph
(`create_graph=true`). This is also why training memory and cost greatly exceed
energy-only inference.

### 20.5 Gradient clipping and AdamW

For global gradient norm $\|g\|_2$ and threshold $G$:

$$
g\leftarrow g\min\left(1,\frac G{\|g\|_2}\right).
$$

`GradClip_Fraction` is the fraction of batches with $\|g\|_2>G$. Frequent
initial clipping may be normal; persistent values near 100% indicate a problem
with learning rate, scales, or curvature.

AdamW maintains exponential moments and applies decoupled weight decay:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}=(1-\eta\lambda_w)\theta_t
-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

`learning_rate` is $\eta$ and `weight_decay` is $\lambda_w$; parameters not
exposed by configuration use LibTorch defaults.

## 21. Quantitative examples and checklists

### 21.1 TEL22 Morse softening

At fixed $D$ and $r_0$, changing $a_0=0.300$ to $a_1=0.255$ preserves depth
$-D$ but changes minimum curvature by

$$
\frac{k_1}{k_0}=\left(\frac{a_1}{a_0}\right)^2
=0.85^2=0.7225.
$$

Local stiffness decreases by $27.75\%$. The residual target still changes in
every frame because the full Morse force is not a global rescaling; retraining
remains mandatory.

### 21.2 Reading an IBI update

If one bin has $P_i/P_*=2$, $k_BT=2.49$ kJ/mol, and $\alpha=0.2$:

$$
\Delta U=0.2\times2.49\times\ln2\simeq0.345\ \text{kJ/mol}.
$$

The bin is raised because it is overpopulated. If the ratio were $1/2$, the
correction would have equal magnitude and opposite sign before clipping,
smoothing, and tapering.

### 21.3 PaiNN diagnostics

To separate accuracy from numerical correctness:

1. compare validation loss with the zero baseline;
2. inspect train/validation gap and multiple temporal splits;
3. verify energy-force and rigid-body force/torque parity;
4. run multi-$\Delta t$ NVE in FP32;
5. repeat in FP64 to localize the precision floor;
6. compare priors-only and full Hamiltonians from the same mechanical state.

An FP64 closure with $p\simeq2$ establishes conservative numerical consistency,
but does not prove that the learned residual is accurate relative to the
atomistic reference.

### 21.4 Checklist before changing a prior

- verify units, sign, and domain;
- inspect $U$, $U'$, and for NVE the smoothness of $U''$ at joins;
- apply identical exclusions offline and at runtime;
- measure invasion of physical support;
- rebuild dataset and manifest;
- train from scratch or resume only with compatible provenance;
- re-equilibrate without reusing an inconsistent checkpoint;
- certify NVT, FP32 NVE, and FP64 closure when needed.

### 21.5 Checklist for a new IBI coordinate

- choose the correct Jacobian;
- check sample count and effective sample size;
- choose bins and grid without over-resolution;
- inspect support and gaps;
- tune smoothing and tails separately;
- constrain updates and forces for a physical reason;
- use a conservative representation for NVE;
- check distribution, energy-force consistency, and stability after each
  iteration;
- rebuild the PaiNN residual only after promoted priors are fixed.
