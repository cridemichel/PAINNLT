# MLCG Framework v2 — Mathematical reference

This document describes the mathematics implemented by the framework:
atomistic-to-CG mapping, rigid bodies, priors, residual targets, PaiNN,
training, dynamics, and NVE certification. Its purpose is to make every model
parameter interpretable without reconstructing conventions from the source.

The complete operational guide remains [`HOWTO_EN.md`](HOWTO_EN.md); the
Italian version is [`MATHEMATICAL_REFERENCE.md`](MATHEMATICAL_REFERENCE.md).

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
\Delta\mathbf r\leftarrow\Delta\mathbf r-mathbf L\,
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
U_0(r)=D(y^2-2y),\qquad F_0(r)=2Day(y-1).
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
