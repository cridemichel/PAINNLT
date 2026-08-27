# MLCG Framework v2 — Complete Technical Guide

> Dedicated mathematical reference: [`MATHEMATICAL_REFERENCE_EN.md`](MATHEMATICAL_REFERENCE_EN.md).

> Documented state: **17 August 2026**.
> This guide describes the current framework configuration, including the recent fixes for:
> reversible switched non-bonded Morse interactions (pair-specific and type-pair) in ESPResSo, manifest validation tolerant to harmless
> `float32` round trips, ML-only dummy neighbor-list interactions, conservative IBI conversion, **optional regularized angular IBI**, and post-promotion NVE certification in which `sigma_E = std(E)` is again a gating `O(dt^2)` observable.

---

## Contents

1. [Framework purpose and modeling philosophy](#1-framework-purpose-and-modeling-philosophy)
2. [Conventions, units, and notation](#2-conventions-units-and-notation)
3. [CG Hamiltonian decomposition](#3-cg-hamiltonian-decomposition)
4. [Atomistic to coarse-grained mapping](#4-atomistic-to-coarse-grained-mapping)
5. [Reference generalized forces and torques](#5-reference-generalized-forces-and-torques)
6. [Rigid bodies: masses, inertia, Kabsch alignment, and virtual sites](#6-rigid-bodies-masses-inertia-kabsch-alignment-and-virtual-sites)
7. [Analytic priors, DBI, and bonded IBI](#7-analytic-priors-dbi-and-bonded-ibi)
8. [Pair-specific WCA: construction, guardrails, and subtraction](#8-pair-specific-wca-construction-guardrails-and-subtraction)
9. [Residual force-matching target](#9-residual-force-matching-target)
10. [Implemented PaiNN architecture](#10-implemented-painn-architecture)
11. [Network-predicted forces and torques](#11-network-predicted-forces-and-torques)
12. [Loss function and normalization](#12-loss-function-and-normalization)
13. [Energy gauge and energy scale](#13-energy-gauge-and-energy-scale)
14. [Binary dataset format](#14-binary-dataset-format)
15. [Model manifest and provenance](#15-model-manifest-and-provenance)
16. [ESPResSo checkpoints](#16-espresso-checkpoints)
17. [Python environment and trainer build](#17-python-environment-and-trainer-build)
18. [PaiNN/Morse integration into ESPResSo](#18-painnmorse-integration-into-espresso)
19. [`topology_config.json` reference](#19-topology_configjson-reference)
20. [Training configuration reference](#20-training-configuration-reference)
21. [Complete pipeline: scripts and parameters](#21-complete-pipeline-scripts-and-parameters)
22. [CG equilibration in detail](#22-cg-equilibration-in-detail)
23. [CG production in detail](#23-cg-production-in-detail)
24. [NVE certification](#24-nve-certification)
25. [Quantitative interpretation of NVE scaling](#25-quantitative-interpretation-of-nve-scaling)
26. [Tests and consistency checks](#26-tests-and-consistency-checks)
27. [TEL22 tutorial: current configuration](#27-tel22-tutorial-current-configuration)
28. [Troubleshooting](#28-troubleshooting)
29. [Checklist for adapting the framework to a new system](#29-checklist-for-adapting-the-framework-to-a-new-system)

---

# 1. Framework purpose and modeling philosophy

`MLCG_Framework_v2` builds and simulates a coarse-grained (CG) molecular model in which the
potential energy is decomposed into two conceptually distinct terms:

1. **explicit structural/physical priors**, chosen to be simple and interpretable;
2. a **residual PaiNN potential**, learned from atomistic data by force matching.

The core framework is chemistry-agnostic. TEL22 DNA is an optional tutorial/example and is
not part of the assumptions of the generic core.

The current model can be written as

\[
U_{\mathrm{CG}}(\mathbf R,\mathbf Q)
=
U_{\mathrm{prior}}(\mathbf R,\mathbf Q)
+
U_{\mathrm{ML}}(\mathbf R,\mathbf Q;\theta),
\]

where

- \(\mathbf R\) denotes translational CG coordinates;
- \(\mathbf Q\) denotes orientations of multi-site rigid bodies;
- \(U_{\mathrm{prior}}\) contains WCA and bonded terms;
- \(U_{\mathrm{ML}}\) is the residual PaiNN energy;
- \(\theta\) are trainable network parameters.

The force-matching target is therefore not simply the mapped atomistic force. The preprocessing
stage first evaluates the generalized force/torque produced by the explicit priors and subtracts
it from the atomistic reference:

\[
\mathbf F_m^{\mathrm{target}}
=
\mathbf F_m^{\mathrm{ref}}
-
\mathbf F_m^{\mathrm{prior}},
\]

\[
\boldsymbol\tau_m^{\mathrm{target}}
=
\boldsymbol\tau_m^{\mathrm{ref}}
-
\boldsymbol\tau_m^{\mathrm{prior}}.
\]

PaiNN is trained on these residual targets.

A particularly important design choice in the current implementation is that **PaiNN excludes
edges connecting sites belonging to the same CG molecule/rigid body**. The learned model therefore
targets residual intermolecular interactions. Intramolecular structure is represented by explicit
bonded priors and, for multi-site bodies, by rigidification.

This decomposition has several practical consequences:

- a prior used during preprocessing must be reproduced consistently at runtime;
- changing priors changes the residual target and normally requires rebuilding the dataset;
- the PaiNN force must be the gradient of the same scalar energy used for MD if strict NVE
  conservation is to be tested;
- numerical conservation and force-matching accuracy are separate questions: an inaccurate
  potential can still be perfectly conservative.

---

# 2. Conventions, units, and notation

The current framework uses the following working units:

| Quantity | Unit |
|---|---|
| position | nm |
| time | ps |
| mass | amu |
| moment of inertia | amu nm² |
| energy | kJ/mol |
| force | kJ/(mol nm) |
| torque | kJ/mol |
| preprocessing temperature | K |
| runtime thermal energy | `kT` in kJ/mol |

At 300 K,

\[
RT
\simeq
0.008314462618\times300
\simeq
2.494\ \mathrm{kJ/mol}.
\]

The TEL22 wrappers therefore commonly use `kT=2.49`.

The dataset builder reads positions through MDAnalysis and converts Å to nm by division by 10.
For the verified GROMACS/TRR workflow the current code applies a factor of 10 to force values
before assembling targets in kJ/(mol nm). If a different MDAnalysis reader or trajectory format
is introduced, force-unit behavior must be re-audited rather than assumed.

Throughout this guide:

- lowercase \(i,j,k,l\) usually index CG sites;
- lowercase \(a\) indexes atomistic atoms;
- lowercase \(m\) indexes CG molecules/rigid bodies;
- \(r\) is a scalar distance;
- \(\mathbf r\), \(\mathbf R\) are vectors;
- \(\mathbf F\) is force;
- \(\boldsymbol\tau\) is torque;
- \(k_B T\) is represented numerically by `kT` at runtime;
- \(\beta=(RT)^{-1}\) in the molar units used by preprocessing.

---

# 3. CG Hamiltonian decomposition

For a rigid-body CG model with translational momenta \(\mathbf P_m\) and body-frame angular
momenta \(\mathbf L_m\), the conservative Hamiltonian has the schematic form

\[
H
=
K_{\mathrm{trans}}
+
K_{\mathrm{rot}}
+
U_{\mathrm{prior}}
+
U_{\mathrm{ML}},
\]

with

\[
K_{\mathrm{trans}}
=
\sum_m \frac{|\mathbf P_m|^2}{2M_m},
\]

and, in principal axes,

\[
K_{\mathrm{rot}}
=
\frac12\sum_m
\left(
I_{m,x}\omega_{m,x}^2
+I_{m,y}\omega_{m,y}^2
+I_{m,z}\omega_{m,z}^2
\right).
\]

The total conservative force on a generalized coordinate is the negative energy gradient:

\[
\mathbf F
=-\nabla U_{\mathrm{CG}}
=-\nabla U_{\mathrm{prior}}-\nabla U_{\mathrm{ML}}.
\]

This identity is central to the NVE certification. If energy and force are evaluated from
independent interpolants, the numerical trajectory can drift even if each table separately looks
smooth. For this reason the current runtime uses an **analytic switched non-bonded Morse interaction** rather
than independently tabulated Morse energy and force.

The current runtime energy reported for certification is conceptually

\[
E_{\mathrm{tot}}
=
E_{\mathrm{ESPResSo}}
+E_{\mathrm{ML}},
\]

where `E_ESPResSo` contains kinetic plus the classical ESPResSo interaction energy. The historical
CSV field name `E_class` is therefore a legacy/misleading name: in existing logs it is not purely
classical potential energy.

---

# 4. Atomistic to coarse-grained mapping

The mapping is controlled primarily by `preprocessing/topology_config.json` or a system-specific
copy such as `tutorials/tel22/tel22_topology.json`.

## 4.1 Body center

The dynamical center of each mapped molecule is always its atomistic center of mass (COM):

\[
\mathbf R_m
=
\frac{\sum_{a\in m}m_a\mathbf r_a}
     {\sum_{a\in m}m_a}.
\]

This remains true even if the individual CG site mapping method is `COG` or `ATOM`.

That distinction is important: `mapping_method` controls the positions of retained CG sites;
it does not redefine the translational center used for rigid-body mechanics and torque.

## 4.2 CG site positions

The supported mapping methods are:

### `mapping_method = "COM"`

For the atoms selected for a site,

\[
\mathbf r_i^{\mathrm{CG}}
=
\frac{\sum_{a\in i}m_a\mathbf r_a}
     {\sum_{a\in i}m_a}.
\]

This is usually the physically natural choice when site mass distribution matters.

### `mapping_method = "COG"`

\[
\mathbf r_i^{\mathrm{CG}}
=
\frac1{N_i}\sum_{a\in i}\mathbf r_a.
\]

Every selected atom contributes equally.

### `mapping_method = "ATOM"`

The position of the first selected atom is used. This is appropriate for an intentionally retained
representative atom but should not be confused with a mass center.

A selector `"*"` means all atoms in the residue/molecule selection associated with that site.

## 4.3 Site types

Each mapped site receives an integer `site_type`. These integers index the PaiNN embedding table.
They must satisfy

\[
0\le\texttt{site_type}<\texttt{num_species}.
\]

`num_species` therefore means the number of ML site species, not the number of ESPResSo particle
types. Runtime also creates non-ML COM/dummy particle types above the ML range.

For a new chemistry, site-type semantics are entirely defined by configuration. The generic core
does not require DNA residue names or TEL22-specific type numbers.

---

# 5. Reference generalized forces and torques

The reference force on CG molecule \(m\) is the sum of atomistic forces on all atoms belonging to it:

\[
\mathbf F_m^{\mathrm{ref}}
=
\sum_{a\in m}\mathbf f_a.
\]

The reference torque is evaluated about the molecule COM:

\[
\boldsymbol\tau_m^{\mathrm{ref}}
=
\sum_{a\in m}
(\mathbf r_a-\mathbf R_m)\times\mathbf f_a.
\]

This construction is invariant to a uniform translation of the molecule because the lever arm is
measured from the COM.

The torque is physically useful only when the CG representation retains orientation. In the current
model this means multi-site rigid bodies. Single-site molecules do not have an independently
observable CG orientation, and the training torque loss masks them.

A useful conceptual separation is:

- **site forces** are the local gradients produced by PaiNN;
- **molecular force** is their sum over sites in a body;
- **molecular torque** is the sum of lever-arm cross force terms.

---

# 6. Rigid bodies: masses, inertia, Kabsch alignment, and virtual sites

## 6.1 Mass and inertia tensor

For atom positions relative to the molecular COM,

\[
\mathbf x_a=\mathbf r_a-\mathbf R_m,
\]

the inertia tensor is

\[
\mathbf I
=
\sum_a m_a
\left(
|\mathbf x_a|^2\mathbf 1
-
\mathbf x_a\mathbf x_a^T
\right).
\]

The preprocessing code diagonalizes this tensor with a symmetric eigensolver. The eigenvectors are
sorted consistently and the basis is corrected if necessary to be right-handed. The principal
moments become the runtime rigid-body inertias.

A right-handed basis matters because orientation conventions must be consistent between the stored
site geometry, quaternions, body-frame angular velocities, and torque.

## 6.2 Mean multi-site geometry

For a multi-site molecule, raw mapped site clouds fluctuate from frame to frame. The current builder
constructs a representative rigid geometry using iterative Kabsch alignment:

1. select mapped site coordinates for each frame;
2. center them consistently;
3. align each frame to a reference by the optimal rotation minimizing squared displacement;
4. average aligned coordinates;
5. repeat the align/average cycle (currently a small fixed number of iterations, three in the
   implementation used for this framework state);
6. express the final site vectors in the principal-axis body frame.

The same rigidified geometry must be used consistently when:

- fitting WCA distances;
- collecting bonded statistics;
- subtracting prior forces/torques;
- constructing ESPResSo virtual sites at runtime.

Using raw mapped coordinates in one part of the pipeline and rigidified coordinates in another
creates a systematic mismatch in geometry and therefore in force/torque targets.

## 6.3 Runtime representation

ESPResSo creates a **real COM particle for every CG molecule**, plus one or more virtual particles
representing ML sites. This is true even for one-site molecules.

The current conventions include:

- COM/dummy type: `num_species + 1`;
- virtual-site masses/inertias set to a very small value (`1e-5`) to avoid introducing meaningful
  translational mass;
- rotational dynamics enabled only for bodies with more than one retained site;
- one-site bodies must place their sole ML site at the COM to within the strict geometry tolerance
  (approximately `1e-6 nm`).

The separate COM particle is needed for rigid-body mechanics and molecular generalized forces, but
it is **not** an ML species and must not participate in PaiNN neighbor interactions.

---

# 7. Analytic priors, DBI, and bonded IBI

Priors encode stiff/local structure explicitly and reduce what the neural network has to learn.
Whenever a prior is fitted from an equilibrium distribution, the builder uses a Direct Boltzmann
Inversion (DBI)-style approximation.

Let

\[
\beta=\frac1{RT}.
\]

## 7.1 Harmonic bond

The potential is

\[
U(r)=\frac12k(r-r_0)^2,
\]

with radial force

\[
F_r=-k(r-r_0).
\]

For a narrow approximately Gaussian bond distribution, the current auto-fit uses

\[
r_0=\langle r\rangle,
\]

\[
k=\frac{1}{\beta\,\mathrm{Var}(r)}.
\]

Consequences:

- smaller variance -> larger `k` -> stiffer bond;
- increasing preprocessing temperature at fixed variance changes the inferred `k` through \(\beta\);
- a short/nonrepresentative trajectory can strongly bias a stiff DBI constant.

## 7.2 FENE

The implemented force convention is based on displacement from `r0`:

\[
\Delta r=r-r_0,
\]

\[
F_r
=-\frac{k\Delta r}{1-(\Delta r/r_{\max})^2}.
\]

The corresponding logarithmic energy diverges as \(|\Delta r|\to r_{\max}\). FENE therefore provides
a hard extensibility guard and must be parameterized so physically sampled configurations stay well
inside the singular boundary.

## 7.3 Switched non-bonded Morse

The current runtime uses one analytic conservative Morse kernel with gauge

\[
U_0(r)=D\left(e^{-2a(r-r_0)}-2e^{-a(r-r_0)}\right),
\]

so \(U_0(r_0)=-D\) and \(U_0(\infty)=0\). Between `r_switch` and `r_cut` the tail is multiplied by a quintic smoothstep; both energy and force are exactly zero at and beyond the cutoff. Crossing the cutoff is therefore not a topology event: a pair can leave the interaction range and later re-enter/rebind.

The framework exposes **two selection modes** that share this same potential.

### 7.3.1 Pair-specific Morse

Explicit contacts remain in the `bonds` list with `type="morse"`. Each endpoint is addressed as `(mol, site)`: `site=-1` selects the rigid-body COM, while `site>=0` selects the corresponding CG site. `COM<->COM`, `COM<->site`, and `site<->site` contacts are therefore supported:

```json
{
  "mol_i": 1, "site_i": 0,
  "mol_j": 7, "site_j": 2,
  "type": "morse",
  "D": 50.0, "a": 0.3, "r0": 1.57,
  "r_switch": 11.64, "r_cut": 15.0
}
```

These terms are **pair-specific physically**, but are not bonded in ESPResSo bookkeeping. The runtime must not change the physical `particle.type` of a CG site because that value identifies the PaiNN species and the WCA/type-pair interactions. Instead, every endpoint used by explicit Morse receives a **coincident technical virtual marker site**. The marker is attached to the same COM, carries a dedicated ESPResSo type outside the ML-species range, and is used only to dispatch the non-bonded Morse. A force on the marker is transferred to the rigid body at the selected endpoint position, so an off-COM site also generates the correct torque. Pair-specific marker types are placed on the N-square side of the hybrid decomposition so a long explicit-contact cutoff does not inflate the short-range WCA/PaiNN cells. The runtime activates the hybrid decomposition **before** registering long-cutoff pair-specific Morse interactions: ESPResSo validates every new non-bonded cutoff against the current cell system, so this ordering is required to prevent an N-square marker interaction from being rejected by the regular-decomposition range limit.

Markers are not physical CG sites: PaiNN does not see them, and WCA/type-pair Morse remains configured on the original physical sites. Pair-specific Morse therefore adds to those physical terms without creating implicit exclusions. Markers do belong to the ESPResSo particle list: runtime checkpoints and VTF output contain them as technical particles with `type >= num_species+2`; analyses that want only physical CG sites should filter `type < num_species`. Changing pair-specific endpoints therefore requires regenerating the checkpoint.

For backward compatibility, pair-specific contacts retain a 15 nm default `r_cut`; if `r_switch` is omitted it is placed 75% of the way from `r0` to `r_cut`. Omitted `site_i/site_j` fields mean `-1`, so existing COM-COM topologies remain valid. `exclude_wca` defaults to `false`: a tertiary Morse contact is not covalent connectivity and does not create a WCA exclusion.

### 7.3.2 Type-pair Morse

Transferable bead-type attractions use the top-level `morse_type_pairs` section:

```json
"morse_type_pairs": [
  {
    "type_i": 1, "type_j": 3,
    "D": 4.0, "a": 2.0, "r0": 0.55,
    "r_switch": 0.90, "r_cut": 1.20
  }
]
```

`type_i` and `type_j` are the **physical CG site types** from `mapping.site_types`, not the technical marker types used by pair-specific contacts. One record applies to every non-excluded site pair carrying those types, using ordinary ESPResSo `non_bonded_inter[type_i,type_j]` semantics. `r_cut` is mandatory for this mode because it contributes directly to the regular neighbor-search cutoff; `r_switch` is optional and uses the same 75% default.

ESPResSo particle exclusions suppress all non-bonded potentials, not only WCA. Type-pair Morse is therefore excluded intra-rigid-body, on explicitly excluded 1-2 site pairs, and on 1-3 pairs excluded by WCA policy v3. Preprocessing applies exactly the same mask when subtracting the prior from reference forces and torques.

### 7.3.3 Coexistence and double counting

Both modes may coexist, but they are **additive**. If an explicit endpoint pair has a pair-specific contact while the corresponding physical sites are also covered by a type-pair Morse entry, both contributions enter the Hamiltonian. This can be intentional, but should not happen accidentally. Use pair-specific Morse for native/topological/tertiary contacts and type-pair Morse for generic transferable bead-type attractions.

For both modes, the prior subtracted during preprocessing must match the runtime interaction exactly. After changing Morse priors, regenerate dataset/priors, retrain the residual model, re-equilibrate, and repeat NVE certification.

## 7.4 Harmonic angle

For angle \(\theta\),

\[
U(\theta)
=
\frac12k_\theta(\theta-\theta_0)^2,
\]

with the DBI approximation

\[
\theta_0=\langle\theta\rangle,
\qquad
k_\theta=\frac{1}{\beta\,\mathrm{Var}(\theta)}.
\]

Angles are stored in radians. A small angular variance can imply a very large bending stiffness and
therefore a restrictive stable MD timestep.

## 7.5 Cosine dihedral

The preprocessing convention is

\[
U(\phi)
=
K\left[1-\cos(n\phi-\phi_0)\right].
\]

`n` is periodicity and `phi0` is based on a circular mean. During preprocessing, current code uses
small central finite differences for dihedral force subtraction to avoid hidden sign/index convention
mismatches between different analytic implementations. Runtime ESPResSo uses the corresponding
analytic dihedral interaction.

The present TEL22 tutorial priors contain no dihedrals, but the generic framework supports them.

## 7.6 Tabulated bonded priors: initial DBI and iterative IBI

Version 2 now provides a **bonded DBI/IBI** workflow for site-addressable distances, angles, and dihedrals. The implementation is molecule-agnostic: `ibi/` contains no TEL22-specific topology, and the same `(molecule, site)` endpoints are used to evaluate coordinates in the mapped atomistic target dataset and in sampled CG trajectories.

This first IBI layer is intentionally bonded-only. It does **not** yet implement non-bonded/RDF IBI.

### 7.6.1 Declaring inversion groups

A bonded entry in the seed priors can be marked `type="ibi"` or `type="dbi"`:

```json
{
  "bonds": [
    {
      "name": "backbone",
      "type": "ibi",
      "mol_i": 0, "site_i": 0,
      "mol_j": 1, "site_j": 0
    }
  ]
}
```

- `type="ibi"` creates an initial Direct Boltzmann Inversion table and updates it iteratively;
- `type="dbi"` creates the initial table only and keeps it fixed during later iterations;
- entries in the same bonded category that share a `name` are pooled into one target distribution and share one table;
- endpoints may address COMs (`site=-1`) or physical CG sites (`site>=0`).

While the **seed dataset** is built, `ibi/dbi` terms are not subtracted; they only declare which geometry is to be inverted. Generated entries are converted to `type="tabulated"` and retain `ibi_mode="ibi"` or `ibi_mode="dbi"`.

### 7.6.2 Initial DBI only

```bash
python3 ibi/build_dbi_priors.py \
  --dataset cg_dataset_seed.bin \
  --priors cg_priors_seed.json \
  --outdir ibi_priors_dbi \
  --ibi-config ibi_settings.json
```

This command uses only coordinates from the binary dataset; residual-force columns do not enter the Boltzmann inversion. Generated table paths are relative to the output priors JSON, so the output directory remains self-contained when moved.

### 7.6.3 Iterative IBI

```bash
python3 ibi/run_ibi_loop.py \
  --dataset cg_dataset_seed.bin \
  --priors cg_priors_seed.json \
  --config training/cg_model_config.json \
  --rb_info rigid_bodies_info.json \
  --pypresso espresso/build/pypresso \
  --iterations 5 \
  --outdir ibi_priors \
  --ibi-config ibi_settings.json
```

At each iteration the driver:

1. starts from the current DBI/IBI table set;
2. runs **priors-only NVT** CG sampling with `simulation/run_cg_md.py`, without PaiNN;
3. discards burn-in and writes a structured NPZ trajectory containing COMs and physical CG sites only; technical pair-specific Morse markers are excluded;
4. evaluates exactly the same bond/angle/dihedral coordinates used for the target;
5. applies \(\Delta U=\alpha k_BT\ln[P_i/P_{target}]\) only to `ibi` groups;
6. leaves `dbi` groups fixed;
7. writes metrics and the next `cg_priors.json`.

A minimal settings file is:

```json
{
  "kT": 2.49,
  "alpha": 0.25,
  "simulation": {
    "dt": 0.0005,
    "burn_in_steps": 8000,
    "steps": 40000,
    "log_interval": 40
  }
}
```

Here `steps` is the sampled production segment after `burn_in_steps`; both values must be multiples of `log_interval`. The self-contained final file is `ibi_priors/cg_priors_final.json`, with final tables under `ibi_priors/final/`.

### 7.6.4 ESPResSo table conventions

Table files have `x energy force` columns, but the third column has interaction-specific semantics:

- `TabulatedDistance`: `force = -dU/dr`;
- `TabulatedAngle`: `force = +dU/dtheta`, on the exact `0..pi` domain;
- `TabulatedDihedral`: the column is the ESPResSo **torsional force factor**, not raw `-dU/dphi`; away from the geometric singularities,

\[
\mathrm{factor}(\phi)=-\frac{dU/d\phi}{\sin\phi}.
\]

The v1 implementation treated the tabulated dihedral column as ordinary `-dU/dphi`; that convention is deliberately not carried into v2. Dihedral tables span `0..2*pi`. Because the ESPResSo torsional geometry is numerically singular at `phi=0` and `phi=pi`, dihedral targets with substantial probability or steep slopes close to those points require extra care.

Run the explicit runtime/preprocessing parity diagnostic before generating residual targets with IBI tables:

```bash
espresso/build/pypresso simulation/diagnose_tabulated_prior_parity.py
```

It must finish with `[PASS]`.

### 7.6.5 Mandatory post-IBI residual rebuild

The final table changes the explicit prior and therefore changes the residual target that PaiNN must learn. **Do not train from the old residual dataset.** Re-run `preprocessing/build_cg_dataset.py` on the original atomistic trajectory using:

```text
--priors ibi_priors/cg_priors_final.json
```

Pass 2 then subtracts the same site-aware tabulated bond/angle/dihedral forces and torques used by the runtime. Only this regenerated dataset should be used for residual PaiNN training.

ESPResSo interpolates tabulated energy and force arrays separately. The framework therefore continues to reject strict **NVE certification** by default when explicit tabulated priors are active; IBI sampling is NVT. This is separate from the already certified analytic pre-IBI baseline.

### 7.6.6 Conservative IBI conversion for strict NVE

For converged bond and angle priors the framework also supports
`type="conservative_spline"`.  A single cubic-Hermite energy spline `U(q)` is
the source of truth; ESPResSo obtains `dU/dq` from that same polynomial instead
of interpolating energy and force as independent tables.  The currently
certified scope is **bond + angle**; tabulated dihedrals are deliberately outside
this conservative phase.

For the TEL22 IBI tutorial run:

```bash
cd tutorials/tel22_IBI
bash ./20_install_conservative_spline.sh
bash ./21_convert_best_ibi_to_conservative.sh
bash ./22_validate_conservative_spline.sh
```

`20_install_conservative_spline.sh` rebuilds ESPResSo, checks the Python
bindings, and runs a synthetic runtime smoke test that compares ESPResSo forces
with `-grad U` obtained by finite differences of ESPResSo's own bonded energy.
`22_validate_conservative_spline.sh` then gates the actual converted tables on
`U/dU_dq` consistency and preprocessing/runtime energy-force parity for every
unique conservative spline.

Passing these gates does not make the old PaiNN model valid for the new prior:
the explicit Hamiltonian has changed.  Rebuild the residual dataset against the
**exact** conservative `cg_priors.json`, retrain PaiNN, repeat the matched
structural checks, and only then run strict NVE timestep-scaling and long-window
drift certification.

#### 7.6.6.1 Post-conversion gate and conservative residual training

After `22_validate_conservative_spline.sh` finishes with `[PASS]`, the TEL22 IBI
workflow continues fail-closed:

```bash
bash ./13_rebuild_residual_dataset.sh
bash ./16_check_ibi_training_inputs.sh
```

When `ibi_conservative/cg_priors.json` exists, scripts `13`, `16`, `03`, `18`,
and `19` prefer it automatically over the historical tabulated priors. The
rebuild writes `ibi_residual_build_manifest.json`, which SHA256-binds the
residual dataset, rigid-body metadata, conservative priors, every referenced
spline table, and both `validation_report.json` and
`runtime_parity_report.json`. Preflight `16` must finish with `[PASS]` before
training.

Residual training must start from the newly defined target:

```bash
bash ./03_train_model.sh
```

If the default output `tel22_model_ibi.pt` already exists, the script refuses to
overwrite it or to resume implicitly. When the residual dataset has been rebuilt
for conservative priors, **do not** `--resume` from a model trained against an
older residual target. Select a new artifact instead, for example:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./03_train_model.sh
```

The model manifest records the training dataset/config hashes. If a non-default
model name is used, propagate the same `IBI_MODEL` to downstream runtime gates,
for example:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  OVERWRITE=1 bash ./diagnostics/scripts/18_validate_postibi_runtime.sh

IBI_MODEL=tel22_model_ibi_conservative.pt \
  OVERWRITE=1 bash ./diagnostics/scripts/19_validate_ibi_ml_ab.sh
```

Strict NVE certification belongs only at the end of this chain: first produce a
new residual model, preserve matching provenance, and validate the runtime using
the same `conservative priors + residual PaiNN` Hamiltonian.


#### 7.6.6.2 NVE certification of the conservative IBI-only candidate

If the matched A/B gate in step `19` shows that the PaiNN residual does not
improve structure, the physical candidate to certify can be the **IBI-only**
branch. Run the dedicated gate with:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./diagnostics/scripts/23_certify_conservative_ibi_nve.sh --overwrite
```

The model selected by `IBI_MODEL` is **not activated** either while preparing
the dedicated NVT checkpoint or in the NVE trajectories. The older
`diagnostics/ml/postibi_runtime_validation/equilibrated_postibi.npz` is only the **source
state**: step `23` first runs Langevin NVT with `--disable_ml` and writes
`diagnostics/nve/nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz`.
Only this new checkpoint, thermalized under the Hamiltonian actually being
certified, is reused identically for every NVE timestep.

The dedicated checkpoint records its Hamiltonian mode, sampling ensemble, and
the SHA256 of the source checkpoint. The certifier revalidates these fields
fail-closed before the first NVE run. Every NVE trajectory passes `--disable_ml`,
turns the thermostat off, sets force cap to zero, and uses velocity Verlet.

Before dynamics, `simulation/conservative_nve_preflight.py` revalidates that
`ibi_conservative/cg_priors.json`, every referenced spline,
`validation_report.json`, and `runtime_parity_report.json` are byte-identical to
the Phase-2 validated artifacts. Legacy `tabulated` bonded priors are rejected,
and the conservative scope remains **bond + angle**; a conservative dihedral
fails closed until a dedicated torsional kernel/parity gate exists.

Defaults are 5 ps per timestep with
`0.001 0.0015 0.002 0.003 0.004 0.005 ps`. The generic certification criteria
remain `sigma_E ~ dt^p`, `1.7 <= p <= 2.3`, `R2 >= 0.97`, and relative block
mean drift `<= 1e-4`. Outputs live under
`diagnostics/nve/nve_certification_conservative_ibi_only/`, and the report records
`hamiltonian_mode=conservative_classical_model_provenance_ml_disabled`.

This gate certifies only `WCA + Morse + conservative bonded IBI` over the sampled
state region. It does not promote the PaiNN residual; if a future ML model wins
the matched A/B test, the complete ML-active Hamiltonian must be certified
separately.


#### 7.6.6.3 Fine-dt and short-time NVE diagnostics (not a certification)

If step `23` passes the drift gate but fails the global `sigma_E ~ dt^p` fit, do
not immediately modify the spline kernel or priors. Run the dedicated
**diagnostic-only** gate first:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
NVE_DIAG_DURATION_PS=2 \
  bash ./diagnostics/scripts/24_diagnose_conservative_ibi_nve_scaling.sh --overwrite
```

Step `24` reuses the provenance-bound IBI-only NVT checkpoint prepared by step
`23`; it does not re-thermalize the system and it never activates PaiNN. The
default scan is
`0.00025 0.0005 0.00075 0.001 0.0015 0.002 0.003 0.004 0.005 ps`, with
separate `sigma_E` and `rms_delta_E` power-law fits for:

- **fine**: `dt <= 0.001 ps`;
- **coarse**: `dt >= 0.0015 ps`;
- **global**: all timesteps.

For the fine regime only, energy errors are also measured at the common physical
times `0.012 0.024 0.048 0.096 ps`. These times are exact multiples of every
default fine timestep, so the diagnostic reads actual samples and performs
**no energy interpolation**. Each local window reports power-law fits for
`|Delta E(t)|`, prefix RMS `Delta E`, and prefix `sigma_E`.

Outputs:

```text
diagnostics/nve/nve_diagnostic_conservative_ibi_only/nve_diagnostic_report.json
diagnostics/nve/nve_diagnostic_conservative_ibi_only/nve_diagnostic_runs.csv
```

This step is deliberately **diagnostic-only**: it returns success after the
trajectories are generated and analyzed even if the strict reference criterion
would remain `FAIL`. It does not change step-23 thresholds and cannot be used as
a certification. Recovery of `p ~= 2` with good `R2` in the fine/short-time
fits is evidence for a fine-timestep asymptotic regime; failure to recover the
expected order even there calls for Hamiltonian-component diagnostics rather
than relaxed certification thresholds.


#### 7.6.6.4 Historical step-26 composite conservative IBI-only certification

After steps `22`, `23`, and `25`, assemble the final verdict without rerunning
dynamics:

```bash
cd tutorials/tel22_IBI
bash ./diagnostics/scripts/26_finalize_conservative_ibi_nve_certification.sh
```

The final gate requires all of the following simultaneously:

- conservative-spline finite-difference validation: PASS;
- preprocessing <-> ESPResSo energy/force parity: PASS;
- consistent provenance for priors and the dedicated IBI-only NVT checkpoint;
- NVE relative block-mean drift below threshold;
- Richardson state convergence consistent with second order for position,
  velocity, orientation, and `omega_body`.

By default every state metric must satisfy `1.7 <= median p <= 2.3` and
`median R2 >= 0.95`. In the **historical step-26** composite, the step-23
`sigma_E ~ dt^p` fit remained `diagnostic_only`. That was useful for separating
trajectory order from the energy-scaling anomaly, but it is **not the current
final production criterion**: steps 27--34 localized IBI-angle stiffness, and
the post-promotion step-34 certification again requires `sigma_E = O(dt^2)` in
addition to Richardson state convergence.

Output:

```text
diagnostics/nve/nve_final_certification_conservative_ibi_only/
    conservative_ibi_nve_certification_report.json
```

A PASS certifies only `WCA + Morse + bonded conservative IBI` with PaiNN
disabled. An ML-active Hamiltonian requires a separate certification.


### 7.6.7 Regularized angular IBI: supported option, not a universal default

For IBI angles that have already been converted to `type="conservative_spline"`,
the framework supports an optional regularization path. This is an explicit
post-IBI prior transformation: **it is not an automatic `run_ibi_loop.py`
flag**, it is not a different integrator, and it does not relax any NVE
certification requirement.

The numerical motivation is that the IBI update

\[
\Delta U(\theta)=\alpha k_B T\ln\frac{P_{\mathrm{sim}}(\theta)}
                                      {P_{\mathrm{target}}(	heta)}
\]

can absorb small short-wavelength components from finite sampling or marginal
representability. A tiny energy perturbation on a short angular scale can create
a large `U''(theta)` and therefore artificially fast angular modes. Raw IBI is
still usable with a sufficiently small timestep; regularization is an optional
way to recover a wider clean timestep range while retaining the target
structure.

The implementation is `ibi/generate_angle_smoothing_candidate.py`. For each
angle table it:

1. analytically removes the configured quadratic endpoint wall;
2. Gaussian-smooths only the de-walled `U_body(theta)`;
3. restores the same wall and barrier;
4. constructs a C2 `CubicSpline` for the regularized energy;
5. exports energy and nodal derivatives into the same conservative Hermite
   runtime representation.

Bonds and dihedrals are not smoothed by this command. ESPResSo still evaluates
energy and force from the same Hermite polynomial.

Generic example:

```bash
python3 ibi/generate_angle_smoothing_candidate.py \
  --source-priors ibi_conservative/cg_priors.json \
  --ibi-config ibi_settings.json \
  --body-sigma-rad 0.0075 \
  --output-dir ibi_angle_candidate \
  --dry-run

python3 ibi/generate_angle_smoothing_candidate.py \
  --source-priors ibi_conservative/cg_priors.json \
  --ibi-config ibi_settings.json \
  --body-sigma-rad 0.0075 \
  --output-dir ibi_angle_candidate \
  --overwrite
```

The output is deliberately tagged `validated=false` and never modifies the
source priors. `body_sigma_rad` is system-specific: `0.0075 rad` is validated
for the current TEL22 model, not a transferable default.

Candidate selection must combine at least:

- structural angle/bond distributions and L1 deltas versus the unregularized
  prior;
- occupied `P95/P99/max |U''|` rather than table extrema alone;
- a **contiguous** `sigma_E/dt^2` plateau plus `sigma_E ~ dt^p`, rather than
  ranking solely by a global exponent close to 2;
- independent replicas and a longer structural validation before promotion.

Regularizing a prior changes the residual decomposition. Any residual dataset or
PaiNN model built before regularization becomes **stale for ML-active use** and
must be rebuilt/retrained before the network can be enabled again.

### 7.6.8 TEL22 validated `smooth_0p0075` path and post-promotion certification

For TEL22, diagnostics 29--32 localized the historical timestep restriction to
the IBI angles: the original harmonic-angle control retained quadratic scaling,
whereas the IBI-angle branches exhibited a much larger occupied curvature. The
local sweep selected `smooth_0p0075_wall_current` as the best structural/NVE
compromise.

Step 33 validates one candidate without further tuning:

```bash
cd tutorials/tel22_IBI
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./33_validate_final_ibi_angle_candidate.sh --overwrite
```

Validated TEL22 result:

```text
common sigma_E exponent p = 1.947046
within-replica R2          = 0.984844
full clean range           = 3/3 replicas through dt=0.005 ps
Delta weighted angle L1    = +0.009561
Delta weighted bond L1     = -0.019844
angle P99 |U''| reduction  = 2.416x
```

Only after that PASS does step 34 transactionally promote the candidate into the
production priors, preserve the previous prior set as a backup, and recertify the
**production path**:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./34_promote_and_certify_ibi_angle_prior.sh --promote
```

The current certification simultaneously requires finite-difference/parity,
preflight, energy scaling, and Richardson convergence. `sigma_E` is gating
again:

```text
fresh production sigma_E: p=1.877261, R2=0.984412,
                          C2 spread=1.487, max dt=0.005 ps, PASS
Richardson median p:      position=2.004, velocity=2.215,
                          orientation=2.035, omega_body=2.019, PASS
FINAL:                    pass=True, ML_active=False
```

The final certificate therefore covers only
`WCA + Morse + bonded conservative regularized IBI` with PaiNN disabled. The
pre-promotion residual dataset and PaiNN model are stale and are not part of the
certified Hamiltonian.

---

# 8. Pair-specific WCA: construction, guardrails, and subtraction

The repulsive Weeks-Chandler-Andersen interaction is the Lennard-Jones potential shifted at its
minimum:

\[
U_{\mathrm{WCA}}(r)
=
\begin{cases}
4\epsilon\left[\left(\frac\sigma r\right)^{12}
-\left(\frac\sigma r\right)^6\right]+\epsilon,
& r<r_c,\\
0,&r\ge r_c,
\end{cases}
\]

with

\[
r_c=2^{1/6}\sigma,
\qquad
\sigma=\frac{r_c}{2^{1/6}}.
\]

The radial force is

\[
F_r
=
\frac{24\epsilon}{r}
\left[
2\left(\frac\sigma r\right)^{12}
-\left(\frac\sigma r\right)^6
\right].
\]

The current recommended route is **pair-specific automatic WCA fitting**, enabled by

```json
"wca_sigma": "auto"
```

rather than one global hard-core diameter for every site type pair.

## 8.1 Topological exclusions — policy v3

The current policy separates the **molecule-level topological relation** from the **specific site pair that should lose the WCA core**:

- all virtual-site pairs within one rigid body are excluded;
- for a topological 1-2 pair with `exclude_wca=true`, only the explicitly bonded virtual-site pair(s) are excluded; all other cross-body site pairs retain WCA;
- a COM-COM 1-2 bond does not automatically suppress WCA for any virtual-site pair;
- explicit 1-3 angle endpoints retain the all-sites exclusion under policy v3;
- Morse defaults to `exclude_wca=false`, because it is typically a dissociative contact/restraint rather than covalent connectivity.

The policy is serialized in `cg_priors.json` and the runtime explicitly requires:

```text
policy_version = 3
direct_scope = bonded_site_pairs_only
one_three_scope = molecule_pair_all_sites
pair_source = explicit_topology_pairs_v3
```

Both `direct_pairs` and `direct_site_pairs` are stored. Runtime cross-checks those records against the bonded priors and rejects legacy or inconsistent prior files.

### Why 1-2 exclusions must be site-aware

Applying a 1-2 exclusion to **every** cross-site pair between two rigid bodies is generally too broad. A bonded `site0-site0` pair does not imply that, for example, `site2-site3` should be allowed to interpenetrate without repulsion. In the TEL22 case, the legacy all-sites 1-2 policy produced a reproducible short-range collapse of non-bonded sites on backbone-adjacent residues. An A/B run that retained WCA on all 1-2 cross-pairs except the actually bonded site pair removed that failure mode.

The correction must be **symmetric between preprocessing and runtime**. Changing exclusions only during MD changes the Hamiltonian relative to the priors subtracted from training targets. After a WCA-policy change, regenerate dataset/priors, retrain, re-equilibrate, and repeat NVE certification.

## 8.2 Automatic pair-specific fit

For each eligible site-type pair \((i,j)\), the builder collects distances using the **rigidified CG
geometry**, not arbitrary raw site coordinates.

A low distance quantile \(q_{ij}\) and the exact observed minimum \(r_{\min,ij}\) are tracked.

The code estimates type radii \(R_i\) by minimizing a weighted consistency objective of the form

\[
L_R
=
\sum_{ij}
w_{ij}
\left(R_i+R_j-q_{ij}\right)^2,
\]

with sample-count weighting approximately

\[
w_{ij}=\frac{N_{ij}}{N_{ij}+1000}.
\]

For sparsely sampled pairs the raw pair quantile is shrunk toward the additive type-radius estimate:

\[
\alpha_{ij}=\frac{N_{ij}}{N_{ij}+1000},
\]

\[
r_{c,ij}^{\mathrm{raw}}
=
\alpha_{ij}q_{ij}
+(1-\alpha_{ij})(R_i+R_j).
\]

The onset is then guarded so it does not exceed the low empirical support quantile.

Interpretation:

- large `N_ij`: the pair-specific empirical distance distribution dominates;
- small `N_ij`: the additive type-radius estimate stabilizes the fit;
- lower `wca_quantile_percent`: less invasive repulsion, but greater sensitivity to rare-tail
  statistics.

## 8.3 Physical-support guard

Define a guard distance

\[
r_{\mathrm{guard}}
=f_{\mathrm{guard}}r_c.
\]

The cutoff is constrained so this guard remains below the physically observed minimum by a margin:

\[
r_{\mathrm{guard}}
\le
m_{\mathrm{phys}}r_{\min}.
\]

Equivalently,

\[
r_c
\le
\frac{m_{\mathrm{phys}}r_{\min}}
     {f_{\mathrm{guard}}}.
\]

In configuration these quantities are controlled by:

- `wca_guard_fraction` -> \(f_{\mathrm{guard}}\);
- `wca_physical_guard_margin` -> \(m_{\mathrm{phys}}\).

This prevents WCA from imposing a hard core that contradicts configurations already present in the
training data.

## 8.4 Epsilon calibration

Rather than treating \(\epsilon\) as an arbitrary global constant, the current auto-fit calibrates the
barrier at the guard point:

\[
U(r_{\mathrm{guard}})
=
\texttt{wca_guard_kbt}\,k_BT.
\]

Thus `wca_guard_kbt` is the most direct physical control on repulsive barrier height in units of
thermal energy.

Increasing it increases \(\epsilon\), increases short-range forces, and can require a smaller MD
timestep.

### Note on `wca_epsilon` and `wca_overrides`

In the current pair-specific automatic path, generated `wca_pairs` are the authoritative runtime
policy. Legacy/global settings such as `wca_epsilon` and carried `wca_overrides` should not be
assumed to alter every auto-fitted pair unless the corresponding code path explicitly applies them.
For reproducible production, inspect the generated `cg_priors.json` rather than inferring the final
interaction from template defaults.

---

# 9. Residual force-matching target

Once the explicit prior is defined, preprocessing evaluates its generalized force and torque on the
same mapped configuration and stores the residual target:

\[
\mathbf F_m^{\mathrm{target}}
=
\mathbf F_m^{\mathrm{ref}}
-
\mathbf F_m^{\mathrm{prior}},
\]

\[
\boldsymbol\tau_m^{\mathrm{target}}
=
\boldsymbol\tau_m^{\mathrm{ref}}
-
\boldsymbol\tau_m^{\mathrm{prior}}.
\]

This means the dataset and priors form a coupled artifact pair. If WCA, bonded constants, rigid
geometry, or exclusions are changed, the old residual target is generally no longer correct.

The full runtime prediction is reconstructed as

\[
\mathbf F^{\mathrm{runtime}}
=
\mathbf F^{\mathrm{prior}}
+
\mathbf F^{\mathrm{ML}},
\]

and similarly for torque.

Force matching minimizes a conditional mean-square objective. A low training loss does not imply
that instantaneous atomistic forces are deterministic functions of the retained CG state. Eliminated
atomistic degrees of freedom can generate irreducible conditional fluctuations. Diagnostic nearest-
state half-pair MSE values used in the TEL22 development are therefore **noise proxies**, not rigorous
mathematical lower bounds on achievable force-matching error.

---

# 10. Implemented PaiNN architecture

The architecture variant currently accepted by the framework is

```text
painn_canonical_context_silu_v2
```

Each ML site has:

- a scalar feature \(\mathbf s_i\in\mathbb R^D\);
- an equivariant vector feature \(\mathbf v_i\in\mathbb R^{3\times D}\).

Scalar features are initialized from a learned species embedding. Vector features start at zero.

## 10.1 Embedding

For site type \(z_i\),

\[
\mathbf s_i^{(0)}=\mathrm{Embedding}(z_i),
\qquad
\mathbf v_i^{(0)}=0.
\]

`hidden_channels` sets \(D\).

## 10.2 Radial basis functions

Distances are expanded with Gaussian radial basis functions whose centers cover `[0, cutoff]`:

\[
\mathrm{RBF}_k(d)
=
\exp\left[-\frac{(d-\mu_k)^2}{\sigma_{\mathrm{rbf}}^2}\right]c(d).
\]

The width scale in the current implementation is tied to the cutoff and number of radial functions,
approximately

\[
\sigma_{\mathrm{rbf}}\sim\frac{r_c}{N_{\mathrm{RBF}}}.
\]

Increasing `num_rbf` improves radial resolution but increases radial-filter cost and parameter count.

## 10.3 Toxvaerd cutoff

To smoothly suppress interactions near the cutoff, define

\[
x=\frac{r_c-d}{r_c}.
\]

For \(d\le r_c\),

\[
c(d)
=
\frac{x^4}{x^4+\alpha^4},
\]

and \(c(d)=0\) outside the cutoff.

`toxvaerd_alpha` controls the width/sharpness of the switching region. It is part of the effective
architecture and must remain consistent between training, manifest, and runtime.

## 10.4 Message block

The scalar path applies an MLP of the schematic form

```text
Linear(D,D) -> SiLU -> Linear(D,3D)
```

while radial features are projected with

```text
Linear(num_rbf,3D,bias=False)
```

and multiplied with the learned context features. Messages are aggregated by summation over incoming
neighbors.

The graph is built with periodic minimum-image distances, and physical site pairs are represented in
both directions as required by the message-passing implementation.

## 10.5 Update block

The update block uses two learned linear transforms of vector features and a stabilized vector norm.
For a vector-channel tensor \(\mathbf v\), the norm is evaluated schematically as

\[
\|\mathbf v\|_\epsilon
=
\sqrt{\sum_{\alpha=x,y,z}v_\alpha^2+\epsilon},
\qquad
\epsilon=10^{-8}.
\]

Scalar/vector context is then processed by an MLP of the form

```text
Linear(2D,D) -> SiLU -> Linear(D,3D)
```

to update invariant and equivariant channels.

## 10.6 Readout

The scalar energy readout is

```text
Linear(D,D/2) -> SiLU -> Linear(D/2,1)
```

and site contributions are summed into the scalar ML energy.

`n_layers` controls how many message/update blocks are stacked. Increasing it increases effective
graph depth and compute/memory cost.

---

# 11. Network-predicted forces and torques

The network predicts a scalar energy. Site forces are obtained by differentiating this same energy
with respect to site coordinates:

\[
\mathbf f_i^{\mathrm{ML}}
=
-\frac{\partial U_{\mathrm{ML}}}{\partial\mathbf r_i}.
\]

This is not an independently trained vector head. Energy-force consistency is therefore built into
the architecture.

During force training, the backward graph must retain derivatives through the force computation;
`create_graph=true` is therefore required, making training effectively involve second derivatives of
energy with respect to coordinates and parameters.

Molecular force is obtained by summing site forces:

\[
\mathbf F_m^{\mathrm{ML}}
=
\sum_{i\in m}\mathbf f_i^{\mathrm{ML}}.
\]

For a multi-site body, the predicted torque is

\[
\boldsymbol\tau_m^{\mathrm{ML}}
=
\sum_{i\in m}
(\mathbf r_i-\mathbf R_m)
\times
\mathbf f_i^{\mathrm{ML}}.
\]

The trainer caches graph connectivity because dataset coordinates and the architecture cutoff are
fixed. This avoids rebuilding an \(O(N^2)\) candidate-pair search on every epoch.

The current graph policy excludes edges where the two sites belong to the same CG molecule. Thus
PaiNN does not learn the explicit bonded intramolecular prior a second time.

---

# 12. Loss function and normalization

Training force and torque scales are computed from the training subset:

\[
F_{\mathrm{RMS}}
=
\sqrt{\left\langle
(F^{\mathrm{target}})^2
\right\rangle},
\]

and, over multi-site bodies only,

\[
\tau_{\mathrm{RMS}}
=
\sqrt{\left\langle
(\tau^{\mathrm{target}})^2
\right\rangle}.
\]

The normalized force loss is

\[
L_F
=
\frac{\mathrm{MSE}(\mathbf F^{\mathrm{pred}},\mathbf F^{\mathrm{target}})}
     {F_{\mathrm{RMS}}^2}.
\]

The torque loss is

\[
L_\tau
=
\frac{\mathrm{MSE}_{\mathrm{multi-site}}
(\boldsymbol\tau^{\mathrm{pred}},\boldsymbol\tau^{\mathrm{target}})}
{\tau_{\mathrm{RMS}}^2}.
\]

The total objective is

\[
L
=
L_F
+\lambda_\tau L_\tau
+\lambda_{\mathrm{Lip}}L_{\mathrm{Lip}}.
\]

The optional force-norm regularizer is

\[
L_{\mathrm{Lip}}
=
\frac{\left\langle\|\mathbf f_i\|^2\right\rangle}
{F_{\mathrm{RMS}}^2}.
\]

Despite the configuration name `lipschitz_lambda`, this is a force-norm penalty; it is **not** a
rigorous estimate or enforcement of a global Lipschitz constant.

`torque_weight` is \(\lambda_\tau\). Setting it to zero activates a force-only path and avoids
unnecessary torque work.

The trainer also reports a validation **zero predictor** baseline. In normalized units, a model that
cannot extract meaningful signal should approach the error of predicting zero residual force/torque.
Comparison to this baseline is more informative than looking only at raw loss magnitude.

### Gradient clipping

`grad_clip_norm` bounds the global parameter-gradient norm before the optimizer step. It can protect
against rare unstable batches, but if most batches are clipped it changes the optimization regime.
When `report_grad_norms=true`, inspect mean, median/P50, P95, maximum, and clipped-batch fraction.

A useful interpretation is:

- rare clipping -> safety guard;
- nearly every batch clipped -> learning rate / scaling / architecture may be poorly conditioned.

---

# 13. Energy gauge and energy scale

Energy is defined only up to an additive constant as far as forces are concerned. The current model
uses the gauge

```text
isolated_species_zero_v1
```

which measures raw isolated-site offsets for each species and subtracts them so isolated species have
a controlled zero reference.

If

\[
U'(\mathbf r)=U(\mathbf r)-C,
\]

then

\[
-\nabla U'=-\nabla U.
\]

Thus this gauge changes energy reporting but not forces.

The model also stores an `energy_scale` registered buffer, set from the training force RMS in the
current training convention. It improves numerical conditioning/scale handling but does not alter the
physical definition that force is the coordinate gradient of energy.

Gauge, scale, and architecture metadata must be preserved in the model/manifest so C++ training,
export, and ESPResSo runtime agree.

---

# 14. Binary dataset format

The trainer consumes a compact binary format written by preprocessing.

Global header:

```text
int32  num_frames
```

For each frame:

```text
int32   num_molecules
int32   num_total_sites
float32 box[3]
```

For each molecule:

```text
int32   molecule_id
int32   num_sites
float32 center[3]
float32 target_force[3]
float32 target_torque[3]
```

For each site:

```text
int32   site_type
float32 position[3]
```

The reader performs safety checks including:

- finite numeric values;
- valid molecule IDs/counts;
- site-type range;
- consistency of total site count;
- no unexpected trailing bytes.

Because coordinates/targets are serialized as `float32`, harmless representational differences can
appear when values are later compared to decimal JSON configuration values. This is why manifest
validation uses a physically tight but `float32`-compatible tolerance for architecture floats.

---

# 15. Model manifest and provenance

A trained model should be accompanied by

```text
model.pt.manifest.json
```

The current manifest schema records information such as:

- framework identifier `MLCG_Framework_v2`;
- schema version;
- architecture variant and hyperparameters;
- cutoff and Toxvaerd alpha;
- energy gauge;
- effective training configuration;
- split/provenance metadata;
- SHA256 and file size of relevant artifacts such as model, dataset, and config.

The runtime validates the manifest before using a model. This protects against silent mismatches such
as changing `cutoff`, `num_species`, or architecture after training.

Architecture floats are compared using a tolerance suitable for a single-precision round trip,
approximately

```text
rel_tol = 1e-6
abs_tol = 1e-8
```

so a pair such as

```text
1.2616000175476074  vs  1.2616
```

is accepted, while physically meaningful changes remain rejected.

`--allow_missing_model_manifest` exists for legacy/diagnostic use. It should not be the normal
production path.

---

# 16. ESPResSo checkpoints

The current checkpoint (`equilibrated.npz`) is more than a coordinate snapshot. The schema stores
state and provenance information including, as applicable:

- particle positions;
- velocities;
- quaternions;
- body angular velocities;
- box;
- particle IDs and types;
- molecule IDs;
- architecture/runtime parameters;
- timestep and thermal parameters;
- hashes of the dataset/config/priors/rigid-body/model/manifest inputs.

This makes a production run reproducible and prevents accidentally restarting a state with a
different potential.

Runtime normally refuses mismatched provenance. Escape hatches include:

- `--allow_legacy_checkpoint` for an old schema;
- `--allow_checkpoint_mismatch` for an intentional diagnostic bypass.

Both options weaken reproducibility guarantees and should be documented whenever used.

---

# 17. Python environment and trainer build

## 17.1 Python environment

The preprocessing/runtime stack requires the packages used by the repository, notably the scientific
Python stack and MDAnalysis for reading atomistic trajectories. A typical environment is created with
whatever package manager is appropriate for the host, then verified with the framework tests.

The key operational requirement is that the Python interpreter used for preprocessing sees the same
MDAnalysis behavior expected by the unit-conversion code.

Run the Python tests from the repository root with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## 17.2 LibTorch trainer

The C++ trainer lives under `training/` and is built with CMake against LibTorch.

A typical build pattern is:

```bash
cd training
mkdir -p build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . -j
```

The resulting executables include the PaiNN trainer and diagnostics such as `eval_parity` when enabled
by the current CMake configuration.

Apple Silicon notes:

- training may select MPS when supported;
- CPU remains a valuable reference backend;
- NVE certification should prefer CPU because the smallest energy fluctuations can be obscured by
  lower-precision accelerator arithmetic.

---

# 18. PaiNN/Morse integration into ESPResSo

The custom runtime code under

```text
simulation/espresso_plugin/
```

must be copied into the matching ESPResSo source tree and ESPResSo rebuilt.

The helper

```bash
simulation/espresso_plugin/copy_plugin_files.sh
```

performs this synchronization. The current macOS-friendly version treats already-identical files as
`[SKIP]` rather than relying on `cp` behavior for same-source/destination paths.

After changing the PaiNN architecture header, C++ potential implementation, Python/Cython binding, or
switched Morse plugin files, **rebuild ESPResSo**. An old `pypresso` binary does not automatically pick up
new framework source files.

## 18.1 PaiNN plugin properties

The current plugin:

- evaluates only particles satisfying `type < num_species`;
- uses ESPResSo neighbor/PBC displacement information;
- deduplicates physical pairs/periodic aliases;
- evaluates a scalar PaiNN energy and its consistent forces;
- is certified for single-rank use in this workflow;
- supports CPU and accelerator backends, subject to precision/backend availability.

The system box must be large enough for the cutoff plus skin without ambiguous self images. A useful
runtime guard is

\[
L_{\min}>2(r_c+\text{skin}).
\]

The current scripts commonly use `skin = 0.4 nm`.

## 18.2 Dummy interaction for the neighbor list

A zero-strength SoftSphere interaction is used as a mechanism to make the PaiNN cutoff visible to
ESPResSo's nonbonded neighbor machinery. The critical current rule is:

```python
for i in range(num_species):
    for j in range(i, num_species):
        ... SoftSphere(a=0, ...)
```

It must apply **only to ML site types**.

Do not extend this loop to COM/dummy types. For a one-site molecule the sole ML virtual site lies
exactly on its COM. A singular SoftSphere functional form evaluated at `r=0` can produce `NaN` even
with nominal amplitude `a=0` (`0 * inf`/undefined arithmetic). This was the source of the observed
TEL22 step-zero terms

```text
('non_bonded', 0, 9): nan
('non_bonded', 1, 9): nan
```

and was fixed by restricting the dummy interaction to `0 .. num_species-1`.

## 18.3 Apple MPS runtime memory policy

The PaiNN bridge applies a device-specific policy when the effective runtime
device is Apple MPS. If `MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS` is unset, the
default is:

```text
emptyCache after every 100 successfully completed PaiNN force calls
```

The cache is emptied only after the internal force evaluation has returned, so
the temporary tensors and per-call autograd graph have already been destroyed.
`emptyCache()` can therefore release unused MPS allocator blocks, but not the
model, its weights, the MD state, or any live tensor. A force call is not
formally identical to one MD step: initialization and specialized workflows may
perform additional evaluations.

The environment variable remains an explicit override:

```bash
# Framework MPS default
unset MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS

# Disable periodic cache emptying
export MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS=0

# Select a custom MPS cadence
export MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS=50
```

The value must be a non-negative integer. CPU and CUDA do not apply this policy
and retain their previous behavior. Every MPS initialization logs the effective
cadence and its source, for example:

```text
[PaiNN] MPS diagnostic emptyCache cadence: 100 successful force calls (MPS default)
[PaiNN] MPS diagnostic emptyCache cadence: 0 successful force calls (environment override)
```

The value `100` was validated on the TEL22 runtime. In the matched 5000-step
comparison, peak physical footprint decreased from 27136 MiB (26.5 GiB) to
8704 MiB (8.5 GiB) without a measurable timing penalty. In the 20000-step
candidate run, RSS stabilized after the allocation ramp and the final footprint
was 9011 MiB (8.8 GiB), with transient peaks below the baseline. These
measurements motivate the framework default, but they are not a universal
guarantee for every model, PyTorch release, or machine. `emptyCache()` manages
unused memory; by itself it neither proves the absence of a leak nor fixes
numerical problems in a potential. An earlier per-force-call Objective-C
autorelease-pool experiment produced no material benefit and was removed; it is
not part of the current policy.

This policy applies to **PaiNN inference inside ESPResSo**. It is separate from
the trainer configuration option `mps_empty_cache_every_batches` described in
section 20.5. After changing the bridge, synchronize and rebuild ESPResSo; the
`train_painn` executable does not need to be rebuilt:

```bash
bash simulation/espresso_plugin/copy_plugin_files.sh
cmake --build espresso/build --parallel
```

The TEL22 diagnostics `25_test_mps_memory_growth.sh` and
`26_test_mps_empty_cache_ab.sh` measure a single policy and a controlled `0`
versus positive-cadence comparison, respectively. Their reports are diagnostic:
interpret ramp-up, plateau behavior, and sparse `vmmap` sampling rather than
relying only on a global regression.

---

# 19. `topology_config.json` reference

The chemistry-neutral template is

```text
preprocessing/topology_config.json
```

A system tutorial should provide its own copy.

## 19.1 `temperature`

Temperature in kelvin used for

\[
\beta=1/(RT),
\]

DBI constants, and WCA thermal calibration. It should represent the atomistic distribution from
which the training data were sampled.

## 19.2 `mapping.mapping_method`

Allowed values:

```text
COM
COG
ATOM
```

This controls retained-site geometry, not the mass-weighted dynamical body COM.

## 19.3 `mapping.residues`

Defines included residue/molecule names and the atom selections assigned to each CG site.

Example:

```json
"residues": {
  "MOL": {
    "CG_A": ["A1", "A2"],
    "CG_B": ["B1", "B2"]
  }
}
```

Residue types not represented in this map are not mapped as CG molecules.

## 19.4 `mapping.site_types`

Maps site names to integer PaiNN species. These values must agree with `num_species` in the training
configuration.

## 19.5 `prior_geometry.default_angle_site`

Selects the default site index used at angle vertices.

- `-1` means the COM;
- `>=0` means a **site index within the molecule**, not a PaiNN site type.

## 19.6 `bonds`

Common fields include:

```text
mol_i, mol_j
site_i, site_j
type
name
exclude_wca
```

`harmonic` and `fene` represent structural bonded connectivity. A `type="morse"` record instead denotes a **pair-specific Morse contact between explicit endpoints** that the runtime converts into a selective non-bonded interaction. Morse fields are:

```text
D, a, r0
r_switch   # optional
r_cut      # optional for pair-specific mode
```

For pair-specific Morse, `site_i/site_j=-1` (or omitted) select the COM, while values `>=0` select CG/virtual sites of the corresponding molecule. The runtime uses coincident technical virtual markers without changing the physical site types; `exclude_wca` defaults to `false`. The switched kernel is exactly zero at and beyond `r_cut`.

## 19.7 `morse_type_pairs`

Optional section for ordinary non-bonded Morse selected by physical CG site type:

```json
"morse_type_pairs": [
  {
    "type_i": 0,
    "type_j": 1,
    "D": 4.0,
    "a": 2.0,
    "r0": 0.55,
    "r_switch": 0.90,
    "r_cut": 1.20
  }
]
```

`type_i/type_j` must exist in `mapping.site_types`; duplicate unordered pairs are rejected. `r_cut` is mandatory. ESPResSo non-bonded exclusions also apply to these Morse interactions.

## 19.8 `angles`

Typical fields:

```text
mol_i, mol_j, mol_k
site_i, site_j, site_k
type = harmonic
k
theta0
name
exclude_wca
```

`theta0` is in radians.

## 19.9 `dihedrals`

Typical fields:

```text
mol_i, mol_j, mol_k, mol_l
site_i, site_j, site_k, site_l
type = cosine
k
n
phi0
name
```

## 19.10 `wca_sigma`

Recommended current value:

```json
"wca_sigma": "auto"
```

This activates the pair-specific WCA policy and writes explicit `wca_pairs` to generated priors.

## 19.11 `wca_quantile_percent`

Low percentile of physically sampled distances used to define repulsive support.

Smaller values:

- make WCA less invasive;
- rely more strongly on rare lower-tail samples;
- can become statistically noisy in short trajectories.

It must be in the valid low-percentile range accepted by preprocessing (conceptually between 0 and
50%).

## 19.12 `wca_guard_fraction`

Defines

\[
r_{\mathrm{guard}}
=f_{\mathrm{guard}}r_c.
\]

The neutral template uses approximately `0.8`. Smaller values calibrate barrier height deeper inside
the repulsive core.

## 19.13 `wca_guard_kbt`

Defines

\[
U(r_{\mathrm{guard}})
=
\texttt{wca_guard_kbt}\,k_BT.
\]

Larger values increase the repulsive energy/force scale and can reduce the maximum stable timestep.

## 19.14 `wca_physical_guard_margin`

Sets the margin relative to the exact minimum sampled distance. A value near `0.98` leaves a small
support margin while preventing the fitted WCA guard from invading known physical data.

## 19.15 `decoy_target_fraction`

The recommended default is

```text
0.0
```

Legacy decoy frames used an all-zero residual target without a molecule-level loss mask. They are
therefore disabled in the normal production training path.

## 19.16 `allow_unmasked_zero_target_decoys`

Keep `false` unless intentionally reproducing a legacy diagnostic/ablation.

## 19.17 `decoy_random_seed`

Random seed for legacy decoy generation if that path is explicitly enabled.

## 19.18 `rigid_bodies`

Optional per-residue rigid-body controls can include:

```text
auto_align_sites
sites.<name>.relative_pos_nm
```

With `auto_align_sites=true`, representative geometry is inferred from aligned data. With it disabled,
a complete explicit body geometry must be supplied.

---

# 20. Training configuration reference

The neutral template is

```text
training/cg_model_config.json
```

and TEL22 uses a tutorial-specific configuration such as

```text
tutorials/tel22/tel22_training_config.json
```

## 20.1 Architecture parameters

### `architecture_variant`

Must match the implemented/exported architecture, currently

```text
painn_canonical_context_silu_v2
```

Changing it is a model-definition change, not a tuning-only change.

### `num_species`

Number of learned site embeddings. Must satisfy

\[
\texttt{num_species}>
\max(\texttt{site_type}).
\]

Runtime COM/dummy types are intentionally outside this range.

### `hidden_channels`

Feature width \(D\).

Increasing it generally increases:

- model capacity;
- parameter count;
- activation memory;
- message/update compute cost.

It does not increase geometric cutoff.

### `n_layers`

Number of PaiNN message/update blocks. More layers increase message-passing depth and cost.

### `num_rbf`

Number of Gaussian radial basis functions. More functions provide finer radial resolution but
increase radial-filter cost.

### `cutoff`

PaiNN cutoff in nm. It controls graph density and the physical environment visible to the network.

Increasing it can:

- include more intermolecular context;
- increase edge count and memory strongly;
- require a larger simulation box to satisfy minimum-image guards.

It must match the trained model manifest at runtime.

### `toxvaerd_alpha`

Controls the smooth cutoff envelope. It changes radial weighting near `cutoff` and is therefore an
architecture-level parameter that must be reproduced at inference.

## 20.2 Optimization parameters

### `epochs`

Maximum number of passes over the training subset. Early stopping may terminate earlier.

Epoch count should be interpreted together with `batch_size`: the number of optimizer updates per
epoch changes as the batch size changes.

### `learning_rate`

Initial Adam/AdamW step scale used by the current trainer. Too high can destabilize second-derivative
force training; too low can make the model appear stuck.

### `weight_decay`

Parameter regularization applied by the optimizer. It is distinct from the force-norm regularizer.

### `batch_size`

Number of frames combined per optimizer batch.

Trade-offs:

- larger batch -> fewer optimizer updates per epoch, more memory, lower gradient noise;
- smaller batch -> more updates, lower per-batch memory, potentially noisier gradients.

### `grad_clip_norm`

Global norm threshold for gradient clipping. It should be interpreted using reported pre-clip gradient
statistics rather than tuned blindly.

### `reduce_lr_patience`

Number of plateau epochs before reducing the learning rate. The current scheduler reduces LR
multiplicatively (halving in the present implementation) until a small lower bound is reached.

### `early_stopping_patience`

Validation plateau length tolerated before stopping. Best-model weights are saved based on validation
behavior rather than blindly using the final epoch.

## 20.3 Loss parameters

### `torque_weight`

Sets \(\lambda_\tau\) in

\[
L=L_F+\lambda_\tau L_\tau+\lambda_{\mathrm{Lip}}L_{\mathrm{Lip}}.
\]

Set it to zero for a true force-only run. A nonzero value is meaningful only if the dataset contains
multi-site orientation-bearing bodies.

### `lipschitz_lambda`

Weight of the normalized force-norm penalty. It should be interpreted as a regularizer on force
magnitude, not as a proof of a Lipschitz bound.

## 20.4 Split and diagnostic parameters

### `diagnostic_overfit_frames`

Restricts training to a tiny number of frames for an intentional overfit diagnostic. This tests
implementation/capacity, not generalization.

### `physical_validation_only`

Controls validation selection policy for datasets containing diagnostic/nonphysical samples.
Production validation should represent the physical distribution of interest.

### `include_decoys_in_train`

Legacy/diagnostic control. Normal current training keeps unmasked zero-target decoys disabled.

### `validation_fraction`

Fraction of available physical frames assigned to validation for random split mode.

### `validation_split_mode`

Supported policies include conceptually:

```text
random
tail
```

A random split measures interpolation over shuffled states. A tail split is more sensitive to
temporal/configurational extrapolation when the trajectory is time ordered.

### `validation_tail_frames`

Explicit tail size when tail splitting is selected.

### `split_seed`

Reproducibility seed for random split/shuffle decisions.

### `shuffle_each_epoch`

Controls training-order reshuffling. It does not alter the cached geometrical graph for each frame.

## 20.5 Diagnostics and MPS

### `report_grad_norms`

Enables per-epoch summaries of pre-clip gradient norm, including central/upper quantiles and clipped
fraction. This is strongly recommended while tuning force training.

### `mps_empty_cache_every_batches`

Optional Apple MPS memory-management interval. More frequent cache emptying may reduce peak retained
memory at the cost of synchronization/allocator overhead.

The trainer backend selection is currently ordered roughly as CUDA, then MPS where available, then
CPU. Backend choice should be recorded because numerical behavior and performance differ.

---

# 21. Complete pipeline: scripts and parameters

For the TEL22 tutorial, the high-level workflow is

```text
01_run_gromacs.sh
      ↓
02_build_dataset.sh
      ↓
03_train_model.sh
      ↓
04_equilibrate.sh
      ↓
diagnostics/scripts/06_certify_nve.sh
      ↓
05_run_espresso.sh
```

NVE certification is deliberately placed before treating a model/runtime combination as a verified
conservative production setup.

For a different molecule/system, the same core stages apply but the atomistic preparation,
`topology_config.json`, training configuration, and artifact paths must be changed.

## 21.1 `01_run_gromacs.sh` — atomistic trajectory

Purpose: generate atomistic configurations and, critically, **atomistic forces** for force matching.

The TEL22 reference workflow performs approximately:

1. check that `gmx` is available;
2. obtain PDB 143D;
3. retain the first NMR model;
4. run `pdb2gmx` with AMBER99SB-ILDN and TIP3P;
5. insert 10 molecular copies into an 8 nm box;
6. update the topology molecule count;
7. solvate;
8. neutralize and add 0.15 M KCl;
9. energy minimization;
10. NVT equilibration;
11. NPT equilibration;
12. production MD;
13. verify that `nstfout > 0`;
14. run production `mdrun`;
15. verify `md.trr` exists;
16. make the trajectory whole with
    `gmx trjconv -pbc whole -force`;
17. run `gmx check`;
18. retain `md_whole.trr` and the compatible topology/structure (`md.gro`).

The full reference production used during framework development is of order 1 ns with force-bearing
frames roughly every 1 ps. A much shorter trajectory can test the mechanics of the pipeline but is
not evidence that the ML model has statistically converged.

### Important GROMACS/MDP parameters

Most physical parameters live in `mdp/*.mdp`, not as shell-script arguments.

Key fields include:

- `dt`: atomistic integration step;
- `nsteps`: duration;
- `nstxout`: coordinate output stride;
- `nstvout`: velocity output stride;
- `nstfout`: force output stride;
- thermostat/barostat settings;
- electrostatics and PME settings;
- constraints;
- nonbonded cutoffs.

For force matching,

```text
nstfout > 0
```

is mandatory. The final `gmx check` output should list at least

```text
Coords
Forces
Box
```

and velocities if the chosen workflow needs them.

### Download mirror note

If the primary structural download endpoint is unavailable, use an official wwPDB mirror rather than
changing the molecular structure. During the TEL22 smoke test, the PDBj wwPDB mirror was a working
fallback.

### Short smoke trajectory

A previously used smoke configuration used about 20 ps NVT + 20 ps NPT + 50 ps production with one
frame per ps, producing 51 force-bearing production frames. This is appropriate for verifying file
formats/build/runtime plumbing only; it is not adequate as a final scientific force-matching dataset.

## 21.2 `02_build_dataset.sh`

Wrapper around

```text
preprocessing/build_cg_dataset.py
```

Typical environment variables:

```bash
AA_TOPOLOGY=md.gro
AA_TRAJECTORY=md_whole.trr
PYTHON_BIN=python3
```

Usage:

```bash
cd tutorials/tel22
bash 02_build_dataset.sh
```

Outputs:

```text
tel22_dataset.bin
cg_priors.json
rigid_bodies_info.json
```

### Builder CLI

Conceptually:

```bash
python3 preprocessing/build_cg_dataset.py \
  --topology TOP.gro \
  --trajectory TRAJ.trr \
  --config topology_config.json \
  --output dataset.bin \
  --priors-output cg_priors.json \
  --rb-info-output rigid_bodies_info.json
```

Important parameters:

| Parameter | Meaning |
|---|---|
| `--topology`, `-c` | atomistic topology/coordinates readable by MDAnalysis |
| `--trajectory`, `-f` | atomistic trajectory containing forces |
| `--config`, `-j` | CG mapping/topology configuration |
| `--priors`, `-p` | reuse an existing prior file instead of re-inferring it |
| `--output`, `-o` | output binary training dataset |
| `--priors-output` | generated prior JSON path |
| `--rb-info-output` | generated rigid-body metadata path |
| `--clip_forces` | optional component-wise force clipping diagnostic/safety parameter |

`--clip_forces` changes the target distribution and therefore should not be enabled casually. If used,
the clipping threshold and motivation must be recorded.

### When to use `--priors`

Use it when rebuilding a dataset while requiring **exactly the same prior definition**. The supplied
file must include the explicit current WCA pair/exclusion policy and bonded parameters compatible with
the mapped geometry.

## 21.3 `03_train_model.sh`

Wrapper around the C++ trainer.

Typical override:

```bash
TRAINER=/path/to/train_painn
```

Default tutorial path is conceptually

```text
../../training/build/train_painn
```

Usage:

```bash
bash 03_train_model.sh
```

Equivalent trainer invocation:

```bash
training/build/train_painn \
    tel22_dataset.bin \
    tel22_model.pt \
    tel22_training_config.json
```

The output normally includes:

```text
tel22_model.pt
tel22_model.pt.manifest.json
cg_training_log.csv
```

### `--resume`

Resume is explicit:

```bash
train_painn dataset.bin model.pt config.json --resume
```

The trainer intentionally avoids silently overwriting an existing model. Resume also validates
compatible artifacts/configuration before continuing.

A resume run should use the same dataset/model definition unless a deliberately supported migration
path says otherwise.

## 21.4 `training/create_model_manifest.py`

Utility for creating or refreshing the sidecar manifest without retraining weights:

```bash
python3 training/create_model_manifest.py \
  --model model.pt \
  --config config.json \
  --dataset dataset.bin
```

Use it for a compatible artifact migration or to restore missing provenance metadata, not to hide an
actual architecture mismatch.

## 21.5 `training/eval_parity`

C++ diagnostic executable used to verify parity when changing components such as:

- architecture code;
- radial basis/cutoff behavior;
- periodic boundary handling;
- energy gauge;
- serialization;
- ESPResSo integration.

It is a code-parity test, not an NVE conservation test and not a physical-validation metric.

## 21.6 `simulation/espresso_plugin/copy_plugin_files.sh`

Copies the custom framework files into the configured ESPResSo source tree.

Run it after changes to files such as:

```text
PaiNN_Architecture.hpp
PaiNN_ML_Potential.*
painn.pyx
morse_switched.hpp / `install_switched_morse_nonbonded.py`
```

Then rebuild ESPResSo. Merely modifying the framework copy does not modify an already compiled
`pypresso` binary.

## 21.7 `04_equilibrate.sh`

Wrapper around `simulation/equilibrate.py`.

Typical environment:

```bash
PYRESSO=../../espresso/build/pypresso
DEVICE=auto
```

Usage:

```bash
PYRESSO=../../espresso/build/pypresso \
bash 04_equilibrate.sh
```

Output:

```text
equilibrated.npz
```

### `equilibrate.py` CLI

| Parameter | Default | Effect |
|---|---:|---|
| `--model` | required | PaiNN weights |
| `--config` | required | network config |
| `--priors` | required | runtime priors |
| `--rb_info` | required | masses/inertias/rigid geometry |
| `--dataset` | required | topology/initial mapped frame source |
| `--dt` | `0.002` ps | warm-up timestep |
| `--out_checkpoint` | `equilibrated.npz` | output checkpoint |
| `--device` | `auto` | `cpu`, `mps`, `cuda`, or `auto` |
| `--kT` | `2.49` | Langevin thermal energy |
| `--steps_sd` | `5000` | classical steepest-descent steps |
| `--steps_md` | `2000` | classical thermostatted MD steps |
| `--steps_ml_capped` | `2000` | ML-on capped warm-up steps |
| `--steps_ml_uncapped` | `2000` | final uncapped ML warm-up steps |
| `--warmup_chunk` | `100` | progress/force-cap ramp chunk size |
| `--toxvaerd_alpha` | config | runtime architecture override, manifest-checked |
| `--allow_missing_model_manifest` | false | legacy bypass |
| `--allow_unsafe_mpi` | false | bypass single-rank guard |

Architecture overrides must remain consistent with the trained model/manifest.

## 21.8 `05_run_espresso.sh`

Production wrapper around `simulation/run_cg_md.py`.

Typical variables:

```bash
PYRESSO=../../espresso/build/pypresso
DEVICE=auto
CG_STEPS=20000
CG_DT=0.001
```

Usage:

```bash
PYRESSO=../../espresso/build/pypresso \
CG_DT=0.001 \
CG_STEPS=20000 \
bash 05_run_espresso.sh
```

The simulated physical time is

\[
t_{\mathrm{sim}}
=
\texttt{CG_STEPS}\times\texttt{CG_DT}.
\]

For the example defaults,

\[
20000\times0.001=20\ \mathrm{ps}.
\]

Choose `CG_DT` based on stability/conservation evidence, not only on speed.

## 21.9 `simulation/run_cg_md.py`

Current runtime arguments include:

| Parameter | Meaning |
|---|---|
| `--model` | PaiNN model; may be omitted for a classical-only diagnostic |
| `--config` | architecture configuration |
| `--priors` | runtime prior JSON |
| `--rb_info` | rigid-body metadata |
| `--dataset` | topology/initial metadata source |
| `--checkpoint` | state checkpoint to load |
| `--dt` | integration timestep |
| `--steps` | number of integration steps |
| `--log_interval` | normal output/log stride |
| `--device` | `cpu/mps/cuda/auto` |
| `--kT` | Langevin thermal energy when thermostat is used |
| `--init_kT` | initialize velocities from a Maxwell-Boltzmann distribution |
| `--nve` | disable thermostat and run conservative VV dynamics |
| `--no_log` | disable optional normal logging |
| `--no_vtf` | disable VTF trajectory output |
| `--energy_file` | explicit energy CSV path |
| `--trajectory_file` | explicit trajectory path |
| `--toxvaerd_alpha` | manifest-checked runtime override |
| `--allow_missing_model_manifest` | legacy bypass |
| `--allow_legacy_checkpoint` | accept legacy checkpoint schema |
| `--allow_checkpoint_mismatch` | intentionally bypass provenance mismatch |
| `--allow_unsafe_mpi` | bypass single-rank certification guard |
| `--allow_nonconservative_tables` | explicitly permit tabulated priors in NVE diagnostics |

The conservative current path uses analytic Morse, so Morse itself is not considered a
nonconservative table.

At NVE startup the runtime explicitly uses velocity Verlet, sets force cap to zero, turns the
thermostat off, and performs a zero-step force recalculation before recording the initial state.

### `--init_kT`

For translational components,

\[
v_\alpha
\sim
\mathcal N\left(0,\frac{kT}{M}\right).
\]

For principal-axis angular velocities,

\[
\omega_\alpha
\sim
\mathcal N\left(0,\frac{kT}{I_\alpha}\right).
\]

When production follows equilibration, loading the checkpoint velocities is normally preferable to
reinitializing them.

### Safety diagnostics

The energy logger records quantities such as:

```text
Step
Time_ps
E_tot
E_kin
E_kin_trans
E_kin_rot
E_class   (legacy name for ESPResSo total)
E_ml
min_dist
min_pair
min_pids
f_max
torque_max
```

The runtime fails fast when any ESPResSo energy component or PaiNN energy is non-finite. This is
important because a step-zero `NaN` is a potential-definition/initial-state problem, not timestep
drift.

## 21.10 `diagnostics/scripts/06_certify_nve.sh`

Wrapper for the dedicated NVE timestep-scaling test.

Current recommended environment defaults are

```bash
PYRESSO=../../espresso/build/pypresso
NVE_DTS="0.001 0.002 0.005 0.01"
NVE_DURATION_PS=5.0
```

Recommended invocation:

```bash
NVE_DTS="0.001 0.002 0.005 0.01" \
NVE_DURATION_PS=5.0 \
PYRESSO=../../espresso/build/pypresso \
bash diagnostics/scripts/06_certify_nve.sh --overwrite
```

The certifier requires at least three positive timestep values. All timestep cases cover the same
physical duration, and energy is sampled every integration step for the current `sigma_E` protocol.

`--overwrite` allows existing per-timestep output directories to be regenerated.

## 21.11 `simulation/certify_nve.py`

The orchestrator:

1. validates inputs and timestep list;
2. hashes relevant input artifacts;
3. rejects explicitly nonconservative/tabulated priors unless deliberately overridden;
4. defaults to CPU for certification;
5. converts each physical duration to an integer number of steps;
6. runs `run_cg_md.py --nve` for every timestep;
7. requests energy output every step;
8. loads each energy series;
9. computes per-run metrics;
10. fits the timestep power law;
11. writes JSON/CSV reports;
12. returns a nonzero exit status for numerical/subprocess failure.

The certification task is about **integration/conservative consistency**, not whether the learned
potential is physically accurate.

## 21.12 `simulation/nve_analysis.py`

This module evaluates the energy series.

Primary fluctuation metric:

\[
\sigma_E
=
\sqrt{\frac1N\sum_{n=1}^{N}
(E_n-\bar E)^2}.
\]

The older

\[
\mathrm{RMS}(E-E_0)
\]

is retained only as a diagnostic because it depends on a single initial reference value and is more
sensitive to oscillation phase/offset.

The module also computes a relative early-vs-late block drift and the log-log power-law fit described
in the NVE section below.

## 21.13 `tutorials/plot_metrics_cg.py`

Plotting/inspection utility for training metrics. It can be used to inspect quantities such as:

- train and validation loss;
- normalized force/torque components;
- MAE/error trends;
- overfitting behavior.

It does not alter model parameters or define the force field.

---

## 21.14 `preprocessing/geometry_utils.py` — internal module, not a CLI

This module is not invoked directly from the shell. It centralizes numerically testable geometry primitives used by preprocessing, including:

- symmetric inertia-tensor diagonalization and principal-moment sorting;
- correction to a right-handed principal-axis basis;
- orthorhombic minimum-image displacements;
- minimum-image distance matrices.

With principal axes stored as columns of matrix \(\mathbf A\) in the space-fixed frame, the convention is

\[
\mathbf r_{\mathrm{body}}
=
\mathbf A^T\mathbf r_{\mathrm{space}}.
\]

Changing this convention requires simultaneous verification of rigid-site geometry, runtime quaternion/orientation handling, and torque.

---

## 21.15 `simulation/framework_utils.py` — shared validation and provenance

This is also an internal module rather than a command-line stage. It is imported by equilibration and production and centralizes guardrails such as:

- model-manifest schema and validation;
- PaiNN architecture and energy-gauge identifiers;
- SHA256 hashing of artifacts;
- validation of explicit 1–2/1–3 WCA exclusion policy;
- construction of runtime excluded molecule-pair sets;
- checkpoint save/validation with particle signatures;
- configuration, hash, box, and restart-state consistency checks.

Manifest architecture floats use the current `float32`-compatible tolerance (approximately `rel_tol=1e-6`, `abs_tol=1e-8`), while strings, integers, and booleans remain exact comparisons.

These checks should stay centralized rather than being reimplemented differently in shell wrappers, `equilibrate.py`, and `run_cg_md.py`.

---

# 22. CG equilibration in detail

The current equilibration intentionally introduces the force field in stages rather than starting an
uncapped ML simulation from a potentially imperfect mapped configuration.

## Phase 1 — Classical steepest descent

PaiNN is not yet active. The current representative settings are approximately:

```text
f_max = 10000
gamma = 50
max_displacement = 0.001 nm
steps = steps_sd (default 5000)
```

Purpose:

- remove severe bonded/WCA overlaps;
- avoid asking the ML potential to resolve obviously bad initial geometry;
- make subsequent velocity-Verlet integration safer.

`max_displacement` controls how far a minimization update can move a particle in one iteration.

## Phase 2 — Classical thermostatted velocity Verlet with force cap

The classical potential is integrated with a Langevin thermostat and an initially finite force cap.
Representative current settings use strong damping (`gamma` and rotational `gamma` around 50) and a
cap that is gradually relaxed/ramped over chunks.

`steps_md` controls the total classical MD warm-up length and `warmup_chunk` controls the granularity
of progress/cap changes.

Physical interpretation:

- strong damping removes excess kinetic energy quickly;
- a cap protects against a single extreme force during early relaxation;
- the cap is not a physical production interaction and must eventually be removed.

## PaiNN activation

After classical relaxation, the PaiNN plugin is initialized with the trained model and the same
architecture values recorded in the manifest.

This is where a manifest mismatch must be treated as a configuration error rather than silently
ignored.

## Phase 3 — ML-on thermostatted MD with force cap

The full classical + ML force field is activated under a temporary cap and strong damping.
`steps_ml_capped` controls the duration.

This phase allows the system to adapt to residual forces learned by PaiNN without immediately exposing
the integrator to an arbitrarily large force spike.

## Phase 4 — ML-on uncapped thermostatted MD

The final warm-up sets

```text
force_cap = 0
```

and uses much weaker damping (representatively around `gamma=1`, including rotational damping) for
`steps_ml_uncapped` steps.

A production checkpoint should be generated only after this uncapped stage has completed and forces
have been recalculated.

### When to lengthen equilibration

Increase relevant stages if diagnostics show:

- persistent large `f_max` or torque peaks;
- close-contact collapse;
- violent temperature transients;
- geometry still moving systematically after the cap is removed;
- large changes between capped and uncapped behavior.

Do not lengthen equilibration simply to conceal a fundamentally unstable prior or ML potential.

---

# 23. CG production in detail

Production typically starts from `equilibrated.npz` and uses `run_cg_md.py` through
`05_run_espresso.sh`.

For thermostatted production, velocity Verlet and Langevin dynamics evolve the state with the chosen
`kT`, translation/rotation masses, priors, and PaiNN residual force.

For NVE, the thermostat is disabled and no force cap is allowed.

Important numerical controls are:

- `dt`: integration step;
- `steps`: simulation duration through `steps * dt`;
- `device`: backend and arithmetic precision;
- `log_interval`: normal monitoring frequency;
- output frequency/trajectory size;
- box size relative to cutoff + skin.

A timestep that merely avoids crashing is not necessarily a good timestep. Use the NVE scaling test
to identify a conservative/asymptotic regime and then choose a production value with margin.

## 23.1 Energy logging

For diagnostic runs, inspect not only total energy but also:

- kinetic partition into translation and rotation;
- ESPResSo/classical contribution;
- ML energy;
- minimum nonbonded distance and involved types/IDs;
- maximum force;
- maximum torque.

A finite total energy can occasionally hide compensating pathologies in components; conversely,
component cancellation is exactly what a correct conservative system should exhibit as kinetic and
potential energies exchange.

In a healthy NVE trajectory,

\[
\Delta K
+
\Delta U_{\mathrm{prior}}
+
\Delta U_{\mathrm{ML}}
\]

should remain small compared with the individual energy exchanges.

---

# 24. NVE certification

NVE certification answers a narrow but important question:

> Are the runtime forces and energies sufficiently conservative and is velocity Verlet showing the
> expected timestep convergence for the complete CG force field?

It does **not** answer whether PaiNN reproduces atomistic thermodynamics or kinetics accurately.

## 24.1 Current protocol

Use CPU and a fixed physical duration for every timestep. The current recommended grid spans one
decade without entering very small `float32`-dominated steps:

```text
dt = 0.001 ps
dt = 0.002 ps
dt = 0.005 ps
dt = 0.010 ps
```

with

```text
T = 5.0 ps
```

for each case.

The number of integration steps is therefore:

| dt (ps) | steps in 5 ps | energy samples including step 0 |
|---:|---:|---:|
| 0.001 | 5000 | 5001 |
| 0.002 | 2500 | 2501 |
| 0.005 | 1000 | 1001 |
| 0.010 | 500 | 501 |

Energy is saved **at every step**. Sampling only a handful of energies per run gives a poor estimate
of fluctuation variance and can distort the fitted order.

Steps much below 0.001 ps are not automatically better for this diagnostic: once truncation error is
small enough, single-precision/roundoff and energy-evaluation noise can flatten the observed scaling.

The largest point, 0.01 ps, is deliberately useful as a stress point but may already be outside the
stable/asymptotic regime for a stiff system.

## 24.2 Primary metric

For each timestep, compute the population standard deviation of total energy:

\[
\sigma_E(\Delta t)
=
\sqrt{
\frac1N
\sum_{n=1}^{N}
(E_n-\bar E)^2
}.
\]

For a second-order symplectic method such as velocity Verlet in the asymptotic regime, the bounded
energy error is expected to scale approximately as

\[
\sigma_E
=C\,\Delta t^p,
\qquad p\approx2.
\]

Fit in log space:

\[
\log\sigma_E
=
\log C
+p\log\Delta t.
\]

The fit reports:

- slope/order `p`;
- prefactor `C`;
- coefficient of determination \(R^2\);
- adjacent timestep/error ratios.

## 24.3 Drift as a separate diagnostic

A bounded symplectic energy oscillation and a secular drift are different phenomena. The certifier
therefore also compares early and late energy blocks.

For example, using the first and last 20% of samples,

\[
D_{\mathrm{rel}}
=
\frac{|\langle E\rangle_{\mathrm{late}}
-\langle E\rangle_{\mathrm{early}}|}
{E_{\mathrm{scale}}}.
\]

The exact characteristic energy scale is defined by `nve_analysis.py`; the default certification
threshold used in this framework state is of order

```text
1e-4
```

for the relative block drift.

A trajectory can have small standard deviation but still drift systematically; therefore both
metrics matter.

## 24.4 Fit thresholds

The original certification thresholds, retained as practical defaults unless explicitly changed,
are approximately:

```text
1.7 <= p <= 2.3
R2 >= 0.97
relative block drift <= 1e-4
```

These are engineering acceptance windows around ideal second-order behavior, not universal theorems
for every system size and precision.

If the largest timestep is visibly outside the asymptotic regime, inspect a fit over the stable
subset before concluding that the potential is nonconservative. Conversely, do not simply delete a
bad point without a physical/numerical reason.

---

# 25. Quantitative interpretation of NVE scaling

### `p ≈ 2`, high R², low drift

This is the expected signature of second-order velocity-Verlet energy error and strongly supports
conservative consistency of the implemented force field over the tested state region.

### `p < 2` only at the smallest timestep

Likely causes include:

- arithmetic roundoff/precision floor;
- ML energy accumulation noise;
- insufficient fluctuation amplitude relative to numerical noise.

This is one reason the current standard grid stops at `dt=0.001 ps` rather than automatically pushing
to `0.0005 ps` or below.

### Largest timestep strongly off the power law

`dt=0.01 ps` may be outside the asymptotic/stable regime, especially with stiff angles/WCA/Morse
terms. This suggests a practical upper timestep limit; it does not by itself prove an energy-force
inconsistency.

### `NaN` at step 0

This is **not** an integration-order problem. No trajectory has yet been advanced. Decompose
`system.analysis.energy()` and identify the exact non-finite term.

The TEL22 case exposed nonbonded type pairs involving the COM dummy and was fixed by excluding COM
types from the zero-strength SoftSphere neighbor-list interaction.

### Drift grows but `sigma_E` scales correctly

Investigate a small systematic bias, thermostat/cap accidentally active, incomplete force
recalculation, precision, or a force contribution not derived from the reported energy.

### Force blows up / minimum distance collapses

This is more likely a stability/physical-support issue than a subtle integrator-order issue. Inspect
WCA support, bonded singularities, ML extrapolation, and the equilibrated starting state.

### Historical short-run result

Before adopting the per-step `sigma_E` protocol, the corrected runtime already produced an older
`RMS(E-E0)` scaling result around

```text
p = 2.147
R2 = 0.9993
drift_pass = true
```

for a shorter set of timesteps. That result is useful supporting evidence, but the current standard
certification should use the fixed-duration, per-step `sigma_E` procedure above.

---

# 26. Tests and consistency checks

## 26.1 Python tests

Install the test dependency once in the active Python environment:

```bash
python3 -m pip install -r requirements-test.txt
```

From the repository root:

```bash
python3 -m pytest -q
```

`pytest.ini` sets `testpaths = tests`, so the no-argument command and
`python3 -m pytest -q tests` have the same scope. This prevents accidental
collection of an optional bundled `espresso/` source tree, whose upstream tests
require the ESPResSo build/test environment and are not part of the MLCG
framework suite.

Run this after changing preprocessing, runtime utilities, NVE analysis, or guardrails.

## 26.2 Whitespace/patch sanity

```bash
git diff --check
```

No output means no whitespace errors detected. A warning such as

```text
new blank line at EOF
```

can be fixed by ensuring exactly one final newline:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("simulation/nve_analysis.py")
p.write_text(p.read_text().rstrip() + "\n")
PY
```

## 26.3 Plugin/PBC regression

When modifying the ESPResSo plugin, run the plugin's PBC/pair-deduplication regression tests before
production. The exact command depends on the local ESPResSo build tree, but the purpose is to verify:

- minimum-image displacement;
- no duplicated physical edge;
- single-rank expected behavior;
- cutoff consistency.

## 26.4 Morse smoke test

After rebuilding ESPResSo, run `tutorials/tel22/diagnostics/scripts/08_diagnose_morse_reversibility.sh --assert-expected`. The diagnostic verifies `U(r_cut)=F(r_cut)=0`, cutoff crossing without exceptions, and re-entry/rebinding. Unit tests additionally cover `morse_type_pairs` parsing and runtime configuration.

## 26.5 Runtime site-site Morse force/torque diagnostic

After changing pair-specific Morse endpoint or virtual-marker logic, also run:

```bash
cd tutorials/tel22
bash ./diagnostics/scripts/09_diagnose_morse_site_torque.sh --assert-expected
```

The test is independent of TEL22: it constructs two synthetic rigid bodies, each with an off-COM physical CG virtual site, creates the technical markers through the same production helpers, and activates one `site<->site` pair-specific Morse interaction. It compares the ESPResSo energy, COM forces, and lab-frame torques against the analytic prediction `tau=(r_site-r_COM) x F`. It also checks marker/site coincidence and verifies that the physical CG-site types are unchanged. This is the runtime certification of the generic `COM/site` endpoint path beyond TEL22's current COM-COM contacts.

## 26.6 Prior audit

After dataset generation, inspect the generated prior file. A useful quick audit is:

```bash
python3 - <<'PY'
import json, math
from collections import Counter

with open("cg_priors.json") as f:
    p = json.load(f)

print("bonds:", len(p.get("bonds", [])))
print("bond types:", Counter(x.get("type", "harmonic") for x in p.get("bonds", [])))
print("morse type pairs:", len(p.get("morse_type_pairs", [])))
print("angles:", len(p.get("angles", [])))
print("dihedrals:", len(p.get("dihedrals", [])))

bad = []
def walk(x, path="root"):
    if isinstance(x, dict):
        for k, v in x.items(): walk(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): walk(v, f"{path}[{i}]")
    elif isinstance(x, float) and not math.isfinite(x):
        bad.append((path, x))
walk(p)
print("nonfinite:", bad)
PY
```

Also inspect WCA pair counts/exclusions and verify no accidental tabulated prior remains before strict
NVE certification.

---

# 27. TEL22 tutorial: current configuration

TEL22 is a tutorial/example, not a core hard-coded chemistry.

The currently documented tutorial system has approximately:

```text
10 DNA copies
22 residues per copy
~82 retained CG sites per copy
~820 ML sites per frame
8 PaiNN ML site types
```

Representative current PaiNN settings:

```text
architecture_variant = painn_canonical_context_silu_v2
hidden_channels      = 64
n_layers             = 2
num_rbf              = 32
cutoff               = 1.2616 nm
learning_rate        = 0.001
batch_size           = 4
torque_weight        = 0.5
grad_clip_norm       = 1
```

The short 51-frame smoke dataset generated during the recent pipeline recovery produced priors with:

```text
390 bonds
  210 harmonic
  180 Morse
200 angles
0 dihedrals
0 tabulated priors
```

These counts are useful topology sanity checks, but **51 frames are not a scientifically sufficient
training trajectory**. For real model quality, regenerate a substantially longer representative
atomistic dataset.

The PaiNN conditional-force diagnostics performed during framework development showed substantial
instantaneous unresolved force fluctuations. Temporal averaging reduced the diagnostic noise proxy,
but those nearest-state half-pair metrics are not rigorous lower bounds. NVE certification should not
be interpreted as resolving this model-accuracy question: it verifies conservation/integration, not
force predictability.

## 27.1 Architectural rule: every model-dependent choice is configurable

The IBI/conservative path explicitly separates three classes of information:

- **CORE_INVARIANT**: universal properties of the method, implemented in generic
  code and covered by tests. Examples include `F=-grad(U)`, periodic torsional
  splines, the conservative table schema, the IBI update formula, conservative
  validation and runtime/preprocessing parity.
- **MODEL_PARAMETER**: any choice that can change with molecule, mapping, dataset,
  temperature, Hamiltonian or sampling protocol. It must come from external
  configuration.
- **CALIBRATED_PARAMETER**: a `MODEL_PARAMETER` selected by a sweep or diagnostic
  study, such as a smoothing width or accepted NVE window. It must be configured
  together with calibration provenance.

The TEL22 IBI model-dependent workflow configuration is:

```text
tutorials/tel22_IBI/model_dependent_workflow_config.json
```

Workflow wrappers in steps 11–39 that contain model-dependent decisions load it
through `tutorials/tel22_IBI/model_config.sh`. Step 20 only installs the generic
ESPResSo kernel and therefore has no model-dependent section. The external config
contains bonded/dihedral grouping, `ibi`/`dbi` choices, mixing, histogram/support
settings, sampling and seed policy, regularization/candidate sweeps, replica
counts, timestep grids, fine/coarse windows and validation thresholds.

Validate a workflow config before running it:

```bash
python3 simulation/model_dependent_config.py validate \
  --config tutorials/tel22_IBI/model_dependent_workflow_config.json
```

A different model can supply a different file without editing the generic core:

```bash
IBI_MODEL_DEPENDENT_CONFIG=/path/to/my_model_workflow_config.json \
  bash tutorials/tel22_IBI/diagnostics/scripts/38_test_conservative_in_loop_dihedral_ibi.sh --run
```

Environment overrides remain available, but the model-config provenance sidecar (`model_config_provenance*.json`)
records whether each resolved value came from the model config or from an
`environment_override`. Explicit `ibi_settings.json` files are **authoritative
and complete**: production workflows no longer silently merge missing
model-dependent fields from internal defaults. Missing required settings fail as
configuration errors. Low-level Python helper defaults remain only for API/test
or exploratory use; production/tutorial wrappers supply explicit model values.

TEL22 values such as `sigma_angle=0.0075 rad`, IBI mixing, replica counts, seeds,
structural thresholds and NVE grids are therefore TEL22 configuration/results,
not properties of the method. The JSON `calibration_provenance` block links
calibrated values to the diagnostic/validation step that selected them.

---

# 28. Troubleshooting

## 28.1 `Model manifest mismatch: cutoff`

Example harmless mismatch:

```text
manifest=1.2616000175476074
runtime=1.2616
```

This can arise from a `float32` round trip. Current manifest validation uses approximately

```text
rel_tol=1e-6
abs_tol=1e-8
```

for architecture floats. Do not replace this with a loose tolerance large enough to hide a real
cutoff change.

## 28.2 `E_tot = NaN` at step 0

If `E_ml` is finite but the ESPResSo total is not, enumerate all entries of

```python
system.analysis.energy()
```

and print every non-finite key.

For the TEL22 issue the bad terms were nonbonded pairs involving types `(0,9)` and `(1,9)`. Type 9 was
the COM dummy; one-site ML particles lay exactly on that COM. Restricting the dummy SoftSphere loop to
ML types fixed the problem.

The key diagnostic principle is:

```text
NaN at step 0 -> inspect potential/state definition first
NaN after integration -> then investigate timestep/dynamic instability
```

## 28.3 NVE rejects tabulated priors

Strict certification is intended for a conservative energy/force path. Explicit independently
tabulated energy/force priors are rejected by default.

Use analytic supported interactions when possible. The current Morse implementation is analytic and
should not require `--allow_nonconservative_tables`.

That override exists only for intentional diagnostic work and invalidates a strict conservation
claim.

## 28.4 `At least three energy samples are required`

The run produced too few logged energy points. Under the current standard protocol, energy is sampled
every step and 5 ps runs provide hundreds to thousands of samples, so this should not occur.

For an ad-hoc short diagnostic, increase duration and/or reduce the logging stride.

## 28.5 Low fitted NVE `p` from a very short run

Do not infer nonconservativity from a fit based on a handful of energy samples or only a few hundredths
of a ps. Use the fixed 5 ps protocol and sample every step.

Also inspect whether the smallest timestep has reached a precision floor and whether the largest
point has left the asymptotic regime.

## 28.6 Box too small

If

\[
L_{\min}\le2(r_c+\mathrm{skin}),
\]

the minimum-image neighbor interpretation is unsafe for the plugin's assumptions. Increase the box,
reduce cutoff where physically justified, or change the simulation design.

Do not bypass the guard simply to make the run start.

## 28.7 Very large residual forces

Potential causes include:

- force-unit mismatch;
- incorrect atom selection/mapping;
- inconsistent prior subtraction;
- short/noisy atomistic statistics;
- ML extrapolation;
- training instability;
- a physically under-resolved CG state.

Check the zero-predictor baseline and train/validation behavior before increasing network size.

## 28.8 Torque on one-site molecules

One-site bodies have no retained orientation. The torque loss should exclude them. If they appear in
a torque objective, inspect the multi-site mask and rigid-body metadata.

## 28.9 MPI / multiple ranks

The current plugin workflow is certified for single rank. `--allow_unsafe_mpi` is an explicit bypass,
not evidence of multi-rank correctness.

Use a single ESPResSo rank for reference training/production/NVE validation until a dedicated
multi-rank pair/energy regression has been established.

## 28.10 MPS versus CPU

MPS is useful for throughput where supported, but strict energy scaling can reach fluctuations small
enough for single-precision arithmetic to matter. Use CPU for NVE certification and compare backends
before making a precision-sensitive claim.

If an MPS run shows physical-footprint growth, first check the
`MPS diagnostic emptyCache cadence` line in the runtime log. Section 18.3
documents the runtime policy, its overrides, and the matched diagnostics.

---

## 28.11 Diagnose the actual NVE trajectory order

When NVE drift is tiny but `sigma_E(dt)` is non-monotonic, do not infer the
Velocity-Verlet order from the energy-oscillation amplitude alone. In the
conservative-IBI workflow run step 25:

```bash
cd tutorials/tel22_IBI
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./diagnostics/scripts/25_diagnose_conservative_ibi_state_convergence.sh --overwrite
```

The test reuses the provenance-bound IBI-only NVT checkpoint from step 23 and
keeps PaiNN disabled. The default NVE ladder is dyadic: `0.001, 0.0005,
0.00025, 0.000125 ps`, with a `0.0000625 ps` reference trajectory. Runs stop at
`0.096 ps` and share exact samples every `0.012 ps`.

`run_cg_md.py --state_sample_npz` stores real-particle IDs, positions, velocities,
quaternions and `omega_body`, plus input hashes, Hamiltonian mode and the source
checkpoint SHA256. Postprocessing compares both each finite-dt trajectory with
the finest reference and Richardson `dt`/`dt/2` pairs. The latter determine the
order: a second-order method should satisfy `error(dt,dt/2) ~ dt^2`. Separate
fits are reported for position, velocity, orientation and angular velocity.
Position differences use the minimum-image convention and quaternion-angle
errors are invariant under `q -> -q`.

Step 25 remains diagnostic and never rewrites the historical step-23 result.
Its report is consumed by the historical step-26 composite together with
kernel/parity, provenance, and drift. Its purpose is to separate trajectory
order from an energy-scaling anomaly. A Richardson PASS does **not** justify
ignoring a reproducibly non-quadratic `sigma_E(dt)` law; localize that
discrepancy before certifying the prior.

## 28.12 Localize non-quadratic `sigma_E(dt)` in conservative IBI

For conservative IBI, use tutorial steps 27--29 to distinguish spline/kernel
issues from a changed bonded frequency scale. If the original analytic-angle
control retains `sigma_E ~ dt^2`, bond-only IBI remains acceptable, and the
angle-only/full-IBI branches lose the clean timestep range together with a large
increase in occupied `|U''|`, treat angle stiffness as the primary numerical
restriction.

At that point there are two valid choices:

- keep the raw conservative IBI angle and use a timestep inside its demonstrated
  second-order regime; or
- use the optional regularized-angle path in section 7.6.7, validate one
  candidate structurally and across NVE replicas, promote it explicitly, and
  require the fresh step-34 `sigma_E = O(dt^2)` gate on the production path.

Do not use the historical step-26 non-gating treatment of `sigma_E` as a final
production certification.

---

# 29. Checklist for adapting the framework to a new system

1. **Atomistic reference**
   - equilibrate the intended physical state;
   - write coordinates, box, and forces at a useful cadence;
   - verify trajectory force units explicitly.

2. **Mapping**
   - define retained residue/molecule types;
   - choose `COM`, `COG`, or `ATOM` site construction intentionally;
   - assign deterministic site-type integers;
   - ensure `num_species` covers every type.

3. **Rigid bodies**
   - decide which molecules are one-site vs multi-site;
   - verify principal inertias are positive/meaningful;
   - inspect Kabsch-averaged geometry;
   - confirm one-site position coincides with COM.

4. **Bonded priors**
   - choose harmonic/FENE/Morse/angle/dihedral definitions based on physical topology;
   - audit automatically inferred constants;
   - avoid singular parameter ranges near sampled configurations.

5. **WCA**
   - prefer explicit auto-generated pair-specific `wca_pairs`;
   - inspect quantiles and exact minima;
   - inspect topological exclusions;
   - verify `r_guard` remains inside physical support;
   - understand `wca_guard_kbt` as a barrier-height control.

6. **Dataset**
   - reject NaN/Inf;
   - confirm reference force is not accidentally zero;
   - inspect residual target RMS scales;
   - use a trajectory long enough for the intended scientific claim.

7. **Training configuration**
   - set correct `num_species` and cutoff;
   - choose capacity (`hidden_channels`, `n_layers`, `num_rbf`) based on evidence;
   - use a reproducible validation split;
   - include torque only for orientation-bearing bodies;
   - monitor gradient clipping rather than assuming its threshold is harmless.

8. **Training diagnostics**
   - compare validation error with the zero predictor;
   - inspect train-validation gap;
   - use tiny-overfit only as an implementation/capacity check;
   - do not interpret a conditional-noise proxy as a rigorous theoretical floor.

9. **Manifest**
   - generate/preserve it;
   - do not bypass architecture mismatches in normal production.

10. **ESPResSo build**
    - copy plugin files after every relevant source change;
    - rebuild `pypresso`;
    - keep analytic Morse enabled;
    - restrict dummy SoftSphere interactions to ML types;
    - use a single rank for the certified reference path.

11. **Equilibration**
    - pass through classical minimization/warm-up;
    - introduce ML under a temporary cap;
    - complete a final uncapped stage;
    - save the provenance-checked checkpoint.

12. **NVE certification**
    - CPU;
    - same 5 ps duration for each timestep;
    - energy every step;
    - use the current `0.001–0.01 ps` grid;
    - fit `sigma_E ~ dt^p` and expect a second-order regime;
    - inspect drift independently;
    - distinguish a high-dt instability from a low-dt roundoff floor.

13. **Production**
    - choose `dt` inside the verified stable/conservative regime with margin;
    - monitor minimum distances, force/torque maxima, and energy;
    - preserve config/model/prior/checkpoint hashes with results.

---

## Minimal TEL22 command sequence

```bash
cd tutorials/tel22

# 1. Atomistic trajectory
bash 01_run_gromacs.sh

# 2. Build CG dataset + priors + rigid-body metadata
bash 02_build_dataset.sh

# 3. Train PaiNN
bash 03_train_model.sh

# 4. Equilibrate CG system
PYRESSO=../../espresso/build/pypresso \
bash 04_equilibrate.sh

# 5. Conservative NVE certification
NVE_DTS="0.001 0.002 0.005 0.01" \
NVE_DURATION_PS=5.0 \
PYRESSO=../../espresso/build/pypresso \
bash diagnostics/scripts/06_certify_nve.sh --overwrite

# 6. Production example
CG_DT=0.001 \
CG_STEPS=20000 \
PYRESSO=../../espresso/build/pypresso \
bash 05_run_espresso.sh
```

The sequence above is a TEL22 example. For a generic system, replace the system-specific mapping and
configuration files and pass the corresponding artifacts explicitly to `build_cg_dataset.py`,
`train_painn`, `equilibrate.py`, and `run_cg_md.py`.
