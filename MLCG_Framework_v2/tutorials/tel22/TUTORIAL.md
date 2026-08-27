# TEL22 reference example

For the equations, sign conventions, units, and parameter sensitivities used by
the framework, see [`../../MATHEMATICAL_REFERENCE_EN.md`](../../MATHEMATICAL_REFERENCE_EN.md)
or the [Italian version](../../MATHEMATICAL_REFERENCE.md).

This directory is an **application example**, not part of the MLCG core API.
No code under `preprocessing/`, `training/`, `simulation/` or `tests/` imports
TEL22 files.

The repository intentionally does not bundle atomistic trajectories, GROMACS
working files, trained weights or diagnostic/ablation outputs.

Artifact cleanup and retention policy are documented in `../TEL22_CLEANUP.md`.
Use `bash ../cleanup_tel22_artifacts.sh --dry-run` before deleting local
outputs; destructive cleanup is never the default. Prepare an AA
trajectory containing forces separately, then run:

```bash
AA_TOPOLOGY=md.gro AA_TRAJECTORY=md_whole.trr bash 02_build_dataset.sh
bash 03_train_model.sh
PYRESSO=/path/to/pypresso DEVICE=auto bash 04_equilibrate.sh
PYRESSO=/path/to/pypresso DEVICE=auto bash 05_run_espresso.sh
```

Inputs kept under version control:

- `tel22_topology.json`: TEL22-specific mapping and priors configuration;
- `tel22_training_config.json`: one TEL22 training profile;
- scripts `02`-`05`: thin wrappers around the generic framework commands.

Generated files such as `tel22_dataset.bin`, `cg_priors.json`,
`rigid_bodies_info.json`, `*.pt`, manifests, checkpoints and trajectories are
runtime artifacts and are intentionally excluded from the source tree.

## Apple MPS runtime memory

When `DEVICE=mps`, or when `DEVICE=auto` selects Apple MPS, the PaiNN ESPResSo
bridge empties unused MPS allocator blocks after every 100 successful force
calls by default. CPU and CUDA runs are unchanged. The startup log records the
effective policy:

```text
[PaiNN] MPS diagnostic emptyCache cadence: 100 successful force calls (MPS default)
```

No environment variable is required for the validated default. To disable it
for a controlled comparison, or to select another cadence:

```bash
# Disable on MPS
MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS=0 \
PYRESSO=/path/to/pypresso DEVICE=mps bash 05_run_espresso.sh

# Example custom cadence
MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS=50 \
PYRESSO=/path/to/pypresso DEVICE=mps bash 05_run_espresso.sh
```

The default `100` was selected from TEL22 matched memory diagnostics and a
20000-step NVT validation. It reduces retained MPS memory substantially, but it
does not change forces, energies, checkpoints, or the Hamiltonian: cache release
occurs only after per-call tensors are dead. Strict NVE certification still uses
CPU by default because accelerator precision, rather than allocator retention,
is the relevant concern there.

After changing the bridge source, synchronize it into ESPResSo and rebuild only
ESPResSo:

```bash
bash simulation/espresso_plugin/copy_plugin_files.sh
cmake --build espresso/build --parallel
```

The trainer executable is unaffected. For memory regression or allocator A/B
tests, use `diagnostics/scripts/25_test_mps_memory_growth.sh` and
`diagnostics/scripts/26_test_mps_empty_cache_ab.sh`.

## NVE energy-conservation certification

After equilibration, certify the complete conservative Hamiltonian from the same
`equilibrated.npz` checkpoint at several Velocity-Verlet time steps:

```bash
PYRESSO=../../espresso/build/pypresso bash diagnostics/scripts/06_certify_nve.sh
```

The wrapper defaults to CPU, uses the same physical duration for every
timestep, and samples total energy at **every integration step**. The current
default grid is
`dt = 0.001, 0.002, 0.005, 0.01 ps` with `5 ps` per run. It writes
`nve_certification/nve_certification_report.json` and returns a non-zero exit
status unless both conditions pass:

1. the population standard deviation of total energy follows
   `sigma_E = std(E_tot) ~ dt^p` with the configured exponent/r-squared
   guardrails (defaults: `1.7 <= p <= 2.3`, `R^2 >= 0.97`);
2. the difference between the mean total energy in the final and initial 20%
   blocks is below the configured relative threshold (default `1e-4`).

`RMS(E_tot-E_tot(0))` is retained only as a secondary diagnostic and is not the
quantity fitted for the certification order. The `0.01 ps` point is a useful
stress point and may lie outside the asymptotic regime for a stiff model; steps
much below `0.001 ps` can instead become roundoff-limited.

Useful overrides:

```bash
NVE_DURATION_PS=10 \
NVE_DTS="0.001 0.002 0.005 0.01" \
NVE_MAX_RELATIVE_DRIFT=1e-5 \
PYRESSO=../../espresso/build/pypresso \
bash diagnostics/scripts/06_certify_nve.sh --overwrite
```

The certification path refuses explicitly tabulated bonded priors. Analytic
Morse, harmonic, FENE, harmonic-angle and cosine-dihedral priors remain allowed.

## 6b. Diagnose a short-range contact

If an NVE/production run approaches the short-range safety guardrail, diagnose the
actual minimum-distance PID pair before changing WCA parameters or exclusions:

```bash
bash diagnostics/scripts/06b_diagnose_short_range.sh
```

By default this reads `nve_certification/dt_0p005/energy.csv`, uses a 0.20 nm
short-range threshold, and writes
`nve_certification/short_range_diagnostic_0p005.json`. Override the selected run
or threshold with, for example:

```bash
NVE_DT_TAG=0p002 SHORT_RANGE_THRESHOLD_NM=0.25 bash diagnostics/scripts/06b_diagnose_short_range.sh
```

The diagnostic reconstructs the same runtime particle-ID order as
`simulation/run_cg_md.py`, maps the closest PIDs to molecule/site/type, classifies
the molecule pair as `1-2`, `1-3`, or `nonbonded` using the explicit lists in
`cg_priors.json`, and reports the nominal pair-specific WCA energy and force at
the observed distance. Under policy v3 a `1-2` contact is excluded only when the
specific virtual-site pair is explicitly bonded; other `1-2` site pairs retain
WCA. For an excluded site pair the nominal WCA values are reported as
counterfactual. This is a post-processing test only and does not modify the
Hamiltonian.

## WCA topology policy v3: selective 1-2 exclusions

The production WCA decomposition now uses the same site-aware rule in dataset
construction, equilibration, and MD:

- intra-rigid-body virtual-site pairs are excluded;
- for topological 1-2 molecule pairs, only the explicitly bonded virtual-site
  pair(s) are excluded from WCA; all other cross-body site pairs retain WCA;
- topological 1-3 molecule pairs retain the all-sites exclusion.

This changes the Hamiltonian decomposition relative to older `cg_priors.json`
files (`explicit_topology_pairs_v2`).  Old priors are intentionally rejected.
After applying this patch, regenerate `tel22_dataset.bin`, `cg_priors.json`, and
`rigid_bodies_info.json` with `02_build_dataset.sh`, retrain with
`03_train_model.sh`, and create a new `equilibrated.npz` with
`04_equilibrate.sh` before running production MD or NVE certification.

## TEL22 WCA v3 validation history

The site-aware 1-2 policy was introduced after a reproducible short-range failure under the legacy molecule-level exclusion. The closest pair was a non-bonded `CG_DG_B2/CG_DG_B3` site pair belonging to adjacent backbone residues: the actual backbone bond was `site0-site0`, but the old all-sites 1-2 exclusion also removed WCA from the B2/B3 cross-pair. Runs at different timesteps reached essentially the same short-range event at the same physical time, which identified it as a Hamiltonian/topology problem rather than a Velocity-Verlet instability.

A selective A/B runtime test retained WCA on 1-2 cross-pairs while excluding only the explicit bonded site pair. It passed beyond the previous failure time. The permanent v3 implementation then applied that same decomposition in preprocessing, equilibration, and production. After rebuilding priors/dataset, retraining, and re-equilibrating, a 2 ps smoke trajectory completed successfully without the old 1-2 collapse.

The lesson is general: **do not infer an all-sites nonbonded exclusion from a molecule-level bonded relation when the CG objects are multi-site rigid bodies**. Store and apply the exclusion at the same resolution as the actual bonded coordinate.

## Morse contacts and TEL22 unfolding

TEL22 uses harmonic site-0 bonds and harmonic backbone angles for covalent-chain connectivity, while the G-quartet tertiary contacts are represented by **pair-specific reversible Morse contacts**. Each 22-residue copy contains 18 such contacts: three groups of four guanines with six pair contacts per group. There are no dihedral priors.

The TEL22 Morse records remain in `bonds` with `type="morse"`, but at runtime they are **not ESPResSo bonded interactions**. TEL22 currently selects explicit COM-COM endpoints (`site_i=site_j=-1`). The generic framework also accepts COM-site and site-site endpoints. Runtime pair specificity is implemented with coincident technical virtual markers attached to the selected rigid bodies, so physical CG-site types seen by PaiNN/WCA are never changed. The hybrid cell system is selected before these long-cutoff marker interactions are registered; otherwise ESPResSo would validate the 15 nm marker cutoff against the default regular decomposition and reject the TEL22 box before the N-square routing becomes active. For a site endpoint, the marker occupies exactly that site position and its force is transferred to the parent body with the corresponding torque. The validated switched form has `U(r0)=-D`, smoothly reaches `U=F=0` at `r_cut`, can cross the cutoff without an exception, and can re-enter/rebind without topology changes. The old `MorseBond` broken-bond behavior is retained only as a diagnostic regression path.

Before regenerating a production checkpoint after changing the pair-specific marker machinery, the framework also provides `diagnostics/scripts/09_diagnose_morse_site_torque.sh --assert-expected`. Unlike the TEL22 model itself, this diagnostic creates a synthetic `site<->site` contact with both endpoints displaced from their rigid-body COMs. It verifies the switched-Morse energy, equal-and-opposite translational forces, the two analytic rigid-body torques, marker/site coincidence, and preservation of the physical CG site types. It therefore exercises the generic site-addressable path that TEL22's current COM-COM contacts do not cover.

TEL22 does **not** enable `morse_type_pairs` by default. That optional generic mechanism is intended for transferable bead-type attractions and would act on every non-excluded virtual-site pair carrying the selected CG types. Enabling it in TEL22 would add those energies/forces on top of the existing pair-specific quartet contacts, so it should only be done deliberately and requires regenerating dataset/priors, retraining the residual model, re-equilibrating, and repeating NVE certification.

Because the TEL22 pair-specific contacts select COM endpoints, their Morse forces act at the COMs. More generally, pair-specific marker sites do not create or consume WCA exclusions: WCA remains attached to the physical CG sites even when a Morse endpoint is one of those sites. WCA v3 therefore remains independent, so tertiary contacts can dissociate while short-range excluded volume stays active. Marker particles are technical (`type >= num_species+2`) and appear in ESPResSo checkpoints/VTF output; TEL22 analysis that intends to inspect only physical CG sites should filter to `type < num_species`. Any change in the pair-specific endpoint list changes the particle inventory and requires a new equilibration checkpoint.

The current TEL22 parameters are intentionally strong and broad:

```text
D = 50 kJ/mol
a = 0.3 nm^-1
1/a = 3.33 nm
```

At 300 K, one contact depth `D` is about 20 `kBT`. Six pair contacts in one idealized quartet plane therefore correspond to an energetic scale of roughly `6D = 300 kJ/mol` before entropy, the ML residual, backbone priors, and other interactions are considered. Unfolding is allowed by the topology, but it may still be rare on short trajectories; `D`, `a`, and the switching range must be calibrated to the intended thermodynamics and kinetics.

Do not define unfolding as "ESPResSo bond deletion". For TEL22, use geometric/contact-state observables (and, when needed, free-energy or kinetic analysis). Crossing `r_cut` simply means that the switched Morse contribution is zero until the pair re-enters the interaction range.


## Artifact cleanup and deduplication audit

See `../TEL22_CLEANUP.md`. Before sharing files between `tel22` and `tel22_IBI`,
run `python3 tutorials/audit_tel22_dedup.py` from the repository root. The
phase-2 audit computes SHA256/reference evidence and is non-destructive.
