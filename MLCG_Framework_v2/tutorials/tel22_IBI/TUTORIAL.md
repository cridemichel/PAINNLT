# TEL22 reference example

This directory is an **application example**, not part of the MLCG core API.
No code under `preprocessing/`, `training/`, `simulation/` or `tests/` imports
TEL22 files.

The repository intentionally does not bundle atomistic trajectories, GROMACS
working files, trained weights or diagnostic/ablation outputs. Prepare an AA
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

## NVE energy-conservation certification

After equilibration, certify the complete conservative Hamiltonian from the same
`equilibrated.npz` checkpoint at several Velocity-Verlet time steps:

```bash
PYRESSO=../../espresso/build/pypresso bash 06_certify_nve.sh
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
bash 06_certify_nve.sh --overwrite
```

The certification path refuses explicitly tabulated bonded priors. Analytic
Morse, harmonic, FENE, harmonic-angle and cosine-dihedral priors remain allowed.

## 6b. Diagnose a short-range contact

If an NVE/production run approaches the short-range safety guardrail, diagnose the
actual minimum-distance PID pair before changing WCA parameters or exclusions:

```bash
bash 06b_diagnose_short_range.sh
```

By default this reads `nve_certification/dt_0p005/energy.csv`, uses a 0.20 nm
short-range threshold, and writes
`nve_certification/short_range_diagnostic_0p005.json`. Override the selected run
or threshold with, for example:

```bash
NVE_DT_TAG=0p002 SHORT_RANGE_THRESHOLD_NM=0.25 bash 06b_diagnose_short_range.sh
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

Before regenerating a production checkpoint after changing the pair-specific marker machinery, the framework also provides `09_diagnose_morse_site_torque.sh --assert-expected`. Unlike the TEL22 model itself, this diagnostic creates a synthetic `site<->site` contact with both endpoints displaced from their rigid-body COMs. It verifies the switched-Morse energy, equal-and-opposite translational forces, the two analytic rigid-body torques, marker/site coincidence, and preservation of the physical CG site types. It therefore exercises the generic site-addressable path that TEL22's current COM-COM contacts do not cover.

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

## TEL22 bonded IBI sandbox

`tutorials/tel22_IBI` is a dedicated experimental copy of the TEL22 tutorial.
The certified analytic-prior baseline remains in `tutorials/tel22`; the IBI
sandbox deliberately keeps its generated `cg_priors.json` unchanged and writes
all inversion inputs/outputs under separate names.

The first TEL22 IBI experiment replaces only the **named harmonic backbone
bonds and angles** with tabulated bonded priors.  The 180 pair-specific Morse
contact records, WCA parameters/exclusions, mapping and rigid-body definition
are not selected and must remain unchanged.  There are currently no TEL22
dihedral priors.

### 1. Prepare an explicit IBI seed

After `02_build_dataset.sh` has generated `tel22_dataset.bin`, `cg_priors.json`
and `rigid_bodies_info.json`, create a separate seed:

```bash
bash 10_prepare_ibi_seed.sh
```

`ibi_selection.json` enumerates every selected pooled group and its expected
entry count.  `prepare_ibi_seed.py` fails closed if a group is missing, changes
count, or is no longer harmonic.  The expected first TEL22 selection is:

- 210 backbone bond entries in five pooled groups;
- 200 backbone angle entries in six pooled groups;
- 180 Morse records left untouched.

The output is `cg_priors_ibi_seed.json`; stale harmonic parameters (`k`, `r0`,
`theta0`) are removed only from the selected entries so they cannot be mistaken
for active terms.

### 2. Build DBI tables without running dynamics

Before the first iterative run, generate the Direct Boltzmann Inversion tables
only:

```bash
bash 11_build_dbi_preview.sh
```

This reads **coordinates only** from `tel22_dataset.bin`; the residual-force
columns of that dataset are not used for DBI.  Therefore the target dataset may
be the already generated analytic-prior dataset.  The preview is written to
`ibi_dbi_preview/` and should be inspected for support ranges and pathological
tails before launching NVT sampling.  Replacing an existing preview requires
`OVERWRITE=1`.

### 3. Iterative IBI convergence run

After the one-iteration validation has completed successfully, the sandbox
defaults to a less noisy convergence probe:

```bash
OVERWRITE=1 bash 12_run_ibi.sh
```

Defaults are now `IBI_ITERATIONS=5`, `alpha=0.15`, `dt=0.0005 ps`, 2 ps burn-in
and **16 ps sampled production per iteration**, with link-cell/hybrid neighbor
search.  The output directory defaults to `ibi_run_16ps/`, so the earlier
8-ps validation run under `ibi_run/` is preserved for comparison.

The script also runs `summarize_ibi_convergence.py`.  It reports the unweighted
mean L1 across pooled `type=ibi` groups for every sampled iteration and copies
the best **evaluated** prior set into:

```text
ibi_run_16ps/best/cg_priors.json
```

The distinction between an update number and an evaluated prior set matters.
Sampling iteration `i` is generated with the priors recorded in its
`source_priors`, and only afterwards is update `i` written.  Consequently, an
L1 printed for sampling iteration 5 evaluates (normally) `iteration_004`, not
the newly written `iteration_005`.  The final post-update priors are therefore
not automatically treated as the best set unless they are sampled in a later
iteration.  `ibi_convergence_summary.json` records this mapping explicitly.

Do not infer convergence from the last iteration alone.  Inspect both the
group-wise L1 values and the target/sampled distributions, together with the
stability of the resulting tables.  `alpha=0.15` remains deliberately
conservative while this sampling-length test is performed.

### 4. Continue from the best evaluated prior set

After the first 5 x 16 ps convergence block, continue from the best evaluated
set instead of restarting from the DBI seed or from the unevaluated final
update:

```bash
OVERWRITE=1 bash 14_continue_ibi.sh
```

The continuation defaults to five additional iterations with the same
`alpha=0.15`, `dt=0.0005 ps`, 2 ps burn-in and 16 ps production.  It reads
`ibi_run_16ps/best/cg_priors.json`, copies that state into an immutable
`resume_start/`, and writes new sampling/update files under
`ibi_run_16ps_continue/`.  The sampling iteration offset is inferred from the
parent report, so the new samples are numbered 6--10 and use non-overlapping
velocity/thermostat seeds.

The convergence summarizer combines the parent and continuation reports before
choosing `ibi_run_16ps_continue/best/cg_priors.json`.  The first continuation
sample therefore re-evaluates the previous best set with a new stochastic
trajectory before applying another update.  This preserves the distinction
between an evaluated potential and a post-update potential that has not yet
been sampled.

Override `IBI_PARENT_DIR`, `IBI_OUTDIR`, `IBI_ITERATIONS` or
`IBI_ITERATION_OFFSET` only deliberately.  A continuation run fails closed if
the supplied resume priors are not already tabulated entries carrying
`ibi_mode=ibi/dbi`; it never silently performs a new DBI inversion.

### 5. Rebuild the residual force-matching dataset

Once an IBI result has been selected, the PaiNN residual target must be rebuilt
with those exact evaluated tables:

```bash
bash 13_rebuild_residual_dataset.sh
```

After Phase 2 has passed, the script preferentially selects
`ibi_conservative/cg_priors.json`; otherwise it retains the historical fallback
to `ibi_run_16ps_continue/best/cg_priors.json` and then
`ibi_run_16ps/best/cg_priors.json`.  It writes
`tel22_dataset_ibi_residual.bin`.  Override `IBI_PRIORS` explicitly only when a
different evaluated iteration is intended.  The rebuild is mandatory before
training a new residual PaiNN model; using the old force-matching dataset would
double-count the replaced bonded prior contribution.

## Independent read-only validation of the selected best IBI priors

After the continuation run has selected `ibi_run_16ps_continue/best/cg_priors.json`, validate that exact evaluated prior set with a fresh NVT trajectory and independent random seeds:

```bash
OVERWRITE=1 bash ./15_validate_best_ibi.sh
```

This step does **not** update any IBI table.  The validator hashes `cg_priors.json` and every referenced tabulated table before and after the ESPResSo run and fails if any source artifact changes.  It reports the same pooled L1 distribution metrics used by the IBI loop and compares the independent validation mean against the historical best in `ibi_run_16ps_continue/ibi_convergence_summary.json`.

The default validation sampling uses the same 2 ps burn-in and 16 ps production window as the converged IBI runs, but different velocity/thermostat seeds (`271828` and `161803`).  Only after this independent validation is consistent with the selected best set should `13_rebuild_residual_dataset.sh` be used to freeze the priors into the residual force-matching dataset.

## Residual-build provenance and training preflight

After the read-only validation, rebuild the residual dataset with:

```bash
bash ./13_rebuild_residual_dataset.sh
```

The rebuild also writes `ibi_residual_build_manifest.json`.  This manifest
cryptographically binds the residual dataset and `rigid_bodies_info_ibi.json`
to the exact selected priors, every referenced prior table, the atomistic
topology/trajectory and the mapping configuration used by preprocessing.  Two
validation modes are supported: the legacy read-only NVT validation of explicit
tabulated IBI priors, and the Phase-2 conservative validation.  In conservative
mode the manifest additionally binds both `validation_report.json` and the
persisted ESPResSo/preprocessing `runtime_parity_report.json`.

Before training, run the fail-closed preflight explicitly if desired:

```bash
bash ./16_check_ibi_training_inputs.sh
```

It must finish with `[PASS]`.  `03_train_model.sh` runs the same preflight
automatically and trains on `tel22_dataset_ibi_residual.bin`.  With Phase 2
artifacts present, the selected prior must resolve to
`ibi_conservative/cg_priors.json`, and the manifest must bind the corresponding
`validation_report.json` and `runtime_parity_report.json` as well as every
referenced conservative spline table.

The default model name is `tel22_model_ibi.pt`; `tel22_model.pt` is always
protected. Training also fails closed if its selected output already exists,
preventing an accidental implicit resume. If the residual target has changed --
for example after converting the selected IBI priors to the conservative spline
representation -- do not resume an older checkpoint. Start a new model artifact:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./03_train_model.sh
```

The C++ PaiNN trainer consumes the residual dataset and training config directly;
the rigid-body metadata and priors are checked here because they define the
matching runtime Hamiltonian and must remain bound to the dataset used for
training. The resulting `${IBI_MODEL}.manifest.json` binds the model to the
training dataset/config hashes.

## Paired multi-seed baseline vs post-IBI training benchmark

A single 41/10 random train/validation split is too small to decide whether a
small change in best normalized validation loss is systematic.  After the
post-IBI model has trained successfully, run a paired multi-seed benchmark:

```bash
OVERWRITE=1 bash ./17_benchmark_training_multiseed.sh
```

The default seeds are `11 42 73`.  For a stronger five-seed check use:

```bash
MULTISEED_SEEDS="11 23 42 73 101" OVERWRITE=1 \
  bash ./17_benchmark_training_multiseed.sh
```

For every seed, the generic `training/multiseed_benchmark.py` trains both the
pre-IBI `tel22_dataset.bin` case and the post-IBI
`tel22_dataset_ibi_residual.bin` case with the same base training configuration
except for `split_seed`.  Each run gets its own config, model, manifest and log
under `training_multiseed_benchmark/<case>/seed_<N>/`; existing benchmark output
is never reused unless `OVERWRITE=1` is supplied deliberately.

Before any post-IBI benchmark run starts, the wrapper repeats
`16_check_ibi_training_inputs.sh`, so a stale residual dataset or changed IBI
table cannot silently enter the comparison.  The benchmark writes
`benchmark_runs.csv` and `benchmark_summary.json`, reports per-case mean/sample
standard deviation of the best validation loss, and computes the paired
`IBI - baseline` delta at each identical seed.  A negative paired delta favors
the post-IBI case.  Interpret the mean paired delta together with its spread and
win count rather than a single seed.

## 18. Validate the complete post-IBI runtime Hamiltonian

After the validated IBI priors have been used to rebuild the residual dataset and
the matching residual PaiNN model has been trained, run the runtime gate with the
same model name used during training. For the default model:

```bash
OVERWRITE=1 bash ./18_validate_postibi_runtime.sh
```

For a distinct conservative-residual artifact:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  OVERWRITE=1 bash ./18_validate_postibi_runtime.sh
```

The script fails closed unless the model manifest proves that the model was
trained on the exact residual dataset/config selected at runtime and the
residual-build manifest proves that the same dataset, validated IBI priors
(including every referenced table), and rigid-body metadata belong together.
It then creates a provenance-bearing checkpoint with the complete `IBI + PaiNN`
Hamiltonian and runs a short NVT smoke trajectory.

Outputs are written under `postibi_runtime_validation/`:

- `runtime_preflight.json`: cryptographic model/dataset/config/prior/RB provenance;
- `equilibrated_postibi.npz`: checkpoint whose own metadata carries the runtime hashes;
- `nvt_energy.csv` and `nvt_smoke_report.json`: finite force/torque and stability checks;
- `nvt_sample.npz` and `runtime_structure_report.json`: bonded target-distribution L1
  values for the complete `IBI + PaiNN` Hamiltonian;
- `equilibrate.log` and `nvt_run.log`: ESPResSo logs.

The structural L1 comparison is diagnostic and does not impose a universal
threshold in the generic core. This NVT smoke run is also **not** an NVE
energy-conservation certification. With legacy tabulated priors, strict NVE
remains unavailable because energy and force are interpolated independently;
with Phase-2 `conservative_spline` priors that particular obstruction has been
removed, but the complete `conservative IBI + PaiNN residual` Hamiltonian must
still pass its dedicated NVE timestep-scaling and drift gates.

### Matched IBI-only vs IBI+PaiNN structural A/B gate

After `18_validate_postibi_runtime.sh` has produced a provenance-validated equilibrated checkpoint, run:

```bash
OVERWRITE=1 bash ./19_validate_ibi_ml_ab.sh
```

The default matched test uses the same checkpoint, Langevin seed, timestep, and sampling schedule in both branches. Each branch receives 1 ps of branch-specific NVT burn-in followed by 8 ps of structural production. Branch A retains the model in checkpoint/model provenance but disables PaiNN forces; branch B activates the same PaiNN model. The comparison report is written to `ibi_ml_ab_validation/ab_structure_comparison.json`.

Longer production can be requested without changing source code, for example:

```bash
AB_PRODUCTION_PS=16 OVERWRITE=1 bash ./19_validate_ibi_ml_ab.sh
```

This A/B test is a structural diagnostic for the exact priors selected by the
residual provenance.  It does not itself make an NVE conservation claim.

### Phase 2: conservative IBI spline representation

Once the tabulated IBI priors have passed structural validation, convert their
**energy** columns into a single-source conservative cubic-Hermite
representation before attempting strict NVE certification. The converter fits a
shape-preserving PCHIP to `U(q)` and stores `q, U(q), dU/dq`; the ESPResSo
plugin evaluates energy and force from that same cubic polynomial rather than
from independently interpolated columns.

This first conservative implementation intentionally certifies **bond and angle
IBI priors only**. A tabulated dihedral causes conversion to fail closed until a
separate torsional endpoint/singularity convention is certified.

Install the ESPResSo extension and rebuild once:

```bash
bash ./20_install_conservative_spline.sh
```

Step 20 now finishes with a synthetic `pypresso` runtime smoke test for both
`ConservativeSplineDistance` and `ConservativeSplineAngle`.  The smoke test is
independent of any IBI output and verifies directly in ESPResSo that the bonded
force is the negative Cartesian finite-difference gradient of the bonded energy.
Do not proceed if either the binding check or this conservative runtime check
fails.

Then convert the frozen best IBI prior set without modifying it:

```bash
bash ./21_convert_best_ibi_to_conservative.sh
```

The output is written separately under `ibi_conservative/`, including
`cg_priors.json` and `conversion_report.json`. The report records SHA256 hashes
of the source and converted artifacts plus dense-grid energy/force fidelity
metrics.

Validate the exact converted tables:

```bash
bash ./22_validate_conservative_spline.sh
```

This performs two hard consistency checks: finite-difference verification of
`dU/dq` against the same Hermite energy spline and ESPResSo runtime versus
preprocessing force/energy parity for every unique converted bond/angle table.
The latter is persisted as `ibi_conservative/runtime_parity_report.json`, with
hashes of the conservative priors and all referenced tables, so the residual
training provenance can fail closed if any artifact changes after the gate.
The original tabulated-to-spline fidelity metrics remain diagnostic because a
nonzero force change is expected when replacing independently interpolated
energy/force columns by a genuinely conservative representation.

Passing this gate does **not** yet certify the final PaiNN Hamiltonian. The
residual dataset must next be rebuilt against `ibi_conservative/cg_priors.json`,
PaiNN must be retrained on that exact residual target, the matched structural
A/B diagnostic should be repeated, and only then should strict NVE timestep
scaling and long-window drift be measured.


### Phase 2 post-gate sequence

After step 22 passes, use the conservative artifacts for the complete residual
training chain:

```bash
bash ./13_rebuild_residual_dataset.sh
bash ./16_check_ibi_training_inputs.sh
```

The expected selected prior is now `ibi_conservative/cg_priors.json`. Step 13
writes `ibi_residual_build_manifest.json`; step 16 verifies hashes of the
residual dataset, rigid-body metadata, priors and all bound validation artifacts.
Do not train unless step 16 ends with `[PASS]`.

For a fresh conservative-residual model, use a distinct filename whenever an
older IBI model already exists:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./03_train_model.sh
```

An existing output is intentionally an error. Do not use `--resume` merely to
bypass that guard after the residual target has changed: a resumed model would
mix optimization history from a different Hamiltonian decomposition.

When a custom model filename is used, propagate it explicitly through the
post-training runtime gates:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  OVERWRITE=1 bash ./18_validate_postibi_runtime.sh

IBI_MODEL=tel22_model_ibi_conservative.pt \
  OVERWRITE=1 bash ./19_validate_ibi_ml_ab.sh
```

Only after these gates validate the same model/prior/dataset provenance should
strict NVE certification be attempted. The conservative spline gate certifies
the bonded prior kernel; it does not by itself certify the learned residual
model or the complete production Hamiltonian.
