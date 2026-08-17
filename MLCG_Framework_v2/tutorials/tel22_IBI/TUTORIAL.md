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


## 23. Strict NVE certification of the conservative IBI-only candidate

When the matched step-19 A/B comparison selects the IBI-only branch over the
PaiNN-residual branch, certify that exact conservative classical Hamiltonian:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./23_certify_conservative_ibi_nve.sh --overwrite
```

Step 23 intentionally keeps the model path only as a checkpoint-provenance
anchor. PaiNN is disabled via `--disable_ml` both during checkpoint preparation
and in every NVE trajectory; the tested Hamiltonian is therefore the selected
conservative IBI priors plus the other explicit conservative runtime priors
(WCA/Morse), not IBI+PaiNN.

`postibi_runtime_validation/equilibrated_postibi.npz` is now only the **source
checkpoint**. Before the timestep scan, step 23 runs a matched IBI-only Langevin
NVT and writes
`nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz`.
The NVE scan starts from this new state for every timestep. Its metadata records
`hamiltonian_mode=conservative_classical_model_provenance_ml_disabled`,
`sampling_ensemble=NVT_Langevin`, and the SHA256 of the source checkpoint; all
three are validated fail-closed before NVE. No checkpoint-mismatch escape hatch
is needed.

Before launching the timestep scan, the new fail-closed preflight verifies the
current conservative `cg_priors.json`, every referenced spline table, the
Phase-2 finite-difference validation report, and the persisted ESPResSo/runtime
parity report. Any post-validation edit to those artifacts aborts the run.
Legacy `tabulated` bonded priors and conservative dihedrals are rejected.

Defaults:

```text
NVE_EQ_DT=0.0005
NVE_EQ_DURATION_PS=5.0
NVE_EQ_KT=2.49
NVE_DTS="0.001 0.0015 0.002 0.003 0.004 0.005"
NVE_DURATION_PS=5.0
NVE_DEVICE=cpu
NVE_NEIGHBOR_SEARCH=link-cell
NVE_SLOPE_MIN=1.7
NVE_SLOPE_MAX=2.3
NVE_MIN_R2=0.97
NVE_MAX_RELATIVE_DRIFT=1e-4
```

The output directory is `nve_certification_conservative_ibi_only/`. Step 23
preserves the original strict reference rule: both the `sigma_E ~ dt^p` scaling
gate and the block-drift gate must pass. This historical result is never
rewritten. For conservative IBI, the final composite decision is made only
after the direct Richardson state-order test in step 25 and is assembled by
step 26. The report records
`hamiltonian_mode=conservative_classical_model_provenance_ml_disabled` and hashes
the Phase-2 preflight/validation/parity artifacts in addition to the normal
runtime inputs.

This certification does **not** certify `IBI + PaiNN`. A future residual model
that wins the matched A/B structural test must undergo a separate ML-active NVE
certification.

## 24. Diagnose failed NVE scaling before changing the conservative kernel

A step-23 result with `drift_pass=True` but poor/non-monotonic timestep scaling is
not sufficient evidence that the conservative spline itself is wrong. Before
changing priors or ESPResSo code, probe the fine-timestep and short-time regimes:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
NVE_DIAG_DURATION_PS=2 \
  bash ./24_diagnose_conservative_ibi_nve_scaling.sh --overwrite
```

The diagnostic uses the exact IBI-only NVT checkpoint already prepared by step
23 and keeps PaiNN disabled. Defaults:

```text
NVE_DIAG_DTS="0.00025 0.0005 0.00075 0.001 0.0015 0.002 0.003 0.004 0.005"
NVE_DIAG_DURATION_PS=2.0
NVE_DIAG_FINE_MAX_DT=0.001
NVE_DIAG_COARSE_MIN_DT=0.0015
NVE_DIAG_LOCAL_TIMES_PS="0.012 0.024 0.048 0.096"
```

The JSON report contains global/fine/coarse power-law fits for both `sigma_E`
and `rms_delta_E`. For fine timesteps it also evaluates exact-sample short-time
errors at the listed physical times and fits `|Delta E(t)|`, prefix RMS error,
and prefix `sigma_E` versus `dt`. The local times are chosen to be commensurate
with every default fine timestep; no interpolation is allowed.

Outputs are written below `nve_diagnostic_conservative_ibi_only/`. This is a
**diagnostic**, not a replacement certification: its process exit status does
not promote the model when the strict step-23 fit fails. Use the fine and local
fits to decide whether an asymptotic `dt^2` regime exists before moving to
Hamiltonian-component isolation.


## 25. Diagnose the actual Velocity-Verlet state-convergence order

If step 24 shows very small drift and rapidly vanishing short-time energy error,
but `sigma_E` remains non-monotonic, do not infer the integrator order from the
energy-fluctuation amplitude alone. Run the short-time state-convergence test:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./25_diagnose_conservative_ibi_state_convergence.sh --overwrite
```

The default dyadic timestep ladder is:

```text
0.001
0.0005
0.00025
0.000125
reference = 0.0000625 ps
```

Every trajectory starts from the same provenance-checked IBI-only NVT checkpoint,
runs in NVE with PaiNN disabled, and is sampled at the same physical times
`0.012, 0.024, ..., 0.096 ps`. `run_cg_md.py` writes a dedicated real-particle
state NPZ containing particle IDs, positions, velocities, quaternions and
body-frame angular velocities. The NPZ also binds the input hashes, Hamiltonian
mode and source-checkpoint SHA256.

The report contains two complementary comparisons:

1. each finite-dt trajectory versus the finest reference trajectory;
2. Richardson differences between each `dt` trajectory and its `dt/2` partner.

The second comparison is the order diagnostic. For a second-order integrator,
these pair differences should obey `error(dt,dt/2) ~ dt^2`. Fits are reported
independently for translational position, translational velocity, rigid-body
orientation (quaternion geodesic angle) and body-frame angular velocity at every
common physical time. Periodic position differences use the minimum-image
convention, and quaternion sign degeneracy is removed before computing angles.

Outputs are written below
`nve_state_convergence_conservative_ibi_only/`, principally
`state_convergence_report.json` and `run_plan.json`. This remains a diagnostic:
it does not overwrite or relax the strict step-23 certification result. A clean
`p ~= 2` state convergence together with tiny NVE drift is evidence that the
trajectory integrator is behaving at second order even when long-window
`sigma_E` is a noisy/non-monotonic proxy for the shadow-energy amplitude.


## 26. Final composite NVE certification for conservative IBI-only

After steps 22, 23 and 25 have produced their provenance-bound reports, assemble
the final conservative-IBI NVE verdict without rerunning dynamics:

```bash
bash ./26_finalize_conservative_ibi_nve_certification.sh
```

The final gate is deliberately composite. It requires all of the following:

1. conservative spline finite-difference validation: PASS;
2. ESPResSo/runtime vs preprocessing energy/force parity: PASS;
3. provenance consistency for priors and the dedicated IBI-only checkpoint;
4. NVE relative block-mean energy drift below the configured threshold;
5. Richardson state convergence consistent with second order for position,
   velocity, rigid-body orientation and body-frame angular velocity.

The default state-order window is `1.7 <= median p <= 2.3` with median
`R2 >= 0.95` for every required state metric. The long-window `sigma_E ~ dt^p`
fit from step 23 is preserved in the final JSON, including its original
PASS/FAIL status and exponent, but is explicitly marked `diagnostic_only` and
is not a final gating criterion. This avoids silently rewriting the historical
step-23 result while using the direct trajectory-convergence measurement for
the integrator-order claim.

The final report is written to:

```text
nve_final_certification_conservative_ibi_only/
    conservative_ibi_nve_certification_report.json
```

A successful summary has the form:

```text
[CONSERVATIVE IBI NVE COMPOSITE CERTIFICATION]
conservative_kernel : PASS
runtime_parity      : PASS
provenance          : PASS
energy_drift        : PASS
VV_second_order     : PASS
sigma_E_scaling     : DIAGNOSTIC (...)
[PASS] Conservative IBI NVE certification
```

This certifies only the classical `WCA + Morse + bonded conservative IBI`
Hamiltonian with PaiNN disabled. It must not be reused as evidence for an
ML-active Hamiltonian.


## 27. Localize a non-quadratic `sigma_E(dt)` before changing the kernel

A clean Richardson state order does **not** waive the classical molecular-dynamics
expectation that Velocity-Verlet energy fluctuations enter at `O(dt^2)` for a
sufficiently regular Hamiltonian. If step 23/24 continues to show a poor or
non-monotonic `sigma_E(dt)` law, run the localization suite instead of promoting
that discrepancy to a non-gating metric:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./27_diagnose_conservative_ibi_energy_scaling.sh --dry-run

IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./27_diagnose_conservative_ibi_energy_scaling.sh --overwrite
```

The default full-system scan is deliberately finer than the production scan:

```text
ENERGY_LOC_DTS="0.001 0.00075 0.0005 0.000375 0.00025 0.0001875 0.000125"
ENERGY_LOC_DURATION_PS=0.25
ENERGY_LOC_MICRO_DURATION_PS=0.096
```

Increase `ENERGY_LOC_DURATION_PS` to `1` or `2` ps only after the short
localization run identifies which branch deserves a longer measurement.

The suite performs several independent probes:

1. **Spline smoothness:** evaluates the left/right second derivative of every
   unique Hermite table at every internal knot and records `Delta U''`.
2. **Minimal bond dynamics:** compares an exact single Hermite segment with a
   real three-node table that contains one internal knot.
3. **Minimal angle dynamics:** repeats inside-cell/crossing tests on ordinary
   point particles and on virtual sites attached to rotating rigid bodies.
4. **Rigid torque finite differences:** rotates a real rigid body by `+/-eps`
   with ESPResSo `ParticleHandle.rotate()` and compares `-dU/dphi` with
   `torque_lab`.
5. **Full-system generalized gradients:** perturbs selected COM positions and
   orientations at the real TEL22 checkpoint and compares finite-difference
   energy derivatives with runtime forces/torques.
6. **Hamiltonian isolation:** runs matched `no_ibi`, `bonds_only`,
   `angles_only`, and `full` priors. Disabled conservative terms are replaced by
   zero-strength analytic bonds/angles rather than deleted so the WCA topology
   remains unchanged. These diagnostic variants deliberately use
   `--allow_checkpoint_mismatch` because the mechanical checkpoint was prepared
   under the full IBI-only Hamiltonian.
7. **Knot-event correlation:** records every conservative coordinate at the
   full trace timestep and asks whether `|E[n+1]-E[n]|` is enhanced on steps
   that cross spline knots, including a weight based on the crossed `|Delta U''|`.
8. **Energy decomposition:** records translational/rotational kinetic energy and
   ESPResSo bonded/non-bonded energy separately.
9. **Neighbor-search A/B:** compares the normal link-cell/hybrid path with an
   all-pairs `nsquare` path from the same state.
10. **Time reversibility:** propagates forward, reverses `v` and `omega_body`,
    propagates for the same number of steps, and compares with the initial
    positions/orientations and sign-reversed velocities.

The main artifact is:

```text
conservative_ibi_energy_localization/
    localization_report.json
```

Useful decision patterns are:

```text
no_ibi p ~= 2, bonds_only fails       -> distance spline/knot path
bonds_only p ~= 2, angles_only fails  -> angle path
point-angle passes, rigid-angle fails -> virtual-site torque/rotation coupling
inside-cell passes, crossing fails    -> spline knot regularity
link-cell differs from nsquare        -> neighbor traversal / interaction bookkeeping
FD torque mismatch                    -> energy/torque inconsistency
large knot-crossing |dE| ratio        -> direct evidence that knot events drive energy error
```

Step 27 is diagnostic-only and does not modify steps 23-26. Until the origin of
a reproducible non-quadratic `sigma_E(dt)` is understood, do not treat the
composite step-26 PASS as resolving that specific energy-scaling discrepancy.

## 28. Separate timestep, window-length and initial-state effects in `sigma_E`

Step 27 shows that the full conservative IBI Hamiltonian recovers an approximately
quadratic `sigma_E(dt)` law on a short, fine-timestep scan. The remaining question
is why the earlier 2 ps single-trajectory scans were irregular. Do not change the
kernel to answer that question; measure the estimator directly across independent
initial states and observation windows:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./28_diagnose_sigma_energy_replicas.sh --dry-run

IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./28_diagnose_sigma_energy_replicas.sh --overwrite
```

The default diagnostic uses four independently Langevin-branched IBI-only
checkpoints and the power-of-two timestep grid

```text
0.001 0.0005 0.00025 0.000125 ps
```

Each replica/timestep pair is integrated **once** to `2 ps`. The reported
`0.125, 0.25, 0.5, 1, 2 ps` measurements are exact prefixes of that same energy
trace. Consequently, differences between durations cannot be caused by starting a
different NVE trajectory. `sigma_E` is the raw population standard deviation of
`E_tot` (`ddof=0`); no linear detrending or high-pass filtering is applied.

Replica initial conditions are prepared from the same provenance-bound IBI-only
checkpoint by a short NVT branch with independent Langevin seeds. The default is
`1 ps` at `dt=0.0005 ps`, `kT=2.49 kJ/mol`. Increase
`SIGMA_REPLICA_COUNT` from `4` to `8` for a stronger production-quality estimate.

For every observation duration the report contains:

1. the ordinary `sigma_E(dt)` fit for each replica separately;
2. fits of the arithmetic mean, median and geometric mean `sigma_E` across replicas;
3. a fixed-effects log-log regression
   `log sigma[r,dt] = alpha[r] + p log(dt)`, where each replica has its own
   intercept but all replicas share the timestep exponent `p`;
4. a deterministic replica bootstrap confidence interval for that common slope when at least three replicas are available;
5. the coefficient of variation of `sigma_E` across replicas at every timestep.

The fixed-effects slope is the most direct test of whether state-dependent
prefactors were obscuring the common `dt^2` law. A result near `p=2` at all window
lengths, with increasing scatter among individual replicas at long windows, would
identify the old failure as a single-trajectory estimator problem. A systematic
loss of the common slope as the physical window grows would instead indicate a
real long-time effect that still requires investigation.

Outputs are written below:

```text
sigma_energy_replica_window_diagnostic/
    run_plan.json
    sigma_energy_replica_report.json
    sigma_energy_replica_observations.csv
    sigma_energy_replica_aggregate.csv
    replicas/
```

Step 28 is diagnostic-only. It does not modify the historical step-23 result or
any certification artifact.

### Recover an interrupted step-28 run without launching more dynamics

If the long default run is interrupted after at least two replicas have completed
all requested timestep trajectories, reuse those files directly:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./28_diagnose_sigma_energy_replicas.sh --analyze-existing
```

`--analyze-existing` performs zero integration steps. It validates each existing
replica checkpoint and requires every requested `dt` energy trace to reach the
maximum requested duration. Incomplete replicas are reported as `SKIP`; only
complete provenance-valid replicas are analyzed. At least two complete replicas
are required. With exactly two replicas the fixed-effects/common-slope and
per-replica fits are still produced, but the replica bootstrap interval is
explicitly disabled because two replicas are not enough for a useful bootstrap
uncertainty estimate. Do **not** combine this recovery mode with `--overwrite`.


## 29. Explain why conservative IBI narrows the clean Verlet timestep range

The original TEL22 Hamiltonian showed an approximately quadratic `sigma_E(dt)`
law through the historical `0.001--0.005 ps` scan, whereas conservative IBI
recovers a clean `dt^2` law only on the finer scan tested in steps 27--28. Do not
assume that this is generically "outside the asymptotic regime": isolate which
bonded replacement changed the numerical frequency scale.

Run:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./29_diagnose_ibi_timestep_range.sh --dry-run

IBI_MODEL=tel22_model_ibi_conservative.pt \
  bash ./29_diagnose_ibi_timestep_range.sh --overwrite
```

Unlike step 27's `no_ibi` control, which replaces conservative bonded terms by
zero-strength analytic interactions, step 29 restores the **actual original
TEL22 harmonic priors**. It constructs four topology-matched Hamiltonians:

```text
reference        = configured baseline bonds + configured baseline angles
ibi_bonds_only   = conservative IBI bonds + original TEL22 angles
ibi_angles_only  = original TEL22 bonds + conservative IBI angles
full_ibi         = conservative IBI bonds + conservative IBI angles
```

All nonbonded/WCA/exclusion/dihedral terms are required to be identical between
the old and IBI prior files; the diagnostic fails closed otherwise. Common
Morse terms must also be unchanged.

Each variant is briefly thermalized from the same mechanical source checkpoint
under its own Hamiltonian and then scanned on the historical timestep grid:

```text
IBI_TIMESTEP_DTS="0.001 0.0015 0.002 0.003 0.004 0.005"
IBI_TIMESTEP_DURATION_PS=1.0
IBI_TIMESTEP_BRANCH_DT=0.0005
IBI_TIMESTEP_BRANCH_DURATION_PS=0.25
```

For every variant the report includes the global `sigma_E = C dt^p` fit,
adjacent local exponents, and `sigma_E/dt^2`. A flat `sigma_E/dt^2` is the most
direct visualization of the quadratic regime and identifies the timestep at
which higher-order/stiff-mode effects become visible.

During the final half of each short NVT branch, step 29 also evaluates the local
bonded curvature `U''(q)` at the **coordinates actually visited**, rather than
using extrema from unused parts of an IBI table. It reports `P50/P95/P99/max` of
`|U''|` for bonds, angles, and each named bonded group. Since the compared
Hamiltonians have the same geometry and masses, the square root of an IBI/old
curvature ratio is reported as a frequency-scale proxy. For example, a `P99`
curvature ratio near `25` corresponds to an approximate five-fold increase in
the local fastest frequency scale and would quantitatively explain a roughly
five-fold reduction in the clean Verlet timestep range.

Decision patterns:

```text
reference passes, ibi_bonds_only degrades   -> bond IBI introduces the restrictive scale
reference passes, ibi_angles_only degrades  -> angle IBI introduces the restrictive scale
both isolated variants pass, full_ibi fails -> bond/angle mode coupling is implicated
same branch that degrades has large U'' ratio -> stiffness explanation is quantitatively supported
scaling degrades without a curvature increase -> investigate spline regularity/runtime representation further
the configured reference itself no longer passes            -> runtime/regression issue, not an IBI-specific stiffness effect
```

Main artifact:

```text
ibi_timestep_range_diagnostic/timestep_range_localization_report.json
```

Step 29 is diagnostic-only and does not alter any certification artifact or the
conservative spline implementation.

## 30. Localize and regularize the stiff IBI angle priors offline

Step 29 identifies the angle replacement as the dominant restriction on the
historical Verlet timestep range: the original TEL22 angles retain clean
quadratic `sigma_E(dt)` scaling, bond-only IBI remains close to second order,
while angle-only and full IBI degrade strongly.  Before changing the runtime or
rerunning a long NVE scan, separate three possible sources of angle stiffness:

1. the explicit quadratic endpoint wall added by the IBI builder;
2. short-wavelength structure/noise in the IBI potential body;
3. the C1 PCHIP/Hermite derivative representation used by the conservative
   conversion.

Run the offline audit:

```bash
bash ./30_diagnose_regularize_ibi_angles.sh --dry-run
bash ./30_diagnose_regularize_ibi_angles.sh --overwrite
```

The default runtime sample is the already-generated step-29 full-IBI NVT sample:

```text
ibi_timestep_range_diagnostic/full_ibi/nvt_structured_sample.npz
```

No new MD is run.  The mapped target geometry is reconstructed from
`tel22_dataset_ibi_residual.bin`; only positions/sites are used, so the residual
force labels are irrelevant to this diagnostic.

For each named IBI angle group, step 30 reports:

- runtime-vs-target angular-distribution L1;
- target and runtime probability inside the configured endpoint wall zones;
- `P50/P95/P99/max |U''|` on target and runtime coordinates;
- the corresponding frequency-scale proxy versus the original TEL22 harmonic
  angle constant;
- the largest curvature hot spots with their angle, wall/interior classification,
  and local target/runtime density;
- the jump in `U''` across Hermite knots.

The current wall is also separated analytically from the potential.  With the
standard settings `wall_width=0.1 rad` and `wall_k=5000`, the endpoint barrier is
`0.5*k*w^2 = 25 kJ/mol`, about `10 kT` at `kT=2.49`.

Step 30 then writes five **unvalidated** conservative candidate prior sets:

```text
c2_raw_wall_current
smooth_0p01_wall_current
smooth_0p02_wall_current
smooth_0p02_wall_1p5x_same_barrier
smooth_0p02_wall_2x_same_barrier
```

`c2_raw_wall_current` keeps the same nodal energy and the current wall but derives
nodal slopes from a C2 cubic spline.  It isolates the effect of the PCHIP/C1
slope representation.  The `smooth_*` candidates first subtract the known
quadratic wall, Gaussian-smooth only the remaining IBI body by the stated angular
width, and then re-add the wall.  The widened-wall candidates reduce wall
curvature while preserving the endpoint barrier energy by scaling
`k_new = k_old * (w_old/w_new)^2`.

For every candidate the report gives occupied-curvature reduction and the
potential perturbation in units of `kT` on both target and runtime samples.  This
is an **offline screen**, not structural validation: changing an angle potential
can change the full coupled distribution even when the direct `Delta U` is small.
Do not replace `ibi_conservative/cg_priors.json` with a candidate at this stage.
The next step, after inspecting the report, is a matched short NVT structural
comparison plus the same coarse NVE timestep scan for only the most defensible
candidate(s).

Outputs:

```text
ibi_angle_regularization_diagnostic/
    angle_regularization_report.json
    profiles/<angle_group>.csv
    candidates/<candidate>/cg_priors.json
    candidates/<candidate>/angle_conservative_*.dat
```

The candidate directories are self-contained with respect to bonded table files
and are explicitly tagged `validated=false` in their provenance metadata.

## 31. Matched short-MD validation of the two defensible angle regularization candidates

Step 30 shows that changing only the C1/PCHIP slope representation has little
impact on the occupied angle curvature, while smoothing the de-walled IBI body by
`0.01-0.02 rad` reduces the high-curvature tail with a small direct potential
perturbation.  Do **not** promote either offline candidate yet.  Validate the two
least invasive candidates against the current conservative IBI Hamiltonian under
one matched MD protocol:

```bash
bash ./31_validate_ibi_angle_regularization.sh --dry-run
bash ./31_validate_ibi_angle_regularization.sh --overwrite
```

The three variants are:

```text
current       ibi_conservative/cg_priors.json
smooth_0p01  ibi_angle_regularization_diagnostic/candidates/smooth_0p01_wall_current/cg_priors.json
smooth_0p02  ibi_angle_regularization_diagnostic/candidates/smooth_0p02_wall_current/cg_priors.json
```

Each starts from the same source checkpoint and receives the same short NVT
protocol (`0.25 ps`, `dt=0.0005 ps`, `kT=2.49`, same thermostat seed).  The final
half of each branch is sampled structurally.  Step 31 then reports, for bonds and
angles separately, the per-group target-vs-runtime distribution L1 together with
target-count-weighted mean L1.  It also re-evaluates the angle `P95/P99/max |U''|`
on the **new candidate-specific NVT ensemble**, so the curvature reduction from
step 30 is checked after the ensemble is allowed to respond to the modified
potential.

The NVT checkpoint for each variant seeds the historical coarse NVE scan:

```text
dt = 0.001 0.0015 0.002 0.003 0.004 0.005 ps
T  = 1 ps per dt
```

For each variant the report includes `sigma_E`, the global `sigma_E=C dt^p` fit,
`R2`, `sigma_E/dt^2`, the `C2` spread, adjacent local exponents, and two simple
contiguous clean-range indicators. `max_clean_dt_factor_1p5`, for example, is the
largest timestep reachable from the smallest scanned `dt` before
`sigma_E/dt^2` departs by more than a factor 1.5 from its smallest-dt value.
This is a comparison diagnostic, not a new universal certification threshold.

The final comparison against `current` reports:

- change in weighted angle and bond L1;
- actual runtime `P99 |U''|` reduction;
- improvement in `C2` spread;
- movement of the coarse-range exponent toward or away from 2;
- change in the contiguous clean timestep indicator.

Default cost is only about `10350` integration steps total (three variants), and
`--resume` reuses complete NVT/NVE artifacts after an interruption:

```bash
bash ./31_validate_ibi_angle_regularization.sh --resume
```

Main artifact:

```text
ibi_angle_regularization_validation/angle_candidate_validation_report.json
```

Step 31 remains diagnostic-only.  A candidate should be considered for promotion
only after it improves the timestep behavior **without materially worsening the
matched structural distributions**; final production/certification decisions are
separate from this screen.

## 32. Local sweep around the useful angle-body smoothing scale

Step 31 establishes two points that must be kept separate.  Moderate `0.01 rad`
body smoothing materially improves the **contiguous** `sigma_E/dt^2` plateau
with negligible structural cost, while the stronger `0.02 rad` candidate lowers
occupied curvature much further but its local timestep behavior remains
irregular.  Therefore do not optimize the global log-log exponent alone and do
not assume that lower `P99 |U''|` is monotonically better.

Run a narrow local sweep around `0.01 rad`:

```bash
bash ./32_optimize_ibi_angle_smoothing.sh --dry-run
bash ./32_optimize_ibi_angle_smoothing.sh --overwrite
```

The default new candidates are:

```text
0.0075 rad
0.0125 rad
0.0150 rad
```

The already-computed `0.0100 rad` candidate is reused directly from
`ibi_angle_regularization_validation/angle_candidate_validation_report.json`.
Its NVE points are re-fitted on the same common timestep subset as the new
candidates, so no MD is repeated merely to make the comparison fair.

Each **new** candidate uses the same wall as the selected conservative IBI prior,
changes only the de-walled angle-potential body, and receives the same matched
short NVT protocol used in step 31.  The default NVE grid is deliberately small:

```text
dt = 0.001 0.002 0.003 0.004 0.005 ps
T  = 1 ps per dt
```

With a `0.25 ps` NVT branch at `dt=0.0005 ps`, the three new candidates require
about `8349` integration steps total.  The reused `0.01` point adds zero MD cost.
Use `--resume` after an interruption:

```bash
bash ./32_optimize_ibi_angle_smoothing.sh --resume
```

The NVE-only ranking is intentionally lexicographic.  It first maximizes the
largest **contiguous** timestep range for which `sigma_E/dt^2` stays within a
factor `1.5` of its smallest-dt value.  Ties are then broken by the C2 spread
inside that clean prefix, the prefix distance from `p=2`, and only later by the
global C2 spread / exponent / `R2`.  This prevents an oscillatory sequence from
winning merely because compensating points happen to give a global `p` close to
2.

Structural diagnostics remain separate from this NVE ranking.  For every
candidate the report gives weighted angle/bond L1 relative to the same target,
actual occupied `P99 |U''|`, and deltas versus the current unsmoothed prior.  The
highest NVE rank is therefore **not** an automatic promotion decision: reject a
candidate if the matched structural response is materially worse.

New candidate priors are generated under the step-32 output directory and are
explicitly tagged `validated=false`; `ibi_conservative/cg_priors.json` is never
modified.

Main artifact:

```text
ibi_angle_smoothing_sweep/angle_smoothing_sweep_report.json
```

After this local sweep choose at most one smoothing scale for the longer final
structural/NVE validation.  Do not continue tuning the smoothing bandwidth once
a clean plateau and acceptable structure have been identified.

## 33. Replica/structure validation of the selected 0.0075-rad candidate

After the local sweep, validate only `smooth_0p0075` on independent thermal
branches before changing production priors:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt bash ./33_validate_final_ibi_angle_candidate.sh --dry-run
IBI_MODEL=tel22_model_ibi_conservative.pt bash ./33_validate_final_ibi_angle_candidate.sh --overwrite
```

Step 33 combines the reused step-32 branch with two independent NVT/NVE branches,
fits a common fixed-effects `sigma_E ~ dt^p` exponent across all three replicas,
and performs a longer matched current-vs-candidate NVT structural comparison.
Passing this step validates the candidate **for promotion consideration only**;
it does not modify `ibi_conservative/` and does not certify the production path.

The validated TEL22 result is:

```text
[NVE] common p=1.947046 R2within=0.984844
      full-clean=3/3 medianC2spread=1.389 pass=True
[STRUCT] dAngleL1=+0.009561 dBondL1=-0.019844
         maxGroupAngleDelta=+0.055364 kineticRelDelta=4.001e-03
         P99U2red=2.416x pass=True
[FINAL] pass=True
```

This is evidence for the **specific** `0.0075 rad` candidate; it is not evidence
that the same smoothing bandwidth should be used for another model.

## 34. Explicit promotion and post-promotion Hamiltonian certification

Only after step 33 passes, promote the reviewed `smooth_0p0075` candidate and
certify the exact production path:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt bash ./34_promote_and_certify_ibi_angle_prior.sh --dry-run
IBI_MODEL=tel22_model_ibi_conservative.pt bash ./34_promote_and_certify_ibi_angle_prior.sh --promote
```

The promotion is fail-closed on the reviewed candidate SHA256
`c31f6d0d53f053071ab694f91d8271c83fc90a90ada291ba60c206adf82a3799`
and on the passing step-33 report.  Before changing production it copies the
current `ibi_conservative/` tree to `ibi_conservative_pre_smooth_0p0075/`.
The promoted JSON metadata are rewritten to record validation/promotion, while
every runtime bonded table must remain byte-identical to the validated candidate.

The pre-promotion residual dataset and PaiNN model are explicitly marked stale
for ML-active use.  Step 34 certifies only the classical Hamiltonian with PaiNN
disabled; residual labels must be rebuilt and PaiNN retrained before any later
ML-active calculation using the promoted priors.

Post-promotion certification does **not** reuse the historical step-26 decision.
It regenerates conservative finite-difference validation and ESPResSo runtime
parity from `ibi_conservative/cg_priors.json`, then creates a fresh promoted-prior
NVT checkpoint and runs:

```text
fresh sigma_E scan : dt = 0.001 0.002 0.003 0.004 0.005 ps, 1 ps each
fresh Richardson   : dt_ref = 0.0000625 ps, duration = 0.096 ps
```

Here `sigma_E ~ dt^2` is gating again (`1.8 <= p <= 2.2`, `R2 >= 0.95`,
`C2 spread <= 2`, full scan through `0.005 ps`, relative block drift `<=2e-5`).
Richardson position/velocity/orientation/body-omega convergence must also remain
second order.  The final report additionally binds the three-replica step-33
evidence to production through table SHA256 identity.

If execution is interrupted after promotion, use:

```bash
IBI_MODEL=tel22_model_ibi_conservative.pt bash ./34_promote_and_certify_ibi_angle_prior.sh --resume
```

Main artifact:

```text
ibi_promoted_final_certification/promoted_ibi_final_certification_report.json
```

The validated production result is:

```text
[SIGMA] p=1.877261 R2=0.984412 C2spread=1.487 maxdt=0.005 pass=True
[RICHARDSON] position:p=2.004/R2=1.000 velocity:p=2.215/R2=0.998
             orientation:p=2.035/R2=1.000 omega_body:p=2.019/R2=0.999 pass=True
[FINAL] pass=True ML_active=False
```

The single fresh `sigma_E` scan is not perfectly pointwise quadratic, but it
passes the pre-declared second-order gate and is consistent with the independent
step-33 result (`common p=1.947046`, `3/3` clean through `0.005 ps`).

A final `pass=true` certifies `WCA + Morse + bonded conservative regularized
smooth_0p0075 IBI` with PaiNN disabled. Promotion/certification does not silently
rebuild or promote an ML residual model.

### Regularized angular IBI is optional

The TEL22 result must not be interpreted as a universal rule that every angular
IBI potential needs `0.0075 rad` smoothing. There are two valid operating modes:

- **raw/conservative angular IBI:** keep the converged IBI potential unchanged
  and choose a timestep inside its demonstrated clean second-order regime;
- **regularized conservative angular IBI:** remove short-wavelength angle-body
  structure only when diagnostics show that it creates an unnecessarily stiff
  numerical scale, then validate and promote the regularized candidate as a new
  prior.

The smoothing bandwidth is a model parameter selected by structural and NVE
validation, not an IBI constant. A candidate with lower `|U''|` is not
necessarily better; the step-31/32 diagnostics showed that excessive smoothing
can produce a less regular `sigma_E/dt^2` sequence even when the curvature is
smaller.

## 35–39. Periodic conservative dihedral IBI: infrastructure and matched structural evidence

The dihedral work is intentionally **test-only** at this stage. Nothing in steps
35–39 promotes torsional priors into `ibi_conservative/`. The model-specific
torsional grouping strategy is declared by configuration; for TEL22 the current
strategy is `consecutive_angle_types`, producing six pooled backbone groups. The
generic seed builder does not assume this strategy and fails if an unsupported
strategy is requested.

Step 35 established the periodic conservative dihedral path and isolated the six
torsional IBI groups while keeping the inherited bond/angle conservative tables
byte-identical. Step 37 then showed why legacy `TabulatedDihedral` must not be
used as the sampling representation for a prior that will later be promoted as
a single-source conservative potential: energy and force-factor are interpolated
independently, whereas `ConservativeSplineDihedral` derives Cartesian forces from
the same periodic `U(phi)`.

Step 38 therefore runs torsional IBI **conservative-in-the-loop** from
`iteration_000` onward. After correcting the final sampling protocol so that all
three priors are sampled apples-to-apples, the single-seed sequence was:

```text
U0  0.512339
U1  0.547941
U2  0.525388
```

The large earlier short-NVT value near `1.38` was a protocol-mismatch artifact
and must not be used as evidence of IBI divergence.

Step 39 completed a matched `prior x seed-pair` replica matrix. For the current
TEL22 configuration (`3` paired replicas), the measured mean dihedral L1 values
were:

```text
U0  mean=0.550617  SD=0.036116
U1  mean=0.527872  SD=0.023729
U2  mean=0.528039  SD=0.028747
```

Paired differences were:

```text
U1-U0  mean=-0.022745  SD=0.012796  lower L1 in 3/3 seed pairs
U2-U0  mean=-0.022578  SD=0.009686  lower L1 in 3/3 seed pairs
U2-U1  mean=+0.000167  SD=0.009393  mixed sign
```

Thus the current evidence supports a small, consistent improvement from the
first conservative-in-loop IBI update, while the second update adds no measurable
improvement at this sampling resolution. `U1` is therefore the simpler candidate
for any *next diagnostic*, but **neither U1 nor U2 is promotion-ready**. The
three-replica study is a diagnostic uncertainty estimate, not a universal
statistical or certification threshold.

Run the matched replica diagnostic with:

```bash
bash ./39_test_conservative_dihedral_ibi_replicas.sh --dry-run
bash ./39_test_conservative_dihedral_ibi_replicas.sh --run
```

The number of replicas is model-dependent and comes from external configuration;
the generic matrix code also infers `U0..UN` from the IBI report rather than
assuming exactly three priors.

## Model-dependent workflow configuration and provenance

From this point onward, and retroactively for the IBI/conservative workflows
already introduced, every model-dependent choice is external configuration. The
TEL22 file is:

```text
model_dependent_workflow_config.json
```

It is loaded by `model_config.sh`. The architectural classification is:

```text
CORE_INVARIANT       universal physical/methodological rule -> generic code/tests
MODEL_PARAMETER      system/dataset/model-dependent choice  -> external config
CALIBRATED_PARAMETER parameter selected by diagnostics       -> config + provenance
```

Examples of `MODEL_PARAMETER`/`CALIBRATED_PARAMETER` are IBI mixing, histogram
support, torsional grouping, sampling lengths, seed policy, regularization
sweeps, `sigma_angle=0.0075 rad`, replica count, structure gates, NVE timestep
grids and NVE acceptance windows. None of these values is a universal property
of conservative IBI.

Validate the TEL22 config with:

```bash
python3 ../../simulation/model_dependent_config.py validate \
  --config model_dependent_workflow_config.json
```

A different system can use the same workflow code with another config:

```bash
IBI_MODEL_DEPENDENT_CONFIG=/path/to/my_model_workflow_config.json \
  bash ./38_test_conservative_in_loop_dihedral_ibi.sh --run
```

Explicit environment overrides are permitted but are provenance-visible. Each
retrofitted workflow writes a model-config provenance sidecar (`model_config_provenance*.json`), recording the config
path/SHA256, selected sections, configured values, resolved values and whether a
value came from the config or an environment override. The TEL22 config also
records calibration provenance for the validated angle smoothing candidate and
its promoted prior SHA256.

An explicitly supplied `ibi_settings.json` is authoritative: missing required
model-dependent fields are configuration errors rather than invitations to merge
unknown internal defaults. This prevents a workflow from changing scientific
policy silently when moved to a new molecular model.
