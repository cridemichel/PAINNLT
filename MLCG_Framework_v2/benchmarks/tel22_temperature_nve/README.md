# TEL22 iso-configurational temperature NVE diagnostic

This benchmark asks a narrow question: does the non-ideal TEL22 FP32 NVE exponent depend on
the kinetic amplitude explored from the same mechanical configuration?

It leaves the production Hamiltonian untouched:

```text
H = cg_priors.json + tel22_model.pt
```

and reuses `simulation/certify_nve.py`.  The 300 K branch uses `tutorials/tel22/equilibrated.npz`
**byte-for-byte**.  Lower-temperature branches copy the same positions and orientations and
rescale translational velocities and body-frame angular velocities as

```text
v(T)     = v(source)     * sqrt(T / T_source)
omega(T) = omega(source) * sqrt(T / T_source)
```

The default source-temperature label is 300 K and the default sweep is 300, 100, and 30 K.

## Scope

This is deliberately an **iso-configurational amplitude diagnostic**, not a set of independently
equilibrated canonical ensembles.  It isolates the effect of reducing kinetic excursion amplitude
while holding the initial TEL22 structure fixed.  Therefore:

- if `p` moves substantially toward 2 as the kinetic amplitude is reduced, that is consistent with
  amplitude/configuration-dependent stiffness or strongly anharmonic regions being responsible;
- it does **not** by itself prove that a harmonic high-frequency mode is responsible, because a
  harmonic mode's frequency is temperature independent;
- if the smallest-dt `C2 = sigma_E/dt^2` excess grows as temperature is reduced, a numerical/FP32
  floor becomes a stronger explanation;
- if `p` is insensitive to the rescaling, intrinsic stiffness or another TEL22-specific mechanism
  remains possible.

No gate is relaxed.  An individual strict NVE `FAIL` is recorded but does not abort the remaining
temperatures; operational failures do abort.

## Default protocol

```text
Hamiltonian       : TEL22 production priors + trained PaiNN
source checkpoint : tutorials/tel22/equilibrated.npz
T labels          : 300, 100, 30 K
precision         : float32
device            : CPU
neighbor search   : link-cell
dt [ps]           : .001 .0015 .002 .003 .004 .005
duration / dt     : 2.0 ps
sampling          : every integration step
thermostat        : OFF (NVE)
```

This intentionally matches the accepted TEL22 FP32 scaling protocol apart from the velocity/
angular-velocity rescaling.

## Run

From the repository root:

```bash
cd benchmarks/tel22_temperature_nve
python3 selftest.py
./run.sh --dry-run
./run.sh --overwrite
```

For the recommended FP64 low-temperature control after the FP32 sweep:

```bash
TEL22_TEMP_NVE_TEMPERATURES="300 30" \
TEL22_TEMP_NVE_PRECISIONS="float64" \
./run.sh --overwrite
```

Or run both precisions over all temperatures:

```bash
TEL22_TEMP_NVE_PRECISIONS="float32 float64" ./run.sh --overwrite
```

The source-temperature label can be overridden explicitly if needed:

```bash
TEL22_TEMP_NVE_SOURCE_T_K=300 ./run.sh --overwrite
```

## Outputs

Generated artifacts are ignored by Git:

```text
results/
  checkpoints/
    equilibrated_T100K.npz
    equilibrated_T30K.npz
  T300K_float32/
    nve_certification_report.json
    ...
  T100K_float32/
  T30K_float32/
  temperature_sweep_manifest.json
  temperature_nve_summary.json
```

The summary reports, for each temperature, `p`, `R2`, `C2` spread, maximum block-mean drift,
and a small-dt floor proxy:

```text
C2(dt_min) / median[C2(three largest dt)]
```

A value near 1 is consistent with clean quadratic scaling over the sampled grid; an elevated
value at the smallest timestep is a useful signature of a possible small-dt floor.

For the smallest-dt trajectory the analyzer also reports the initial, mean, and final-block
kinetic energy and `Ekin_mean/Ekin0`.  This is a guard against over-interpreting the temperature
label: if a velocity-rescaled low-T state rapidly converts the original 300 K configurational
energy into kinetic energy, the trajectory is no longer a clean low-amplitude probe and should
be followed by a separately equilibrated low-temperature checkpoint.

## Important interpretation note

Do not describe 100 K or 30 K here as fully equilibrated thermodynamic states.  Only the initial
kinetic amplitudes are rescaled.  If this test indicates a strong temperature/amplitude dependence,
a second-stage diagnostic can independently equilibrate TEL22 at the selected temperatures and
repeat the NVE scan.
