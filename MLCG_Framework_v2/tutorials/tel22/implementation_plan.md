# Goal Description

We will upgrade the ML-CG framework to support a configurable, mathematically robust $C^3$ continuous interaction cutoff, while making the use of linear biases optional.

## Handling the Dimensionality of $\alpha$
The Toxvaerd smoothing function is $S(r) = \frac{(r_c - r)^n}{(r_c - r)^n + \alpha^n}$.
Since $r$ and $r_c$ have units of length (e.g. nanometers), $\alpha$ must also be a length.
To make $\alpha$ a universal, **dimensionless** configuration parameter that works regardless of the chosen units, we will define the dimensionless parameter `toxvaerd_alpha` (e.g., $0.1$) as the ratio $\tilde{\alpha} = \alpha / r_c$.
The formula can be rewritten in a strictly dimensionless form by defining $x = (r_c - r) / r_c = 1 - r/r_c$:
$$ S(x) = \frac{x^n}{x^n + \tilde{\alpha}^n} $$
This solves the dimensionality problem elegantly: `toxvaerd_alpha` becomes a pure percentage of the cutoff radius!

## Architectural Choices
1. **Toxvaerd Cutoff**: We will replace the Cosine Cutoff with the Toxvaerd function ($n=4$) everywhere in the network (in `expansion_rbf` and optionally in the envelope).
2. **`use_bias` flag**: Allows enabling/disabling the bias in `filter_mlp`.
3. **`apply_envelope` flag**: We will keep this as a configurable alternative! If `use_bias=true`, the user can still activate `apply_envelope=true` to dynamically suppress the bias at the boundary using the Toxvaerd envelope.

## Proposed Changes

### `espresso/src/core/nonbonded_interactions/PaiNN_Architecture.hpp`
- Add `bool m_use_bias` and `double m_toxvaerd_alpha` to constructors.
- Modify `filter_mlp`: `torch::nn::LinearOptions(num_rbf, dim * 3).bias(m_use_bias)`
- Replace `cos_cutoff` with:
  ```cpp
  auto x = (cutoff_radius - d_ij) / cutoff_radius; // Dimensionless distance to cutoff
  auto x_n = torch::pow(x, 4);
  auto toxvaerd_cutoff = x_n / (x_n + std::pow(m_toxvaerd_alpha, 4));
  ```
- Use `toxvaerd_cutoff` in `expansion_rbf`.
- If `m_apply_envelope` is true, multiply $w$ by `toxvaerd_cutoff` in `PaiNNMessageImpl::forward`.

### `MLCG_Framework/training/train_painn.cpp` & Python Scripts
- Add `use_bias` (default `false`) and `toxvaerd_alpha` (default `0.1`) to the JSON config parser.
- Add `--use_bias` and `--toxvaerd_alpha` to `equilibrate.py` and `run_cg_md.py` argparse.

## Verification Plan

### Automated Tests
1. Recompile the C++ components.
2. We will run the training with `use_bias: false` and `apply_envelope: false`.
3. We will run `run_nve_scaling_envelope.py` to confirm the Verlet scaling is perfect with the dimensionless Toxvaerd cutoff.
