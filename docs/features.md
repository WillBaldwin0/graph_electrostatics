# Electrostatic Features

## Interface

### Classes

- `GTOElectrostaticFeatures`: single-channel electrostatic features.

### Constructor

`GTOElectrostaticFeatures` takes:

- `density_max_l`: maximum multipole order for the source density.
- `density_smearing_width`: Gaussian width $\sigma$ for the source density basis.
- `feature_max_l`: maximum angular order for the projection basis.
- `feature_smearing_widths`: Gaussian widths for the projection basis.
- `include_self_interaction`: whether to include self-interaction terms.
- `kspace_cutoff`: cutoff for generating reciprocal vectors.
- `integral_normalization`: normalization used in the projection integrals.
- `quadrupole_feature_corrections`: include additional quadrupole corrections for
  non-periodic systems.
- `pbc_handling`: boundary-condition/evaluator mode.

Example:

```python
from graph_longrange.features import GTOElectrostaticFeatures

features_block = GTOElectrostaticFeatures(
    density_max_l=1,
    density_smearing_width=0.5,
    feature_max_l=1,
    feature_smearing_widths=[0.4, 0.8],
    include_self_interaction=False,
    kspace_cutoff=kspace_cutoff,
    quadrupole_feature_corrections=False,
    integral_normalization="receiver",
    pbc_handling="mixed_periodic",
)
```

### Forward arguments

`forward(...)` expects:

- `k_vectors`: flattened tensor `[n_k_total, 3]` of reciprocal vectors.
- `k_norm2`: flattened tensor `[n_k_total]` of squared norms.
- `k_vector_batch`: `[n_k_total]` mapping each k-vector to a graph id.
- `k0_mask`: `[n_k_total]`, `1.0` at the $\mathbf{k}=\mathbf{0}$ entry.
- `source_feats`: multipoles, `[n_nodes, m_dim]`.
- `node_positions`: `[n_nodes, 3]`.
- `batch`: `[n_nodes]` graph id for each node.
- `volume`: `[n_graph]` cell volumes.
- `pbc`: `[n_graph, 3]` periodic flags.

### Precompute and dynamic paths

The class supports a cached path:

- `precompute_geometry(...)` caches geometry-dependent tensors.
- `forward_dynamic(cache, source_feats)` computes features for new multipoles.

Calling `forward(...)` does both in one call. Caching is useful when the geometry is
fixed but the multipoles change.

Example:

```python
cache = features_block.precompute_geometry(
    k_vectors=k_vectors,
    k_norm2=k_norm2,
    k_vector_batch=k_vector_batch,
    k0_mask=k0_mask,
    node_positions=positions,
    batch=batch,
    volume=volume,
    pbc=pbc,
)

features = features_block.forward_dynamic(
    cache=cache,
    source_feats=multipoles,
)
```

Caches are mode-specific. A real-space cache must be used with a real-space dynamic
path, and a periodic cache must be used with a periodic dynamic path. The implementation
should check `cache["mode"]` for this; it should not inspect tensor values
such as `pbc` inside the hot dynamic path.

### Outputs and ordering

The output shape is `[n_nodes, n_features]`.

Feature channels are permuted into the ordering expected by existing downstream code.
This is handled internally by `output_permutation`.

## Boundary-condition modes

Boundary-condition handling is selected explicitly at construction time through
`pbc_handling`. Explicit modes bind fixed precompute and dynamic paths and avoid
data-dependent Python control flow in normal training, inference, and `torch.compile`
paths.

| `pbc_handling` | Evaluator | Correction | Intended use |
| --- | --- | --- | --- |
| `realspace` | real space | none beyond the real-space implementation | FFF open boundary |
| `pbc` | k-space | none | TTT, or deliberate uncorrected periodic path |
| `slab` | k-space | slab dipole correction | TTF slabs |
| `molecule_in_box` | k-space | molecule correction | FFF molecules evaluated through a periodic box |
| `mixed_periodic` | k-space | tensor-masked TTF/FFF corrections | mixed periodic-path batches |
| `auto` | runtime-selected | real-space for all-FFF batches, periodic mixed path otherwise | mixed datasets with charged non-periodic batches |

`auto` is useful when training batches may alternate between periodic systems and
highly charged non-periodic configurations, where a periodic calculation plus molecule
corrections is not accurate enough. If all graphs in the batch are non-periodic, `auto`
uses the real-space evaluator. If any graph is periodic, it uses the periodic
mixed-batch evaluator for the full batch. This mode inspects tensor data at runtime and
is not intended for compiled or performance-critical paths.

The explicit modes do not validate that `pbc` matches `pbc_handling` inside
`forward(...)`. Dataset preparation or higher-level code should enforce that the chosen
mode matches the batch.

### Realspace vs molecule-in-box

`realspace` and `molecule_in_box` are intentionally different.

- `realspace` uses `RealSpaceFiniteDifferenceElectrostaticFeatures`.
- `molecule_in_box` uses the periodic Fourier-space projection and then applies the
  molecule correction.

Use `molecule_in_box` when boxed molecules should stay on the periodic code path. Use
`realspace` for the open-boundary alternative.

### Corrections

Non-periodic corrections are applied after the k-space projection:

- `pbc`: no non-periodic correction.
- `slab`: TTF slab correction.
- `molecule_in_box`: FFF molecule correction.
- `mixed_periodic`: per-graph tensor-masked slab or molecule corrections.
- `realspace`: real-space evaluator only.

Optional quadrupole feature corrections are controlled by
`quadrupole_feature_corrections=True`.

### Self-interaction

If `include_self_interaction=False`, self-interaction terms are subtracted from the
projected features. If `include_self_interaction=True`, they are left in place.

### Multipole convention

Input multipoles are assumed to follow the Condon-Shortley phase convention. This
differs from some real-harmonic conventions used in e3nn; convert your inputs if they
are produced in a different basis.

## ESP evaluation

`compute_esps(cache, source_feats, pbc)` reconstructs electrostatic potentials at the
atomic positions from a periodic cache.

Current support is limited to:

- `pbc`
- `slab`

ESP reconstruction is undefined for `realspace`, `molecule_in_box`, and
`mixed_periodic` until explicit support is added.

## Implementation

The background for derivations is in `docs/maths/`.

### Definition

The electrostatic potential is:

$$
v(\mathbf{r}) =
\int
\frac{\rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}
d\mathbf{r}'.
$$

Features are projections of this potential onto local GTOs:

$$
v_{i,nlm} =
\int
v(\mathbf{r}) \phi_{nlm}(\mathbf{r}-\mathbf{r}_i)
d\mathbf{r}.
$$

### Periodic k-space evaluation

The periodic path mirrors the energy computation but includes a projection step.

1. Build the density coefficients:

$$
\tilde{\rho}(\mathbf{k}) =
\frac{(2\pi)^3}{\Omega}
\sum_{ilm}
p_{ilm}\,\tilde{\phi}_{nlm}(\mathbf{k})\,e^{-i\mathbf{k}\cdot \mathbf{r}_i}.
$$

2. Apply the Coulomb kernel:

$$
\tilde{v}(\mathbf{k}) =
\frac{4\pi}{\epsilon_0}
\frac{\tilde{\rho}(\mathbf{k})}{k^2}.
$$

3. Project onto the feature basis:

$$
v_{i,nlm} =
\frac{1}{(2\pi)^3}
\sum_{\mathbf{k}}
\tilde{v}(\mathbf{k})^{\star}
\tilde{\phi}_{nlm}(\mathbf{k})
e^{-i\mathbf{k}\cdot\mathbf{r}_i}.
$$

In code, only one half-space of k-vectors is stored, and real/imaginary parts are
handled explicitly. The k=0 term is masked by `k0_mask`.

### Real-space evaluation

`pbc_handling="realspace"` delegates to
`RealSpaceFiniteDifferenceElectrostaticFeatures`. This uses finite differences for
dipoles and is intended for open boundary conditions.
