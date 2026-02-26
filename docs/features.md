# Electrostatic Features

This file documents the feature computation provided by
`graph_longrange.features.GTOElectrostaticFeatures` and
`graph_longrange.features.GTOElectrostaticFeaturesMultiChannel`.

## Interface

### Classes

- `GTOElectrostaticFeatures`: single-channel features.
- `GTOElectrostaticFeaturesMultiChannel`: independent channels with no cross-coupling
  between channels. This is a vectorized wrapper around the same mathematics.

### Constructor

`GTOElectrostaticFeatures` takes:
- `density_max_l`: maximum multipole order for the density expansion.
- `density_smearing_width`: Gaussian width $\sigma$ for the density basis.
- `feature_max_l`: maximum order for the feature projection basis.
- `feature_smearing_widths`: list of Gaussian widths for projection.
- `include_self_interaction`: whether to include self-interaction terms.
- `kspace_cutoff`: cutoff for generating reciprocal vectors.
- `integral_normalization`: normalization used in the projection integrals.
- `quadrupole_feature_corrections`: include additional quadrupole corrections for
  non-periodic systems.

The multi-channel class takes the same arguments plus the optional
`quadrupole_feature_corrections` flag.

### Forward arguments

`forward(...)` expects:
- `k_vectors`: flattened tensor `[n_k_total, 3]` of reciprocal vectors.
- `k_norm2`: flattened tensor `[n_k_total]` of squared norms.
- `k_vector_batch`: `[n_k_total]` mapping each k-vector to a graph id.
- `k0_mask`: `[n_k_total]`, `1.0` at the $\mathbf{k}=\mathbf{0}$ entry.
- `source_feats`: multipoles, `[n_nodes, m_dim]` or `[n_nodes, 1, m_dim]`.
  The multi-channel class also accepts `[n_nodes, n_channels, m_dim]`.
- `node_positions`: `[n_nodes, 3]`.
- `batch`: `[n_nodes]` graph id for each node.
- `volume`: `[n_graph]` cell volumes.
- `pbc`: `[n_graph, 3]` periodic flags.
- `force_pbc_evaluator`: forces k-space evaluation even if all `pbc` are false.

### Precompute and dynamic paths

The class supports a cached path:
- `precompute_geometry(...)` caches geometry-dependent tensors.
- `forward_dynamic(cache, source_feats, pbc)` computes features for new multipoles.

Calling `forward(...)` does both in one call. Caching is useful when the geometry
is fixed but the multipoles change.

### Outputs and ordering

The output is `[n_nodes, n_features]` for the single-channel class and
`[n_nodes, n_channels, n_features]` for the multi-channel class. The feature ordering
interleaves radial channels by angular order to match legacy downstream code.

### Dispatch and boundary conditions

If any graph in the batch has `pbc=True` (or `force_pbc_evaluator=True`), the periodic
k-space evaluator is used for the full batch. If all graphs have `pbc=False`, the
real-space evaluator is used and no k-grid is required.

### Corrections

Non-periodic corrections are applied after the k-space projection:
- Slabs (TTF): dipole correction.
- Molecules in boxes (FFF): monopole and dipole corrections.
- Optional quadrupole feature corrections via `quadrupole_feature_corrections=True`.

No correction is applied for fully periodic systems (TTT). Correction terms are selected
per-graph from `pbc`.

### Self-interaction

If `include_self_interaction=False`, the self-interaction terms are subtracted from the
projected features. If `include_self_interaction=True`, the self-interaction terms are
left in place.

### Multipole convention

Input multipoles are assumed to follow the Condon–Shortley phase convention. This
differs from some real-harmonic conventions used in e3nn; convert your inputs if they
are produced in a different basis.

## Implementation

The background for derviations is in `docs/maths/`.

### Definition

The electrostatic potential is

$$
v(\mathbf{r}) = \int \frac{\rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}\,d\mathbf{r}'.
$$

Features are projections of this potential onto local GTOs:

$$
v_{i,nlm} = \int v(\mathbf{r}) \phi_{nlm}(\mathbf{r}-\mathbf{r}_i)\,d\mathbf{r}.
$$

### Periodic (k-space) evaluation

The periodic path mirrors the energy computation but includes a projection step.
Using Fourier series coefficients:

1. Build the density coefficients:

$$
\tilde{\rho}(\mathbf{k}) = \frac{(2\pi)^3}{\Omega}
\sum_{ilm} p_{ilm}\,\tilde{\phi}_{nlm}(\mathbf{k})\,e^{-i\mathbf{k}\cdot \mathbf{r}_i}.
$$

2. Apply the Coulomb kernel:

$$
\tilde{v}(\mathbf{k}) = \frac{4\pi}{\epsilon_0}\frac{\tilde{\rho}(\mathbf{k})}{k^2}.
$$

3. Project onto the feature basis:

$$
v_{i,nlm} = \frac{1}{(2\pi)^3}\sum_{\mathbf{k}}
\tilde{v}(\mathbf{k})\,\tilde{\phi}_{nlm}(\mathbf{k})\,e^{-i\mathbf{k}\cdot\mathbf{r}_i}.
$$

In code, only one half-space of k-vectors is stored and real/imaginary parts are handled
explicitly. The k=0 term is masked by `k0_mask`.

### Real-space evaluation

If all graphs have `pbc=False` (and `force_pbc_evaluator=False`), the class dispatches
to `RealSpaceFiniteDifferenceElectrostaticFeatures`. This uses finite differences for
dipoles and is intended for open boundary conditions.

### ESP evaluation

`compute_esps(cache, source_feats, pbc)` reconstructs the electrostatic potential at
node positions from the Fourier coefficients. It currently supports only TTT and TTF
geometries; for slabs (TTF) a dipole correction is added, and other cases raise an error.
