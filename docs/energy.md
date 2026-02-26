# Electrostatic Energy

This file documents the energy computation provided by
`graph_longrange.energy.GTOElectrostaticEnergy`.

## Interface

### Constructor

`GTOElectrostaticEnergy` takes:
- `density_max_l`: maximum multipole order for the density expansion.
- `density_smearing_width`: Gaussian width $\sigma$ for the density basis.
- `kspace_cutoff`: cutoff for generating reciprocal vectors.
- `include_self_interaction`: whether to include self-interaction terms.
- `include_pbc_corrections`: whether to apply slab/molecule corrections.

### Forward arguments

`forward(...)` expects:
- `k_vectors`: flattened tensor `[n_k_total, 3]` of reciprocal vectors.
- `k_norm2`: flattened tensor `[n_k_total]` of squared norms.
- `k_vector_batch`: `[n_k_total]` mapping each k-vector to a graph id.
- `k0_mask`: `[n_k_total]`, `1.0` at the $\mathbf{k}=\mathbf{0}$ entry.
- `source_feats`: `[n_nodes, m_dim]` or `[n_nodes, 1, m_dim]` multipoles.
- `node_positions`: `[n_nodes, 3]`.
- `batch`: `[n_nodes]` graph id for each node.
- `volume`: `[n_graph]` cell volumes.
- `pbc`: `[n_graph, 3]` periodic flags.
- `force_pbc_evaluator`: forces k-space evaluation even if all `pbc` are false.

### Dispatch and boundary conditions

If any graph in the batch has `pbc=True` (or `force_pbc_evaluator=True`), the periodic
k-space evaluator is used for the full batch. If all graphs have `pbc=False`, the
real-space evaluator is used and no k-grid is required.

### Batching

The implementation uses flattened k-vectors grouped by graph, with `k_vector_batch`
used to mask cross-graph contributions. This is the fast path on GPU and is the only
batching strategy used by the current implementation.

### Corrections

When `include_pbc_corrections=True`, the following corrections are applied:
- Slabs (TTF): dipole correction.
- Molecules in boxes (FFF): monopole and dipole corrections.

No correction is applied for fully periodic systems (TTT). The correction terms are
selected per-graph from `pbc`.

### Multipole convention

Input multipoles are assumed to follow the Condon–Shortley phase convention. This
differs from some real-harmonic conventions used in e3nn; convert your inputs if they
are produced in a different basis.

## Implementation

### Definition

The electrostatic (Hartree) energy of a smooth charge density is

$$
E = \frac{1}{2}\iint \frac{\rho(\mathbf{r}) \rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}
d\mathbf{r}\,d\mathbf{r}'.
$$

The density is built by expanding atomic multipoles in a Gaussian type orbital (GTO)
basis (see `docs/maths/densities_and_projections.md` for the full derivation).

### Periodic (k-space) evaluation

For periodic systems, the code uses Fourier series coefficients on the reciprocal lattice
to compute the energy efficiently. The background for derviations is in `docs/maths/`.

1. Build the density coefficients:

$$
\tilde{\rho}(\mathbf{k}) = \frac{(2\pi)^3}{\Omega}
\sum_{ilm} p_{ilm}\,\tilde{\phi}_{nlm}(\mathbf{k})\,e^{-i\mathbf{k}\cdot \mathbf{r}_i}.
$$

2. Apply the Coulomb kernel in k-space:

$$
\tilde{v}(\mathbf{k}) = \frac{4\pi}{\epsilon_0}\frac{\tilde{\rho}(\mathbf{k})}{k^2},
\quad \mathbf{k}\neq\mathbf{0}.
$$

3. Combine with Parseval to form the energy:

$$
E = \frac{\Omega}{2(2\pi)^6}\sum_{\mathbf{k}\neq 0}
\frac{4\pi}{\epsilon_0 k^2}\left|\tilde{\rho}(\mathbf{k})\right|^2.
$$

In code, only one half-space of k-vectors is stored (typically $\mathbf{k}_x>0$),
and real/imaginary parts are handled explicitly. The k=0 term is masked by `k0_mask`.

The functions `assemble_fourier_series_batch` and `apply_coulomb_kernel_batch`
implement steps (1) and (2), and `energy_product_batch` implements step (3).

### Real-space evaluation

If all graphs have `pbc=False` (and `force_pbc_evaluator=False`), the class dispatches
to `RealSpaceFiniteDiffereneEnergy` and no k-grid is required. This uses finite
differences for dipoles and is intended for open boundary conditions.
