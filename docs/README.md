# Core functionality

`graph_longrange` provides PyTorch blocks for computing electrostatic quantities from
smooth atom-centered multipole densities. The main outputs are:

- electrostatic energies
- electrostatic feature projections onto local GTOs
- interpolation and auxiliary reciprocal-space quantities

## charge density expanded in GTOs

Consider a set of atoms at positions $`\{x_i\}_i`$, with atomic multipoles in spherical
notation $`p_{ilm}`$. We introduce a Gaussian type orbital (GTO) basis:

$$
\phi_{nlm}(\mathbf{r}) =
C_{l\sigma_n} e^{-\frac{r^2}{2\sigma_n^2}} r^{l} Y_{lm}(\hat{\mathbf{r}})
$$

Using this basis, a smooth charge density is constructed as:

$$
\rho(\mathbf{r}) =
\sum_{ilm} p_{ilm} \phi_{nlm}(\mathbf{r}-\mathbf{r}_i)
$$

The width $\sigma_n$ controls the smoothness of the density and damps the Coulomb
interaction at short distances.

## electrostatic energy

The electrostatic energy is:

$$
E =
\frac{1}{2}
\iint
\frac{\rho(\mathbf{r}) \rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}
d\mathbf{r}d\mathbf{r}'.
$$

The code provides `GTOElectrostaticEnergy` for evaluating this quantity with real-space
or Fourier-space paths.

### example

```python
import torch
from scipy.constants import pi
from graph_longrange import compute_k_vectors_flat
from graph_longrange.energy import GTOElectrostaticEnergy

# inputs: positions [n_nodes, 3], batch [n_nodes], pbc [n_graph, 3]
# cell vectors [n_graph, 3, 3], volume [n_graph]
r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
volume = torch.linalg.det(cell)
k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
    cutoff=kspace_cutoff,
    cell_vectors=cell,
    r_cell_vectors=r_cell,
)

# For a default cutoff, you can use:
# from graph_longrange.gto_utils import gto_basis_kspace_cutoff
# kspace_cutoff = gto_basis_kspace_cutoff(sigmas=[0.5], max_l=1)

energy_block = GTOElectrostaticEnergy(
    density_max_l=1,
    density_smearing_width=0.5,
    kspace_cutoff=kspace_cutoff,
    include_self_interaction=False,
    pbc_handling="mixed_periodic",
)

energy = energy_block(
    k_vectors=k_vectors,
    k_norm2=k_norm2,
    k_vector_batch=k_vector_batch,
    k0_mask=k0_mask,
    source_feats=multipoles,  # [n_nodes, m_dim]
    node_positions=positions,
    batch=batch,
    volume=volume,
    pbc=pbc,
)
```

For homogeneous datasets, prefer the most specific explicit mode: `realspace`, `pbc`,
`slab`, or `molecule_in_box`. Use `mixed_periodic` when a batch intentionally contains
multiple periodic-path boundary-condition types.

If derivatives with respect to the cell are needed, compute both `r_cell` and `volume`
from the same differentiable `cell` tensor. The k-space path is differentiable with
respect to the returned `k_vectors`, but the reciprocal cutoff is a hard truncation, so
derivatives are only piecewise defined when the selected k-grid changes.

## electrostatic features

Electrostatic features are local projections of the electrostatic potential onto
atom-centered GTOs:

$$
v_{i,nlm} =
\int v(\mathbf{r}) \phi_{nlm}(\mathbf{r}-\mathbf{r}_i) d\mathbf{r}.
$$

They are computed by `GTOElectrostaticFeatures`.

### example

```python
from scipy.constants import pi
from graph_longrange import compute_k_vectors_flat
from graph_longrange.features import GTOElectrostaticFeatures

r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
volume = torch.linalg.det(cell)
k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
    cutoff=kspace_cutoff,
    cell_vectors=cell,
    r_cell_vectors=r_cell,
)

# For a default cutoff, you can use:
# from graph_longrange.gto_utils import gto_basis_kspace_cutoff
# kspace_cutoff = gto_basis_kspace_cutoff(sigmas=[0.4, 0.8], max_l=1)

features_block = GTOElectrostaticFeatures(
    density_max_l=1,
    density_smearing_width=0.5,
    feature_max_l=1,
    feature_smearing_widths=[0.4, 0.8],
    include_self_interaction=False,
    kspace_cutoff=kspace_cutoff,
    pbc_handling="mixed_periodic",
)

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

You can also call `features_block(...)` directly, which performs both precompute and
dynamic steps in one call. The cache is useful when the geometry is fixed but the source
features change.

Caches are mode-specific. A real-space cache must be used with the real-space dynamic
path, and a periodic cache must be used with a periodic dynamic path.

## Bounadry Conditions

Boundary-condition handling is selected explicitly with `pbc_handling`.

Available modes:

- `realspace`: open-boundary real-space evaluator.
- `pbc`: periodic Fourier evaluator with no non-periodic correction.
- `slab`: periodic Fourier evaluator plus the TTF slab dipole correction.
- `molecule_in_box`: periodic Fourier evaluator plus the FFF molecule correction.
- `mixed_periodic`: periodic Fourier evaluator with tensor-masked per-graph corrections.
- `auto`: runtime-selected evaluator; inspects `pbc` at runtime.

Use explicit modes when the boundary-condition path is known. `auto` is useful when
training batches may alternate between periodic systems and highly charged
non-periodic configurations, where a periodic calculation plus molecule corrections is
not accurate enough. Because `auto` branches on `pbc` at runtime, explicit modes remain
preferred for compiled or performance-critical paths.

## current limitations and notes

`pbc_handling="auto"` uses the real-space evaluator when all graphs in the batch are
non-periodic. If any graph is periodic, it uses the periodic mixed-batch evaluator.
This can be useful for datasets that contain both periodic systems and highly charged
non-periodic configurations, provided the batching strategy keeps the highly charged
non-periodic systems in all-nonperiodic batches.

## interpolator functionality

The interpolator evaluates a real-valued Fourier series at arbitrary points. This is
useful for visualizing the density or potential on a grid that does not coincide with
the atom positions.

### example

```python
from graph_longrange.kspace import evaluate_fourier_series_at_points_flat

# fourier_coefficients: [n_k_total, 2] for a real series (Re, Im)
values = evaluate_fourier_series_at_points_flat(
    k_vectors=k_vectors,
    k_vector_batch=k_vector_batch,
    fourier_coefficients=fourier_coefficients,
    sample_points=probe_points,  # [n_points, 3]
    sample_batch=probe_batch,    # [n_points]
    k0_mask=k0_mask,
)
```

See `kspace.md` for details on the k-grid and coefficient conventions.
