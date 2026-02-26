# Core functionality

## charge density expanded in GTOs

Consider a set of atoms at positions $\{x_i\}_i$, which have atomic multipoles in spherical notation $p_{ilm}$. We introduce a gaussian type orbital (GTO) basis:

$$\phi_{nlm}(\mathbf{r}) = C_{l\sigma_n} e^{-\frac{r^2}{2\sigma_n^2}} r^{l} Y_{lm}(\hat{\mathbf{r}})$$

using this we construct a smooth charge density $\rho$. Given a single $n$, corresponding to $\sigma_n$, the charge density is defined as:

$$\rho(\mathbf{r}) = \sum_{ilm} p_{ilm} \phi_{nlm}(\mathbf{r}-\mathbf{r}_i)$$

Varying $n$ controls the smoothness of the density and provides Coulomb energy damping at short distances.

## electrostatic energy

You want to compute the electrostatic energy:

$$E =\frac{1}{2}\iint \frac{\rho(\mathbf{r}) \rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}d\mathbf{r}d\mathbf{r}'$$

The code provides one class, `GTOElectrostaticEnergy`, for computing this quantity in open or closed boundary conditions, for batches of configs, and with add-ons such self interaction terms or slab dipole corrections.

### example

```python
import torch
from scipy.constants import pi
from graph_longrange import compute_k_vectors_flat
from graph_longrange.energy import GTOElectrostaticEnergy

# inputs: positions [n_nodes, 3], batch [n_nodes], pbc [n_graph, 3]
# cell vectors [n_graph, 3, 3], volume [n_graph]
r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
    cutoff=kspace_cutoff,
    cell_vectors=cell,
    r_cell_vectors=r_cell,
)
#
# For a default cutoff, you can use:
# from graph_longrange.gto_utils import gto_basis_kspace_cutoff
# kspace_cutoff = gto_basis_kspace_cutoff(sigmas=[0.5], max_l=1)

energy_block = GTOElectrostaticEnergy(
    density_max_l=1,
    density_smearing_width=0.5,
    kspace_cutoff=kspace_cutoff,
    include_self_interaction=False,
    include_pbc_corrections=True,
)

energy = energy_block(
    k_vectors=k_vectors,
    k_norm2=k_norm2,
    k_vector_batch=k_vector_batch,
    k0_mask=k0_mask,
    source_feats=multipoles,  # [n_nodes, m_dim] or [n_nodes, 1, m_dim]
    node_positions=positions,
    batch=batch,
    volume=volume,
    pbc=pbc,
)
```

If all `pbc` are `False`, the class dispatches to the real-space evaluator; you do not
need to build the k-grid for that case.

## electrostatic features

Electrostatic features are local projections of the potential onto atom-centered GTOs.
They are computed by `GTOElectrostaticFeatures` (single channel) or
`GTOElectrostaticFeaturesMultiChannel` (independent channels).

### example

```python
from graph_longrange import compute_k_vectors_flat
from graph_longrange.features import GTOElectrostaticFeatures
from scipy.constants import pi

r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
    cutoff=kspace_cutoff,
    cell_vectors=cell,
    r_cell_vectors=r_cell,
)
#
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
    pbc=pbc,
)
```

You can also call `features_block(...)` directly, which performs both precompute and
dynamic steps in one call. The cache is useful when the geometry is fixed but the
source features change.

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
