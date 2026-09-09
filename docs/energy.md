# Electrostatic Energy

## Interface

`GTOElectrostaticEnergy` takes:

- `density_max_l`: maximum multipole order for the density expansion.
- `density_smearing_width`: Gaussian width $\sigma$ for the density basis.
- `kspace_cutoff`: cutoff for generating reciprocal vectors.
- `include_self_interaction`: whether to include self-interaction terms.
- `pbc_handling`: boundary-condition/evaluator mode.

Example:

```python
from graph_longrange.energy import GTOElectrostaticEnergy

energy_block = GTOElectrostaticEnergy(
    density_max_l=1,
    density_smearing_width=0.5,
    kspace_cutoff=kspace_cutoff,
    include_self_interaction=False,
    pbc_handling="mixed_periodic",
)
```

`forward(...)` expects:

- `k_vectors`: flattened tensor `[n_k_total, 3]` of reciprocal vectors.
- `k_norm2`: flattened tensor `[n_k_total]` of squared norms.
- `k_vector_batch`: `[n_k_total]` mapping each k-vector to a graph id.
- `k0_mask`: `[n_k_total]`, `1.0` at the $\mathbf{k}=\mathbf{0}$ entry.
- `source_feats`: `[n_nodes, m_dim]` multipoles.
- `node_positions`: `[n_nodes, 3]`.
- `batch`: `[n_nodes]` graph id for each node.
- `volume`: `[n_graph]` cell volumes.
- `pbc`: `[n_graph, 3]` periodic flags.

Example:

```python
energy = energy_block(
    k_vectors=k_vectors,
    k_norm2=k_norm2,
    k_vector_batch=k_vector_batch,
    k0_mask=k0_mask,
    source_feats=multipoles,
    node_positions=positions,
    batch=batch,
    volume=volume,
    pbc=pbc,
)
```

### Boundary-condition modes

Boundary-condition handling is selected explicitly at construction time through
`pbc_handling`. Explicit modes bind a fixed evaluator path and avoid data-dependent
Python control flow in normal training, inference, and `torch.compile` paths.

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

- `realspace` uses the open-boundary real-space evaluator.
- `molecule_in_box` uses the periodic Fourier-space evaluator and then applies the
  molecule correction.

Use `molecule_in_box` when boxed molecules should stay on the periodic code path. Use
`realspace` for the open-boundary alternative.

### Batching

The periodic implementation uses flattened k-vectors grouped by graph, with
`k_vector_batch` used to mask cross-graph contributions. This is the fast path on GPU.

`mixed_periodic` still uses one periodic evaluator path for the whole batch. Correction
terms are selected with tensor masks rather than Python-side per-graph dispatch.

### Corrections

The explicit mode determines which correction is applied:

- `pbc`: no non-periodic correction.
- `slab`: TTF slab dipole correction.
- `molecule_in_box`: FFF monopole/dipole molecule correction.
- `mixed_periodic`: per-graph tensor-masked slab or molecule corrections.
- `realspace`: real-space evaluator only.

Selecting `pbc` for a molecular system intentionally gives an uncorrected periodic
calculation. This is uncommon, but is available for users who deliberately want that
behavior.

### Self-interaction

If `include_self_interaction=False`, self-interaction terms are subtracted from the
periodic energy. If `include_self_interaction=True`, they are left in place.

### Multipole convention

Input multipoles are assumed to follow the Condon-Shortley phase convention. This
differs from some real-harmonic conventions used in e3nn; convert your inputs if they
are produced in a different basis.

## Implementation

### Definition

The electrostatic/Hartree energy of a smooth charge density is:

$$
E =
\frac{1}{2}
\iint
\frac{\rho(\mathbf{r}) \rho(\mathbf{r}')}{4\pi\epsilon_0|\mathbf{r}-\mathbf{r}'|}
d\mathbf{r}\,d\mathbf{r}'.
$$

The density is built by expanding atomic multipoles in a GTO basis:

$$
\rho(\mathbf{r}) =
\sum_{ilm} p_{ilm} \phi_{nlm}(\mathbf{r}-\mathbf{r}_i).
$$

See `docs/maths/densities_and_projections.md` for the density conventions and
normalization details.

### Periodic k-space evaluation

For periodic modes, the code uses Fourier-series coefficients on the reciprocal lattice.

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
\frac{\tilde{\rho}(\mathbf{k})}{k^2},
\quad \mathbf{k}\neq\mathbf{0}.
$$

3. Combine with Parseval to form the energy:

$$
E =
\frac{\Omega}{2(2\pi)^6}
\sum_{\mathbf{k}\neq 0}
\frac{4\pi}{\epsilon_0 k^2}
\left|\tilde{\rho}(\mathbf{k})\right|^2.
$$

In code, only one half-space of k-vectors is stored, and real/imaginary parts are
handled explicitly. The k=0 term is masked by `k0_mask`.

The functions `assemble_fourier_series_batch` and `apply_coulomb_kernel_batch`
implement the Fourier assembly and Coulomb application. `energy_product_batch`
implements the Parseval contraction.