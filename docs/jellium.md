# Jellium Slabs

This repo includes a jellium slab option for modeling electrochemical systems. The idea is to add a uniform charge density slab inside the simulation cell to compensate a net surface charge. The slab is static and is only used at inference time; it is not a physical atomistic component.

## Fourier-Series Representation

The slab is represented as a uniform charge density on a finite interval along the z direction:

- The slab spans the user-provided bounds `(z_lower, z_upper)` in the cell.
- The total slab charge is chosen to cancel the net atomic charge in the cell.

The Fourier series of the sharp-edged slab is computed analytically for k-vectors with `k_x = k_y = 0`, and is zero elsewhere. The k=0 term is set to the mean density so that the total charge matches the desired slab charge. The sharp slab is then smoothed by multiplying its Fourier series by a Gaussian factor in k-space, matching the density smearing width used for the atomic GTO basis.

## Energy and Features

Two new classes are provided:

- `JelliumSlabSolvatedEnergy`
- `JelliumSlabSolvatedFeatures`

Both classes construct a total k-space density:

```
rho_total(k) = rho_atoms(k) + rho_slab_smoothed(k)
```

The energy is computed from this combined density using the standard Coulomb kernel in k-space. This means the reported energy corresponds to the electrostatic energy of the combined atomic density plus the jellium slab density.

The features are computed by projecting the resulting electrostatic potential onto the local GTO feature basis, just as in `GTOElectrostaticFeatures`, and then applying a dipole correction using the combined dipole of the atoms and slab.

## Minimal Usage Example

```python
from graph_longrange.jellium_energy import JelliumSlabSolvatedEnergy
from graph_longrange.jellium_features import JelliumSlabSolvatedFeatures

# slab bounds are absolute z coordinates inside the cell
slab_bounds = (12.0, 22.0)

energy_block = JelliumSlabSolvatedEnergy(
    density_max_l=1,
    density_smearing_width=1.5,
    kspace_cutoff=8.0,
    slab_bounds=slab_bounds,
    include_self_interaction=False,
)

feature_block = JelliumSlabSolvatedFeatures(
    density_max_l=1,
    density_smearing_width=1.5,
    feature_max_l=1,
    feature_smearing_widths=[1.5, 2.0],
    kspace_cutoff=8.0,
    slab_bounds=slab_bounds,
    include_self_interaction=False,
    integral_normalization="receiver",
)
```
