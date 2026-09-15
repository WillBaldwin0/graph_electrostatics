# External electrostatic sources

Source-target field features live in `external_source_features.py`; damped
real-space helpers live in `external_source_realspace.py`. The ordinary
`GTOElectrostaticFeatures` block retains its same-set API. Wrap an existing
block with `GTOElectrostaticExternalSourceFeatures.from_features(base)` to use
`forward_source_target`, or the split `precompute_geometry_source_target` and
`forward_dynamic_source_target` calls. The wrapper reuses the existing basis
objects and parameters without rebuilding them.

`GTOElectrostaticCrossEnergy` evaluates the electrostatic interaction between
two disjoint sets of GTO multipoles.  It is intended for dynamic environments
such as electrostatic ML/MM embedding, while remaining independent of MACE and
OpenMM.

The result is

```text
E_cross(A, B) = E(A + B) - E(A) - E(B)
```

and therefore contains neither `A-A` nor `B-B` interactions.  The optimized
implementation evaluates the damped bipartite interaction directly in real
space and uses separately assembled Fourier densities in reciprocal space.
Both paths use the same `multipoles` normalization as
`GTOElectrostaticEnergy`.

```python
from graph_longrange.external_source_energy import GTOElectrostaticCrossEnergy

cross = GTOElectrostaticCrossEnergy(
    density_max_l=1,
    density_smearing_width=1.0,
    kspace_cutoff=8.0,
    pbc_handling="auto",
)

energy_ab = cross(
    k_vectors=k_vectors,
    k_norm2=k_norm2,
    k_vector_batch=k_vector_batch,
    k0_mask=k0_mask,
    source_feats=source_multipoles,
    source_positions=source_positions,
    source_batch=source_batch,
    target_feats=environment_multipoles,
    target_positions=environment_positions,
    target_batch=environment_batch,
    volume=volume,
    pbc=pbc,
)
```

The two position tensors remain in the autograd graph, so differentiating this
energy produces equal-and-opposite reaction forces on the two subsystems.

## Drop-in energy and feature wrappers

For host models that already call `GTOElectrostaticEnergy` and
`GTOElectrostaticFeatures`, graph_longrange also provides wrappers that retain
the same-set calculation and add a dynamic external contribution:

```python
from graph_longrange.external_source_energy import (
    GTOElectrostaticExternalSourceEnergy,
)
from graph_longrange.external_source_features import (
    GTOElectrostaticExternalSourceFeatures,
)

model.coulomb_energy = GTOElectrostaticExternalSourceEnergy.from_energy(
    model.coulomb_energy
)
model.electric_potential_descriptor = (
    GTOElectrostaticExternalSourceFeatures.from_features(
        model.electric_potential_descriptor,
        external_scale=0.5,  # two equal spin channels in PolarMACE
    )
)

for block in (model.coulomb_energy, model.electric_potential_descriptor):
    block.set_external_sources(
        external_feats=environment_multipoles,
        external_positions=environment_positions,
        external_batch=environment_batch,
    )
```

The energy wrapper returns `E(A) + E_cross(A, B)`, which is equal to
`E(A + B) - E(B)`.  The feature wrapper returns the normal same-set features
plus `external_scale` times the source-to-target field.  Adapters remain
responsible for clearing the transient external sources after each evaluation.
