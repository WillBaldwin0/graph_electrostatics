# `graph_longrange`

`graph_longrange` provides blah in pytorch, for use in X. The repo works mostly with Gaussian type orbitals, and provides the following fundamental operations. All the code is pytorch and for systems up to a few thousand atoms on a single GPU, the code is reasonably fast compared to many modern MLIPs.

- [x] **Coulomb Energy** from a set of atoms with **atomic multipole moments**
- [x] **Atom Centered Electrostatic Features**, similar to the LODE descriptor 
- [x] Damped Coulomb Interactions via **Gaussian Type Orbitals**, instead of point charges
- [x] Consistent **Realspace and Periodic** Implementations
- [x] Correction terms to energy and features for handling **slab geometries**, as well as molecules in boxes
- [x] Efficient **batching** for pytorch training
- [x] Considerable **precopmutation** for electrostatic features to speed up self-consistent-field loops
- [x] Interpolation functions for probing the potential and density around atomistic systems

Additional functionality will also be coming in the near future, including:

- [ ] Confined Jellium slabs of charge, useful when simulating electrochemical interfaces

This repo currently supports several long-range MLIP architectures including the MACE-POLAR-1 foundation model.
