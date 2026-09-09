import pytest
import torch
from scipy.constants import pi

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import (
    GTOElectrostaticFeatures,
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    project_to_features_batch,
)
from graph_longrange.jellium_energy import (
    JelliumSlabSolvatedEnergy,
    _smooth_slab_density,
    _slab_fourier_series_flat,
)
from graph_longrange.jellium_features import (
    JelliumSlabSolvatedFeatures,
    _slab_dipole_correction_node_fields_from_total_dipole,
)
from graph_longrange.kspace import (
    compute_k_vectors_flat,
    evaluate_fourier_series_at_points_flat,
)
from graph_longrange.slabs import get_nonperiodic_charge_dipole


@pytest.mark.parametrize(
    "cell_length, slab_bounds, charges",
    [
        (18.0, (2.0, 8.0), [-1.20, -0.30, -0.40, -0.10]),
        (14.0, (3.0, 9.0), [0.50, -0.75, 0.40, -0.15]),
    ],
)
@pytest.mark.parametrize("density_max_l", [0, 1])
def test_slab_fourier_series_reconstruction(
    cell_length, slab_bounds, charges, density_max_l
):
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    device = torch.device("cpu")

    cell = torch.diag(
        torch.tensor([cell_length, cell_length, cell_length], dtype=dtype, device=device)
    ).unsqueeze(0)
    r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        cutoff=12.0, cell_vectors=cell, r_cell_vectors=r_cell
    )

    node_positions = torch.tensor(
        [
            [2.0, 2.0, -1.0],
            [5.0, 4.0, 0.0],
            [7.5, 6.0, 0.0],
            [9.0, 8.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    batch = torch.zeros(node_positions.size(0), dtype=torch.long, device=device)
    volume = torch.det(cell).reshape(1)

    source_feats = torch.zeros(
        (node_positions.size(0), (density_max_l + 1) ** 2),
        dtype=dtype,
        device=device,
    )
    source_feats[:, 0] = torch.tensor(charges, dtype=dtype, device=device)
    if density_max_l == 1:
        source_feats[:, 1:4] = torch.tensor(
            [
                [0.10, -0.20, 0.30],
                [-0.15, 0.25, -0.05],
                [0.20, 0.05, -0.10],
                [-0.05, -0.10, 0.15],
            ],
            dtype=dtype,
            device=device,
        )

    total_atomic_charge = source_feats[:, 0].sum()
    slab_charge = -total_atomic_charge
    lower, upper = slab_bounds
    density = _slab_fourier_series_flat(
        total_charge=slab_charge,
        lower=torch.tensor(lower, dtype=dtype, device=device),
        upper=torch.tensor(upper, dtype=dtype, device=device),
        k_vectors=k_vectors,
        k0_mask=k0_mask,
        volume=volume,
    )

    k0_coefficients = density[k0_mask > 0.0]
    expected_k0_real = (2 * pi) ** 3 * slab_charge / volume[0]
    torch.testing.assert_close(
        k0_coefficients[:, 0],
        expected_k0_real.reshape(1),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        k0_coefficients[:, 1],
        torch.zeros_like(k0_coefficients[:, 1]),
        rtol=1e-12,
        atol=1e-12,
    )

    z_inside = 0.5 * (lower + upper)
    z_outside_low = lower - 2.5
    z_outside_high = upper + 2.5
    sample_points = torch.tensor(
        [
            [0.0, 0.0, z_inside],
            [0.0, 0.0, z_outside_low],
            [0.0, 0.0, z_outside_high],
        ],
        dtype=dtype,
        device=device,
    )
    sample_batch = torch.zeros(sample_points.size(0), dtype=torch.long, device=device)

    rho = evaluate_fourier_series_at_points_flat(
        k_vectors=k_vectors,
        k_vector_batch=k_vector_batch,
        fourier_coefficients=density,
        sample_points=sample_points,
        sample_batch=sample_batch,
        k0_mask=k0_mask,
    )

    volume_of_jellium = (upper - lower) * cell_length * cell_length
    expected_inside = slab_charge / volume_of_jellium
    atol = 1e-5
    rtol = 1e-5
    zero_tol = torch.maximum(
        torch.tensor(atol, dtype=dtype, device=device),
        torch.abs(expected_inside) * rtol,
    )

    assert torch.isclose(rho[0], expected_inside, atol=atol, rtol=rtol)
    assert torch.abs(rho[1]) < zero_tol
    assert torch.abs(rho[2]) < zero_tol


@pytest.mark.parametrize("density_max_l", [0, 1])
def test_zero_charge_jellium_matches_standard_slab_blocks(density_max_l):
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    device = torch.device("cpu")

    cell_length = 16.0
    slab_bounds = (4.0, 10.0)
    density_smearing_width = 0.7
    feature_smearing_widths = [0.8, 1.4]
    feature_max_l = density_max_l
    kspace_cutoff = 8.0

    cell = torch.diag(
        torch.tensor([cell_length, cell_length, cell_length], dtype=dtype, device=device)
    ).unsqueeze(0)
    r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        cutoff=kspace_cutoff, cell_vectors=cell, r_cell_vectors=r_cell
    )

    node_positions = torch.tensor(
        [
            [2.0, 3.0, 1.0],
            [5.0, 4.0, 3.0],
            [7.5, 6.0, 6.0],
            [11.0, 8.0, 9.0],
        ],
        dtype=dtype,
        device=device,
    )
    batch = torch.zeros(node_positions.size(0), dtype=torch.long, device=device)
    volume = torch.det(cell).reshape(1)
    pbc = torch.tensor([[True, True, False]], dtype=torch.bool, device=device)

    source_feats = torch.zeros(
        (node_positions.size(0), (density_max_l + 1) ** 2),
        dtype=dtype,
        device=device,
    )
    source_feats[:, 0] = torch.tensor(
        [0.75, -0.25, -0.375, -0.125],
        dtype=dtype,
        device=device,
    )
    torch.testing.assert_close(
        source_feats[:, 0].sum(),
        torch.zeros((), dtype=dtype, device=device),
        rtol=0.0,
        atol=0.0,
    )
    if density_max_l == 1:
        source_feats[:, 1:4] = torch.tensor(
            [
                [0.10, -0.20, 0.30],
                [-0.15, 0.25, -0.05],
                [0.20, 0.05, -0.10],
                [-0.05, -0.10, 0.15],
            ],
            dtype=dtype,
            device=device,
        )

    standard_energy_block = GTOElectrostaticEnergy(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=False,
        pbc_handling="slab",
    )
    jellium_energy_block = JelliumSlabSolvatedEnergy(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        kspace_cutoff=kspace_cutoff,
        slab_bounds=slab_bounds,
        include_self_interaction=False,
    )

    standard_energy = standard_energy_block(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        source_feats=source_feats,
        node_positions=node_positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    jellium_energy = jellium_energy_block(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        source_feats=source_feats,
        node_positions=node_positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    torch.testing.assert_close(jellium_energy, standard_energy, rtol=1e-9, atol=1e-10)

    standard_features_block = GTOElectrostaticFeatures(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_smearing_widths,
        include_self_interaction=False,
        kspace_cutoff=kspace_cutoff,
        pbc_handling="slab",
    )
    jellium_features_block = JelliumSlabSolvatedFeatures(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_smearing_widths,
        kspace_cutoff=kspace_cutoff,
        slab_bounds=slab_bounds,
        include_self_interaction=False,
    )

    standard_features = standard_features_block(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        source_feats=source_feats,
        node_positions=node_positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    jellium_features = jellium_features_block(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        source_feats=source_feats,
        node_positions=node_positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    torch.testing.assert_close(
        jellium_features, standard_features, rtol=1e-9, atol=1e-10
    )


def _jellium_features_direct_recompute(
    block: JelliumSlabSolvatedFeatures,
    cache: dict,
    source_feats: torch.Tensor,
) -> torch.Tensor:
    density = assemble_fourier_series_batch(
        source_feats=source_feats,
        cosines=cache["cosines"],
        sines=cache["sines"],
        density_basis_fs=cache["density_basis_fs"],
        volume_per_k=cache["volume_per_k"],
    )

    total_charge_atoms = source_feats[:, 0].sum()
    total_charge = -total_charge_atoms
    density = density + cache["smoothed_unit_slab_density"] * total_charge

    potential = apply_coulomb_kernel_batch(
        density=density,
        k_factor_coulomb=cache["k_factor_coulomb"],
    )
    features_si = project_to_features_batch(
        potential=potential,
        feature_basis_fs=cache["feature_basis_fs"],
        cosines=cache["cosines"],
        sines=cache["sines"],
        k_factor_proj=cache["k_factor_proj"],
    )
    features_flat = features_si.reshape(features_si.size(0), -1)
    features_flat = block._permute_output_channels(features_flat)

    if not block.include_self_interaction:
        si_terms = block.self_interaction_terms(source_feats)
        features_flat = features_flat - si_terms

    total_charge_atoms, total_dipole_atoms = get_nonperiodic_charge_dipole(
        source_feats,
        cache["node_positions"],
        cache["batch"],
    )
    z_center = 0.5 * (block.slab_lower + block.slab_upper)
    slab_com = torch.zeros(
        (1, 3),
        device=cache["node_positions"].device,
        dtype=cache["node_positions"].dtype,
    )
    slab_com[0, 2] = z_center
    total_dipole = total_dipole_atoms - slab_com * total_charge_atoms.unsqueeze(-1)
    node_fields = _slab_dipole_correction_node_fields_from_total_dipole(
        total_dipole=total_dipole,
        volumes=cache["volumes"],
        node_positions=cache["node_positions"],
        batch=cache["batch"],
    )
    correction_terms = block.displaced_interactions(
        batch=cache["batch"],
        positions=cache["node_positions"],
        node_fields=node_fields,
    )
    return features_flat + correction_terms


@pytest.mark.parametrize(
    "charges",
    [
        [0.75, -0.25, -0.375, -0.125],
        [0.75, -0.25, 0.10, -0.05],
        [-0.80, -0.30, 0.10, -0.20],
    ],
)
def test_jellium_feature_unit_slab_cache_matches_direct_recompute(charges):
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    device = torch.device("cpu")

    cell_length = 16.0
    slab_bounds = (4.0, 10.0)
    density_max_l = 1
    feature_max_l = 1
    density_smearing_width = 0.7
    feature_smearing_widths = [0.8, 1.4]
    kspace_cutoff = 8.0

    cell = torch.diag(
        torch.tensor([cell_length, cell_length, cell_length], dtype=dtype, device=device)
    ).unsqueeze(0)
    r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        cutoff=kspace_cutoff, cell_vectors=cell, r_cell_vectors=r_cell
    )

    node_positions = torch.tensor(
        [
            [2.0, 3.0, 1.0],
            [5.0, 4.0, 3.0],
            [7.5, 6.0, 6.0],
            [11.0, 8.0, 9.0],
        ],
        dtype=dtype,
        device=device,
    )
    batch = torch.zeros(node_positions.size(0), dtype=torch.long, device=device)
    volume = torch.det(cell).reshape(1)
    pbc = torch.tensor([[True, True, False]], dtype=torch.bool, device=device)

    source_feats = torch.zeros(
        (node_positions.size(0), (density_max_l + 1) ** 2),
        dtype=dtype,
        device=device,
    )
    source_feats[:, 0] = torch.tensor(charges, dtype=dtype, device=device)
    source_feats[:, 1:4] = torch.tensor(
        [
            [0.10, -0.20, 0.30],
            [-0.15, 0.25, -0.05],
            [0.20, 0.05, -0.10],
            [-0.05, -0.10, 0.15],
        ],
        dtype=dtype,
        device=device,
    )

    block = JelliumSlabSolvatedFeatures(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_smearing_widths,
        kspace_cutoff=kspace_cutoff,
        slab_bounds=slab_bounds,
        include_self_interaction=False,
    )
    cache = block.precompute_geometry(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        node_positions=node_positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )

    actual = block.forward_dynamic(cache=cache, source_feats=source_feats)
    expected = _jellium_features_direct_recompute(
        block=block,
        cache=cache,
        source_feats=source_feats,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
