"""Tests for the analytical real-space Gaussian multipole electrostatics.

Organized in four layers:
1. kernel-level oracles: symbolic derivatives, series/closed-form branch
   consistency, zero-distance and point-multipole limits;
2. first-principles quadrature oracles: radial shell integration for the
   potential, Gauss-Hermite quadrature for the pair energy;
3. consistency with the existing modules: the (already exact) l=0 path,
   first-order Richardson extrapolation of the finite-difference modules
   (their displaced-charge scheme is a forward difference, so the leading
   error is O(offset)), and the GTO self-interaction block;
4. symmetry, gradients, and batch semantics.
"""

import math

import numpy as np
import pytest
import scipy.integrate
import scipy.special
import sympy
import torch
from scipy.constants import pi

from graph_longrange.realspace_electrostatics import (
    RealSpaceAnalyticalElectrostaticFeatures,
    RealSpaceAnalyticalEnergy,
    RealSpaceFiniteDifferenceElectrostaticFeatures,
    RealSpaceFiniteDiffereneEnergy,
    _smeared_coulomb_kernels_closed_form,
    _smeared_coulomb_kernels_series,
    smeared_coulomb_kernels,
)
from graph_longrange.utils import FIELD_CONSTANT

torch.set_default_dtype(torch.float64)

DENSITY_SMEARING_WIDTH = 1.5
PROJECTION_SMEARING_WIDTHS = [1.0, 2.0]

# e3nn l=1 component order is (y, z, x); these permute between the two orders.
E3NN_TO_CARTESIAN_COLUMNS = [3, 1, 2]
CARTESIAN_TO_E3NN_COMPONENTS = [1, 2, 0]
E3NN_TO_CARTESIAN_COMPONENTS = [2, 0, 1]


def random_system(num_graphs=1, atoms_per_graph=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    num_atoms = num_graphs * atoms_per_graph
    positions = 1.5 * torch.randn((num_atoms, 3), generator=generator)
    source_feats = 0.5 * torch.randn((num_atoms, 4), generator=generator)
    batch = torch.repeat_interleave(
        torch.arange(num_graphs, dtype=torch.long), atoms_per_graph
    )
    return source_feats, positions, batch


def random_rotation_matrix(seed=42):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((3, 3), generator=generator)
    orthogonal, upper_triangular = torch.linalg.qr(matrix)
    orthogonal = orthogonal * torch.sign(torch.diagonal(upper_triangular))
    if torch.det(orthogonal) < 0:
        orthogonal[:, 0] = -orthogonal[:, 0]
    return orthogonal


def rotate_source_features(source_feats, rotation):
    rotated = source_feats.clone()
    dipoles_cartesian = source_feats[:, E3NN_TO_CARTESIAN_COLUMNS]
    rotated[:, E3NN_TO_CARTESIAN_COLUMNS] = dipoles_cartesian @ rotation.T
    return rotated


def rotate_feature_vector_blocks(features, rotation, num_radial):
    """Rotate the l=1 blocks of a [n, 4 * num_radial] feature tensor."""
    rotated = features.clone()
    vectors = features[:, num_radial:].reshape(features.shape[0], num_radial, 3)
    vectors_cartesian = vectors[..., E3NN_TO_CARTESIAN_COMPONENTS]
    rotated_cartesian = vectors_cartesian @ rotation.T
    rotated[:, num_radial:] = rotated_cartesian[
        ..., CARTESIAN_TO_E3NN_COMPONENTS
    ].reshape(features.shape[0], num_radial * 3)
    return rotated


# ---------------------------------------------------------------------------
# 1. Kernel-level oracles
# ---------------------------------------------------------------------------


def test_kernels_match_symbolic_derivatives():
    distance_symbol, width_symbol = sympy.symbols("R Sigma", positive=True)
    kernel_0 = sympy.erf(distance_symbol / (sympy.sqrt(2) * width_symbol)) / (
        distance_symbol
    )
    kernel_1 = -sympy.diff(kernel_0, distance_symbol) / distance_symbol
    kernel_2 = -sympy.diff(kernel_1, distance_symbol) / distance_symbol
    symbolic_kernels = [kernel_0, kernel_1, kernel_2]

    for combined_width in [0.5, 1.5, 2.5]:
        for scaled_distance_squared in np.logspace(-6, 2, 9):
            distance = combined_width * math.sqrt(2.0 * scaled_distance_squared)
            computed = smeared_coulomb_kernels(
                torch.tensor([distance**2]), combined_width, highest_order=2
            )
            for order in range(3):
                expected = float(
                    symbolic_kernels[order].evalf(
                        30,
                        subs={
                            distance_symbol: distance,
                            width_symbol: combined_width,
                        },
                    )
                )
                assert computed[order][0].item() == pytest.approx(
                    expected, rel=1e-12
                ), f"order {order}, t={scaled_distance_squared}, width={combined_width}"


def test_kernel_branch_consistency_values_and_derivatives():
    combined_width = DENSITY_SMEARING_WIDTH
    scaled = torch.linspace(0.25, 0.6, 21, dtype=torch.float64)

    scaled_series = scaled.clone().requires_grad_(True)
    scaled_closed = scaled.clone().requires_grad_(True)
    series = _smeared_coulomb_kernels_series(scaled_series, combined_width, 2)
    closed = _smeared_coulomb_kernels_closed_form(scaled_closed, combined_width, 2)

    for order in range(3):
        torch.testing.assert_close(
            series[order], closed[order], rtol=5e-12, atol=0.0
        )
        series_grad = torch.autograd.grad(
            series[order].sum(), scaled_series, retain_graph=True
        )[0]
        closed_grad = torch.autograd.grad(
            closed[order].sum(), scaled_closed, retain_graph=True
        )[0]
        torch.testing.assert_close(series_grad, closed_grad, rtol=5e-11, atol=0.0)


def test_kernels_at_zero_distance():
    combined_width = 2.0
    kernels = smeared_coulomb_kernels(
        torch.tensor([0.0]), combined_width, highest_order=2
    )
    for order in range(3):
        expected = (
            math.sqrt(2.0 / pi)
            / (combined_width ** (2 * order + 1))
            / (2 * order + 1)
        )
        assert kernels[order][0].item() == pytest.approx(expected, rel=1e-14)


def test_kernels_far_field_point_multipole_limit():
    combined_width = 1.5
    distance = 20.0 * combined_width
    kernels = smeared_coulomb_kernels(
        torch.tensor([distance**2]), combined_width, highest_order=2
    )
    assert kernels[0][0].item() == pytest.approx(1.0 / distance, rel=1e-10)
    assert kernels[1][0].item() == pytest.approx(1.0 / distance**3, rel=1e-10)
    assert kernels[2][0].item() == pytest.approx(3.0 / distance**5, rel=1e-10)


def test_kernels_float32_accuracy():
    combined_width = DENSITY_SMEARING_WIDTH
    scaled = np.logspace(-6, 2, 25)
    distances_squared = 2.0 * combined_width**2 * scaled
    reference = smeared_coulomb_kernels(
        torch.tensor(distances_squared, dtype=torch.float64),
        combined_width,
        highest_order=2,
    )
    single_precision = smeared_coulomb_kernels(
        torch.tensor(distances_squared, dtype=torch.float32),
        combined_width,
        highest_order=2,
    )
    for order in range(3):
        relative_error = (
            single_precision[order].double() - reference[order]
        ).abs() / reference[order].abs()
        assert relative_error.max().item() < 1e-5


# ---------------------------------------------------------------------------
# 2. First-principles quadrature oracles
# ---------------------------------------------------------------------------


def _gaussian_density_radial(radius, smearing_width):
    return (2 * pi * smearing_width**2) ** -1.5 * np.exp(
        -(radius**2) / (2 * smearing_width**2)
    )


def test_potential_matches_radial_shell_quadrature():
    """Potential of a unit charge / unit dipole Gaussian on the z-axis."""
    smearing_width = 1.3

    def monopole_potential(radius):
        inner, _ = scipy.integrate.quad(
            lambda s: s**2 * _gaussian_density_radial(s, smearing_width),
            0.0,
            radius,
            epsabs=1e-14,
        )
        outer, _ = scipy.integrate.quad(
            lambda s: s * _gaussian_density_radial(s, smearing_width),
            radius,
            np.inf,
            epsabs=1e-14,
        )
        return 4 * pi * (inner / radius + outer)

    def dipole_potential_on_axis(radius):
        # density mu_z * (z / width^2) * gaussian = f(s) * cos(theta)
        def radial_part(s):
            return s / smearing_width**2 * _gaussian_density_radial(s, smearing_width)

        inner, _ = scipy.integrate.quad(
            lambda s: s**3 * radial_part(s), 0.0, radius, epsabs=1e-14
        )
        outer, _ = scipy.integrate.quad(radial_part, radius, np.inf, epsabs=1e-14)
        return 4 * pi / 3 * (inner / radius**2 + outer * radius)

    for radius in [0.3, 1.0, 3.0, 6.0]:
        kernels = smeared_coulomb_kernels(
            torch.tensor([radius**2]), smearing_width, highest_order=1
        )
        assert kernels[0][0].item() == pytest.approx(
            monopole_potential(radius), rel=1e-10
        )
        # analytic on-axis dipole potential per unit moment: radius * B_1
        assert radius * kernels[1][0].item() == pytest.approx(
            dipole_potential_on_axis(radius), rel=1e-10
        )


def test_pair_energy_matches_gauss_hermite_quadrature():
    """The full l<=1 pair formula against 3D quadrature, unequal widths."""
    width_1, width_2 = 1.2, 1.7
    combined_width = math.sqrt(width_1**2 + width_2**2)

    rng = np.random.default_rng(7)
    charge_1, charge_2 = 0.7, -0.4
    dipole_1 = rng.normal(size=3) * 0.5
    dipole_2 = rng.normal(size=3) * 0.5
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    nodes, weights = np.polynomial.hermite.hermgauss(32)
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    grid_weights = (
        weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    ).reshape(-1)

    def single_center_potential(points, center, charge, dipole, smearing_width):
        displacement = points - center
        distance = np.linalg.norm(displacement, axis=-1)
        kernel_0 = scipy.special.erf(
            distance / (math.sqrt(2) * smearing_width)
        ) / distance
        gaussian_term = (
            math.sqrt(2 / pi)
            / smearing_width
            * np.exp(-(distance**2) / (2 * smearing_width**2))
        )
        kernel_1 = (kernel_0 - gaussian_term) / distance**2
        return charge * kernel_0 + (displacement @ dipole) * kernel_1

    for separation_over_width in [0.0, 0.3, 1.0, 3.0]:
        center_1 = np.zeros(3)
        center_2 = separation_over_width * combined_width * direction
        separation = center_2 - center_1

        # quadrature: E = integral rho_1(r) V_2(r) d^3r, r = center_1 + sqrt(2) w u
        points = center_1 + math.sqrt(2) * width_1 * grid
        density_polynomial = charge_1 + (points - center_1) @ dipole_1 / width_1**2
        potential_values = single_center_potential(
            points, center_2, charge_2, dipole_2, width_2
        )
        quadrature_energy = pi**-1.5 * np.sum(
            grid_weights * density_polynomial * potential_values
        )

        kernels = smeared_coulomb_kernels(
            torch.tensor([float(separation @ separation)]),
            combined_width,
            highest_order=2,
        )
        kernel_0, kernel_1, kernel_2 = (kernel[0].item() for kernel in kernels)
        formula_energy = (
            charge_1 * charge_2 * kernel_0
            - (charge_1 * dipole_2 - charge_2 * dipole_1) @ separation * kernel_1
            + dipole_1 @ dipole_2 * kernel_1
            - (dipole_1 @ separation) * (dipole_2 @ separation) * kernel_2
        )
        assert formula_energy == pytest.approx(quadrature_energy, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Consistency with the existing modules
# ---------------------------------------------------------------------------


def test_l0_energy_matches_finite_difference_module():
    # The finite-difference l=0 path is analytically exact up to its 1e-6
    # distance regularizer, which biases the kernel at the ~1e-6 relative level.
    source_feats, positions, batch = random_system(seed=1)
    monopole_feats = source_feats[:, :1]
    analytical_energy = RealSpaceAnalyticalEnergy(0, DENSITY_SMEARING_WIDTH)(
        monopole_feats, positions, batch
    )
    finite_difference_energy = RealSpaceFiniteDiffereneEnergy(
        0, DENSITY_SMEARING_WIDTH
    )(monopole_feats, positions, batch)
    torch.testing.assert_close(
        analytical_energy, finite_difference_energy, rtol=5e-6, atol=1e-12
    )


def test_l0_features_match_finite_difference_module():
    source_feats, positions, batch = random_system(seed=2)
    monopole_feats = source_feats[:, :1]
    analytical_features = RealSpaceAnalyticalElectrostaticFeatures(
        0, DENSITY_SMEARING_WIDTH, 0, PROJECTION_SMEARING_WIDTHS
    )(monopole_feats, positions, batch)[0]
    finite_difference_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
        0, DENSITY_SMEARING_WIDTH, 0, PROJECTION_SMEARING_WIDTHS
    )(monopole_feats, positions, batch)[0]
    scale = finite_difference_features.abs().max()
    assert (
        analytical_features - finite_difference_features
    ).abs().max() < 5e-6 * scale


def test_energy_finite_difference_error_is_first_order_with_richardson():
    """The displaced-charge scheme is a forward difference: O(offset) error.

    First-order Richardson extrapolation 2 E(a/2) - E(a) must converge as
    O(offset^2) to the analytical energy.
    """
    source_feats, positions, batch = random_system(seed=3)
    analytical_energy = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)(
        source_feats, positions, batch
    )[0]

    offsets = [0.04, 0.02, 0.01]
    energies = {
        offset: RealSpaceFiniteDiffereneEnergy(
            1, DENSITY_SMEARING_WIDTH, offset=offset
        )(source_feats, positions, batch)[0]
        for offset in offsets
    }
    errors = {
        offset: (energies[offset] - analytical_energy).item() for offset in offsets
    }

    assert errors[0.04] / errors[0.02] == pytest.approx(2.0, rel=0.05)
    assert errors[0.02] / errors[0.01] == pytest.approx(2.0, rel=0.05)

    # The Richardson residual cannot be pushed below the bias of the 1e-6
    # distance regularizer inside the finite-difference kernel (~1e-6 relative,
    # absent from the analytical module), so only smallness is asserted here.
    richardson_coarse = (2 * energies[0.02] - energies[0.04] - analytical_energy).item()
    richardson_fine = (2 * energies[0.01] - energies[0.02] - analytical_energy).item()
    assert abs(richardson_coarse) < 0.05 * abs(errors[0.04])
    assert abs(richardson_fine) < 0.02 * abs(errors[0.02])


def test_features_finite_difference_error_is_first_order_with_richardson():
    source_feats, positions, batch = random_system(seed=4)
    analytical_features = RealSpaceAnalyticalElectrostaticFeatures(
        1, DENSITY_SMEARING_WIDTH, 1, PROJECTION_SMEARING_WIDTHS
    )(source_feats, positions, batch)[0]

    offsets = [0.1, 0.05, 0.025]
    features = {
        offset: RealSpaceFiniteDifferenceElectrostaticFeatures(
            1, DENSITY_SMEARING_WIDTH, 1, PROJECTION_SMEARING_WIDTHS, offset=offset
        )(source_feats, positions, batch)[0]
        for offset in offsets
    }
    errors = {
        offset: (features[offset] - analytical_features).norm().item()
        for offset in offsets
    }
    assert errors[0.1] / errors[0.05] == pytest.approx(2.0, rel=0.05)
    assert errors[0.05] / errors[0.025] == pytest.approx(2.0, rel=0.05)

    richardson = (
        2 * features[0.05] - features[0.1] - analytical_features
    ).norm().item()
    assert richardson < 0.05 * errors[0.1]


@pytest.mark.parametrize("density_max_l", [0, 1])
@pytest.mark.parametrize("smearing_width", [0.8, 1.5])
def test_self_energy_matches_gto_self_interaction_block(
    density_max_l, smearing_width
):
    """Single atom: the energy is purely the (independently integrated) self term."""
    source_feats, positions, _ = random_system(seed=5, atoms_per_graph=1)
    source_feats = source_feats[:, : (density_max_l + 1) ** 2]
    batch = torch.zeros(1, dtype=torch.long)

    module_energy = RealSpaceAnalyticalEnergy(
        density_max_l, smearing_width, include_self_interaction=True
    )(source_feats, positions, batch)[0]

    combined_width = math.sqrt(2.0) * smearing_width
    charge = source_feats[0, 0]
    analytic_energy = (
        0.5
        * FIELD_CONSTANT
        / (4 * pi)
        * charge**2
        * math.sqrt(2 / pi)
        / combined_width
    )
    if density_max_l >= 1:
        dipole_norm_squared = (source_feats[0, 1:] ** 2).sum()
        analytic_energy = analytic_energy + (
            0.5
            * FIELD_CONSTANT
            / (4 * pi)
            * dipole_norm_squared
            * math.sqrt(2 / pi)
            / (3 * combined_width**3)
        )
    assert module_energy.item() == pytest.approx(analytic_energy.item(), rel=1e-8)


def test_energy_and_features_are_mutually_consistent():
    """E = 1/2 sum_i (q_i V_i + mu_i . grad V_i) using moment-normalized features
    projected with the density smearing width (same combined width as the energy)."""
    source_feats, positions, batch = random_system(seed=6)
    energy = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)(
        source_feats, positions, batch
    )[0]
    features = RealSpaceAnalyticalElectrostaticFeatures(
        1,
        DENSITY_SMEARING_WIDTH,
        1,
        [DENSITY_SMEARING_WIDTH],
        integral_normalization="multipoles",
    )(source_feats, positions, batch)[0]
    potential = features[:, 0]
    potential_gradient_e3nn = features[:, 1:]
    energy_from_features = 0.5 * (
        source_feats[:, 0] * potential
        + (source_feats[:, 1:] * potential_gradient_e3nn).sum(-1)
    ).sum()
    assert energy_from_features.item() == pytest.approx(energy.item(), rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Symmetry, gradients, and batch semantics
# ---------------------------------------------------------------------------


def test_energy_rotation_invariance():
    source_feats, positions, batch = random_system(seed=8)
    rotation = random_rotation_matrix()
    rotated_positions = positions @ rotation.T
    rotated_feats = rotate_source_features(source_feats, rotation)

    analytical_module = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)
    energy = analytical_module(source_feats, positions, batch)[0]
    rotated_energy = analytical_module(rotated_feats, rotated_positions, batch)[0]
    assert abs((rotated_energy - energy).item()) < 1e-12 * max(
        1.0, abs(energy.item())
    )

    # Document the bug being fixed: the finite-difference module violates
    # rotational invariance at the O(offset) level.
    finite_difference_module = RealSpaceFiniteDiffereneEnergy(
        1, DENSITY_SMEARING_WIDTH, offset=0.02
    )
    violation = (
        finite_difference_module(rotated_feats, rotated_positions, batch)[0]
        - finite_difference_module(source_feats, positions, batch)[0]
    )
    assert abs(violation.item()) > 1e-6


def test_features_rotation_equivariance():
    source_feats, positions, batch = random_system(seed=9)
    rotation = random_rotation_matrix(seed=43)
    rotated_positions = positions @ rotation.T
    rotated_feats = rotate_source_features(source_feats, rotation)

    module = RealSpaceAnalyticalElectrostaticFeatures(
        1, DENSITY_SMEARING_WIDTH, 1, PROJECTION_SMEARING_WIDTHS
    )
    features = module(source_feats, positions, batch)[0]
    features_of_rotated = module(rotated_feats, rotated_positions, batch)[0]
    rotated_features = rotate_feature_vector_blocks(
        features, rotation, len(PROJECTION_SMEARING_WIDTHS)
    )
    torch.testing.assert_close(
        features_of_rotated, rotated_features, rtol=1e-10, atol=1e-12
    )


def test_energy_translation_and_permutation_invariance():
    source_feats, positions, batch = random_system(seed=10, atoms_per_graph=4)
    module = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)
    energy = module(source_feats, positions, batch)[0]

    translated_energy = module(
        source_feats, positions + torch.tensor([3.0, -2.0, 5.0]), batch
    )[0]
    torch.testing.assert_close(translated_energy, energy, rtol=1e-12, atol=1e-14)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_energy = module(
        source_feats[permutation], positions[permutation], batch
    )[0]
    torch.testing.assert_close(permuted_energy, energy, rtol=1e-12, atol=1e-14)


def test_batch_semantics_no_cross_graph_leakage():
    energy_module = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)
    features_module = RealSpaceAnalyticalElectrostaticFeatures(
        1, DENSITY_SMEARING_WIDTH, 1, PROJECTION_SMEARING_WIDTHS
    )

    molecules = [random_system(seed=seed, atoms_per_graph=count)[:2]
                 for seed, count in [(11, 2), (12, 3), (13, 4)]]
    source_feats = torch.cat([feats for feats, _ in molecules])
    positions = torch.cat([pos for _, pos in molecules])
    batch = torch.repeat_interleave(
        torch.arange(3, dtype=torch.long), torch.tensor([2, 3, 4])
    )

    batched_energies = energy_module(source_feats, positions, batch)
    batched_features = features_module(source_feats, positions, batch)[0]

    node_offset = 0
    for graph_index, (feats, pos) in enumerate(molecules):
        single_batch = torch.zeros(feats.shape[0], dtype=torch.long)
        single_energy = energy_module(feats, pos, single_batch)[0]
        torch.testing.assert_close(
            batched_energies[graph_index], single_energy, rtol=1e-13, atol=1e-14
        )
        single_features = features_module(feats, pos, single_batch)[0]
        torch.testing.assert_close(
            batched_features[node_offset : node_offset + feats.shape[0]],
            single_features,
            rtol=1e-13,
            atol=1e-14,
        )
        node_offset += feats.shape[0]


def test_directed_sign_convention():
    """Charge on one atom, dipole on the other, hand-computed signs.

    Convention: separation = positions[receiver] - positions[sender]. With a
    charge q at the origin and a dipole mu_z along +z at distance d, the pair
    energy is E = -(q * mu_z * d) * B_1(d) * FIELD_CONSTANT / (4 pi): a dipole
    pointing away from a positive charge is repelled (positive end further out
    in a decaying potential -> negative interaction energy).
    """
    distance = 2.0
    charge = 1.0
    dipole_z = 0.5
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    batch = torch.zeros(2, dtype=torch.long)
    source_feats = torch.zeros((2, 4))
    source_feats[0, 0] = charge
    source_feats[1, 2] = dipole_z  # e3nn order (y, z, x): column 2 is z

    combined_width = math.sqrt(2.0) * DENSITY_SMEARING_WIDTH
    kernels = smeared_coulomb_kernels(
        torch.tensor([distance**2]), combined_width, highest_order=1
    )
    expected_energy = (
        -FIELD_CONSTANT / (4 * pi) * charge * dipole_z * distance
        * kernels[1][0].item()
    )
    energy = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)(
        source_feats, positions, batch
    )[0]
    assert energy.item() == pytest.approx(expected_energy, rel=1e-12)
    assert energy.item() < 0.0

    # Feature check: the potential of the positive sender charge decays with
    # distance, so its gradient at the receiver (at +z) has a negative
    # z-component.
    features = RealSpaceAnalyticalElectrostaticFeatures(
        1,
        DENSITY_SMEARING_WIDTH,
        1,
        [DENSITY_SMEARING_WIDTH],
        integral_normalization="multipoles",
    )(source_feats, positions, batch)[0]
    receiver_gradient_z = features[1, 2]  # (y, z, x) block: column 2 is z
    assert receiver_gradient_z.item() < 0.0
    projection_kernels = smeared_coulomb_kernels(
        torch.tensor([distance**2]), combined_width, highest_order=1
    )
    expected_gradient_z = (
        -FIELD_CONSTANT / (4 * pi) * charge * distance
        * projection_kernels[1][0].item()
    )
    assert receiver_gradient_z.item() == pytest.approx(expected_gradient_z, rel=1e-12)


def test_energy_gradients_and_double_backward():
    source_feats, positions, batch = random_system(seed=14)
    module = RealSpaceAnalyticalEnergy(1, DENSITY_SMEARING_WIDTH)

    positions_input = positions.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda pos: module(source_feats, pos, batch).sum(),
        (positions_input,),
        eps=1e-6,
        atol=1e-8,
    )
    assert torch.autograd.gradgradcheck(
        lambda pos: module(source_feats, pos, batch).sum(),
        (positions_input,),
        eps=1e-6,
        atol=1e-7,
    )

    # training-like double backward: force-weighted loss back to a parameter
    feature_parameter = torch.nn.Parameter(source_feats.clone())
    positions_leaf = positions.clone().requires_grad_(True)
    energy = module(feature_parameter, positions_leaf, batch).sum()
    forces = torch.autograd.grad(energy, positions_leaf, create_graph=True)[0]
    loss = forces.pow(2).sum()
    loss.backward()
    assert feature_parameter.grad is not None
    assert torch.isfinite(feature_parameter.grad).all()


def test_features_gradcheck():
    source_feats, positions, batch = random_system(seed=15)
    module = RealSpaceAnalyticalElectrostaticFeatures(
        1, DENSITY_SMEARING_WIDTH, 1, PROJECTION_SMEARING_WIDTHS
    )
    positions_input = positions.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda pos: module(source_feats, pos, batch)[0].pow(2).sum(),
        (positions_input,),
        eps=1e-6,
        atol=1e-8,
    )


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        RealSpaceAnalyticalEnergy(2, DENSITY_SMEARING_WIDTH)
    with pytest.raises(ValueError):
        RealSpaceAnalyticalEnergy(1, -1.0)
    with pytest.raises(ValueError):
        RealSpaceAnalyticalElectrostaticFeatures(1, DENSITY_SMEARING_WIDTH, 1, [])
    with pytest.raises(ValueError):
        RealSpaceAnalyticalElectrostaticFeatures(
            1, DENSITY_SMEARING_WIDTH, 1, [1.0], integral_normalization="bogus"
        )
    source_feats, positions, batch = random_system(seed=16)
    with pytest.raises(ValueError):
        RealSpaceAnalyticalEnergy(0, DENSITY_SMEARING_WIDTH)(
            source_feats, positions, batch
        )
