"""Tests for the finite-difference / analytical real-space evaluator toggle.

The analytical modules themselves are covered by
test_realspace_analytical_electrostatics.py; this file covers the wiring in
GTOElectrostaticEnergy / GTOElectrostaticFeatures and, above all, the
backward-compatibility contract: the flag must not change what a default block
computes, must not change its state dict, and must not break models that were
pickled or checkpointed before the flag existed.
"""

import io

import pytest
import torch

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.realspace_electrostatics import (
    RealSpaceAnalyticalElectrostaticFeatures,
    RealSpaceAnalyticalEnergy,
    RealSpaceFiniteDifferenceElectrostaticFeatures,
    RealSpaceFiniteDiffereneEnergy,
)

torch.set_default_dtype(torch.float64)

DENSITY_MAX_L = 1
DENSITY_SMEARING_WIDTH = 1.5
FEATURE_MAX_L = 1
FEATURE_SMEARING_WIDTHS = [1.0, 2.0]
KSPACE_CUTOFF = 3.0


def make_energy_block(**kwargs):
    return GTOElectrostaticEnergy(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        kspace_cutoff=KSPACE_CUTOFF,
        include_self_interaction=False,
        pbc_handling="realspace",
        **kwargs,
    )


def make_features_block(**kwargs):
    return GTOElectrostaticFeatures(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        feature_max_l=FEATURE_MAX_L,
        feature_smearing_widths=FEATURE_SMEARING_WIDTHS,
        include_self_interaction=False,
        kspace_cutoff=KSPACE_CUTOFF,
        pbc_handling="realspace",
        **kwargs,
    )


def random_system(num_graphs=2, atoms_per_graph=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    num_atoms = num_graphs * atoms_per_graph
    positions = 2.0 * torch.randn((num_atoms, 3), generator=generator)
    source_feats = 0.5 * torch.randn((num_atoms, 4), generator=generator)
    batch = torch.repeat_interleave(
        torch.arange(num_graphs, dtype=torch.long), atoms_per_graph
    )
    return source_feats, positions, batch


def run_energy_block(block, source_feats, positions, batch):
    num_graphs = int(batch.max()) + 1
    return block(
        k_vectors=torch.zeros((0, 3)),
        k_norm2=torch.zeros((0,)),
        k_vector_batch=torch.zeros((0,), dtype=torch.long),
        k0_mask=torch.zeros((0,), dtype=torch.bool),
        source_feats=source_feats,
        node_positions=positions,
        batch=batch,
        volume=torch.ones(num_graphs),
        pbc=torch.zeros((num_graphs, 3), dtype=torch.bool),
    )


def run_features_block(block, source_feats, positions, batch):
    num_graphs = int(batch.max()) + 1
    return block(
        k_vectors=torch.zeros((0, 3)),
        k_norm2=torch.zeros((0,)),
        k_vector_batch=torch.zeros((0,), dtype=torch.long),
        k0_mask=torch.zeros((0,), dtype=torch.bool),
        source_feats=source_feats,
        node_positions=positions,
        batch=batch,
        volume=torch.ones(num_graphs),
        pbc=torch.zeros((num_graphs, 3), dtype=torch.bool),
    )


def test_default_selects_finite_difference():
    """The default must stay finite difference: existing weights were fit to it."""
    assert isinstance(
        make_energy_block().realspace_energy, RealSpaceFiniteDiffereneEnergy
    )
    assert isinstance(
        make_features_block().realspace_features,
        RealSpaceFiniteDifferenceElectrostaticFeatures,
    )


def test_analytical_flag_selects_analytical_modules():
    assert isinstance(
        make_energy_block(realspace_method="analytical").realspace_energy,
        RealSpaceAnalyticalEnergy,
    )
    assert isinstance(
        make_features_block(realspace_method="analytical").realspace_features,
        RealSpaceAnalyticalElectrostaticFeatures,
    )


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unsupported realspace_method"):
        make_energy_block(realspace_method="quadrature")
    with pytest.raises(ValueError, match="Unsupported realspace_method"):
        make_features_block(realspace_method="quadrature")


def test_analytical_block_matches_standalone_analytical_module():
    """The block must forward to the analytical module without rescaling it."""
    source_feats, positions, batch = random_system()

    block_energy = run_energy_block(
        make_energy_block(realspace_method="analytical"), source_feats, positions, batch
    )
    reference_energy = RealSpaceAnalyticalEnergy(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        include_self_interaction=False,
    )(source_feats=source_feats, positions=positions, batch=batch)
    torch.testing.assert_close(block_energy, reference_energy)

    block_features = run_features_block(
        make_features_block(realspace_method="analytical"),
        source_feats,
        positions,
        batch,
    )
    reference_features, _, _ = RealSpaceAnalyticalElectrostaticFeatures(
        density_max_l=DENSITY_MAX_L,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        projection_max_l=FEATURE_MAX_L,
        projection_smearing_widths=FEATURE_SMEARING_WIDTHS,
        include_self_interaction=False,
        integral_normalization="receiver",
    )(source_feats=source_feats, node_positions=positions, batch=batch)
    torch.testing.assert_close(block_features, reference_features)


def test_setter_swaps_evaluator_and_changes_output():
    """set_realspace_method must rebuild the evaluator and take effect immediately."""
    source_feats, positions, batch = random_system()

    energy_block = make_energy_block()
    finite_difference_energy = run_energy_block(
        energy_block, source_feats, positions, batch
    )
    energy_block.set_realspace_method("analytical")
    assert isinstance(energy_block.realspace_energy, RealSpaceAnalyticalEnergy)
    analytical_energy = run_energy_block(energy_block, source_feats, positions, batch)
    torch.testing.assert_close(
        analytical_energy,
        run_energy_block(
            make_energy_block(realspace_method="analytical"),
            source_feats,
            positions,
            batch,
        ),
    )
    # Same physics, different discretization: close but not identical.
    assert not torch.allclose(analytical_energy, finite_difference_energy)

    features_block = make_features_block()
    features_block.set_realspace_method("analytical")
    assert isinstance(
        features_block.realspace_features, RealSpaceAnalyticalElectrostaticFeatures
    )
    torch.testing.assert_close(
        run_features_block(features_block, source_feats, positions, batch),
        run_features_block(
            make_features_block(realspace_method="analytical"),
            source_feats,
            positions,
            batch,
        ),
    )

    # And back again, so the toggle is not one-way.
    energy_block.set_realspace_method("finite_difference")
    torch.testing.assert_close(
        run_energy_block(energy_block, source_feats, positions, batch),
        finite_difference_energy,
    )


def test_setter_preserves_dtype_and_keeps_integer_buffers_integral():
    """A block cast after construction must survive an evaluator swap."""
    energy_block = make_energy_block().to(dtype=torch.float32)
    energy_block.set_realspace_method("analytical")
    for name, buffer in energy_block.realspace_energy.named_buffers():
        if buffer.is_floating_point():
            assert buffer.dtype == torch.float32, name
        else:
            assert not buffer.is_floating_point(), name
    assert (
        energy_block.realspace_energy.self_interaction.select_indices.dtype
        == torch.long
    )


def test_default_state_dict_is_unchanged_by_the_flag():
    """Old checkpoints must still load strictly into a default-built block."""
    energy_keys = set(make_energy_block().state_dict())
    assert "realspace_energy.x" in energy_keys
    assert "realspace_energy.y" in energy_keys
    assert "realspace_energy.z" in energy_keys

    features_keys = set(make_features_block().state_dict())
    assert "realspace_features.l0_factors" in features_keys
    assert "realspace_features.total_width_factors" in features_keys


def test_analytical_state_dict_is_a_subset_of_the_finite_difference_one():
    """Analytical derived constants stay out of the state dict.

    The finite-difference module registers l0_factors / l1_factors of the same
    shape but with offset-dependent values, so shared keys would let a non-strict
    cross-arm load silently install the wrong constants.
    """
    for make_block in (make_energy_block, make_features_block):
        finite_difference_keys = set(make_block().state_dict())
        analytical_keys = set(make_block(realspace_method="analytical").state_dict())
        assert analytical_keys < finite_difference_keys
        assert not any("l0_factors" in key for key in analytical_keys)
        assert not any("l1_factors" in key for key in analytical_keys)
        assert not any("combined_smearing_width" in key for key in analytical_keys)


def test_finite_difference_checkpoint_loads_into_analytical_block_without_corruption():
    """A non-strict cross-arm load must leave the analytical constants intact."""
    finite_difference_state = make_features_block().state_dict()
    analytical_block = make_features_block(realspace_method="analytical")
    expected_l1 = analytical_block.realspace_features.l1_factors.clone()

    result = analytical_block.load_state_dict(finite_difference_state, strict=False)
    assert result.unexpected_keys  # the finite-difference-only buffers
    torch.testing.assert_close(
        analytical_block.realspace_features.l1_factors, expected_l1
    )

    with pytest.raises(RuntimeError):
        analytical_block.load_state_dict(finite_difference_state, strict=True)


def test_legacy_pickle_without_the_flag_still_runs_and_can_be_switched():
    """Whole-model pickles restore __dict__ without calling __init__.

    Such instances have no realspace_method attribute at all, so nothing on the
    forward path may read one, and the setter must still be able to rebuild.
    """
    source_feats, positions, batch = random_system()

    energy_block = make_energy_block()
    del energy_block.realspace_method
    expected = run_energy_block(
        make_energy_block(), source_feats, positions, batch
    )
    torch.testing.assert_close(
        run_energy_block(energy_block, source_feats, positions, batch), expected
    )
    energy_block.set_realspace_method("analytical")
    assert isinstance(energy_block.realspace_energy, RealSpaceAnalyticalEnergy)

    features_block = make_features_block()
    del features_block.realspace_method
    torch.testing.assert_close(
        run_features_block(features_block, source_feats, positions, batch),
        run_features_block(make_features_block(), source_feats, positions, batch),
    )
    features_block.set_realspace_method("analytical")
    assert isinstance(
        features_block.realspace_features, RealSpaceAnalyticalElectrostaticFeatures
    )


@pytest.mark.parametrize("realspace_method", ["finite_difference", "analytical"])
def test_whole_block_pickle_round_trip_preserves_the_evaluator(realspace_method):
    """torch.save of a whole block keeps the selected evaluator and its outputs."""
    source_feats, positions, batch = random_system()

    energy_block = make_energy_block(realspace_method=realspace_method)
    features_block = make_features_block(realspace_method=realspace_method)
    expected_energy = run_energy_block(energy_block, source_feats, positions, batch)
    expected_features = run_features_block(
        features_block, source_feats, positions, batch
    )

    for block, expected, runner, attribute in (
        (energy_block, expected_energy, run_energy_block, "realspace_energy"),
        (
            features_block,
            expected_features,
            run_features_block,
            "realspace_features",
        ),
    ):
        buffer = io.BytesIO()
        torch.save(block, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        assert restored.realspace_method == realspace_method
        assert type(getattr(restored, attribute)) is type(getattr(block, attribute))
        torch.testing.assert_close(
            runner(restored, source_feats, positions, batch), expected
        )


@pytest.mark.parametrize("realspace_method", ["finite_difference", "analytical"])
def test_state_dict_round_trip_within_one_method(realspace_method):
    """Training checkpoints round-trip strictly as long as the flag matches."""
    source_feats, positions, batch = random_system()
    block = make_energy_block(realspace_method=realspace_method)
    reloaded = make_energy_block(realspace_method=realspace_method)
    reloaded.load_state_dict(block.state_dict(), strict=True)
    torch.testing.assert_close(
        run_energy_block(reloaded, source_feats, positions, batch),
        run_energy_block(block, source_feats, positions, batch),
    )
