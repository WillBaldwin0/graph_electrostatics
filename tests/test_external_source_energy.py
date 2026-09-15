"""Tests for the dynamic external-source cross-energy block."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.constants import pi

# PyTorch 2.6+ requires this allowlist before importing e3nn.
torch.serialization.add_safe_globals([slice])

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.external_source_energy import (
    GTOElectrostaticCrossEnergy,
    GTOElectrostaticExternalSourceEnergy,
)
from graph_longrange.kspace import compute_k_vectors_flat
from graph_longrange.utils import FIELD_CONSTANT


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("mode", ["slab", "molecule_in_box", "mixed_periodic"])
@pytest.mark.parametrize("occupied", [1, 2])
@pytest.mark.parametrize("max_l", [0, 1])
def test_corrections_leave_missing_graphs_empty(mode, occupied, max_l):
    block = GTOElectrostaticCrossEnergy(max_l, 0.7, 4.0, pbc_handling=mode)
    features = torch.full((occupied, (max_l + 1) ** 2), 0.3)
    positions = torch.ones((occupied, 3), requires_grad=True)
    volume = torch.tensor([720., 800., 900.])
    pbc = torch.tensor([[True, True, False]] * 3)
    actual = block._correction(features, positions, torch.arange(occupied), volume, pbc, mode, 3)
    expected = torch.cat([
        block._correction(features[i:i+1], positions[i:i+1], torch.zeros(1, dtype=torch.long),
                          volume[i:i+1], pbc[i:i+1], mode, 1)
        for i in range(occupied)
    ] + [volume.new_zeros(3 - occupied)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), positions, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), positions)[0],
    )


def _geometry(pbc):
    cell = torch.diag(torch.tensor([8.0, 9.0, 10.0])).unsqueeze(0)
    rcell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        cutoff=4.0, cell_vectors=cell, r_cell_vectors=rcell
    )
    return {
        "k_vectors": k_vectors,
        "k_norm2": k_norm2,
        "k_vector_batch": k_vector_batch,
        "k0_mask": k0_mask,
        "volume": torch.det(cell),
        "pbc": torch.tensor([pbc], dtype=torch.bool),
    }


def _sets(max_l):
    source_positions = torch.tensor(
        [[0.2, 0.3, 0.5], [1.4, 0.8, 0.1]], requires_grad=True
    )
    target_positions = torch.tensor(
        [[2.1, -0.4, 0.7], [-0.8, 1.7, 0.4], [0.6, 2.4, -0.2]],
        requires_grad=True,
    )
    source_feats = torch.tensor([[0.31], [-0.27]])
    target_feats = torch.tensor([[0.43], [-0.38], [0.11]])
    if max_l == 1:
        source_feats = torch.cat(
            [source_feats, torch.tensor([[0.012, -0.021, 0.034], [-0.015, 0.009, -0.018]])],
            dim=1,
        )
        target_feats = torch.cat(
            [
                target_feats,
                torch.tensor(
                    [[0.0, 0.0, 0.0], [0.013, -0.007, 0.004], [-0.01, 0.02, 0.03]]
                ),
            ],
            dim=1,
        )
    source_batch = torch.zeros(source_positions.shape[0], dtype=torch.long)
    target_batch = torch.zeros(target_positions.shape[0], dtype=torch.long)
    return (
        source_feats,
        source_positions,
        source_batch,
        target_feats,
        target_positions,
        target_batch,
    )


@pytest.mark.parametrize("max_l", [0, 1])
@pytest.mark.parametrize(
    "mode,pbc",
    [
        ("realspace", [False, False, False]),
        ("pbc", [True, True, True]),
        ("slab", [True, True, False]),
        ("molecule_in_box", [False, False, False]),
        ("mixed_periodic", [True, True, False]),
        ("auto", [False, False, False]),
        ("auto", [True, True, True]),
    ],
)
def test_cross_matches_combined_density_identity_and_gradients(max_l, mode, pbc):
    geometry = _geometry(pbc)
    values = _sets(max_l)
    source_feats, source_positions, source_batch, target_feats, target_positions, target_batch = values

    cross_block = GTOElectrostaticCrossEnergy(
        density_max_l=max_l,
        density_smearing_width=0.7,
        kspace_cutoff=4.0,
        pbc_handling=mode,
        reciprocal_chunk_size=1,
        realspace_chunk_size=1,
    )
    actual = cross_block(
        **geometry,
        source_feats=source_feats,
        source_positions=source_positions,
        source_batch=source_batch,
        target_feats=target_feats,
        target_positions=target_positions,
        target_batch=target_batch,
    )
    actual_grads = torch.autograd.grad(
        actual.sum(), (source_positions, target_positions), retain_graph=False
    )

    ref_source_positions = source_positions.detach().clone().requires_grad_(True)
    ref_target_positions = target_positions.detach().clone().requires_grad_(True)
    energy = GTOElectrostaticEnergy(
        density_max_l=max_l,
        density_smearing_width=0.7,
        kspace_cutoff=4.0,
        include_self_interaction=False,
        pbc_handling=mode,
    )
    common = dict(
        k_vectors=geometry["k_vectors"],
        k_norm2=geometry["k_norm2"],
        k_vector_batch=geometry["k_vector_batch"],
        k0_mask=geometry["k0_mask"],
        volume=geometry["volume"],
        pbc=geometry["pbc"],
    )
    mixed = energy(
        **common,
        source_feats=torch.cat([source_feats, target_feats]),
        node_positions=torch.cat([ref_source_positions, ref_target_positions]),
        batch=torch.cat([source_batch, target_batch]),
    )
    source_only = energy(
        **common,
        source_feats=source_feats,
        node_positions=ref_source_positions,
        batch=source_batch,
    )
    target_only = energy(
        **common,
        source_feats=target_feats,
        node_positions=ref_target_positions,
        batch=target_batch,
    )
    expected = mixed - source_only - target_only
    expected_grads = torch.autograd.grad(
        expected.sum(), (ref_source_positions, ref_target_positions)
    )

    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], atol=2e-9, rtol=2e-9)


def test_realspace_monopoles_use_gto_damping_and_physical_charge_normalization():
    geometry = _geometry([False, False, False])
    block = GTOElectrostaticCrossEnergy(
        density_max_l=0,
        density_smearing_width=0.7,
        kspace_cutoff=4.0,
        pbc_handling="realspace",
    )
    source_positions = torch.tensor([[0.0, 0.0, 0.0]])
    target_positions = torch.tensor([[0.35, 0.0, 0.0]])
    batch = torch.zeros(1, dtype=torch.long)
    actual = block(
        **geometry,
        source_feats=torch.tensor([[0.7]]),
        source_positions=source_positions,
        source_batch=batch,
        target_feats=torch.tensor([[-0.4]]),
        target_positions=target_positions,
        target_batch=batch,
    )
    distance = torch.tensor(0.35)
    expected = (
        FIELD_CONSTANT
        / (4.0 * math.pi)
        * 0.7
        * -0.4
        * torch.erf(0.5 * distance / 0.7)
        / (distance + 1e-6)
    )
    torch.testing.assert_close(actual[0], expected, atol=1e-12, rtol=1e-12)


def test_empty_external_set_returns_one_zero_per_graph():
    geometry = _geometry([True, True, True])
    block = GTOElectrostaticCrossEnergy(0, 0.7, 4.0, pbc_handling="pbc")
    source_feats, source_positions, source_batch, _, _, _ = _sets(0)
    actual = block(
        **geometry,
        source_feats=source_feats,
        source_positions=source_positions,
        source_batch=source_batch,
        target_feats=torch.empty((0, 1)),
        target_positions=torch.empty((0, 3)),
        target_batch=torch.empty((0,), dtype=torch.long),
    )
    torch.testing.assert_close(actual, torch.zeros(1))


def test_external_source_energy_wraps_base_plus_cross_and_can_be_cleared():
    geometry = _geometry([False, False, False])
    (
        source_feats,
        source_positions,
        source_batch,
        target_feats,
        target_positions,
        target_batch,
    ) = _sets(1)
    base = GTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=0.7,
        kspace_cutoff=4.0,
        include_self_interaction=False,
        pbc_handling="realspace",
    )
    wrapped = GTOElectrostaticExternalSourceEnergy.from_energy(base)
    base_kwargs = dict(
        **geometry,
        source_feats=source_feats,
        node_positions=source_positions,
        batch=source_batch,
    )
    base_energy = base(**base_kwargs)
    cross_energy = wrapped.cross(
        **geometry,
        source_feats=source_feats,
        source_positions=source_positions,
        source_batch=source_batch,
        target_feats=target_feats,
        target_positions=target_positions,
        target_batch=target_batch,
    )

    wrapped.set_external_sources(target_feats, target_positions, target_batch)
    torch.testing.assert_close(wrapped(**base_kwargs), base_energy + cross_energy)

    wrapped.clear_external_sources()
    torch.testing.assert_close(wrapped(**base_kwargs), base_energy)


def test_external_source_energy_validates_source_lifecycle_and_syncs_pbc_mode():
    base = GTOElectrostaticEnergy(0, 0.7, 4.0, pbc_handling="realspace")
    wrapped = GTOElectrostaticExternalSourceEnergy.from_energy(base)
    with pytest.raises(ValueError, match="either all be provided or all be None"):
        wrapped.set_external_sources(external_feats=torch.ones((1, 1)))

    wrapped.set_pbc_handling("pbc")
    assert wrapped.base.pbc_handling == "pbc"
    assert wrapped.cross.pbc_handling == "pbc"
