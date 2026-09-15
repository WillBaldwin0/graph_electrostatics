"""Tests for dynamic external-source feature composition."""

from __future__ import annotations

import pytest
import torch

torch.serialization.add_safe_globals([slice])

from graph_longrange.external_source_features import (
    GTOElectrostaticExternalSourceFeatures,
)
from graph_longrange.features import GTOElectrostaticFeatures


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _inputs():
    internal_positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]])
    internal_feats = torch.tensor([[0.3], [-0.2]])
    internal_batch = torch.zeros(2, dtype=torch.long)
    external_positions = torch.tensor([[2.0, -0.4, 0.3]])
    external_feats = torch.tensor([[0.45]])
    external_batch = torch.zeros(1, dtype=torch.long)
    geometry = {
        "k_vectors": torch.empty((0, 3)),
        "k_norm2": torch.empty((0,)),
        "k_vector_batch": torch.empty((0,), dtype=torch.long),
        "k0_mask": torch.empty((0,)),
        "node_positions": internal_positions,
        "batch": internal_batch,
        "volume": torch.ones(1),
        "pbc": torch.tensor([[False, False, False]]),
    }
    return (
        geometry,
        internal_feats,
        external_feats,
        external_positions,
        external_batch,
    )


def _base():
    return GTOElectrostaticFeatures(
        density_max_l=0,
        density_smearing_width=0.7,
        feature_max_l=0,
        feature_smearing_widths=[0.8],
        include_self_interaction=False,
        kspace_cutoff=4.0,
        pbc_handling="realspace",
    )


def test_external_source_features_add_scaled_source_target_field():
    geometry, internal_feats, external_feats, external_positions, external_batch = _inputs()
    base = _base()
    base_cache = base.precompute_geometry(**geometry)
    internal = base.forward_dynamic(base_cache, internal_feats)
    source_target = GTOElectrostaticExternalSourceFeatures.from_features(base)
    external_cache = source_target.precompute_geometry_source_target(
        k_vectors=geometry["k_vectors"],
        k_norm2=geometry["k_norm2"],
        k_vector_batch=geometry["k_vector_batch"],
        k0_mask=geometry["k0_mask"],
        src_positions=external_positions,
        src_batch=external_batch,
        tgt_positions=geometry["node_positions"],
        tgt_batch=geometry["batch"],
        volume=geometry["volume"],
        pbc=geometry["pbc"],
    )
    external = source_target.forward_dynamic_source_target(external_cache, external_feats)

    wrapped = GTOElectrostaticExternalSourceFeatures.from_features(
        base, external_scale=0.5
    )
    wrapped.set_external_sources(
        external_feats, external_positions, external_batch
    )
    cache = wrapped.precompute_geometry(**geometry)
    actual = wrapped.forward_dynamic(cache, internal_feats.unsqueeze(-2))
    torch.testing.assert_close(actual, internal + 0.5 * external)

    direct = wrapped(source_feats=internal_feats, **geometry)
    torch.testing.assert_close(direct, actual)


def test_external_source_features_clear_and_validate_lifecycle():
    geometry, internal_feats, external_feats, _, _ = _inputs()
    base = _base()
    wrapped = GTOElectrostaticExternalSourceFeatures.from_features(base)
    assert wrapped.base is base
    assert wrapped.density_basis is base.density_basis
    assert wrapped.realspace_features is base.realspace_features
    assert not hasattr(base, "forward_source_target")
    with pytest.raises(ValueError, match="either all be provided or all be None"):
        wrapped.set_external_sources(external_feats=external_feats)

    wrapped.set_external_sources()
    expected = base(source_feats=internal_feats, **geometry)
    actual = wrapped(source_feats=internal_feats, **geometry)
    torch.testing.assert_close(actual, expected)

    wrapped.set_pbc_handling("pbc")
    assert wrapped.base.pbc_handling == "pbc"
