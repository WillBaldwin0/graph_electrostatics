import pytest
import torch

from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat


@pytest.mark.parametrize(
    ("pbc_handling", "pbc", "n_graphs"),
    [
        ("pbc", [[True, True, True]], 1),
        ("slab", [[True, True, False]], 1),
        ("molecule_in_box", [[False, False, False]], 1),
        ("mixed_periodic", [[True, True, False]], 1),
        ("auto", [[True, True, True]], 1),
        ("realspace", [[False, False, False]], 1),
        ("auto", [True, True, True], 2),
        ("auto", [[True, True, True]], 2),
    ],
)
def test_feature_dtype_follows_inputs_not_process_default(
    pbc_handling, pbc, n_graphs
):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        dtype = torch.float64
        positions = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 1.5, 1.2]], dtype=dtype
        ).repeat(n_graphs, 1)
        source_feats = torch.tensor([[0.4], [-0.4]], dtype=dtype).repeat(
            n_graphs, 1
        )
        batch = torch.arange(n_graphs).repeat_interleave(2)
        cell = 6.0 * torch.eye(3, dtype=dtype).repeat(n_graphs, 1, 1)
        r_cell = 2.0 * torch.pi * torch.linalg.inv(cell).transpose(-1, -2)
        volume = torch.linalg.det(cell)
        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            cutoff=1.2,
            cell_vectors=cell,
            r_cell_vectors=r_cell,
        )
        feature_block = GTOElectrostaticFeatures(
            density_max_l=0,
            density_smearing_width=0.4,
            feature_max_l=1,
            feature_smearing_widths=[0.3],
            include_self_interaction=False,
            kspace_cutoff=1.2,
            pbc_handling=pbc_handling,
        ).to(dtype=dtype)

        features = feature_block(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=source_feats,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=torch.tensor(pbc, dtype=torch.bool),
        )

        assert features.dtype == dtype
        assert features.shape[0] == 2 * n_graphs
        assert torch.isfinite(features).all()
    finally:
        torch.set_default_dtype(previous_dtype)
