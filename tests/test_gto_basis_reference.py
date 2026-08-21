from pathlib import Path

import numpy as np
import pytest
import torch

from graph_longrange.gto_utils import GTOBasis


def test_gto_basis_direct_has_finite_k_vector_gradients():
    torch.set_default_dtype(torch.float64)
    k_vectors = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, -0.2, 0.1], [-0.1, 0.3, 0.5]],
        requires_grad=True,
    )
    k_norm2 = torch.sum(k_vectors.square(), dim=-1)
    k0_mask = torch.tensor([1.0, 0.0, 0.0])
    basis = GTOBasis(
        max_l=1,
        sigmas=[0.5, 1.0],
        kspace_cutoff=3.0,
        normalize="multipoles",
        use_spline=False,
    )

    values = basis(k_vectors, k_norm2, k0_mask)
    gradient = torch.autograd.grad(values.square().sum(), k_vectors)[0]

    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient[1:]) > 0.0


@pytest.mark.parametrize("rtol, atol", [(1e-15, 1e-15)])
def test_gto_basis_matches_old_reference(rtol, atol):
    torch.set_default_dtype(torch.float64)
    reference_path = (
        Path(__file__).resolve().parent / "reference_data" / "gto_basis_old_reference.npz"
    )
    if not reference_path.exists():
        pytest.skip(
            "Reference file missing. Run scripts/generate_old_gto_basis_reference.py first."
        )

    data = np.load(reference_path, allow_pickle=False)
    sigmas = data["sigmas"].tolist()
    max_l = int(data["max_l"])
    kspace_cutoff = float(data["kspace_cutoff"])
    normalize = str(data["normalize"].item())

    if "k_vectors_flat" in data.files:
        k_vectors = torch.from_numpy(data["k_vectors_flat"]).to(torch.get_default_dtype())
        k_norm2 = torch.from_numpy(data["k_norm2_flat"]).to(torch.get_default_dtype())
        k0_mask = torch.from_numpy(data["k0_mask_flat"]).to(torch.get_default_dtype())
        reference_values = torch.from_numpy(data["basis_values_flat"]).to(
            torch.get_default_dtype()
        )
    else:
        k_vectors_dense = torch.from_numpy(data["k_vectors"]).to(torch.get_default_dtype())
        k_norm2_dense = torch.from_numpy(data["k_norm2"]).to(torch.get_default_dtype())
        k_mask = torch.from_numpy(data["k_mask"]).to(torch.bool)
        k_vectors = k_vectors_dense[k_mask, ...]
        k_norm2 = k_norm2_dense[k_mask]
        k0_mask = torch.zeros_like(k_norm2)
        start = 0
        for b in range(k_mask.size(0)):
            n_k = int(k_mask[b].sum().item())
            k0_mask[start] = 1.0
            start += n_k
        reference_values = torch.from_numpy(data["basis_values"]).to(
            torch.get_default_dtype()
        )[k_mask, ...]

    basis = GTOBasis(
        max_l=max_l,
        sigmas=sigmas,
        kspace_cutoff=kspace_cutoff,
        normalize=normalize,
        use_spline=True,
    )

    with torch.no_grad():
        new_values = basis(k_vectors, k_norm2, k0_mask)

    torch.testing.assert_close(new_values, reference_values, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rtol, atol", [(1e-10, 1e-15)])
def test_gto_basis_direct_matches_reference_l1(rtol, atol):
    torch.set_default_dtype(torch.float64)
    reference_path = (
        Path(__file__).resolve().parent / "reference_data" / "gto_basis_old_reference.npz"
    )
    if not reference_path.exists():
        pytest.skip(
            "Reference file missing. Run scripts/generate_old_gto_basis_reference.py first."
        )

    data = np.load(reference_path, allow_pickle=False)
    sigmas = data["sigmas"].tolist()
    max_l_ref = int(data["max_l"])
    kspace_cutoff = float(data["kspace_cutoff"])
    normalize = str(data["normalize"].item())

    if "k_vectors_flat" in data.files:
        k_vectors = torch.from_numpy(data["k_vectors_flat"]).to(torch.get_default_dtype())
        k_norm2 = torch.from_numpy(data["k_norm2_flat"]).to(torch.get_default_dtype())
        k0_mask = torch.from_numpy(data["k0_mask_flat"]).to(torch.get_default_dtype())
        reference_values = torch.from_numpy(data["basis_values_flat"]).to(
            torch.get_default_dtype()
        )
    else:
        k_vectors_dense = torch.from_numpy(data["k_vectors"]).to(torch.get_default_dtype())
        k_norm2_dense = torch.from_numpy(data["k_norm2"]).to(torch.get_default_dtype())
        k_mask = torch.from_numpy(data["k_mask"]).to(torch.bool)
        k_vectors = k_vectors_dense[k_mask, ...]
        k_norm2 = k_norm2_dense[k_mask]
        k0_mask = torch.zeros_like(k_norm2)
        start = 0
        for b in range(k_mask.size(0)):
            n_k = int(k_mask[b].sum().item())
            k0_mask[start] = 1.0
            start += n_k
        reference_values = torch.from_numpy(data["basis_values"]).to(
            torch.get_default_dtype()
        )[k_mask, ...]

    max_l_direct = 1 if max_l_ref >= 1 else max_l_ref
    m_dim_direct = (max_l_direct + 1) ** 2
    reference_slice = reference_values[..., :m_dim_direct, :]

    basis = GTOBasis(
        max_l=max_l_direct,
        sigmas=sigmas,
        kspace_cutoff=kspace_cutoff,
        normalize=normalize,
        use_spline=False,
    )

    with torch.no_grad():
        new_values = basis(k_vectors, k_norm2, k0_mask)

    torch.testing.assert_close(new_values, reference_slice, rtol=rtol, atol=atol)
