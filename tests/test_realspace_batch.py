"""Independent graphs must retain isolated nodes when evaluated in a batch."""

import pytest
import torch

from graph_longrange.realspace_electrostatics import RealSpaceFiniteDiffereneEnergy


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield request.param
    torch.set_default_dtype(previous)


def make_graph(size, max_l, dtype):
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]], dtype=dtype)[:size]
    features = torch.tensor(
        [[1.0, 0.10, -0.20, 0.15], [-1.0, -0.15, 0.10, 0.05]], dtype=dtype
    )[:size, : (max_l + 1) ** 2]
    return features, positions


@pytest.mark.parametrize("max_l", [0, 1])
@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("sizes", [(2, 1), (1, 2), (2, 1, 1), (1, 2, 1), (1, 1, 1)])
def test_batched_energies_match_individual_graphs(dtype, max_l, include_self, sizes):
    model = RealSpaceFiniteDiffereneEnergy(max_l, 0.6, include_self).to(dtype=dtype)
    graphs = [make_graph(size, max_l, dtype) for size in sizes]
    expected = torch.cat(
        [
            model(features, positions, torch.zeros(size, dtype=torch.long))
            for size, (features, positions) in zip(sizes, graphs)
        ]
    )
    features = torch.cat([graph[0] for graph in graphs])
    positions = torch.cat([graph[1] for graph in graphs])
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    actual = model(features, positions, batch)

    assert actual.shape == (len(sizes),)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    if not include_self:
        torch.testing.assert_close(
            actual[torch.tensor(sizes) == 1], torch.zeros(sizes.count(1), dtype=dtype)
        )


@pytest.mark.parametrize("max_l", [0, 1])
@pytest.mark.parametrize("include_self", [False, True])
def test_trailing_isolated_node_preserves_forces(max_l, include_self):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = RealSpaceFiniteDiffereneEnergy(max_l, 0.6, include_self).double()
        features, pair_positions = make_graph(2, max_l, torch.float64)
        pair_positions.requires_grad_(True)
        pair_energy = model(features, pair_positions, torch.tensor([0, 0])).sum()
        pair_forces = -torch.autograd.grad(pair_energy, pair_positions)[0]

        features = torch.cat([features, features[:1]])
        positions = torch.cat(
            [pair_positions.detach(), torch.tensor([[4.0, 0.0, 0.0]])]
        )
        positions.requires_grad_(True)
        batch = torch.tensor([0, 0, 1])
        energy = model(features, positions, batch).sum()
        forces = -torch.autograd.grad(energy, positions)[0]
        torch.testing.assert_close(forces[:2], pair_forces)
        torch.testing.assert_close(forces[2], torch.zeros(3))

        direction = torch.tensor([[0.3, -0.2, 0.1], [-0.1, 0.4, -0.2], [0.1, 0.2, 0.3]])
        direction /= torch.linalg.norm(direction)
        derivative = -torch.sum(forces * direction)
        errors = []
        for step in (1e-2, 1e-3, 1e-4):

            def displaced_energy(offset):
                return model(
                    features, positions.detach() + offset * step * direction, batch
                ).sum()

            p2, p1, m1, m2 = [displaced_energy(k) for k in (2, 1, -1, -2)]
            finite_difference = (-p2 + 8 * p1 - 8 * m1 + m2) / (12 * step)
            errors.append(abs(finite_difference - derivative))
        assert torch.stack(errors).min().item() < 1e-6
    finally:
        torch.set_default_dtype(previous)
