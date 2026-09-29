"""Tests for the vectorized batched pair-list builder.

The builder duplicates every node N times and connects all duplicate pairs
within a graph, except pairs that stem from the same original node. Each case
is checked against a pure-Python reference enumeration.
"""

import itertools

import pytest
import torch

from graph_longrange.realspace_electrostatics import (
    batch_complete_graph_excluding_self_duplicates_vector,
)


def reference_edge_set(batch, num_duplicates):
    """Ground truth from plain Python iteration."""
    duplicated = [
        (node_id * num_duplicates + duplicate, graph, node_id)
        for node_id, graph in enumerate(batch)
        for duplicate in range(num_duplicates)
    ]
    edges = set()
    for (id_a, graph_a, orig_a), (id_b, graph_b, orig_b) in itertools.product(
        duplicated, repeat=2
    ):
        if graph_a == graph_b and orig_a != orig_b:
            edges.add((id_a, id_b))
    return edges


def edge_set_from_tensor(edge_index):
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


@pytest.mark.parametrize("num_duplicates", [1, 4])
@pytest.mark.parametrize(
    "batch",
    [
        [0, 0, 0],
        [0, 0, 1, 1, 1, 2],
        [0],  # single node, no edges
        [0, 1, 2],  # all graphs of size one, no edges
        [0, 1, 0, 1, 2, 0],  # unsorted batch
        [0, 2, 2],  # gap in graph ids
    ],
)
def test_builder_matches_reference(batch, num_duplicates):
    batch_tensor = torch.tensor(batch, dtype=torch.long)
    edge_index = batch_complete_graph_excluding_self_duplicates_vector(
        batch_tensor, num_duplicates
    )
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2
    computed = edge_set_from_tensor(edge_index)
    expected = reference_edge_set(batch, num_duplicates)
    assert computed == expected
    # no duplicate directed edges
    assert edge_index.shape[1] == len(computed)


def test_builder_empty_batch():
    edge_index = batch_complete_graph_excluding_self_duplicates_vector(
        torch.empty(0, dtype=torch.long), 4
    )
    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == torch.long


def test_builder_edge_count_formula():
    # A graph of M nodes duplicated N times has (M*N)^2 - M*N^2 directed edges.
    num_nodes, num_duplicates = 7, 4
    batch = torch.zeros(num_nodes, dtype=torch.long)
    edge_index = batch_complete_graph_excluding_self_duplicates_vector(
        batch, num_duplicates
    )
    expected_count = (num_nodes * num_duplicates) ** 2 - num_nodes * num_duplicates**2
    assert edge_index.shape[1] == expected_count
