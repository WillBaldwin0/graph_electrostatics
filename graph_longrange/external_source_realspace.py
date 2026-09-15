"""Internal damped real-space helpers for disjoint source and target sets."""

import torch
from scipy.constants import pi
from mace.tools.scatter import scatter_sum
from .utils import FIELD_CONSTANT

@torch.no_grad()
def batch_bipartite_pairs(
    src_batch: torch.Tensor, tgt_batch: torch.Tensor
) -> torch.Tensor:
    """Build all (sender, receiver) edges with src_batch[s] == tgt_batch[t].

    Sources and targets live in disjoint node sets, so no self-exclusion is
    needed. Edge order is per-graph, then row-major (sender-major) within each
    graph -- matching the order the symmetric primitive produces when restricted
    to source->target pairs. On CPU that makes the scatter sums bit-identical to
    the concatenated path under zero-coefficient padding; on CUDA `scatter_add_`
    is not order-deterministic, so only round-off agreement is guaranteed there.

    Cost: the full dense n_src x n_tgt product per graph, with no distance
    cutoff. Cheaper than the symmetric (n_src + n_tgt)^2 it replaces, but still
    quadratic -- a large aperiodic MM environment will dominate memory.

    Args:
        src_batch: [N_src] graph ID per source node.
        tgt_batch: [N_tgt] graph ID per target node.

    Returns:
        edge_index: [2, E] -- row 0 = source index, row 1 = target index.
    """
    src_batch = src_batch.long()
    tgt_batch = tgt_batch.long()
    if src_batch.numel() == 0 or tgt_batch.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=src_batch.device)

    G = int(max(int(src_batch.max().item()), int(tgt_batch.max().item()))) + 1
    edges = []
    for g in range(G):
        s_nodes = (src_batch == g).nonzero(as_tuple=False).view(-1)
        t_nodes = (tgt_batch == g).nonzero(as_tuple=False).view(-1)
        if s_nodes.numel() == 0 or t_nodes.numel() == 0:
            continue
        Ds = s_nodes.size(0)
        Dt = t_nodes.size(0)
        row = s_nodes.view(-1, 1).expand(-1, Dt).reshape(-1)
        col = t_nodes.view(1, -1).expand(Ds, -1).reshape(-1)
        edges.append(torch.stack([row, col], dim=0))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=src_batch.device)
    return torch.cat(edges, dim=1)


def charges_features_from_bipartite_graph(
    charges,                # [N_src]
    src_positions,          # [N_src, 3]
    tgt_positions,          # [N_tgt, 3]
    edge_index,             # [2, E] -- row 0 in src space, row 1 in tgt space
    n_targets,
    total_width_factors,    # [1, n_radial]
):
    """Bipartite analogue of charges_features_from_graph for MM->QM fields."""
    sender, receiver = edge_index
    R_ij = src_positions[sender] - tgt_positions[receiver]
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)
    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=n_targets,
    )
    features = FIELD_CONSTANT * features / (4 * pi)
    return features



class _ExternalSourceRealspace:
    """Reuse a real-space block's basis without rebuilding it."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def call_source_target_density_0_feats_0(
        self,
        source_feats: torch.Tensor,   # [N_src, m_dim]
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = batch_bipartite_pairs(src_batch, tgt_batch)
        feats = charges_features_from_bipartite_graph(
            charges=source_feats[:, 0],
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            edge_index=edge_index,
            n_targets=tgt_positions.shape[0],
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )
        return self.l0_factors * feats

    def call_source_target_density_1_feats_1(
        self,
        source_feats: torch.Tensor,   # [N_src, m_dim]
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
    ) -> torch.Tensor:
        # Source 4x duplication encodes source dipoles as displaced charges.
        ext_src_positions = src_positions.repeat_interleave(4, dim=0)
        ext_src_positions[1::4] += self.x
        ext_src_positions[2::4] += self.y
        ext_src_positions[3::4] += self.z
        ext_src_batch = src_batch.repeat_interleave(4)
        src_charges = torch.zeros_like(ext_src_positions[:, 0])
        src_charges[1::4] = source_feats[:, 3] / self.offset
        src_charges[2::4] = source_feats[:, 1] / self.offset
        src_charges[3::4] = source_feats[:, 2] / self.offset
        src_charges[0::4] = source_feats[:, 0] - (
            src_charges[1::4] + src_charges[2::4] + src_charges[3::4]
        )

        # Target 4x duplication encodes the receive-side l=1 finite-difference basis.
        ext_tgt_positions = tgt_positions.repeat_interleave(4, dim=0)
        ext_tgt_positions[1::4] += self.x
        ext_tgt_positions[2::4] += self.y
        ext_tgt_positions[3::4] += self.z
        ext_tgt_batch = tgt_batch.repeat_interleave(4)

        edge_index = batch_bipartite_pairs(ext_src_batch, ext_tgt_batch)
        scalar_features = charges_features_from_bipartite_graph(
            charges=src_charges,
            src_positions=ext_src_positions,
            tgt_positions=ext_tgt_positions,
            edge_index=edge_index,
            n_targets=ext_tgt_positions.shape[0],
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )

        n_tgt = tgt_positions.shape[0]
        all_features = torch.zeros(
            n_tgt,
            4 * self.num_radial,
            dtype=tgt_positions.dtype,
            device=tgt_positions.device,
        )
        all_features[:, : self.num_radial] = self.l0_factors * scalar_features[0::4]
        all_features[:, self.num_radial :: 3] = self.l1_factors * (
            scalar_features[2::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 1 :: 3] = self.l1_factors * (
            scalar_features[3::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 2 :: 3] = self.l1_factors * (
            scalar_features[1::4] - scalar_features[0::4]
        )
        return all_features

    def forward_source_target(
        self,
        source_feats: torch.Tensor,    # [N_src, 1, m_dim] or [N_src, m_dim]
        src_positions: torch.Tensor,
        src_batch: torch.Tensor,
        tgt_positions: torch.Tensor,
        tgt_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Source-target field features at tgt_positions from sources at src_positions.

        Precondition: source and target node sets are disjoint. The block does
        not subtract a self-interaction term, so colocated source/target rows
        will produce wrong answers. Documented; not enforced (O(N^2) check).
        """
        if source_feats.dim() == 3:
            source_feats_2d = source_feats.squeeze(-2)
        else:
            source_feats_2d = source_feats

        if self.density_max_l == 0 and self.projection_max_l == 0:
            return self.call_source_target_density_0_feats_0(
                source_feats_2d, src_positions, src_batch, tgt_positions, tgt_batch
            )
        if self.density_max_l == 1 and self.projection_max_l == 0:
            all_feats = self.call_source_target_density_1_feats_1(
                source_feats_2d, src_positions, src_batch, tgt_positions, tgt_batch
            )
            return all_feats[:, : self.num_radial]
        if self.density_max_l == 0 and self.projection_max_l == 1:
            padded = torch.zeros(
                source_feats_2d.shape[0],
                4,
                dtype=source_feats_2d.dtype,
                device=source_feats_2d.device,
            )
            padded[:, 0] = source_feats_2d[:, 0]
            return self.call_source_target_density_1_feats_1(
                padded, src_positions, src_batch, tgt_positions, tgt_batch
            )
        return self.call_source_target_density_1_feats_1(
            source_feats_2d, src_positions, src_batch, tgt_positions, tgt_batch
        )

