import math
from typing import Literal, NamedTuple

import torch
from mace.tools.scatter import scatter_sum
from scipy.constants import pi

from .gto_utils import (
    GTOSelfInteractionBlock,
    get_Cl_sigma,
)
from .utils import FIELD_CONSTANT

# Sign convention used by the analytical modules (and by charges_energy_from_graph):
# for every directed edge, separation = positions[receiver] - positions[sender].
#
# Component ordering: l=1 coefficients arrive in e3nn order (y, z, x) at feature
# columns (1, 2, 3). Indexing feature columns with E3NN_TO_CARTESIAN_COLUMNS
# yields Cartesian (x, y, z); indexing a Cartesian vector's last axis with
# CARTESIAN_TO_E3NN_COMPONENTS yields e3nn order (y, z, x).
E3NN_TO_CARTESIAN_COLUMNS = [3, 1, 2]
CARTESIAN_TO_E3NN_COMPONENTS = [1, 2, 0]

# Which real-space evaluator the GTO blocks in energy.py / features.py build.
# "finite_difference" is the default everywhere so that existing checkpoints keep
# reproducing the numbers their weights were fitted to.
RealSpaceMethod = Literal["finite_difference", "analytical"]

SQRT_TWO_OVER_PI = math.sqrt(2.0 / pi)

# The upward recursion for the smeared-Coulomb kernels suffers catastrophic
# cancellation for scaled_distance_squared = R^2 / (2 * Sigma^2) << 1, so below
# this crossover the kernels are evaluated from their Taylor series instead.
SERIES_CROSSOVER = 0.5
# Truncation error of the series is bounded by crossover^K / K! with
# K = SERIES_NUM_TERMS; 13 terms give < 1e-12 at the crossover in float64.
SERIES_NUM_TERMS = 13


def _smeared_coulomb_kernels_series(
    scaled_distance_squared: torch.Tensor,
    combined_smearing_width,
    highest_order: int,
) -> list[torch.Tensor]:
    """Taylor-series branch of the smeared-Coulomb kernels (small distances)."""
    accumulators = [
        torch.zeros_like(scaled_distance_squared) for _ in range(highest_order + 1)
    ]
    term = torch.ones_like(scaled_distance_squared)
    for k in range(SERIES_NUM_TERMS):
        for order in range(highest_order + 1):
            accumulators[order] = accumulators[order] + term / (2 * order + 2 * k + 1)
        term = term * (-scaled_distance_squared) / (k + 1)

    kernels = []
    width_power = 1.0 / combined_smearing_width
    for order in range(highest_order + 1):
        kernels.append(SQRT_TWO_OVER_PI * width_power * accumulators[order])
        width_power = width_power / combined_smearing_width**2
    return kernels


def _smeared_coulomb_kernels_closed_form(
    scaled_distance_squared: torch.Tensor,
    combined_smearing_width,
    highest_order: int,
) -> list[torch.Tensor]:
    """Closed-form branch of the smeared-Coulomb kernels (well-separated pairs)."""
    distance = combined_smearing_width * torch.sqrt(2.0 * scaled_distance_squared)
    distance_squared = 2.0 * scaled_distance_squared * combined_smearing_width**2

    kernels = [torch.erf(torch.sqrt(scaled_distance_squared)) / distance]
    gaussian_term = (
        SQRT_TWO_OVER_PI / combined_smearing_width * torch.exp(-scaled_distance_squared)
    )
    width_factor = 1.0
    for order in range(highest_order):
        kernels.append(
            ((2 * order + 1) * kernels[-1] - gaussian_term * width_factor)
            / distance_squared
        )
        width_factor = width_factor / combined_smearing_width**2
    return kernels


class SmearedCoulombKernels(NamedTuple):
    """Radial kernels B_n(R) of the Coulomb interaction between two Gaussians.

    With combined_smearing_width^2 = width_1^2 + width_2^2 the zeroth kernel is
    B_0(R) = erf(R / (sqrt(2) * combined_smearing_width)) / R, the interaction of
    two unit Gaussian charges. Each higher kernel is the radial derivative of
    the previous one, B_{n+1}(R) = -(1/R) dB_n/dR, so with separation vector R
    (receiver minus sender) the multipole interaction terms read:

        charge-charge     q_s q_r B_0
        charge-dipole     (q_s (p_r . R) - q_r (p_s . R)) B_1
        dipole-dipole     (p_s . p_r) B_1 - (p_s . R)(p_r . R) B_2

    Equivalently B_1 R is the field of a unit charge and B_1 I - B_2 R R^T the
    field gradient. All kernels are finite at R = 0. Kernels above the requested
    order are None; every present kernel is broadcast to the common shape of
    distance_squared and combined_smearing_width.
    """

    b0: torch.Tensor
    b1: torch.Tensor | None = None
    b2: torch.Tensor | None = None


def smeared_coulomb_kernels(
    distance_squared: torch.Tensor,
    combined_smearing_width,
    highest_order: int,
) -> SmearedCoulombKernels:
    """Evaluate the smeared-Coulomb kernels B_0..B_highest_order for pairs.

    See SmearedCoulombKernels for the definition of the kernels and how they
    combine into charge and dipole interactions.

    Two branches are needed because neither evaluation is accurate on the
    whole axis. The closed form obtains B_{n+1} from B_n by the upward
    recursion B_{n+1} = ((2n+1) B_n - g(R)) / R^2 with a Gaussian g(R): as
    R -> 0 the numerator is a difference of nearly equal terms divided by a
    vanishing R^2 (and B_0 = erf(.)/R itself is 0/0), so each order loses
    more digits and B_2 is unusable well before R reaches zero. The Taylor
    series in R^2 / (2 Sigma^2) is exact at R = 0 and free of cancellation,
    but it is alternating with terms that grow before they decay, so its cost
    and rounding error grow with the argument and it cannot replace the
    closed form at large R. Below SERIES_CROSSOVER the series is used, where
    SERIES_NUM_TERMS terms reach 1e-12; above it the closed form is used,
    where the recursion has lost at most a few digits. Both branches are
    evaluated on every pair and selected with torch.where so that autograd
    sees a single smooth expression on each side of the crossover.

    Args:
        distance_squared: squared pair distances, broadcastable against
            combined_smearing_width (e.g. [n_edges] with a scalar width, or
            [n_edges, 1] with a [n_radial] width tensor).
        combined_smearing_width: float or tensor of combined Gaussian widths.
        highest_order: largest kernel order to evaluate: 0 for charges only,
            1 for charge-dipole terms, 2 for dipole-dipole terms.

    Returns:
        SmearedCoulombKernels with b0..b_highest_order filled and the rest None.
    """
    if highest_order not in (0, 1, 2):
        raise ValueError(
            f"highest_order must be 0, 1 or 2 (l <= 1), got {highest_order}"
        )
    # Further work (performance): both branches are evaluated for every edge,
    # and the unrolled series alone emits ~40 small elementwise kernels, so
    # batches of many tiny graphs are launch-overhead bound (0.6-0.8x vs the
    # finite-difference modules on B200; every larger regime wins). The whole
    # chain is straight-line elementwise math and should fuse into a few
    # kernels under torch.compile. TODO: verify compilation and double
    # backward before relying on it.
    scaled_distance_squared = distance_squared / (
        2.0 * combined_smearing_width**2
    )
    in_series_branch = scaled_distance_squared < SERIES_CROSSOVER
    # Clamp each branch's argument into its safe domain before torch.where:
    # autograd evaluates both branches, and the closed form is singular at R = 0.
    series_kernels = _smeared_coulomb_kernels_series(
        torch.clamp(scaled_distance_squared, max=SERIES_CROSSOVER),
        combined_smearing_width,
        highest_order,
    )
    closed_form_kernels = _smeared_coulomb_kernels_closed_form(
        torch.clamp(scaled_distance_squared, min=0.5 * SERIES_CROSSOVER),
        combined_smearing_width,
        highest_order,
    )
    return SmearedCoulombKernels(
        *(
            torch.where(in_series_branch, series_kernel, closed_form_kernel)
            for series_kernel, closed_form_kernel in zip(
                series_kernels, closed_form_kernels
            )
        )
    )


@torch.no_grad()
def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor, N: int
) -> torch.Tensor:
    """
    Duplicate each node N times, then for each graph build directed
    edges between every pair of duplicates *unless* they share the same
    original node ID.

    Fully vectorized block-diagonal construction: duplicated nodes are
    grouped by graph with a stable sort. When every graph has the same size
    (single graphs and uniform batches) the pair mesh is a plain batched
    broadcast with no per-edge gathers; otherwise every sender's receiver
    block is enumerated from cumulative per-node edge offsets. All
    dense-edge-sized work is additions and gathers (no integer
    division/modulo, which is emulated and slow for int64 on GPUs). No
    per-graph Python loop; two host syncs per call.

    Args:
        batch (LongTensor): shape [M], graph ID of each original node.
        N (int): number of duplicates per node.

    Returns:
        edge_index (LongTensor[2, E])
    """
    batch = batch.long()
    device = batch.device
    num_nodes = batch.size(0)
    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    original_node_ids = torch.arange(num_nodes, device=device)
    # duplicated per-node graph ID and original-ID
    duplicated_batch = batch.repeat_interleave(N)  # [M*N]
    duplicated_original_ids = original_node_ids.repeat_interleave(N)  # [M*N]

    # group duplicated nodes by graph (stable, so duplicates keep their order)
    grouped_nodes = torch.argsort(duplicated_batch, stable=True)  # [M*N]
    graph_sizes = torch.bincount(duplicated_batch)  # [n_graphs]

    if bool((graph_sizes == graph_sizes[0]).all().item()):
        # equal-size fast path: batched dense mesh via broadcasting
        num_graphs = graph_sizes.size(0)
        size = grouped_nodes.numel() // num_graphs
        nodes = grouped_nodes.view(num_graphs, size)
        senders = nodes.unsqueeze(2).expand(num_graphs, size, size).reshape(-1)
        receivers = nodes.unsqueeze(1).expand(num_graphs, size, size).reshape(-1)
    else:
        node_offsets = torch.cumsum(graph_sizes, dim=0) - graph_sizes  # [n_graphs]

        # per grouped node: its graph's size (= its number of outgoing dense
        # edges) and its graph's first slot in the grouped ordering
        edges_per_node = graph_sizes.repeat_interleave(graph_sizes)  # [M*N]
        graph_start_of_node = node_offsets.repeat_interleave(graph_sizes)  # [M*N]
        edge_start_of_node = torch.cumsum(edges_per_node, dim=0) - edges_per_node

        num_dense_edges = int(edges_per_node.sum().item())
        sender_slots = torch.repeat_interleave(
            torch.arange(edges_per_node.size(0), device=device), edges_per_node
        )  # [E_dense]
        receiver_slots = (
            torch.arange(num_dense_edges, device=device)
            - edge_start_of_node[sender_slots]
            + graph_start_of_node[sender_slots]
        )

        senders = grouped_nodes[sender_slots]
        receivers = grouped_nodes[receiver_slots]

    keep = duplicated_original_ids[senders] != duplicated_original_ids[receivers]
    return torch.stack([senders[keep], receivers[keep]], dim=0)


def charges_energy_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    density_smearing_width,
):
    """
    Computes the energy of a collection of charges considering only specifed edges.
    normalization of the charges is multipoles.
    """
    sender, receiver = edge_index

    R_ij = positions[receiver] - positions[sender]  # [N_edges,3]
    d_ij = torch.linalg.norm(R_ij, dim=-1)  # [N_edges,1]
    smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
        torch.abs(d_ij) + 1e-6
    )

    # charge part
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / (4 * pi)
    )
    # handle the case with no edges
    if edge_energy.numel() == 0:
        return torch.zeros(
            (batch.max() + 1,), dtype=charges.dtype, device=charges.device
        )
    node_energies = scatter_sum(
        src=edge_energy.squeeze(-1), index=receiver, dim=-1, dim_size=charges.shape[0]
    )
    return scatter_sum(src=node_energies, index=batch, dim=-1)  # [n_graphs]


class RealSpaceFiniteDiffereneEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
        offset=0.02,
    ):
        if density_max_l > 1:
            raise ValueError(
                "RealSpaceFiniteDiffereneEnergy only supports l=0 and l=1."
            )

        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.include_self_interaction = include_self_interaction
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            "multipoles",
            "multipoles",
        )

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

    def energy_l0(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
    ) -> torch.Tensor:

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)

        energy = charges_energy_from_graph(
            source_feats.squeeze(-1),
            positions,
            edge_index,
            batch,
            density_smearing_width=self.density_smearing_width,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=-1)
            energy += self_energy * 0.5

        return energy

    def energy_l1(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x
        extended_positions[2::4] += self.y
        extended_positions[3::4] += self.z

        extended_batch = batch.repeat_interleave(4)
        charges = torch.zeros_like(extended_positions[:, 0])

        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)

        energy = charges_energy_from_graph(
            charges,
            extended_positions,
            edge_index,
            extended_batch,
            density_smearing_width=self.density_smearing_width,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=-1)
            energy += self_energy * 0.5

        return energy

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0:
            return self.energy_l0(source_feats, positions, batch)
        else:
            return self.energy_l1(source_feats, positions, batch)


def charges_features_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    total_width_factors,  # [1, n_radial]
):
    """
    Computes the features from a collection of charges, on set of scalar features, considering only specified edges.
    normalization of the charges is multipoles.
    Uses the module-wide sign convention separation = positions[receiver] - positions[sender]
    (only the distance enters here, so this matches the historical behaviour exactly).
    """
    num_nodes = positions.shape[0]
    sender, receiver = edge_index
    R_ij = positions[receiver] - positions[sender]  # [N_edges,3]
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)  # [N_edges,1]
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)

    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=num_nodes,
    )  # [n_nodes, n_radial]

    features = FIELD_CONSTANT * features / (4 * pi)
    return features


class RealSpaceFiniteDifferenceElectrostaticFeatures(torch.nn.Module):
    """Computes field features for L=0,1 charges and features.
    vector charges and features are represented by displaced scalars."""

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: list[float],
        include_self_interaction=False,
        integral_normalization="receiver",
        offset: float = 0.1,
    ):
        super().__init__()

        self.density_max_l = density_max_l
        self.projection_max_l = projection_max_l
        self.include_self_interaction = include_self_interaction
        self.density_smearing_width = density_smearing_width
        self.projection_smearing_widths = projection_smearing_widths
        self.num_radial = len(projection_smearing_widths)

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        projection_smearing_widths_tensor = torch.tensor(
            projection_smearing_widths, dtype=torch.get_default_dtype()
        )
        total_width_factors = torch.pow(
            (density_smearing_width**2 + projection_smearing_widths_tensor**2) / 2, 0.5
        )
        self.register_buffer("total_width_factors", total_width_factors)

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

        l0_factors = [
            get_Cl_sigma(0, sigma, normalize=integral_normalization)
            / get_Cl_sigma(0, sigma, normalize="multipoles")
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l0_factors", torch.tensor(l0_factors, dtype=torch.get_default_dtype())
        )
        l1_factors = [
            3**0.5
            * sigma**2
            * (
                get_Cl_sigma(1, sigma, normalize=integral_normalization)
                / get_Cl_sigma(0, sigma, normalize="multipoles")
            )
            / self.offset
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l1_factors", torch.tensor(l1_factors, dtype=torch.get_default_dtype())
        )

    def call_density_0_feats_0(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_long_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 1
        )
        feats = charges_features_from_graph(
            charges=source_feats[:, 0],
            positions=positions,
            edge_index=edge_long_index,
            batch=batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )  # [n_atoms, n_radial]
        return self.l0_factors * feats

    def call_density_1_feats_1(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x
        extended_positions[2::4] += self.y
        extended_positions[3::4] += self.z

        extended_batch = batch.repeat_interleave(4)
        charges = torch.zeros_like(extended_positions[:, 0])

        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)

        scalar_features = charges_features_from_graph(
            charges=charges,
            positions=extended_positions,
            edge_index=edge_index,
            batch=extended_batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )  # [all_nodes, num_radial]

        all_features = source_feats.new_zeros(
            (batch.size(0), 4 * self.num_radial)
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

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self.call_density_0_feats_0(
                source_feats, node_positions, batch
            )
        elif self.density_max_l == 1 and self.projection_max_l == 0:
            all_feats = self.call_density_1_feats_1(
                source_feats, node_positions, batch
            )
            features = all_feats[:, : self.num_radial]
        elif self.density_max_l == 0 and self.projection_max_l == 1:
            padded_source_feats = torch.zeros(
                source_feats.shape[0],
                4,
                dtype=source_feats.dtype,
                device=source_feats.device,
            )
            padded_source_feats[:, 0] = source_feats[:, 0]
            features = self.call_density_1_feats_1(
                padded_source_feats, node_positions, batch
            )
        else:
            features = self.call_density_1_feats_1(
                source_feats, node_positions, batch
            )

        self_interaction_terms = self.self_interaction(source_feats)
        if self.include_self_interaction:
            features += self_interaction_terms

        return features, self_interaction_terms, None


def _match_tensor_placement(
    module: torch.nn.Module, reference: torch.nn.Module
) -> torch.nn.Module:
    """Put a freshly built real-space module where the one it replaces lives.

    Only floating-point buffers follow the reference dtype (torch.nn.Module.to
    leaves integer buffers such as the self-interaction select_indices alone), so
    a model moved to a device or cast after construction keeps working when its
    evaluator is swapped.
    """
    reference_tensor = next(
        (buffer for buffer in reference.buffers() if buffer.is_floating_point()), None
    )
    if reference_tensor is None:
        return module
    return module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)


def _validate_source_features(source_feats: torch.Tensor, density_max_l: int) -> None:
    expected_columns = (density_max_l + 1) ** 2
    if source_feats.dim() != 2 or source_feats.shape[-1] != expected_columns:
        raise ValueError(
            f"source_feats must have shape [n_nodes, {expected_columns}] for "
            f"density_max_l={density_max_l}, got {tuple(source_feats.shape)}"
        )


class RealSpaceAnalyticalEnergy(torch.nn.Module):
    """Exact real-space electrostatic energy of Gaussian multipole densities (l <= 1).

    Closed-form replacement for RealSpaceFiniteDiffereneEnergy: dipoles interact
    through the analytic interaction tensors built from the smeared-Coulomb
    kernels instead of displaced point charges, so the energy is exactly
    rotationally invariant at all separations (including overlapping densities).

    Conventions: source_feats are in "multipoles" normalization with l=1
    components in e3nn order (y, z, x); pair separations are
    positions[receiver] - positions[sender]; the directed edge sum carries the
    0.5 double-counting factor and the FIELD_CONSTANT / (4 pi) prefactor.

    Further work (performance, applies to the features module as well): forces
    come from autograd, which stores the per-edge intermediates (~2 GB
    backward peak at 3000 atoms). The position gradients are themselves closed
    form (one more kernel order, B_3), so a custom backward composed of
    differentiable ops would cut memory and forward+forces time — it must
    preserve double backward for force-loss training (rerun gradgradcheck and
    the branch-straddling gradient tests).
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
    ):
        super().__init__()
        if density_max_l not in (0, 1):
            raise ValueError("RealSpaceAnalyticalEnergy only supports l=0 and l=1.")
        if density_smearing_width <= 0.0:
            raise ValueError("density_smearing_width must be positive.")

        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.include_self_interaction = include_self_interaction
        # Both densities carry the same smearing width, so the pair kernel width is
        # sqrt(width^2 + width^2).
        self.combined_smearing_width = math.sqrt(2.0) * density_smearing_width

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            "multipoles",
            "multipoles",
        )

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (density_max_l+1)**2]
        positions: torch.Tensor,  # [n_nodes, 3]
        batch: torch.Tensor,  # [n_nodes]
    ) -> torch.Tensor:  # [n_graphs]
        _validate_source_features(source_feats, self.density_max_l)
        num_nodes = positions.shape[0]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
        sender, receiver = edge_index[0], edge_index[1]

        separation = positions[receiver] - positions[sender]  # [n_edges, 3]
        distance_squared = torch.sum(separation * separation, dim=-1)

        highest_order = 2 if self.density_max_l >= 1 else 0
        kernels = smeared_coulomb_kernels(
            distance_squared, self.combined_smearing_width, highest_order
        )

        charges = source_feats[:, 0]
        pair_energy = charges[sender] * charges[receiver] * kernels.b0

        if self.density_max_l >= 1:
            dipoles = source_feats[:, E3NN_TO_CARTESIAN_COLUMNS]  # [n_nodes, 3] (x, y, z)
            sender_dipoles = dipoles[sender]
            receiver_dipoles = dipoles[receiver]
            sender_dipole_along_separation = torch.sum(
                sender_dipoles * separation, dim=-1
            )
            receiver_dipole_along_separation = torch.sum(
                receiver_dipoles * separation, dim=-1
            )
            pair_energy = pair_energy - (
                charges[sender] * receiver_dipole_along_separation
                - charges[receiver] * sender_dipole_along_separation
            ) * kernels.b1
            pair_energy = (
                pair_energy
                + torch.sum(sender_dipoles * receiver_dipoles, dim=-1) * kernels.b1
            )
            pair_energy = (
                pair_energy
                - sender_dipole_along_separation
                * receiver_dipole_along_separation
                * kernels.b2
            )

        edge_energy = 0.5 * FIELD_CONSTANT / (4 * pi) * pair_energy
        node_energies = scatter_sum(
            src=edge_energy, index=receiver, dim=-1, dim_size=num_nodes
        )
        energy = scatter_sum(
            src=node_energies, index=batch, dim=-1, dim_size=num_graphs
        )

        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)
            self_node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            energy = energy + 0.5 * scatter_sum(
                src=self_node_energies, index=batch, dim=-1, dim_size=num_graphs
            )

        return energy


class RealSpaceAnalyticalElectrostaticFeatures(torch.nn.Module):
    """Exact projected-potential features of Gaussian multipole densities (l <= 1).

    Closed-form replacement for RealSpaceFiniteDifferenceElectrostaticFeatures:
    the l=0 feature per radial channel is the source potential smeared with the
    combined width sqrt(density_width^2 + projection_width^2), and the l=1
    feature is its exact gradient with respect to the receiver position, so the
    features are exactly rotationally equivariant.

    Conventions match RealSpaceAnalyticalEnergy; the output layout matches the
    finite-difference module: [n_nodes, num_radial] scalar features followed
    (for projection_max_l >= 1) by num_radial blocks of e3nn-ordered (y, z, x)
    vector features, [n_nodes, 4 * num_radial] in total.

    The per-channel constants are registered as non-persistent buffers: they are
    pure functions of the constructor arguments, and the finite-difference module
    registers l0_factors / l1_factors of the same shape holding offset-dependent
    values, so keeping them out of the state dict stops a non-strict load from
    silently mixing the two conventions.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: list[float],
        include_self_interaction: bool = False,
        integral_normalization: str = "receiver",
    ):
        super().__init__()
        if density_max_l not in (0, 1) or projection_max_l not in (0, 1):
            raise ValueError(
                "RealSpaceAnalyticalElectrostaticFeatures only supports l=0 and l=1."
            )
        if density_smearing_width <= 0.0:
            raise ValueError("density_smearing_width must be positive.")
        if len(projection_smearing_widths) == 0:
            raise ValueError("projection_smearing_widths must not be empty.")
        if any(width <= 0.0 for width in projection_smearing_widths):
            raise ValueError("projection_smearing_widths must be positive.")
        if integral_normalization not in ("multipoles", "receiver", "none"):
            raise ValueError(
                "integral_normalization must be one of 'multipoles', 'receiver', 'none'"
            )

        self.density_max_l = density_max_l
        self.projection_max_l = projection_max_l
        self.include_self_interaction = include_self_interaction
        self.density_smearing_width = density_smearing_width
        self.projection_smearing_widths = projection_smearing_widths
        self.num_radial = len(projection_smearing_widths)

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        combined_smearing_widths = torch.tensor(
            [
                math.sqrt(density_smearing_width**2 + projection_width**2)
                for projection_width in projection_smearing_widths
            ],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer(
            "combined_smearing_widths", combined_smearing_widths, persistent=False
        )

        # The moment-normalized formulas give the smeared potential and its
        # gradient; these per-channel factors convert to the requested receiver
        # normalization (see get_Cl_sigma). Unlike the finite-difference module,
        # no offset-dependent factors appear.
        l0_factors = [
            get_Cl_sigma(0, projection_width, normalize=integral_normalization)
            / get_Cl_sigma(0, projection_width, normalize="multipoles")
            for projection_width in projection_smearing_widths
        ]
        self.register_buffer(
            "l0_factors",
            torch.tensor(l0_factors, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        if projection_max_l >= 1:
            l1_factors = [
                get_Cl_sigma(1, projection_width, normalize=integral_normalization)
                / get_Cl_sigma(1, projection_width, normalize="multipoles")
                for projection_width in projection_smearing_widths
            ]
            self.register_buffer(
                "l1_factors",
                torch.tensor(l1_factors, dtype=torch.get_default_dtype()),
                persistent=False,
            )

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (density_max_l+1)**2]
        node_positions: torch.Tensor,  # [n_nodes, 3]
        batch: torch.Tensor,  # [n_nodes]
    ):
        _validate_source_features(source_feats, self.density_max_l)
        num_nodes = node_positions.shape[0]

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
        sender, receiver = edge_index[0], edge_index[1]

        # separation points from the sending density to the receiving node.
        separation = node_positions[receiver] - node_positions[sender]  # [n_edges, 3]
        distance_squared = torch.sum(
            separation * separation, dim=-1, keepdim=True
        )  # [n_edges, 1]

        if self.density_max_l >= 1 and self.projection_max_l >= 1:
            highest_order = 2
        elif self.density_max_l >= 1 or self.projection_max_l >= 1:
            highest_order = 1
        else:
            highest_order = 0
        kernels = smeared_coulomb_kernels(
            distance_squared, self.combined_smearing_widths, highest_order
        )  # each [n_edges, num_radial]

        charges = source_feats[:, 0]
        sender_charges = charges[sender].unsqueeze(-1)  # [n_edges, 1]

        scalar_edge_features = sender_charges * kernels.b0
        if self.density_max_l >= 1:
            dipoles = source_feats[:, E3NN_TO_CARTESIAN_COLUMNS]  # (x, y, z)
            sender_dipoles = dipoles[sender]  # [n_edges, 3]
            sender_dipole_along_separation = torch.sum(
                sender_dipoles * separation, dim=-1, keepdim=True
            )  # [n_edges, 1]
            scalar_edge_features = (
                scalar_edge_features + sender_dipole_along_separation * kernels.b1
            )

        scalar_features = scatter_sum(
            src=scalar_edge_features, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, num_radial]
        scalar_features = (
            FIELD_CONSTANT / (4 * pi) * self.l0_factors * scalar_features
        )
        feature_blocks = [scalar_features]

        if self.projection_max_l >= 1:
            # Gradient of the smeared potential with respect to the receiver
            # position: [n_edges, num_radial, 3] in Cartesian components.
            gradient_edge_features = (
                -sender_charges.unsqueeze(-1)
                * separation.unsqueeze(1)
                * kernels.b1.unsqueeze(-1)
            )
            if self.density_max_l >= 1:
                gradient_edge_features = (
                    gradient_edge_features
                    + sender_dipoles.unsqueeze(1) * kernels.b1.unsqueeze(-1)
                    - sender_dipole_along_separation.unsqueeze(-1)
                    * separation.unsqueeze(1)
                    * kernels.b2.unsqueeze(-1)
                )

            gradient_features = scatter_sum(
                src=gradient_edge_features, index=receiver, dim=0, dim_size=num_nodes
            )  # [n_nodes, num_radial, 3]
            gradient_features = (
                FIELD_CONSTANT
                / (4 * pi)
                * self.l1_factors.unsqueeze(-1)
                * gradient_features
            )
            vector_features = gradient_features[..., CARTESIAN_TO_E3NN_COMPONENTS]
            feature_blocks.append(vector_features.reshape(num_nodes, -1))

        features = torch.cat(feature_blocks, dim=-1)

        self_interaction_terms = self.self_interaction(source_feats)
        if self.include_self_interaction:
            features = features + self_interaction_terms

        return features, self_interaction_terms, None
