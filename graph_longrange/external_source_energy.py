"""Electrostatic interaction energy between two disjoint source sets.

This module is the dynamic-source analogue of :mod:`jellium_energy`.  It keeps
the electrostatic mathematics in graph_longrange while allowing model adapters
to provide a second set of sources (for example MM point charges) at runtime.
"""

from __future__ import annotations

import copy
import math
from typing import Literal, Optional, Union

import torch
from mace.tools.scatter import scatter_sum

from .energy import GTOElectrostaticEnergy, energy_product_batch
from .features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    compute_coulomb_factor,
)
from .slabs import slab_dipole_correction_energy
from .utils import FIELD_CONSTANT


CrossPBCHandling = Literal[
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
    "auto",
]
ChunkSize = Union[int, Literal["auto"], None]


class GTOElectrostaticCrossEnergy(torch.nn.Module):
    """GTO electrostatic energy between two disjoint sets of multipoles.

    The returned quantity is exactly the cross term

    ``E(source + target) - E(source) - E(target)``.

    It deliberately contains no source-source or target-target contribution
    and no self-interaction contribution.  Inputs use graph_longrange's
    ``multipoles`` normalization and e3nn component ordering, exactly like
    :class:`GTOElectrostaticEnergy`.
    """

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        kspace_cutoff: float,
        pbc_handling: CrossPBCHandling = "mixed_periodic",
        reciprocal_chunk_size: ChunkSize = "auto",
        reciprocal_chunk_budget_bytes: int = 1 << 30,
        realspace_chunk_size: ChunkSize = "auto",
        realspace_chunk_budget_bytes: int = 256 << 20,
    ):
        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.kspace_cutoff = kspace_cutoff
        self.pbc_handling = pbc_handling
        self.reciprocal_chunk_size = reciprocal_chunk_size
        self.reciprocal_chunk_budget_bytes = reciprocal_chunk_budget_bytes
        self.realspace_chunk_size = realspace_chunk_size
        self.realspace_chunk_budget_bytes = realspace_chunk_budget_bytes

        # Reuse the canonical bases, finite-difference representation and PBC
        # corrections instead of maintaining a second normalization convention.
        self.reference_energy = GTOElectrostaticEnergy(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=False,
            pbc_handling=pbc_handling,
        )

    @classmethod
    def from_energy(
        cls,
        energy: GTOElectrostaticEnergy,
        **kwargs,
    ) -> "GTOElectrostaticCrossEnergy":
        """Construct a matching cross block from an existing energy block.

        A deep copy preserves non-default finite-difference buffers and dtype.
        """
        block = cls(
            density_max_l=energy.density_max_l,
            density_smearing_width=energy.density_smearing_width,
            kspace_cutoff=energy.kspace_cutoff,
            pbc_handling=energy.pbc_handling,
            **kwargs,
        )
        block.reference_energy = copy.deepcopy(energy)
        block.reference_energy.include_self_interaction = False
        block.reference_energy.realspace_energy.include_self_interaction = False
        block.reference_energy.set_pbc_handling(energy.pbc_handling)
        return block

    def set_pbc_handling(self, pbc_handling: CrossPBCHandling) -> None:
        self.pbc_handling = pbc_handling
        self.reference_energy.set_pbc_handling(pbc_handling)

    @staticmethod
    def _features_2d(features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 3:
            if features.shape[-2] != 1:
                raise ValueError(
                    "3-D multipole features must have shape [n, 1, m_dim]."
                )
            return features.squeeze(-2)
        if features.dim() != 2:
            raise ValueError("multipole features must be 2-D or 3-D.")
        return features

    @staticmethod
    def _pad_graphs(values: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if values.shape[0] == num_graphs:
            return values
        if values.shape[0] > num_graphs:
            raise ValueError(
                f"per-graph tensor has {values.shape[0]} rows, expected {num_graphs}."
            )
        return torch.cat(
            [values, values.new_zeros((num_graphs - values.shape[0],))], dim=0
        )

    @staticmethod
    def _validate_set(
        name: str,
        features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> None:
        if positions.dim() != 2 or positions.shape[-1] != 3:
            raise ValueError(f"{name}_positions must have shape [n, 3].")
        if features.shape[0] != positions.shape[0] or batch.numel() != positions.shape[0]:
            raise ValueError(
                f"{name} features, positions and batch must have the same length."
            )

    def _as_displaced_charges(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features_2d(features)
        if features.shape[-1] == 1:
            return features[:, 0], positions, batch
        if features.shape[-1] != 4:
            raise ValueError(
                "real-space cross energy supports l<=1 (1 or 4 coefficients), "
                f"got {features.shape[-1]}."
            )

        realspace = self.reference_energy.realspace_energy
        offset = realspace.offset
        displaced_positions = positions.repeat_interleave(4, dim=0)
        displaced_positions[1::4] += realspace.x.to(displaced_positions)
        displaced_positions[2::4] += realspace.y.to(displaced_positions)
        displaced_positions[3::4] += realspace.z.to(displaced_positions)
        displaced_batch = batch.repeat_interleave(4)

        charges = features.new_zeros(displaced_positions.shape[0])
        # graph_longrange/e3nn order is (l0, y, z, x).
        charges[1::4] = features[:, 3] / offset
        charges[2::4] = features[:, 1] / offset
        charges[3::4] = features[:, 2] / offset
        charges[0::4] = features[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )
        return charges, displaced_positions, displaced_batch

    @staticmethod
    def _resolve_chunk(
        setting: ChunkSize,
        n_items: int,
        other_items: int,
        element_size: int,
        budget_bytes: int,
        arrays_per_pair: int,
    ) -> int:
        if isinstance(setting, bool):
            setting = None
        if isinstance(setting, int):
            return max(1, min(setting, max(1, n_items)))
        if n_items <= 0:
            return 1
        denom = max(1, other_items * element_size * arrays_per_pair)
        return max(1, min(n_items, budget_bytes // denom))

    def _forward_realspace(
        self,
        source_feats: torch.Tensor,
        source_positions: torch.Tensor,
        source_batch: torch.Tensor,
        target_feats: torch.Tensor,
        target_positions: torch.Tensor,
        target_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        source_q, source_pos, source_graph = self._as_displaced_charges(
            source_feats, source_positions, source_batch
        )
        target_q, target_pos, target_graph = self._as_displaced_charges(
            target_feats, target_positions, target_batch
        )
        if source_q.numel() == 0 or target_q.numel() == 0:
            return source_positions.new_zeros(num_graphs)

        step = self._resolve_chunk(
            self.realspace_chunk_size,
            target_q.numel(),
            source_q.numel(),
            source_positions.element_size(),
            self.realspace_chunk_budget_bytes,
            arrays_per_pair=6,
        )
        sigma = self.reference_energy.realspace_energy.density_smearing_width
        prefactor = FIELD_CONSTANT / (4.0 * math.pi)
        per_source = source_q.new_zeros(source_q.shape[0])
        for start in range(0, target_q.numel(), step):
            stop = min(start + step, target_q.numel())
            displacement = source_pos[:, None, :] - target_pos[None, start:stop, :]
            distance = torch.linalg.norm(displacement, dim=-1)
            damped_reciprocal = torch.erf(0.5 * distance / sigma) / (
                distance.abs() + 1e-6
            )
            same_graph = source_graph[:, None] == target_graph[None, start:stop]
            pair_energy = (
                prefactor
                * source_q[:, None]
                * target_q[None, start:stop]
                * damped_reciprocal
            )
            per_source = per_source + torch.where(
                same_graph, pair_energy, torch.zeros_like(pair_energy)
            ).sum(dim=1)
        return scatter_sum(
            per_source, source_graph, dim=0, dim_size=num_graphs
        )

    def _assemble_density_chunked(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k0_mask: torch.Tensor,
        k_vector_batch: torch.Tensor,
        volume: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        features = self._features_2d(features)
        density = features.new_zeros((k_vectors.shape[0], 2))
        if features.shape[0] == 0:
            return density
        density_basis = self.reference_energy.density_basis(
            k_vectors, k_norm2, k0_mask
        )
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        for start in range(0, features.shape[0], chunk_size):
            stop = min(start + chunk_size, features.shape[0])
            inner = torch.matmul(k_vectors, positions[start:stop].t())
            graph_mask = (k_vector_batch[:, None] == batch[start:stop][None, :]).to(
                inner.dtype
            )
            density = density + assemble_fourier_series_batch(
                source_feats=features[start:stop],
                cosines=torch.cos(inner) * graph_mask,
                sines=torch.sin(inner) * graph_mask,
                density_basis_fs=density_basis,
                volume_per_k=volume_per_k,
            )
        return density

    def _correction(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        mode: str,
        num_graphs: int,
    ) -> torch.Tensor:
        features = self._features_2d(features)
        if mode == "pbc":
            return volume.new_zeros(num_graphs)
        molecule = self._pad_graphs(
            self.reference_energy.monopole_dipole_correction(
                features, positions, volume, batch
            ),
            num_graphs,
        )
        slab = self._pad_graphs(
            slab_dipole_correction_energy(features, positions, volume, batch),
            num_graphs,
        )
        if mode == "molecule_in_box":
            return molecule
        if mode == "slab":
            return slab
        if mode != "mixed_periodic":
            raise ValueError(f"Unsupported periodic cross-energy mode: {mode!r}.")

        slab_axes = torch.tensor([True, True, False], device=pbc.device)
        is_molecule = ~pbc.any(dim=1)
        is_slab = (pbc == slab_axes).all(dim=1)
        correction = torch.zeros_like(molecule)
        correction = torch.where(is_molecule, molecule, correction)
        return torch.where(is_slab, slab, correction)

    def _forward_periodic(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        source_positions: torch.Tensor,
        source_batch: torch.Tensor,
        target_feats: torch.Tensor,
        target_positions: torch.Tensor,
        target_batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        mode: str,
        num_graphs: int,
    ) -> torch.Tensor:
        target_chunk = self._resolve_chunk(
            self.reciprocal_chunk_size,
            target_positions.shape[0],
            k_vectors.shape[0],
            k_vectors.element_size(),
            self.reciprocal_chunk_budget_bytes,
            arrays_per_pair=4,
        )
        source_chunk = max(1, source_positions.shape[0])
        rho_source = self._assemble_density_chunked(
            source_feats,
            source_positions,
            source_batch,
            k_vectors,
            k_norm2,
            k0_mask,
            k_vector_batch,
            volume,
            source_chunk,
        )
        rho_target = self._assemble_density_chunked(
            target_feats,
            target_positions,
            target_batch,
            k_vectors,
            k_norm2,
            k0_mask,
            k_vector_batch,
            volume,
            target_chunk,
        )
        potential_target = apply_coulomb_kernel_batch(
            density=rho_target,
            k_factor_coulomb=compute_coulomb_factor(k_norm2, k0_mask),
        )
        cross = 2.0 * energy_product_batch(
            density=rho_source,
            potential=potential_target,
            volume=volume,
            k_vector_batch=k_vector_batch,
        )
        cross = self._pad_graphs(cross, num_graphs)

        correction_source = self._correction(
            source_feats,
            source_positions,
            source_batch,
            volume,
            pbc,
            mode,
            num_graphs,
        )
        correction_target = self._correction(
            target_feats,
            target_positions,
            target_batch,
            volume,
            pbc,
            mode,
            num_graphs,
        )
        mixed_feats = torch.cat(
            [self._features_2d(source_feats), self._features_2d(target_feats)], dim=0
        )
        correction_mixed = self._correction(
            mixed_feats,
            torch.cat([source_positions, target_positions], dim=0),
            torch.cat([source_batch, target_batch], dim=0),
            volume,
            pbc,
            mode,
            num_graphs,
        )
        return cross + correction_mixed - correction_source - correction_target

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        source_positions: torch.Tensor,
        source_batch: torch.Tensor,
        target_feats: torch.Tensor,
        target_positions: torch.Tensor,
        target_batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        """Return the source-target interaction energy for each graph."""
        source_feats = self._features_2d(source_feats)
        target_feats = self._features_2d(target_feats)
        source_batch = source_batch.to(dtype=torch.long, device=source_positions.device)
        target_batch = target_batch.to(dtype=torch.long, device=target_positions.device)
        self._validate_set("source", source_feats, source_positions, source_batch)
        self._validate_set("target", target_feats, target_positions, target_batch)
        if source_feats.shape[-1] != target_feats.shape[-1]:
            raise ValueError("source and target multipoles must have the same width.")

        num_graphs = int(volume.reshape(-1).shape[0])
        if source_positions.shape[0] == 0 or target_positions.shape[0] == 0:
            return volume.reshape(-1).new_zeros(num_graphs)

        mode = self.pbc_handling
        if mode == "auto":
            mode = "mixed_periodic" if torch.any(pbc) else "realspace"
        if mode == "realspace":
            return self._forward_realspace(
                source_feats,
                source_positions,
                source_batch,
                target_feats,
                target_positions,
                target_batch,
                num_graphs,
            )
        return self._forward_periodic(
            k_vectors,
            k_norm2,
            k_vector_batch,
            k0_mask,
            source_feats,
            source_positions,
            source_batch,
            target_feats,
            target_positions,
            target_batch,
            volume,
            pbc,
            mode,
            num_graphs,
        )


class GTOElectrostaticExternalSourceEnergy(torch.nn.Module):
    """Add a dynamic external-source cross term to a normal GTO energy block.

    For an internal source set ``A`` and external set ``B``, this wrapper
    returns

    ``E(A) + E_cross(A, B) = E(A + B) - E(B)``.

    External tensors use graph_longrange's ``multipoles`` normalization and
    e3nn component ordering.  They are transient evaluation inputs rather than
    model buffers or checkpoint state.
    """

    def __init__(
        self,
        base: GTOElectrostaticEnergy,
        cross: Optional[GTOElectrostaticCrossEnergy] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.cross = (
            GTOElectrostaticCrossEnergy.from_energy(base)
            if cross is None
            else cross
        )
        self._external_feats: Optional[torch.Tensor] = None
        self._external_positions: Optional[torch.Tensor] = None
        self._external_batch: Optional[torch.Tensor] = None

    @classmethod
    def from_energy(
        cls,
        energy: GTOElectrostaticEnergy,
        **cross_kwargs,
    ) -> "GTOElectrostaticExternalSourceEnergy":
        """Wrap ``energy`` and construct a cross block with matching bases."""
        return cls(
            base=energy,
            cross=GTOElectrostaticCrossEnergy.from_energy(energy, **cross_kwargs),
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def set_pbc_handling(self, pbc_handling: CrossPBCHandling) -> None:
        self.base.set_pbc_handling(pbc_handling)
        self.cross.set_pbc_handling(pbc_handling)

    def set_external_sources(
        self,
        external_feats: Optional[torch.Tensor] = None,
        external_positions: Optional[torch.Tensor] = None,
        external_batch: Optional[torch.Tensor] = None,
    ) -> None:
        """Set one evaluation's external sources, or clear them with all ``None``."""
        values = (external_feats, external_positions, external_batch)
        if all(value is None for value in values):
            self.clear_external_sources()
            return
        if any(value is None for value in values):
            raise ValueError(
                "external_feats, external_positions and external_batch must "
                "either all be provided or all be None."
            )
        self._external_feats = external_feats
        self._external_positions = external_positions
        self._external_batch = external_batch

    def clear_external_sources(self) -> None:
        self._external_feats = None
        self._external_positions = None
        self._external_batch = None

    def forward(self, **kwargs) -> torch.Tensor:
        """Return the native energy plus the current external cross energy."""
        base_kwargs = dict(kwargs)
        # Older host models may still pass this retired graph_longrange option.
        base_kwargs.pop("force_pbc_evaluator", None)
        energy = self.base(**base_kwargs)
        if self._external_feats is None:
            return energy
        cross = self.cross(
            k_vectors=base_kwargs["k_vectors"],
            k_norm2=base_kwargs["k_norm2"],
            k_vector_batch=base_kwargs["k_vector_batch"],
            k0_mask=base_kwargs["k0_mask"],
            source_feats=base_kwargs["source_feats"],
            source_positions=base_kwargs["node_positions"],
            source_batch=base_kwargs["batch"],
            target_feats=self._external_feats,
            target_positions=self._external_positions,
            target_batch=self._external_batch,
            volume=base_kwargs["volume"],
            pbc=base_kwargs["pbc"],
        )
        return energy + cross


__all__ = [
    "GTOElectrostaticCrossEnergy",
    "GTOElectrostaticExternalSourceEnergy",
]
