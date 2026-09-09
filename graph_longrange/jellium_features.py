from typing import List, Optional

import torch
from scipy.constants import pi

from .features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    compute_coulomb_factor,
    project_to_features_batch,
)
from .gto_utils import GTOBasis, GTOSelfInteractionBlock, GTOInternalFieldtoFeaturesBlock
from .jellium_energy import _slab_fourier_series_flat, _smooth_slab_density
from .utils import FIELD_CONSTANT


def _slab_dipole_correction_node_fields_from_total_dipole(
    total_dipole: torch.Tensor,
    volumes: torch.Tensor,
    node_positions: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    A = FIELD_CONSTANT / (4 * pi)
    total_field_z = A * 4 * pi * total_dipole[:, 2] / volumes
    spread_total_field_z = torch.index_select(total_field_z, 0, batch)
    node_fields = torch.zeros(
        (node_positions.shape[0], 4),
        dtype=node_positions.dtype,
        device=node_positions.device,
    )
    node_fields[:, 0] = spread_total_field_z * node_positions[:, 2]
    node_fields[:, 3] = spread_total_field_z
    return node_fields


class JelliumSlabSolvatedFeatures(torch.nn.Module):
    """Electrostatic features with a compensating jellium slab (batch=1 only)."""

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: List[float],
        kspace_cutoff: float,
        slab_bounds: tuple[float, float],
        include_self_interaction: bool,
        integral_normalization: str = "receiver",
    ):
        super().__init__()
        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.feature_basis = GTOBasis(
            max_l=feature_max_l,
            sigmas=feature_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization,
        )
        self.smoothing_basis = GTOBasis(
            max_l=0,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.include_self_interaction = include_self_interaction
        slab_lower = float(slab_bounds[0])
        slab_upper = float(slab_bounds[1])
        if slab_upper <= slab_lower:
            raise ValueError("Jellium slab bounds are invalid (upper <= lower).")
        self.register_buffer("slab_lower", torch.tensor(slab_lower))
        self.register_buffer("slab_upper", torch.tensor(slab_upper))

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        )
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock(
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_receive=integral_normalization,
        )
        self.register_buffer(
            "output_permutation",
            self._build_output_permutation(
                max_l=feature_max_l, n_radial=len(feature_smearing_widths)
            ),
        )

        self.static_quantities = None

    def set_pbc_handling(self, pbc_handling: str) -> None:
        if pbc_handling != "slab":
            raise ValueError(
                "JelliumSlabSolvatedFeatures supports only pbc_handling='slab', "
                f"got {pbc_handling!r}."
            )

    @staticmethod
    def _build_output_permutation(max_l: int, n_radial: int) -> torch.Tensor:
        indices = []
        block = (max_l + 1) ** 2
        for l in range(max_l + 1):
            for c in range(n_radial):
                offset = c * block
                indices += range(l**2 + offset, (l + 1) ** 2 + offset)
        return torch.tensor(indices, dtype=torch.long)

    def _permute_output_channels(self, features_flat: torch.Tensor) -> torch.Tensor:
        return torch.index_select(features_flat, dim=-1, index=self.output_permutation)

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> torch.Tensor:
        self.static_quantities = self.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )
        return self.forward_dynamic(
            cache=self.static_quantities,
            source_feats=source_feats,
        )

    def precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        inner_products = torch.matmul(k_vectors, node_positions.t())
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        feature_basis_fs = self.feature_basis(k_vectors, k_norm2, k0_mask)
        smoothing_basis_fs = self.smoothing_basis(k_vectors, k_norm2, k0_mask)

        volume_per_k = volume.reshape(-1)[k_vector_batch]
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = compute_coulomb_factor(k_norm2, k0_mask)
        k_factor_proj = torch.ones_like(k_norm2)
        k_factor_proj[k0_mask_bool] = 0.5

        cosines_one = torch.ones(
            (k_vectors.size(0), 1), device=k_vectors.device, dtype=k_vectors.dtype
        )
        sines_one = torch.zeros_like(cosines_one)
        ones = torch.ones((1, 1), device=k_vectors.device, dtype=k_vectors.dtype)
        smoothing_density = assemble_fourier_series_batch(
            source_feats=ones,
            cosines=cosines_one,
            sines=sines_one,
            density_basis_fs=smoothing_basis_fs,
            volume_per_k=volume_per_k,
        )
        unit_slab_density = _slab_fourier_series_flat(
            total_charge=torch.ones((), device=k_vectors.device, dtype=k_vectors.dtype),
            lower=self.slab_lower,
            upper=self.slab_upper,
            k_vectors=k_vectors,
            k0_mask=k0_mask,
            volume=volume.reshape(-1),
        )
        smoothed_unit_slab_density = _smooth_slab_density(
            slab_density=unit_slab_density,
            smoothing_density=smoothing_density,
            volume_per_k=volume_per_k,
        )

        return {
            "k_vectors": k_vectors,
            "k_norm2": k_norm2,
            "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask,
            "volume_per_k": volume_per_k,
            "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj,
            "volumes": volume.reshape(-1),
            "batch": batch,
            "node_positions": node_positions,
            "pbc": pbc,
            "cosines": cosines,
            "sines": sines,
            "density_basis_fs": density_basis_fs,
            "feature_basis_fs": feature_basis_fs,
            "smoothing_density": smoothing_density,
            "unit_slab_density": unit_slab_density,
            "smoothed_unit_slab_density": smoothed_unit_slab_density,
        }

    def forward_dynamic(self, cache: dict, source_feats: torch.Tensor) -> torch.Tensor:
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )

        total_charge_atoms = source_feats[:, 0].sum()
        total_charge = -total_charge_atoms

        density = density + cache["smoothed_unit_slab_density"] * total_charge

        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        features_si = project_to_features_batch(
            potential=potential,
            feature_basis_fs=cache["feature_basis_fs"],
            cosines=cache["cosines"],
            sines=cache["sines"],
            k_factor_proj=cache["k_factor_proj"],
        )
        features_flat = features_si.reshape(features_si.size(0), -1)
        features_flat = self._permute_output_channels(features_flat)

        if not self.include_self_interaction:
            si_terms = self.self_interaction_terms(source_feats)
            features_flat = features_flat - si_terms

        total_charge_atoms = source_feats[:, 0].sum().unsqueeze(0)
        total_dipole_atoms = (
            cache["node_positions"] * source_feats[:, 0].unsqueeze(-1)
        ).sum(dim=0, keepdim=True)
        if source_feats.shape[-1] > 1:
            local_dipoles = source_feats[:, 1:4].sum(dim=0, keepdim=True)
            total_dipole_atoms = total_dipole_atoms + local_dipoles[:, [2, 0, 1]]
        z_center = 0.5 * (self.slab_lower + self.slab_upper)
        slab_com = torch.zeros(
            (1, 3), device=cache["node_positions"].device, dtype=cache["node_positions"].dtype
        )
        slab_com[0, 2] = z_center
        total_dipole = total_dipole_atoms - slab_com * total_charge_atoms.unsqueeze(-1)
        node_fields = _slab_dipole_correction_node_fields_from_total_dipole(
            total_dipole=total_dipole,
            volumes=cache["volumes"],
            node_positions=cache["node_positions"],
            batch=cache["batch"],
        )
        correction_terms = self.displaced_interactions(
            batch=cache["batch"],
            positions=cache["node_positions"],
            node_fields=node_fields,
        )
        features_flat = features_flat + correction_terms
        return features_flat
