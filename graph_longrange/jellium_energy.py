
import torch
from scipy.constants import pi

from .energy import energy_product_batch
from .features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
    compute_coulomb_factor,
)
from .gto_utils import GTOBasis, GTOSelfInteractionBlock
from .slabs import get_nonperiodic_charge_dipole
from .utils import FIELD_CONSTANT


def _slab_fourier_series_flat(
    total_charge: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    k_vectors: torch.Tensor,
    k0_mask: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """Fourier series for a sharp-edged slab in a batch=1 cell."""
    width = upper - lower
    A = total_charge / volume[0]

    width_vector = torch.zeros(3, device=k_vectors.device, dtype=k_vectors.dtype)
    width_vector[2] = width
    translation_vector = torch.zeros(3, device=k_vectors.device, dtype=k_vectors.dtype)
    translation_vector[2] = 0.5 * (lower + upper)

    tol = 1e-8
    on_line = (k_vectors[:, 0].abs() < tol) & (k_vectors[:, 1].abs() < tol)
    k_v_norms = torch.einsum("j,nj->n", width_vector, k_vectors)
    k_v_norms = torch.where(on_line, k_v_norms, torch.ones_like(k_v_norms))
    k_v_norms = torch.where(k0_mask > 0.0, torch.ones_like(k_v_norms), k_v_norms)

    phase_width = torch.einsum("j,nj->n", width_vector, k_vectors)
    factors = 2.0 * A * torch.sin(0.5 * phase_width) / k_v_norms
    factors = torch.where(on_line, factors, torch.zeros_like(factors))
    factors = torch.where(k0_mask > 0.0, A, factors)

    phase = torch.einsum("j,nj->n", translation_vector, k_vectors)
    real_coefficients = factors * torch.cos(phase)
    imag_coefficients = -factors * torch.sin(phase)
    coefficients = torch.stack([real_coefficients, imag_coefficients], dim=-1)
    return (2 * pi) ** 3 * coefficients


def _smooth_slab_density(
    slab_density: torch.Tensor,
    smoothing_density: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    real = slab_density[..., 0] * smoothing_density[..., 0] - (
        slab_density[..., 1] * smoothing_density[..., 1]
    )
    imag = slab_density[..., 0] * smoothing_density[..., 1] + (
        slab_density[..., 1] * smoothing_density[..., 0]
    )
    product = torch.stack([real, imag], dim=-1)
    return product * volume_per_k.unsqueeze(-1) / (2 * pi) ** 3


def _slab_dipole_correction_from_total_dipole(
    total_dipole: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    dipole_norms_squared = total_dipole[:, 2] ** 2
    A = FIELD_CONSTANT / (4 * pi)
    return A * 2 * pi * dipole_norms_squared / volume.reshape(-1)


class JelliumSlabSolvatedEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        kspace_cutoff: float,
        slab_bounds: tuple[float, float],
        include_self_interaction: bool = False,
    ):
        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.kspace_cutoff = kspace_cutoff
        self.include_self_interaction = include_self_interaction
        slab_lower = float(slab_bounds[0])
        slab_upper = float(slab_bounds[1])
        if slab_upper <= slab_lower:
            raise ValueError("Jellium slab bounds are invalid (upper <= lower).")
        self.register_buffer("slab_lower", torch.tensor(slab_lower))
        self.register_buffer("slab_upper", torch.tensor(slab_upper))

        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.jellium_smoothing_basis = GTOBasis(
            max_l=0,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=density_max_l,
            sigmas_receive=[density_smearing_width],
            normalize_source="multipoles",
            normalize_receive="multipoles",
        )

    def set_pbc_handling(self, pbc_handling: str) -> None:
        if pbc_handling != "slab":
            raise ValueError(
                "JelliumSlabSolvatedEnergy supports only pbc_handling='slab', "
                f"got {pbc_handling!r}."
            )

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
        inner_products = torch.matmul(k_vectors, node_positions.t())  # [n_k_total, n_nodes]
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        volume_per_k = volume.reshape(-1)[k_vector_batch]
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cosines,
            sines=sines,
            density_basis_fs=density_basis_fs,
            volume_per_k=volume_per_k,
        )

        total_charge_atoms = source_feats[:, 0].sum()
        total_charge = -total_charge_atoms

        slab_density = _slab_fourier_series_flat(
            total_charge=total_charge,
            lower=self.slab_lower,
            upper=self.slab_upper,
            k_vectors=k_vectors,
            k0_mask=k0_mask,
            volume=volume,
        )

        gaussian_basis_fs = self.jellium_smoothing_basis(k_vectors, k_norm2, k0_mask)
        ones = torch.ones((1, 1), device=k_vectors.device, dtype=k_vectors.dtype)
        cosines_one = torch.ones(
            (k_vectors.size(0), 1), device=k_vectors.device, dtype=k_vectors.dtype
        )
        sines_one = torch.zeros_like(cosines_one)
        smoothing_density = assemble_fourier_series_batch(
            source_feats=ones,
            cosines=cosines_one,
            sines=sines_one,
            density_basis_fs=gaussian_basis_fs,
            volume_per_k=volume_per_k,
        )
        density = density + _smooth_slab_density(
            slab_density=slab_density,
            smoothing_density=smoothing_density,
            volume_per_k=volume_per_k,
        )

        k_factor_coulomb = compute_coulomb_factor(k_norm2, k0_mask)
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=k_factor_coulomb,
        )
        energy = energy_product_batch(
            density=density,
            potential=potential,
            volume=volume,
            k_vector_batch=k_vector_batch,
        )

        if not self.include_self_interaction:
            self_fields = self.self_interaction_terms(source_feats)
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = torch.zeros(
                int(volume.shape[0]),
                dtype=node_energies.dtype,
                device=node_energies.device,
            )
            self_energy.index_add_(0, batch, node_energies)
            energy = energy - self_energy * 0.5

        
        total_charge_atoms, total_dipole_atoms = get_nonperiodic_charge_dipole(
            source_feats,
            node_positions,
            batch,
        )
        z_center = 0.5 * (self.slab_lower + self.slab_upper)
        slab_com = torch.zeros(
            (1, 3), device=node_positions.device, dtype=node_positions.dtype
        )
        slab_com[0, 2] = z_center
        total_dipole = total_dipole_atoms - slab_com * total_charge_atoms.unsqueeze(
            -1
        )
        slab_correction = _slab_dipole_correction_from_total_dipole(
            total_dipole=total_dipole,
            volume=volume,
        )
        energy = energy + slab_correction

        return energy
