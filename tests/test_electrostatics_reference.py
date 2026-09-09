from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read
from scipy.constants import pi

from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat


REFERENCE_CONFIGS_PATH = Path(__file__).resolve().parent / "reference_data" / "reference_configs.xyz"
REFERENCE_VALUES_PATH = Path(__file__).resolve().parent / "reference_data" / "electrostatics_reference.npz"

CASE_TO_PBC_HANDLING = {
    "fff": "realspace",
    "fff_force_pbc": "molecule_in_box",
    "ttt": "pbc",
    "mixed": "mixed_periodic",
}


def _load_batched_inputs(density_max_l: int):
    atoms_list = read(REFERENCE_CONFIGS_PATH, ":")
    if len(atoms_list) != 4:
        raise ValueError(
            f"Expected 4 configs in {REFERENCE_CONFIGS_PATH}, found {len(atoms_list)}."
        )

    num_atoms_per_graph = [len(atoms) for atoms in atoms_list]
    if len(set(num_atoms_per_graph)) != 1:
        raise ValueError("All reference configs must contain the same number of atoms.")

    positions = torch.tensor(
        np.concatenate([atoms.positions for atoms in atoms_list], axis=0),
        dtype=torch.get_default_dtype(),
    )
    multipoles = torch.tensor(
        np.concatenate(
            [atoms.arrays["AIMS_atom_multipoles"][:, : (density_max_l + 1) ** 2] for atoms in atoms_list],
            axis=0,
        ),
        dtype=torch.get_default_dtype(),
    )
    source_feats = multipoles
    batch = torch.repeat_interleave(
        torch.arange(len(atoms_list), dtype=torch.long),
        torch.tensor(num_atoms_per_graph, dtype=torch.long),
    )

    cell = torch.tensor(
        np.stack([atoms.cell.array for atoms in atoms_list], axis=0),
        dtype=torch.get_default_dtype(),
    )
    r_cell = 2 * pi * torch.linalg.inv(cell).transpose(-1, -2)
    volume = torch.linalg.det(cell)

    return {
        "n_graphs": len(atoms_list),
        "n_atoms_per_graph": num_atoms_per_graph[0],
        "node_positions": positions,
        "source_feats": source_feats,
        "batch": batch,
        "cell": cell,
        "r_cell": r_cell,
        "volume": volume,
    }


@pytest.mark.parametrize(
    "case_name",
    ["fff", "fff_force_pbc", "ttt", "mixed"],
)
def test_electrostatics_matches_saved_reference(case_name):
    torch.set_default_dtype(torch.float64)

    if not REFERENCE_VALUES_PATH.exists():
        pytest.skip(
            "Reference file missing. Run scripts/generate_electrostatics_reference.py first."
        )

    reference = np.load(REFERENCE_VALUES_PATH, allow_pickle=False)

    density_max_l = int(reference["density_max_l"])
    feature_max_l = int(reference["feature_max_l"])
    density_smearing_width = float(reference["density_smearing_width"])
    feature_smearing_widths = reference["feature_smearing_widths"].tolist()
    kspace_cutoff = float(reference["kspace_cutoff"])
    include_self_interaction = bool(reference["include_self_interaction"])
    quadrupole_feature_corrections = bool(reference["quadrupole_feature_corrections"])
    pbc_handling = CASE_TO_PBC_HANDLING[case_name]

    common_inputs = _load_batched_inputs(density_max_l=density_max_l)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        cutoff=kspace_cutoff,
        cell_vectors=common_inputs["cell"],
        r_cell_vectors=common_inputs["r_cell"],
    )

    energy_block = GTOElectrostaticEnergy(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        kspace_cutoff=kspace_cutoff,
        include_self_interaction=include_self_interaction,
        pbc_handling=pbc_handling,
    )
    features_block = GTOElectrostaticFeatures(
        density_max_l=density_max_l,
        density_smearing_width=density_smearing_width,
        feature_max_l=feature_max_l,
        feature_smearing_widths=feature_smearing_widths,
        include_self_interaction=include_self_interaction,
        kspace_cutoff=kspace_cutoff,
        quadrupole_feature_corrections=quadrupole_feature_corrections,
        pbc_handling=pbc_handling,
    )

    pbc = torch.from_numpy(reference[f"pbc_{case_name}"]).to(dtype=torch.bool)

    with torch.no_grad():
        energy = energy_block(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=common_inputs["source_feats"],
            node_positions=common_inputs["node_positions"],
            batch=common_inputs["batch"],
            volume=common_inputs["volume"],
            pbc=pbc,
        )
        features = features_block(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            source_feats=common_inputs["source_feats"],
            node_positions=common_inputs["node_positions"],
            batch=common_inputs["batch"],
            volume=common_inputs["volume"],
            pbc=pbc,
        )

    expected_energy = torch.from_numpy(reference[f"energy_{case_name}"]).to(
        torch.get_default_dtype()
    )
    expected_features = torch.from_numpy(reference[f"features_{case_name}"]).to(
        torch.get_default_dtype()
    )
    features = features.reshape(
        common_inputs["n_graphs"], common_inputs["n_atoms_per_graph"], -1
    )

    torch.testing.assert_close(energy, expected_energy, rtol=1e-8, atol=1e-10)
    print(energy, expected_energy)
    torch.testing.assert_close(features, expected_features, rtol=1e-8, atol=1e-10)
