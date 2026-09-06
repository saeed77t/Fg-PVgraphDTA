"""SMILES validation and explicitly selectable standardization policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

StandardizationMode = Literal["notebook", "canonical", "fragment_parent"]


@dataclass(frozen=True)
class StandardizedMolecule:
    input_smiles: str
    standardized_smiles: str
    mol: Chem.Mol
    mode: StandardizationMode


def standardize_smiles(smiles: str, mode: StandardizationMode = "notebook") -> StandardizedMolecule:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    input_smiles = smiles.strip()
    mol = Chem.MolFromSmiles(input_smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {input_smiles!r}")
    if mode == "notebook":
        output_smiles = input_smiles
        output_mol = mol
    elif mode == "canonical":
        output_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        output_mol = Chem.MolFromSmiles(output_smiles)
    elif mode == "fragment_parent":
        parent = rdMolStandardize.FragmentParent(mol)
        output_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        output_mol = Chem.MolFromSmiles(output_smiles)
    else:
        raise ValueError(f"Unknown SMILES standardization mode: {mode}")
    if output_mol is None:
        raise RuntimeError("Standardized SMILES could not be parsed back")
    return StandardizedMolecule(input_smiles, output_smiles, output_mol, mode)
