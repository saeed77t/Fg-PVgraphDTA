"""Functional-group definitions reproduced from the uploaded notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from rdkit import Chem

FG_SMARTS = (
    ("CarboxylicAcid", "[CX3](=O)[OX2H1]"),
    ("Ester", "[CX3](=O)[OX2][#6]"),
    ("Amide", "[NX3][CX3](=O)[#6]"),
    ("Anhydride", "[CX3](=O)O[CX3](=O)"),
    ("AcylHalide", "[CX3](=O)[Cl,Br,I,F]"),
    ("Aldehyde", "[CX3H1](=O)[#6]"),
    ("Ketone", "[#6][CX3](=O)[#6]"),
    ("Alcohol", "[#6;!a][OX2H]"),
    ("Phenol", "c[OX2H]"),
    ("Ether", "[OX2]([#6])[#6]"),
    ("Nitrile", "[CX2]#N"),
    ("Nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("Amine_Primary", "[NX3;H2][#6]"),
    ("Amine_Secondary", "[NX3;H1]([#6])[#6]"),
    ("Amine_Tertiary", "[NX3]([#6])([#6])[#6]"),
    ("Thiol", "[#16X2H]"),
    ("Thioether", "[#16X2]([#6])[#6]"),
    ("Sulfoxide", "[#16X3](=O)([#6])[#6]"),
    ("Sulfone", "[#16X4](=O)(=O)([#6])[#6]"),
    ("Aryl", "c1ccccc1"),
)
FG_NAMES = tuple(n for n, _ in FG_SMARTS)
_COMPILED = tuple((n, Chem.MolFromSmarts(s)) for n, s in FG_SMARTS)
if any(p is None for _, p in _COMPILED):
    raise RuntimeError("SMARTS compile failure")
OverlapPolicy = Literal["notebook", "clean"]


def cleanup_group_names(group_names: Sequence[str]) -> list[str]:
    groups = set(group_names)
    if "Ester" in groups:
        groups.discard("Ether")
    if "CarboxylicAcid" in groups:
        groups.discard("Alcohol")
    if "Phenol" in groups:
        groups.discard("Alcohol")
    return sorted(groups)


def detect_functional_groups(mol: Chem.Mol) -> list[str]:
    return cleanup_group_names(
        [name for name, pattern in _COMPILED if mol.HasSubstructMatch(pattern)]
    )


def atom_functional_group_matrix(
    mol: Chem.Mol, overlap_policy: OverlapPolicy = "notebook"
) -> np.ndarray:
    matrix = np.zeros((mol.GetNumAtoms(), len(FG_NAMES)), dtype=np.float32)
    matched = {n: set() for n in FG_NAMES}
    for col, (name, pattern) in enumerate(_COMPILED):
        for match in mol.GetSubstructMatches(pattern):
            for idx in match:
                matrix[idx, col] = 1.0
                matched[name].add(idx)
    if overlap_policy == "notebook":
        return matrix
    if overlap_policy != "clean":
        raise ValueError(overlap_policy)
    name_to_col = {n: i for i, n in enumerate(FG_NAMES)}
    for specific, broad in (
        ("Ester", "Ether"),
        ("CarboxylicAcid", "Alcohol"),
        ("Phenol", "Alcohol"),
    ):
        for idx in matched[specific]:
            matrix[idx, name_to_col[broad]] = 0.0
    return matrix


AA_SMILES_NOTEBOOK = {
    "A": "NCC(C)C(=O)O",
    "R": "NCC(CCCNC(N)=N)C(=O)O",
    "N": "NCC(C(=O)N)C(=O)O",
    "D": "NCC(C(=O)O)C(=O)O",
    "C": "NCC(S)C(=O)O",
    "E": "NCC(CCC(=O)O)C(=O)O",
    "Q": "NCC(CCC(=O)N)C(=O)O",
    "G": "NCC(=O)O",
    "H": "NCC(Cc1c[nH]cn1)C(=O)O",
    "I": "NCC(C(C)CC)C(=O)O",
    "L": "NCC(CC(C)C)C(=O)O",
    "K": "NCC(CCCCN)C(=O)O",
    "M": "NCC(CCSC)C(=O)O",
    "F": "NCC(Cc1ccccc1)C(=O)O",
    "P": "N1CCC(C(=O)O)C1",
    "S": "NCC(CO)C(=O)O",
    "T": "NCC(C(O)C)C(=O)O",
    "W": "NCC(Cc1c2ccccc2[nH]c1)C(=O)O",
    "Y": "NCC(Cc1ccc(O)cc1)C(=O)O",
    "V": "NCC(C(C)C)C(=O)O",
}


def notebook_residue_fg_vector(residue: str) -> np.ndarray:
    smiles = AA_SMILES_NOTEBOOK.get(residue.upper())
    if smiles is None:
        return np.zeros(len(FG_NAMES), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    groups = set(detect_functional_groups(mol))
    return np.asarray([float(n in groups) for n in FG_NAMES], dtype=np.float32)
