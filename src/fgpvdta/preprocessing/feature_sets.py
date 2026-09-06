"""Explicit reviewer feature sets and deterministic random-feature controls.

K20 is historical. K10 selects its first ten entries. K30 appends ten explicitly
new motifs; it is a proposed reviewer extension, not recovered historical code.
"""

import hashlib

import numpy as np
from rdkit import Chem

from .functional_groups import (
    AA_SMILES_NOTEBOOK,
    FG_SMARTS,
    atom_functional_group_matrix,
    notebook_residue_fg_vector,
)

REVIEWER_EXTRA_SMARTS = (
    ("OrganicHalide", "[#6][F,Cl,Br,I]"),
    ("Alkene", "[CX3]=[CX3]"),
    ("Alkyne", "[CX2]#[CX2]"),
    ("Imine", "[CX3]=[NX2]"),
    ("Disulfide", "[SX2][SX2]"),
    ("Sulfonamide", "[SX4](=O)(=O)[NX3]"),
    ("SulfonicAcid", "[SX4](=O)(=O)[OX2H]"),
    ("Phosphate", "[PX4](=O)([OX2])([OX2])[OX2]"),
    ("Phosphonate", "[PX4](=O)([#6])([OX2])[OX2]"),
    ("AromaticHeteroatom", "[a;!#6]"),
)
EXTRA_PATTERNS = tuple(Chem.MolFromSmarts(s) for _, s in REVIEWER_EXTRA_SMARTS)
if any(pattern is None for pattern in EXTRA_PATTERNS):
    raise RuntimeError("Invalid reviewer SMARTS")


def feature_spec(count):
    if count not in (0, 10, 20, 30):
        raise ValueError("Supported FG counts: 0, 10, 20, 30")
    return (FG_SMARTS + REVIEWER_EXTRA_SMARTS)[:count]


def ligand_features(mol, count):
    feature_spec(count)
    base = atom_functional_group_matrix(mol)
    if count <= 20:
        return base[:, :count]
    extra = np.zeros((mol.GetNumAtoms(), 10), dtype=np.float32)
    for col, pattern in enumerate(EXTRA_PATTERNS):
        for match in mol.GetSubstructMatches(pattern):
            extra[list(match), col] = 1
    return np.concatenate((base, extra), axis=1)


def protein_features(sequence, count):
    feature_spec(count)
    rows = []
    for residue in sequence:
        base = notebook_residue_fg_vector(residue)
        if count <= 20:
            rows.append(base[:count])
        else:
            smiles = AA_SMILES_NOTEBOOK.get(residue)
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            extra = [float(mol.HasSubstructMatch(p)) if mol else 0.0 for p in EXTRA_PATTERNS]
            rows.append(np.concatenate((base, extra)))
    return np.asarray(rows, dtype=np.float32)


def random_control(features, entity_id, seed):
    """Shuffle each node's FG bits; preserve its active-bit count, break column meaning.

    The hash makes features independent of iteration order, process hash seeds,
    and labels. This is a row-sparsity-matched control, not a fitted feature model.
    """
    digest = hashlib.sha256(f"{seed}:{entity_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.stack([rng.permutation(row) for row in features]).astype(np.float32)
