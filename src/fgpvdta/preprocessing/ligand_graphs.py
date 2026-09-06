"""Ligand graph construction compatible with the supplied notebooks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

from .functional_groups import FG_NAMES, atom_functional_group_matrix, detect_functional_groups
from .smiles import StandardizationMode, standardize_smiles

ATOM_TYPES = (
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "X",
)


def _oh(v: int, size: int):
    return [float(v == i) for i in range(size)]


def notebook_atom_feature(atom: Chem.Atom) -> np.ndarray:
    feature = np.asarray(
        [float(atom.GetSymbol() == s) for s in ATOM_TYPES]
        + _oh(int(atom.GetDegree()), 11)
        + _oh(int(atom.GetTotalNumHs()), 11)
        + _oh(int(atom.GetImplicitValence()), 11)
        + [float(atom.GetIsAromatic())],
        dtype=np.float32,
    )
    if feature.shape != (78,):
        raise AssertionError(feature.shape)
    return feature


def _directed_bond_edge_index(mol: Chem.Mol) -> torch.Tensor:
    edges = []
    for bond in mol.GetBonds():
        a, b = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        edges.extend([(a, b), (b, a)])
    return (
        torch.empty((2, 0), dtype=torch.long)
        if not edges
        else torch.tensor(edges, dtype=torch.long).T.contiguous()
    )


def build_ligand_graph(
    smiles: str,
    include_functional_groups: bool = False,
    standardization_mode: StandardizationMode = "notebook",
    fg_overlap_policy: Literal["notebook", "clean"] = "notebook",
    drug_id: str | None = None,
) -> Data:
    st = standardize_smiles(smiles, standardization_mode)
    mol = st.mol
    base = np.stack([notebook_atom_feature(a) for a in mol.GetAtoms()])
    features = (
        np.concatenate([base, atom_functional_group_matrix(mol, fg_overlap_policy)], axis=1)
        if include_functional_groups
        else base
    )
    expected = 98 if include_functional_groups else 78
    if features.shape[1] != expected:
        raise AssertionError(features.shape)
    graph = Data(
        x=torch.as_tensor(features, dtype=torch.float32), edge_index=_directed_bond_edge_index(mol)
    )
    graph.num_nodes = mol.GetNumAtoms()
    graph.drug_id = "" if drug_id is None else str(drug_id)
    graph.input_smiles = st.input_smiles
    graph.standardized_smiles = st.standardized_smiles
    graph.standardization_mode = standardization_mode
    graph.functional_groups = detect_functional_groups(mol)
    graph.fg_names = list(FG_NAMES)
    graph.fg_overlap_policy = fg_overlap_policy
    graph.feature_dim = expected
    return graph


def save_ligand_graph(
    graph: Data, path: str | Path, serialization: Literal["pt", "notebook_pickle"] = "pt"
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if serialization == "pt":
        torch.save(graph, path)
        return
    if serialization == "notebook_pickle":
        with path.open("wb") as h:
            pickle.dump(
                {
                    "num_atoms": int(graph.num_nodes),
                    "features": graph.x.cpu().numpy(),
                    "edge_index": graph.edge_index.cpu().numpy().T,
                    "smiles": graph.input_smiles,
                    "fg_names": list(getattr(graph, "fg_names", [])),
                    "ligand_fgs": list(getattr(graph, "functional_groups", [])),
                },
                h,
            )
            return
    raise ValueError(serialization)


def build_ligand_graph_directory(
    interactions_csv: str | Path,
    output_dir: str | Path,
    include_functional_groups: bool,
    standardization_mode: StandardizationMode = "notebook",
    fg_overlap_policy: Literal["notebook", "clean"] = "notebook",
) -> dict[str, Path]:
    frame = pd.read_csv(interactions_csv)
    required = {"Drug_ID", "Drug"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(missing)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for row in frame[["Drug_ID", "Drug"]].drop_duplicates("Drug_ID").itertuples(index=False):
        graph = build_ligand_graph(
            str(row.Drug),
            include_functional_groups,
            standardization_mode,
            fg_overlap_policy,
            str(row.Drug_ID),
        )
        path = out / f"{row.Drug_ID}_graph.pt"
        torch.save(graph, path)
        saved[str(row.Drug_ID)] = path
    return saved
