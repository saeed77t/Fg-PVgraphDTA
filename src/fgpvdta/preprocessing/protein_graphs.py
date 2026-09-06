"""Protein residue-graph construction from sequence, alignment and contact map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .contact_maps import contact_edge_index, load_contact_map
from .protein_features import residue_feature
from .pssm import ProfileMode, compute_residue_profile

CONTACT_SUFFIXES = (".npy", ".npz", ".txt", ".rr")


def find_contact_map(contact_dir: str | Path, target_id: str) -> Path:
    root = Path(contact_dir)
    matches = [
        root / f"{target_id}{s}" for s in CONTACT_SUFFIXES if (root / f"{target_id}{s}").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No PconsC4/contact-map file found for {target_id}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple contact maps found for {target_id}: {matches}")
    return matches[0]


def build_protein_graph(
    target_id: str,
    sequence: str,
    alignment_path: str | Path,
    contact_map_path: str | Path,
    include_functional_groups: bool = False,
    profile_mode: ProfileMode = "notebook",
    contact_edge_mode: str = "threshold",
    contact_threshold: float = 0.5,
    contact_top_k: int | None = None,
    symmetrize_contacts: str = "max",
) -> Data:
    sequence = sequence.strip().upper()
    profile = compute_residue_profile(alignment_path, sequence, mode=profile_mode)
    cm = load_contact_map(contact_map_path, len(sequence), symmetrize_contacts)
    features = np.stack(
        [residue_feature(r, profile[i], include_functional_groups) for i, r in enumerate(sequence)]
    )
    expected = 74 if include_functional_groups else 54
    if features.shape != (len(sequence), expected):
        raise AssertionError(features.shape)
    graph = Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=contact_edge_index(cm, contact_edge_mode, contact_threshold, contact_top_k),
    )
    graph.num_nodes = len(sequence)
    graph.target_id = str(target_id)
    graph.sequence = sequence
    graph.feature_dim = expected
    graph.contact_threshold = float(contact_threshold)
    graph.contact_map_source = str(contact_map_path)
    return graph


def build_protein_graph_directory(
    interactions_csv: str | Path,
    alignment_dir: str | Path,
    contact_map_dir: str | Path,
    output_dir: str | Path,
    include_functional_groups: bool = False,
    profile_mode: ProfileMode = "notebook",
    contact_edge_mode: str = "threshold",
    contact_threshold: float = 0.5,
    contact_top_k: int | None = None,
    symmetrize_contacts: str = "max",
) -> dict[str, Path]:
    frame = pd.read_csv(interactions_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for row in frame[["Target_ID", "Target"]].drop_duplicates("Target_ID").itertuples(index=False):
        tid = str(row.Target_ID)
        aln = Path(alignment_dir) / f"{tid}.aln"
        if not aln.is_file():
            raise FileNotFoundError(f"Missing alignment for target {tid}: {aln}")
        cm = find_contact_map(contact_map_dir, tid)
        graph = build_protein_graph(
            tid,
            str(row.Target),
            aln,
            cm,
            include_functional_groups,
            profile_mode,
            contact_edge_mode,
            contact_threshold,
            contact_top_k,
            symmetrize_contacts,
        )
        path = out / f"{tid}_graph.pt"
        torch.save(graph, path)
        saved[tid] = path
    return saved
