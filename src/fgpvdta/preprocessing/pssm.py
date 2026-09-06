"""Alignment-derived residue profile calculation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

RESIDUE_ORDER = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
)
_PROFILE_INDEX = {r: i for i, r in enumerate(RESIDUE_ORDER)}
ProfileMode = Literal["notebook", "strict"]


def read_alignment_lines(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required MSA/alignment file does not exist: {path}")
    return [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compute_residue_profile(
    alignment_path: str | Path,
    protein_sequence: str,
    pseudocount: float = 0.8,
    mode: ProfileMode = "notebook",
) -> np.ndarray:
    seq = protein_sequence.strip().upper()
    lines = read_alignment_lines(alignment_path)
    if not seq or not lines:
        raise ValueError("Empty sequence/alignment")
    freq = np.zeros((21, len(seq)), dtype=np.float64)
    valid = 0
    for line in lines:
        if len(line) != len(seq):
            continue
        valid += 1
        for pos, res in enumerate(line):
            idx = _PROFILE_INDEX.get(res)
            if idx is not None:
                freq[idx, pos] += 1
    if valid == 0:
        raise ValueError("No alignment line matches target length")
    denom = len(lines) if mode == "notebook" else valid if mode == "strict" else None
    if denom is None:
        raise ValueError(mode)
    profile = ((freq + pseudocount / 21) / (denom + pseudocount)).T.astype(np.float32)
    if profile.shape != (len(seq), 21):
        raise AssertionError(profile.shape)
    return profile
