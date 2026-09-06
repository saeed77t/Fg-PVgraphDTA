"""PconsC4 contact-map loading, validation and edge construction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

SymmetrizeMode = Literal["none", "mean", "max"]
EdgeMode = Literal["threshold", "topk", "threshold_or_topk"]


def validate_contact_map(
    matrix: np.ndarray, sequence_length: int | None = None, symmetrize: SymmetrizeMode = "max"
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Contact map must be square, got {matrix.shape}")
    if matrix.size == 0 or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("Contact probabilities must be nonempty and in [0,1]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Contact map contains NaN/inf")
    if sequence_length is not None and matrix.shape != (sequence_length, sequence_length):
        raise ValueError("Sequence/contact-map mismatch")
    if symmetrize == "mean":
        matrix = 0.5 * (matrix + matrix.T)
    elif symmetrize == "max":
        matrix = np.maximum(matrix, matrix.T)
    elif symmetrize != "none":
        raise ValueError(symmetrize)
    matrix = matrix.copy()
    return matrix


def _load_sparse_contact_text(path: Path, sequence_length: int) -> np.ndarray:
    rec = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (
            not line
            or line.startswith(("#", ">"))
            or line.upper().startswith(
                ("PFRMAT", "TARGET", "AUTHOR", "REMARK", "METHOD", "MODEL", "END")
            )
        ):
            continue
        fields = line.replace(",", " ").split()
        if len(fields) < 3:
            continue
        try:
            rec.append((int(fields[0]), int(fields[1]), float(fields[-1])))
        except ValueError:
            continue
    if not rec:
        raise ValueError(f"No sparse contacts parsed from {path}")
    zero = any(i == 0 or j == 0 for i, j, _ in rec)
    off = 0 if zero else 1
    m = np.zeros((sequence_length, sequence_length), dtype=np.float32)
    for ir, jr, s in rec:
        i, j = ir - off, jr - off
        if not (0 <= i < sequence_length and 0 <= j < sequence_length):
            raise ValueError("Contact index outside sequence")
        m[i, j] = m[j, i] = max(m[i, j], m[j, i], s)
    return m


def load_contact_map(
    path: str | Path, sequence_length: int | None = None, symmetrize: SymmetrizeMode = "max"
) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        m = np.load(path)
    elif path.suffix.lower() == ".npz":
        a = np.load(path)
        keys = list(a.keys())
        if len(keys) != 1:
            raise ValueError("NPZ must contain one array")
        m = a[keys[0]]
    elif path.suffix.lower() == ".rr":
        if sequence_length is None:
            raise ValueError("RR contacts require sequence_length")
        m = _load_sparse_contact_text(path, sequence_length)
    else:
        try:
            dense = np.loadtxt(path, dtype=np.float32)
            if (
                dense.ndim == 2
                and dense.shape[0] == dense.shape[1]
                and (sequence_length is None or dense.shape[0] == sequence_length)
            ):
                m = dense
            else:
                if sequence_length is None:
                    raise ValueError("Sparse contact text requires sequence_length")
                m = _load_sparse_contact_text(path, sequence_length)
        except ValueError:
            if sequence_length is None:
                raise
            m = _load_sparse_contact_text(path, sequence_length)
    return validate_contact_map(m, sequence_length, symmetrize)


def contact_edge_index(
    contact_map: np.ndarray,
    mode: EdgeMode = "threshold",
    threshold: float = 0.5,
    top_k: int | None = None,
) -> torch.Tensor:
    m = validate_contact_map(contact_map, symmetrize="none")
    n = m.shape[0]
    selected = set()
    if mode in {"threshold", "threshold_or_topk"}:
        rows, cols = np.where(np.triu(m >= threshold, k=1))
        selected.update(zip(rows.tolist(), cols.tolist(), strict=True))
    if mode in {"topk", "threshold_or_topk"}:
        if not top_k or top_k <= 0:
            raise ValueError("top_k must be positive")
        for i in range(n):
            row = m[i].copy()
            row[i] = -np.inf
            k = min(top_k, max(0, n - 1))
            if k:
                for j in np.argpartition(row, -k)[-k:].tolist():
                    if np.isfinite(row[j]):
                        selected.add((min(i, j), max(i, j)))
    if mode not in {"threshold", "topk", "threshold_or_topk"}:
        raise ValueError(mode)
    directed = []
    for i, j in sorted(selected):
        if i != j:
            directed.extend([(i, j), (j, i)])
    return (
        torch.empty((2, 0), dtype=torch.long)
        if not directed
        else torch.tensor(directed, dtype=torch.long).T.contiguous()
    )
