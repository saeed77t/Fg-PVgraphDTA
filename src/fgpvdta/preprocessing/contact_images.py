"""Historical contact-map image preprocessing used by PVgraphDTA."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image

from .contact_maps import load_contact_map
from .protein_graphs import find_contact_map

AA_LIST = tuple("ACDEFGHIKLMNPQRSTVWYX")
HISTORICAL_D_MODEL = 32


def sinusoidal_encoding(length: int, d_model: int = HISTORICAL_D_MODEL) -> np.ndarray:
    if length <= 0 or d_model <= 0 or d_model % 2:
        raise ValueError("invalid length/d_model")
    position = np.arange(length)[:, None].astype(np.float32)
    dimension = np.arange(d_model)[None, :].astype(np.float32)
    rates = 1.0 / np.power(10000, (2 * (dimension // 2)) / d_model)
    angles = position * rates
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles


def amino_acid_one_hot(sequence: str, aa_list: Iterable[str] = AA_LIST) -> np.ndarray:
    sequence = sequence.strip().upper()
    alphabet = tuple(aa_list)
    index = {a: i for i, a in enumerate(alphabet)}
    out = np.zeros((len(alphabet), len(sequence)), dtype=np.float32)
    if not sequence:
        raise ValueError("Protein sequence is empty")
    for pos, res in enumerate(sequence):
        if res not in index:
            raise ValueError(f"Residue {res!r} not in historical alphabet")
        out[index[res], pos] = 1.0
    return out


def historical_collapsed_feature_map(
    sequence: str, d_model: int = HISTORICAL_D_MODEL
) -> np.ndarray:
    sequence = sequence.strip().upper()
    L = len(sequence)
    pe1d = sinusoidal_encoding(L, d_model)
    one_hot = amino_acid_one_hot(sequence)
    expected = 2 * d_model + 2 * len(AA_LIST)
    # Algebraically identical to concatenating 106 LxL planes, with O(L^2) memory.
    totals = pe1d.sum(axis=1) + one_hot.sum(axis=0)
    return ((totals[:, None] + totals[None, :]) / expected).astype(np.float32)


def minmax_normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid array")
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    return ((values - lo) / (span + 1e-8)).astype(np.float32)


def build_historical_stacked_rgb(
    contact_map: np.ndarray, sequence: str, d_model: int = HISTORICAL_D_MODEL
) -> np.ndarray:
    sequence = sequence.strip().upper()
    L = len(sequence)
    cm = np.asarray(contact_map, dtype=np.float32)
    if cm.shape != (L, L):
        raise ValueError("Sequence/contact-map mismatch")
    red = minmax_normalize(cm)
    green = minmax_normalize(historical_collapsed_feature_map(sequence, d_model))
    blue = np.zeros_like(red, dtype=np.float32)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    if not np.all(rgb[..., 2] == 0):
        raise AssertionError("Blue must be zero")
    return rgb


def save_historical_stacked_png(
    contact_map: np.ndarray,
    sequence: str,
    output_path: str | Path,
    d_model: int = HISTORICAL_D_MODEL,
    write_metadata: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = build_historical_stacked_rgb(contact_map, sequence, d_model)
    plt.imsave(output_path, rgb, origin="lower")
    if write_metadata:
        meta = {
            "specification": "historical_onehot_pe",
            "source_notebook": "onehot+PE.ipynb",
            "sequence_length": len(sequence),
            "amino_acid_alphabet": "".join(AA_LIST),
            "d_model": d_model,
            "combined_feature_planes": 2 * d_model + 2 * len(AA_LIST),
            "feature_collapse": "mean(axis=0)",
            "red": "minmax(contact_map)",
            "green": "minmax(combined.mean(axis=0))",
            "blue": "exactly zero",
            "native_shape": [len(sequence), len(sequence), 3],
            "save_function": "matplotlib.pyplot.imsave",
            "origin": "lower",
            "matplotlib_version": matplotlib.__version__,
            "numpy_version": np.__version__,
        }
        output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_path


def historical_pv_resize_to_256(
    source_path: str | Path,
    destination_path: str | Path,
    size: int = 256,
    write_metadata: bool = True,
) -> Path:
    source_path, destination_path = Path(source_path), Path(destination_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        input_mode, image_size = image.mode, image.size
        resized = image.resize((size, size))
        resized.save(destination_path)
    if write_metadata:
        meta = {
            "operation": "historical_pv_first_resize",
            "source": str(source_path),
            "input_mode": input_mode,
            "input_size": list(image_size),
            "output_size": [size, size],
            "resample_argument": "omitted",
            "pillow_version": PIL.__version__,
        }
        destination_path.with_suffix(".resize.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return destination_path


def generate_historical_images(
    interactions_csv: str | Path,
    contact_map_dir: str | Path,
    native_output_dir: str | Path,
    resized_output_dir: str | Path | None = None,
    d_model: int = HISTORICAL_D_MODEL,
) -> dict[str, Path]:
    frame = pd.read_csv(interactions_csv)
    native = Path(native_output_dir)
    native.mkdir(parents=True, exist_ok=True)
    resized = Path(resized_output_dir) if resized_output_dir is not None else None
    if resized:
        resized.mkdir(parents=True, exist_ok=True)
    saved = {}
    for row in frame[["Target_ID", "Target"]].drop_duplicates("Target_ID").itertuples(index=False):
        tid, seq = str(row.Target_ID), str(row.Target).strip().upper()
        cm = load_contact_map(find_contact_map(contact_map_dir, tid), len(seq), "none")
        p = native / f"{tid}_stacked.png"
        save_historical_stacked_png(cm, seq, p, d_model)
        if resized:
            p2 = resized / f"{tid}_stacked.png"
            historical_pv_resize_to_256(p, p2)
            saved[tid] = p2
        else:
            saved[tid] = p
    return saved
