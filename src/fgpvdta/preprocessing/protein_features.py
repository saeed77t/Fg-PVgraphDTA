"""Protein residue features reproduced from the uploaded graph notebook."""

from __future__ import annotations

import numpy as np

from .functional_groups import notebook_residue_fg_vector
from .pssm import RESIDUE_ORDER

RES_WEIGHT = {
    "A": 71.08,
    "C": 103.15,
    "D": 115.09,
    "E": 129.12,
    "F": 147.18,
    "G": 57.05,
    "H": 137.14,
    "I": 113.16,
    "K": 128.18,
    "L": 113.16,
    "M": 131.20,
    "N": 114.11,
    "P": 97.12,
    "Q": 128.13,
    "R": 156.19,
    "S": 87.08,
    "T": 101.11,
    "V": 99.13,
    "W": 186.22,
    "Y": 163.18,
    "X": 0.0,
}
RES_PKA = {
    "A": 2.34,
    "C": 1.96,
    "D": 1.88,
    "E": 2.19,
    "F": 1.83,
    "G": 2.34,
    "H": 1.82,
    "I": 2.36,
    "K": 2.18,
    "L": 2.36,
    "M": 2.28,
    "N": 2.02,
    "P": 1.99,
    "Q": 2.17,
    "R": 2.17,
    "S": 2.21,
    "T": 2.09,
    "V": 2.32,
    "W": 2.83,
    "Y": 2.32,
}
RES_PKB = {
    "A": 9.69,
    "C": 10.28,
    "D": 9.60,
    "E": 9.67,
    "F": 9.13,
    "G": 9.60,
    "H": 9.17,
    "I": 9.60,
    "K": 8.95,
    "L": 9.60,
    "M": 9.21,
    "N": 8.80,
    "P": 10.60,
    "Q": 9.13,
    "R": 9.04,
    "S": 9.15,
    "T": 9.10,
    "V": 9.62,
    "W": 9.39,
    "Y": 9.62,
}
RES_PKX = {
    "A": 0.00,
    "C": 8.18,
    "D": 3.65,
    "E": 4.25,
    "F": 0.00,
    "G": 0.00,
    "H": 6.00,
    "I": 0.00,
    "K": 10.53,
    "L": 0.00,
    "M": 0.00,
    "N": 0.00,
    "P": 0.00,
    "Q": 0.00,
    "R": 12.48,
    "S": 0.00,
    "T": 0.00,
    "V": 0.00,
    "W": 0.00,
    "Y": 0.00,
}
RES_PI = {
    "A": 6.00,
    "C": 5.07,
    "D": 2.77,
    "E": 3.22,
    "F": 5.48,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "K": 9.74,
    "L": 5.98,
    "M": 5.74,
    "N": 5.41,
    "P": 6.30,
    "Q": 5.65,
    "R": 10.76,
    "S": 5.68,
    "T": 5.60,
    "V": 5.96,
    "W": 5.89,
    "Y": 5.96,
}
RES_HYDROPHOBIC_PH2 = {
    "A": 47,
    "C": 52,
    "D": -18,
    "E": 8,
    "F": 92,
    "G": 0,
    "H": -42,
    "I": 100,
    "K": -37,
    "L": 100,
    "M": 74,
    "N": -41,
    "P": -46,
    "Q": -18,
    "R": -26,
    "S": -7,
    "T": 13,
    "V": 79,
    "W": 84,
    "Y": 49,
}
RES_HYDROPHOBIC_PH7 = {
    "A": 41,
    "C": 49,
    "D": -55,
    "E": -31,
    "F": 100,
    "G": 0,
    "H": 8,
    "I": 99,
    "K": -23,
    "L": 97,
    "M": 74,
    "N": -28,
    "P": -46,
    "Q": -10,
    "R": -14,
    "S": -5,
    "T": 13,
    "V": 76,
    "W": 97,
    "Y": 63,
}
ALIPHATIC = set("AILMV")
AROMATIC = set("FWY")
POLAR_NEUTRAL = set("CNQST")
ACIDIC_CHARGED = set("DE")
BASIC_CHARGED = set("HKR")


def _prop(table, res):
    return float(table.get(res, 0.0))


def base_residue_feature(residue: str, profile_vector: np.ndarray) -> np.ndarray:
    residue = residue.upper()
    p = np.asarray(profile_vector, dtype=np.float32)
    if p.shape != (21,):
        raise ValueError(p.shape)
    one = np.asarray([float(residue == r) for r in RESIDUE_ORDER], dtype=np.float32)
    flags = np.asarray(
        [
            float(residue in ALIPHATIC),
            float(residue in AROMATIC),
            float(residue in POLAR_NEUTRAL),
            float(residue in ACIDIC_CHARGED),
            float(residue in BASIC_CHARGED),
        ],
        dtype=np.float32,
    )
    props = np.asarray(
        [
            _prop(RES_WEIGHT, residue),
            _prop(RES_PKA, residue),
            _prop(RES_PKB, residue),
            _prop(RES_PKX, residue),
            _prop(RES_PI, residue),
            _prop(RES_HYDROPHOBIC_PH2, residue),
            _prop(RES_HYDROPHOBIC_PH7, residue),
        ],
        dtype=np.float32,
    )
    feat = np.concatenate([one, p, flags, props])
    if feat.shape != (54,):
        raise AssertionError(feat.shape)
    return feat


def residue_feature(
    residue: str, profile_vector: np.ndarray, include_functional_groups: bool = False
) -> np.ndarray:
    base = base_residue_feature(residue, profile_vector)
    if not include_functional_groups:
        return base
    feat = np.concatenate([base, notebook_residue_fg_vector(residue)])
    if feat.shape != (74,):
        raise AssertionError(feat.shape)
    return feat
