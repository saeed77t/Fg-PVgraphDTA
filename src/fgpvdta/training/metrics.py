from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    ci: float
    pearson: float
    spearman: float
    n: int

    def to_dict(self):
        return asdict(self)


class FenwickTree:
    def __init__(self, size):
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index, value=1):
        index += 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index):
        if index < 0:
            return 0
        index += 1
        result = 0
        while index > 0:
            result += int(self.tree[index])
            index -= index & -index
        return result


def concordance_index(targets, predictions):
    y, p = _paired(targets, predictions)
    if len(y) < 2:
        return float("nan")
    vals = np.unique(p)
    rank = {v: i for i, v in enumerate(vals)}
    order = np.argsort(y, kind="mergesort")
    y, p = y[order], p[order]
    tree = FenwickTree(len(vals))
    prior = 0
    comparable = 0
    concordant = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and y[end] == y[start]:
            end += 1
        for pos in range(start, end):
            r = rank[p[pos]]
            low = tree.prefix_sum(r - 1)
            eq = tree.prefix_sum(r) - low
            comparable += prior
            concordant += low + 0.5 * eq
        for pos in range(start, end):
            tree.add(rank[p[pos]])
        prior += end - start
        start = end
    return float(concordant / comparable) if comparable else float("nan")


def regression_metrics(targets, predictions):
    y, p = _paired(targets, predictions)
    mse = float(np.mean((y - p) ** 2))
    pear = (
        float(pearsonr(y, p).statistic)
        if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0
        else float("nan")
    )
    spear = (
        float(spearmanr(y, p).statistic)
        if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0
        else float("nan")
    )
    return RegressionMetrics(mse, concordance_index(y, p), pear, spear, len(y))


def _paired(targets, predictions):
    y = np.asarray(targets, dtype=float).reshape(-1)
    p = np.asarray(predictions, dtype=float).reshape(-1)
    if not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require equal-length, nonempty finite arrays")
    return y, p
