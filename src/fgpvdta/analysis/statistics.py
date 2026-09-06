from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from fgpvdta.training.metrics import regression_metrics

LOWER_IS_BETTER = {"mse"}
HIGHER_IS_BETTER = {"ci", "pearson", "spearman"}


@dataclass(frozen=True)
class PairedSeedStatistics:
    experiment: str
    baseline: str
    proposed: str
    metric: str
    n_pairs: int
    baseline_mean: float
    proposed_mean: float
    mean_improvement: float
    t_statistic: float
    p_value_two_sided: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_samples: int
    bootstrap_seed: int

    def to_dict(self):
        return asdict(self)


def _improvement(b, p, metric):
    return (
        np.asarray(b) - np.asarray(p)
        if metric in LOWER_IS_BETTER
        else np.asarray(p) - np.asarray(b)
    )


def paired_seed_test(
    runs,
    experiment,
    baseline_variant,
    proposed_variant,
    metric,
    expected_seeds: Sequence[int] | None = None,
    bootstrap_samples: int = 50000,
    bootstrap_seed: int = 42,
    confidence: float = 0.95,
):
    subset = runs[runs.experiment == experiment]
    b = subset[subset.variant == baseline_variant][["seed", "split_sha256", metric]]
    p = subset[subset.variant == proposed_variant][["seed", "split_sha256", metric]]
    merged = b.merge(
        p, on="seed", suffixes=("_baseline", "_proposed"), validate="one_to_one"
    ).sort_values("seed")
    if len(merged) != len(b) or len(merged) != len(p):
        raise ValueError("Unmatched seeds")
    if expected_seeds is not None and tuple(merged.seed.astype(int)) != tuple(
        sorted(map(int, expected_seeds))
    ):
        raise ValueError("Unexpected seed set")
    if (
        set(merged.split_sha256_baseline) != set(merged.split_sha256_proposed)
        or len(set(merged.split_sha256_baseline)) != 1
    ):
        raise ValueError("Split mismatch")
    bv = merged[f"{metric}_baseline"].to_numpy(float)
    pv = merged[f"{metric}_proposed"].to_numpy(float)
    t = ttest_rel(pv, bv)
    diff = _improvement(bv, pv, metric).astype(float)
    rng = np.random.default_rng(bootstrap_seed)
    boot = diff[rng.integers(0, len(diff), size=(bootstrap_samples, len(diff)))].mean(axis=1)
    a = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return PairedSeedStatistics(
        experiment,
        baseline_variant,
        proposed_variant,
        metric,
        len(merged),
        float(bv.mean()),
        float(pv.mean()),
        float(diff.mean()),
        float(t.statistic),
        float(t.pvalue),
        float(lo),
        float(hi),
        bootstrap_samples,
        bootstrap_seed,
    )


def hierarchical_paired_bootstrap(
    runs,
    experiment,
    baseline_variant,
    proposed_variant,
    metric,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    confidence: float = 0.95,
):
    subset = runs[runs.experiment == experiment]
    b = {int(r.seed): r for r in subset[subset.variant == baseline_variant].itertuples()}
    p = {int(r.seed): r for r in subset[subset.variant == proposed_variant].itertuples()}
    if set(b) != set(p):
        raise ValueError("Unmatched seeds")
    paired = {}
    for seed in b:
        bf = pd.read_csv(b[seed].predictions_csv).sort_values("row_index").reset_index(drop=True)
        pf = pd.read_csv(p[seed].predictions_csv).sort_values("row_index").reset_index(drop=True)
        if not np.array_equal(bf.row_index, pf.row_index) or not np.allclose(bf.y_true, pf.y_true):
            raise ValueError("Prediction rows/labels mismatch")
        paired[seed] = (bf, pf)
    rng = np.random.default_rng(bootstrap_seed)
    seeds = np.array(sorted(paired))
    effects = []
    for _ in range(bootstrap_samples):
        current = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            bf, pf = paired[int(seed)]
            idx = rng.integers(0, len(bf), size=len(bf))
            bm = regression_metrics(bf.y_true.to_numpy()[idx], bf.y_pred.to_numpy()[idx])
            pm = regression_metrics(pf.y_true.to_numpy()[idx], pf.y_pred.to_numpy()[idx])
            current.append(float(_improvement(getattr(bm, metric), getattr(pm, metric), metric)))
        effects.append(np.mean(current))
    effects = np.asarray(effects)
    a = (1 - confidence) / 2
    lo, hi = np.quantile(effects, [a, 1 - a])
    return {
        "experiment": experiment,
        "baseline": baseline_variant,
        "proposed": proposed_variant,
        "metric": metric,
        "n_seeds": len(seeds),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "mean_bootstrap_improvement": float(effects.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
