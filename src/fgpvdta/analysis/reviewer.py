"""Paired seed tests and target-level inference with explicitly matched predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from fgpvdta.data.schema import write_json

from .aggregate import aggregate_observed_runs, discover_observed_runs


def paired_effects(differences, samples=10000, seed=42):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Statistical inference requires >=2 finite paired units")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    if np.all(values == 0):
        statistic, p_value = 0.0, 1.0
    elif np.ptp(values) == 0:
        statistic, p_value = float("nan"), 0.0
    else:
        result = ttest_1samp(values, 0.0)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    rng = np.random.default_rng(seed)
    # Chunking avoids an unbounded bootstrap_samples x targets allocation.
    boot = np.concatenate(
        [
            values[rng.integers(0, len(values), (min(256, samples - i), len(values)))].mean(axis=1)
            for i in range(0, samples, 256)
        ]
    )
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_pairs": len(values),
        "mean_improvement": float(values.mean()),
        "t_statistic": statistic,
        "p_two_sided": p_value,
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
    }


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_targets(baseline, candidate):
    """Average target MSE across matched seeds, then treat each target as one unit."""
    if set(baseline.seed) != set(candidate.seed) or baseline.empty:
        raise ValueError("Unmatched seed sets")
    effects = []
    expected_targets = None
    for seed in sorted(baseline.seed):
        base_record = baseline[baseline.seed == seed].iloc[0]
        cand_record = candidate[candidate.seed == seed].iloc[0]
        if base_record.split_sha256 != cand_record.split_sha256:
            raise ValueError("Unmatched splits")
        a = pd.read_csv(base_record.predictions_csv, dtype={"Target_ID": str, "Drug_ID": str})
        b = pd.read_csv(cand_record.predictions_csv, dtype={"Target_ID": str, "Drug_ID": str})
        a, b = a.sort_values("row_index"), b.sort_values("row_index")
        keys = ["row_index", "Drug_ID", "Target_ID"]
        if not a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True)):
            raise ValueError("Prediction entity/row IDs mismatch")
        if not np.array_equal(a.y_true.to_numpy(), b.y_true.to_numpy()):
            raise ValueError("Prediction labels mismatch")
        diff = pd.DataFrame(
            {
                "Target_ID": a.Target_ID.to_numpy(),
                "improvement": (a.y_true.to_numpy() - a.y_pred.to_numpy()) ** 2
                - (b.y_true.to_numpy() - b.y_pred.to_numpy()) ** 2,
            }
        )
        per_target = diff.groupby("Target_ID").improvement.mean().sort_index()
        if expected_targets is not None and list(per_target.index) != expected_targets:
            raise ValueError("Different held-out targets across seeds")
        expected_targets = list(per_target.index)
        effects.append(per_target)
    return pd.concat(effects, axis=1).mean(axis=1)


def summarize(results, output, baseline=None, samples=10000, expected_seeds=None):
    runs = discover_observed_runs(results)
    aggregate = aggregate_observed_runs(runs, expected_seeds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "summary.csv", index=False)
    (output / "summary.md").write_text(aggregate.to_markdown(index=False) + "\n", encoding="utf-8")
    statistics = []
    if baseline is not None:
        for experiment, group in runs.groupby("experiment"):
            base = group[group.variant == baseline].sort_values("seed")
            if base.empty:
                raise ValueError(f"Baseline {baseline!r} absent from {experiment}")
            for variant, candidate in group.groupby("variant"):
                if variant == baseline:
                    continue
                candidate = candidate.sort_values("seed")
                target_effect = paired_targets(base, candidate)
                for metric in ("mse", "ci", "pearson", "spearman"):
                    difference = candidate[metric].to_numpy() - base[metric].to_numpy()
                    if metric == "mse":
                        difference *= -1
                    valid = difference[np.isfinite(difference)]
                    row = {
                        "experiment": experiment,
                        "baseline": baseline,
                        "candidate": variant,
                        "metric": metric,
                        "unit": "training_seed",
                        "dropped_pairs": len(difference) - len(valid),
                    }
                    if len(valid) >= 2:
                        row.update(paired_effects(valid, samples))
                    else:
                        row.update(n_pairs=len(valid), status="insufficient_finite_pairs")
                    statistics.append(row)
                target_effect.rename("mse_improvement").to_csv(
                    output / f"{experiment}_{variant}_target_effects.csv"
                )
                row = {
                    "experiment": experiment,
                    "baseline": baseline,
                    "candidate": variant,
                    "metric": "mse",
                    "unit": "target_averaged_over_seeds",
                }
                if len(target_effect) >= 2:
                    row.update(paired_effects(target_effect.to_numpy(), samples))
                else:
                    row.update(n_pairs=len(target_effect), status="insufficient_targets")
                statistics.append(row)
    table = pd.DataFrame(statistics)
    if not table.empty:
        table["p_holm"] = np.nan
        if "p_two_sided" in table:
            valid = table.p_two_sided.notna()
            table.loc[valid, "p_holm"] = holm_adjust(table.loc[valid, "p_two_sided"])
        table.to_csv(output / "paired_statistics.csv", index=False)
    write_json(
        output / "analysis_protocol.json",
        {
            "observed_only": True,
            "bootstrap_samples": samples,
            "bootstrap_seed": 42,
            "confidence": 0.95,
            "target_unit": "one target after averaging MSE across seeds",
            "multiple_testing": "Holm across finite p-values in this invocation",
            "positive_effect": "candidate improves over baseline",
            "expected_seeds": expected_seeds,
            "limitation": "Exact-sequence-disjoint is not sequence-similarity-disjoint; target tests "
            "assume target-level independence, which homologous proteins may violate.",
        },
    )
    return aggregate
