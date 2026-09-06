from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .aggregate import METRICS, aggregate_observed_runs, discover_observed_runs
from .statistics import hierarchical_paired_bootstrap, paired_seed_test


def build_observed_tables(
    results_root,
    output_dir,
    expected_seeds: Sequence[int] | None = None,
    baseline_variant: str | None = None,
    comparison_variants: Sequence[str] | None = None,
    prediction_bootstrap_samples: int = 2000,
    seed_bootstrap_samples: int = 50000,
    bootstrap_seed: int = 42,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = discover_observed_runs(results_root)
    agg = aggregate_observed_runs(runs, expected_seeds)
    agg_path = out / "aggregate_numeric.csv"
    agg.to_csv(agg_path, index=False)
    pub = agg[
        [
            "experiment",
            "variant",
            "mse_mean",
            "mse_sd",
            "ci_mean",
            "ci_sd",
            "pearson_mean",
            "pearson_sd",
            "spearman_mean",
            "spearman_sd",
            "trainable_parameters",
            "runtime_seconds_mean",
            "n_seeds",
        ]
    ].copy()
    pub_path = out / "publication_summary.csv"
    pub.to_csv(pub_path, index=False)
    (out / "publication_summary.md").write_text(
        pub.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    outputs = {"aggregate_numeric_csv": agg_path, "publication_csv": pub_path}
    if baseline_variant is not None:
        comps = list(comparison_variants or [])
        exps = runs.experiment.unique()
        if len(exps) != 1:
            raise ValueError("One experiment expected")
        seed_rows = []
        pred_rows = []
        for proposed in comps:
            for metric in METRICS:
                seed_rows.append(
                    paired_seed_test(
                        runs,
                        str(exps[0]),
                        baseline_variant,
                        proposed,
                        metric,
                        expected_seeds,
                        seed_bootstrap_samples,
                        bootstrap_seed,
                    ).to_dict()
                )
                pred_rows.append(
                    hierarchical_paired_bootstrap(
                        runs,
                        str(exps[0]),
                        baseline_variant,
                        proposed,
                        metric,
                        prediction_bootstrap_samples,
                        bootstrap_seed,
                    )
                )
        sdf = pd.DataFrame(seed_rows)
        pdf = pd.DataFrame(pred_rows)
        sdf.to_csv(out / "paired_seed_statistics.csv", index=False)
        pdf.to_csv(out / "paired_prediction_bootstrap.csv", index=False)
        (out / "paired_seed_statistics.md").write_text(
            sdf.to_markdown(index=False) + "\n", encoding="utf-8"
        )
        (out / "paired_prediction_bootstrap.md").write_text(
            pdf.to_markdown(index=False) + "\n", encoding="utf-8"
        )
    (out / "table_generation_metadata.json").write_text(
        json.dumps(
            {
                "source_results_root": str(results_root),
                "observed_only": True,
                "expected_seeds": expected_seeds,
                "baseline_variant": baseline_variant,
                "comparison_variants": comparison_variants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs
