import json

import numpy as np
import pandas as pd

from fgpvdta.analysis.aggregate import aggregate_observed_runs, discover_observed_runs
from fgpvdta.analysis.statistics import paired_seed_test
from fgpvdta.training.metrics import regression_metrics

SEEDS = [11, 22, 33, 44, 55]


def write(root, variant, seed, scale):
    d = root / variant / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    y = np.arange(1.0, 7.0)
    p = y + np.array([0.2, -0.15, 0.1, -0.2, 0.15, -0.1]) * scale
    pred = pd.DataFrame(
        {
            "row_index": range(6),
            "Drug_ID": [f"D{i}" for i in range(6)],
            "Target_ID": [f"T{i}" for i in range(6)],
            "y_true": y,
            "y_pred": p,
        }
    )
    pp = d / "test_predictions.csv"
    pred.to_csv(pp, index=False)
    m = regression_metrics(y, p).to_dict()
    (d / "result.json").write_text(
        json.dumps(
            {
                "observed": True,
                "experiment": "x",
                "variant": variant,
                "seed": seed,
                "split_sha256": "same",
                "test_metrics": m,
                "total_runtime_seconds": 1.0,
                "peak_gpu_memory_mb": None,
                "trainable_parameters": 10,
                "predictions_csv": str(pp),
            }
        )
    )


def test_stats(tmp_path):
    root = tmp_path / "r"
    for i, s in enumerate(SEEDS):
        write(root, "base", s, 1 + 0.01 * i)
        write(root, "better", s, 0.5 + 0.005 * i)
    runs = discover_observed_runs(root)
    agg = aggregate_observed_runs(runs, SEEDS)
    assert len(agg) == 2
    st = paired_seed_test(runs, "x", "base", "better", "mse", SEEDS, 1000, 1)
    assert st.mean_improvement > 0
