from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fgpvdta.data.download import sha256_file
from fgpvdta.training.metrics import regression_metrics

METRICS = ("mse", "ci", "pearson", "spearman")


def discover_observed_runs(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    files = sorted(root.rglob("result.json"))
    if not files:
        raise RuntimeError(f"No result.json under {root}")
    records = []
    for path in files:
        p = json.loads(path.read_text())
        if p.get("observed") is not True:
            continue
        predictions_path = Path(p["predictions_csv"])
        if not predictions_path.is_absolute():
            predictions_path = path.parent / predictions_path
        predictions = pd.read_csv(predictions_path, dtype={"Drug_ID": str, "Target_ID": str})
        if p.get("predictions_sha256") and sha256_file(predictions_path) != p["predictions_sha256"]:
            raise ValueError(f"Predictions changed: {predictions_path}")
        if predictions.row_index.duplicated().any():
            raise ValueError("Duplicate prediction row IDs")
        m = regression_metrics(predictions.y_true, predictions.y_pred).to_dict()
        missing = set(METRICS) - set(m)
        if missing:
            raise ValueError(f"Missing metrics in {path}: {missing}")
        records.append(
            {
                "experiment": p["experiment"],
                "variant": p["variant"],
                "seed": int(p["seed"]),
                "split_sha256": p["split_sha256"],
                **{k: float(m[k]) for k in METRICS},
                "n_test": int(m["n"]),
                "runtime_seconds": float(p["total_runtime_seconds"]),
                "peak_gpu_memory_mb": p.get("peak_gpu_memory_mb"),
                "trainable_parameters": int(p["trainable_parameters"]),
                "predictions_csv": str(predictions_path.resolve()),
                "result_json": str(path),
            }
        )
    if not records:
        raise ValueError("No observed research runs (synthetic smoke outputs are excluded)")
    frame = pd.DataFrame(records)
    if frame.duplicated(["experiment", "variant", "seed"]).any():
        raise ValueError("Duplicate variant/seed results")
    return frame.sort_values(["experiment", "variant", "seed"]).reset_index(drop=True)


def validate_paired_seed_design(
    runs: pd.DataFrame, expected_seeds: Sequence[int] | None = None
) -> None:
    for experiment, group in runs.groupby("experiment"):
        if group.split_sha256.nunique() != 1:
            raise ValueError(f"Multiple splits in {experiment}")
        seed_sets = {v: tuple(sorted(g.seed.astype(int))) for v, g in group.groupby("variant")}
        if len(set(seed_sets.values())) != 1:
            raise ValueError(f"Unmatched seeds: {seed_sets}")
        if expected_seeds is not None and next(iter(seed_sets.values())) != tuple(
            sorted(map(int, expected_seeds))
        ):
            raise ValueError("Incomplete expected seed set")


def aggregate_observed_runs(
    runs: pd.DataFrame, expected_seeds: Sequence[int] | None = None
) -> pd.DataFrame:
    validate_paired_seed_design(runs, expected_seeds)
    records = []
    for (experiment, variant), g in runs.groupby(["experiment", "variant"]):
        rec = {
            "experiment": experiment,
            "variant": variant,
            "n_seeds": len(g),
            "seeds": ",".join(map(str, sorted(g.seed.astype(int)))),
            "split_sha256": g.split_sha256.iloc[0],
            "n_test": int(g.n_test.iloc[0]),
            "trainable_parameters": int(g.trainable_parameters.iloc[0]),
        }
        for metric in METRICS:
            vals = g[metric].astype(float).to_numpy()
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rt = g.runtime_seconds.astype(float).to_numpy()
        rec["runtime_seconds_mean"] = float(rt.mean())
        rec["runtime_seconds_sd"] = float(rt.std(ddof=1)) if len(rt) > 1 else 0.0
        mem = g.peak_gpu_memory_mb.dropna().astype(float).to_numpy()
        rec["peak_gpu_memory_mb_mean"] = float(mem.mean()) if len(mem) else np.nan
        records.append(rec)
    return pd.DataFrame(records).sort_values(["experiment", "variant"]).reset_index(drop=True)
