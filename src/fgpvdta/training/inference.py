from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from fgpvdta.data.schema import write_json

from .datasets import InteractionBatch
from .metrics import regression_metrics

ModelFamily = Literal["fg", "pv"]


@torch.no_grad()
def predict_loader(model, loader, family: ModelFamily, device):
    device = torch.device(device)
    model = model.to(device).eval()
    records = []
    for batch in loader:
        if not isinstance(batch, InteractionBatch):
            raise TypeError("Loader must return InteractionBatch")
        batch = batch.to(device)
        pred = (
            model(batch.ligand, batch.protein)
            if family == "fg"
            else model(batch.ligand, batch.protein, batch.images, batch.ae_latents).prediction
        )
        y = batch.targets.cpu().numpy().reshape(-1)
        p = pred.cpu().numpy().reshape(-1)
        rows = batch.row_indices.cpu().numpy().reshape(-1)
        for ri, did, tid, yt, yp in zip(rows, batch.drug_ids, batch.target_ids, y, p, strict=False):
            records.append(
                {
                    "row_index": int(ri),
                    "Drug_ID": did,
                    "Target_ID": tid,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )
    return pd.DataFrame(records).sort_values("row_index").reset_index(drop=True)


def export_predictions(predictions, output_csv, output_metrics_json):
    output_csv = Path(output_csv)
    output_metrics_json = Path(output_metrics_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    metrics = regression_metrics(
        predictions.y_true.to_numpy(), predictions.y_pred.to_numpy()
    ).to_dict()
    write_json(output_metrics_json, metrics)
    return metrics
