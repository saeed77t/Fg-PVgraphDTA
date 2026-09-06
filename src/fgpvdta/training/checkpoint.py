from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ResumeState:
    start_epoch: int
    best_metric: float | None
    best_epoch: int | None
    history: list[dict[str, Any]]


def save_checkpoint(path, model, optimizer, epoch, best_metric, best_epoch, history, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "history": history,
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return ResumeState(
        int(payload["epoch"]) + 1,
        payload.get("best_metric"),
        payload.get("best_epoch"),
        list(payload.get("history", [])),
    )
