from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from fgpvdta.seed import seed_everything
from fgpvdta.data.schema import write_json

from .checkpoint import load_checkpoint, save_checkpoint
from .datasets import InteractionBatch
from .metrics import regression_metrics
from .profiling import RuntimeProfiler, count_parameters

ModelFamily = Literal["fg", "pv"]


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    alpha: float = 0.5
    seed: int = 42
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    checkpoint_every: int = 1
    device: str | None = None


@dataclass
class TrainingResult:
    final_epoch: int
    selected_epoch: int
    best_epoch: int | None
    best_validation_mse: float | None
    trainable_parameters: int
    total_parameters: int
    total_runtime_seconds: float
    peak_gpu_memory_mb: float | None
    final_train_metrics: dict
    final_validation_metrics: dict | None
    final_test_metrics: dict | None
    history_path: str
    last_checkpoint: str
    selected_checkpoint: str


def _forward_loss(model, batch, family, alpha):
    if family == "fg":
        prediction = model(batch.ligand, batch.protein)
        info = torch.zeros((), device=prediction.device)
    elif family == "pv":
        out = model(batch.ligand, batch.protein, batch.images, batch.ae_latents)
        prediction = out.prediction
        info = out.info_nce
    else:
        raise ValueError(family)
    mse = F.mse_loss(prediction, batch.targets)
    return prediction, mse + alpha * info, info


def run_epoch(model, loader, family, device, alpha, optimizer=None):
    training = optimizer is not None
    model.train(training)
    preds = []
    targets = []
    total = info_sum = 0.0
    n = 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="train" if training else "eval"):
            if not isinstance(batch, InteractionBatch):
                raise TypeError("Loader must return InteractionBatch")
            batch = batch.to(device)
            pred, loss, info = _forward_loss(model, batch, family, alpha)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            targets.append(batch.targets.detach().cpu().numpy().reshape(-1))
            count = batch.targets.numel()
            total += float(loss.detach().cpu()) * count
            info_sum += float(info.detach().cpu()) * count
            n += count
    if n == 0:
        raise RuntimeError("Zero batches")
    metrics = regression_metrics(np.concatenate(targets), np.concatenate(preds))
    return metrics, total / n, info_sum / n


def train_model(
    model: nn.Module,
    family: ModelFamily,
    train_loader,
    config: TrainingConfig,
    output_dir: str | Path,
    validation_loader=None,
    test_loader=None,
    resume_from=None,
    metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    if config.epochs < 1 or config.learning_rate <= 0 or config.alpha < 0:
        raise ValueError("Invalid training epochs, learning rate or contrastive weight")
    seed_everything(config.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    start = 1
    best = None
    best_epoch = None
    if config.early_stopping_patience > 0 and validation_loader is None:
        raise ValueError("Early stopping requires validation")
    if resume_from is not None:
        state = load_checkpoint(resume_from, model, opt, device)
        start = state.start_epoch
        best = state.best_metric
        best_epoch = state.best_epoch
        history = state.history
    stale = 0
    last = out / "last.pt"
    best_path = out / "best.pt"
    with RuntimeProfiler(device) as prof:
        final = start - 1
        for epoch in range(start, config.epochs + 1):
            final = epoch
            with RuntimeProfiler(device, reset_peak=False) as eprof:
                tm, tl, ti = run_epoch(model, train_loader, family, device, config.alpha, opt)
                vm = vl = None
                if validation_loader is not None:
                    vm, vl, _ = run_epoch(model, validation_loader, family, device, 0.0, None)
            record = {
                "epoch": epoch,
                "train_total_loss": tl,
                "train_info_nce": ti,
                "train_metrics": tm.to_dict(),
                "validation_total_loss": vl,
                "validation_metrics": None if vm is None else vm.to_dict(),
                "epoch_runtime_seconds": eprof.profile.elapsed_seconds,
                "peak_gpu_memory_mb_so_far": eprof.profile.peak_gpu_memory_mb,
            }
            history.append(record)
            improved = False
            if vm is not None:
                if best is None or vm.mse < best - config.early_stopping_min_delta:
                    best = vm.mse
                    best_epoch = epoch
                    stale = 0
                    improved = True
                else:
                    stale += 1
            if improved:
                save_checkpoint(best_path, model, opt, epoch, best, best_epoch, history, metadata)
            if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
                save_checkpoint(last, model, opt, epoch, best, best_epoch, history, metadata)
            if config.early_stopping_patience > 0 and stale >= config.early_stopping_patience:
                break
    save_checkpoint(last, model, opt, final, best, best_epoch, history, metadata)
    selected = last
    selected_epoch = final
    if validation_loader is not None:
        if best_epoch is None or not best_path.is_file():
            raise RuntimeError("Missing best checkpoint")
        load_checkpoint(best_path, model, None, device)
        selected = best_path
        selected_epoch = best_epoch
    train_metrics, _, _ = run_epoch(model, train_loader, family, device, 0.0, None)
    val_metrics = None
    test_metrics = None
    if validation_loader is not None:
        val_metrics, _, _ = run_epoch(model, validation_loader, family, device, 0.0, None)
    if test_loader is not None:
        test_metrics, _, _ = run_epoch(model, test_loader, family, device, 0.0, None)
    history_path = out / "history.json"
    write_json(history_path, history)
    result = TrainingResult(
        final,
        selected_epoch,
        best_epoch,
        best,
        count_parameters(model, True),
        count_parameters(model, False),
        prof.profile.elapsed_seconds,
        prof.profile.peak_gpu_memory_mb,
        train_metrics.to_dict(),
        None if val_metrics is None else val_metrics.to_dict(),
        None if test_metrics is None else test_metrics.to_dict(),
        str(history_path),
        str(last),
        str(selected),
    )
    write_json(out / "training_summary.json", asdict(result))
    return result
