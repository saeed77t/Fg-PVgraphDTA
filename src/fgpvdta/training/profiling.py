from __future__ import annotations

import time
from dataclasses import dataclass

import torch


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@dataclass
class RuntimeProfile:
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None

    @property
    def peak_gpu_memory_mb(self):
        return None if self.peak_gpu_memory_bytes is None else self.peak_gpu_memory_bytes / 1024**2


class RuntimeProfiler:
    def __init__(self, device, reset_peak=True):
        self.device = device
        self.reset_peak = reset_peak
        self.profile = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            if self.reset_peak:
                torch.cuda.reset_peak_memory_stats(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.profile = RuntimeProfile(
            time.perf_counter() - self.start,
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else None,
        )
