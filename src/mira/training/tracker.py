"""Training-loop bookkeeping: throughput/loss tracking, periodic events, and timing."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import torch

from mira.data.batch import VideoActionBatch

from .metrics.distributed_metric import DistributedMetric

logger = logging.getLogger(__name__)


class TrainingTracker:
    """Tracks training statistics: throughput, losses, and ETA."""

    def __init__(
        self,
        world_size: int,
        device: str | int | torch.device = "cpu",
        total_steps: int | None = None,
    ):
        self.world_size = world_size
        self.device = device
        self.total_steps = total_steps
        self.training_start_time: float | None = None
        self.current_start_time: float | None = time.time()
        self.total_n_frames_processed = 0

        self.cur_n_frames_processed = 0
        self.cur_losses: dict[str, DistributedMetric] = defaultdict(
            lambda: DistributedMetric(device=self.device)
        )

    def on_batch_processed(self, batch: VideoActionBatch, losses: dict[str, torch.Tensor]) -> None:
        if self.training_start_time is None:
            self.training_start_time = time.time()
        if self.current_start_time is None:
            self.current_start_time = time.time()

        n_frames_in_batch = batch.video.shape[0] * batch.video.shape[1]

        self.total_n_frames_processed += n_frames_in_batch

        self.cur_n_frames_processed += n_frames_in_batch
        for k, v in losses.items():
            self.cur_losses[k].update(v)

    def get_stats(self, step: int) -> dict[str, float]:
        if self.current_start_time is None:
            return {}

        elapsed_time = time.time() - self.current_start_time
        if elapsed_time == 0:
            return {}

        n_frames_processed_per_sec = self.cur_n_frames_processed / elapsed_time
        losses_to_log = {
            f"train/{k}": metric.compute_and_reset().item() for k, metric in self.cur_losses.items()
        }

        self.current_start_time = time.time()
        self.cur_n_frames_processed = 0

        eta_stats: dict[str, float] = {}
        if self.total_steps is not None and step > 0:
            if self.training_start_time is not None:
                total_elapsed = time.time() - self.training_start_time
                steps_remaining = self.total_steps - step
                time_per_step = total_elapsed / step
                eta_seconds = steps_remaining * time_per_step
                eta_stats["System/eta_hours"] = eta_seconds / 3600

        return {
            "System/n_frames_processed": self.total_n_frames_processed * self.world_size,
            "System/throughput_fps_total": n_frames_processed_per_sec * self.world_size,
            "System/throughput_fps_per_gpu": n_frames_processed_per_sec,
            **losses_to_log,
            **eta_stats,
        }


def periodic_event(
    step: int, interval: int | str, total_steps: int | None = None, include_0: bool = True
) -> bool:
    """Determine if an action should occur at the current step based on interval or percentage."""
    if step == 0:
        return include_0

    if isinstance(interval, int):
        return step % interval == 0
    elif isinstance(interval, str) and interval.endswith("%"):
        percent = float(interval[:-1])
        if percent <= 0 or percent > 100:
            raise ValueError("interval percentage must be between 1 and 100")

        if total_steps is None:
            raise ValueError("total_steps must be provided when using percentage intervals")

        steps_interval = max(1, int(total_steps * (percent / 100)))
        return step % steps_interval == 0
    else:
        raise ValueError(f"Invalid interval: {interval}. Must be int or percentage string (e.g. '10%')")


class display_execution_time:
    """Context manager that logs the wall-clock time taken by the wrapped block."""

    def __init__(self, task_name: str, print_output: bool = True):
        self.task_name = task_name
        self.print_output = print_output
        self.start_time: float | None = None
        self.elapsed_time_ms: int | None = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self) -> display_execution_time:
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        assert self.start_time is not None
        end_time = time.monotonic()
        self.elapsed_time_ms = int((end_time - self.start_time) * 1000)
        if self.print_output:
            if self.elapsed_time_ms < 1000:
                self.logger.info("%s took %dms", self.task_name, self.elapsed_time_ms)
            else:
                self.logger.info("%s took %.2fs", self.task_name, self.elapsed_time_ms / 1000.0)
        return False  # Don't suppress exceptions
