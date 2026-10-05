"""Elapsed CPU/GPU time for explicitly declared training work only."""

import math
import time
from contextlib import contextmanager


TIMING_POLICY_VERSION = 1
TRAINING_TIME_SCOPES = ("full_training", "since_resume")


def validate_seconds(value, name="training_time_seconds"):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return seconds


def checkpoint_training_time(checkpoint):
    """Never infer pure training history from legacy logs or checkpoints."""
    if checkpoint is None:
        return 0.0, "full_training"
    if not isinstance(checkpoint, dict) or checkpoint.get("version") == 2:
        return 0.0, "since_resume"
    if checkpoint.get("version") != 3:
        raise ValueError(f"Unsupported checkpoint version: {checkpoint.get('version')!r}")
    if checkpoint.get("timing_policy_version") != TIMING_POLICY_VERSION:
        raise ValueError("Checkpoint uses an unsupported training timing policy")
    scope = checkpoint.get("training_time_scope")
    if scope not in TRAINING_TIME_SCOPES:
        raise ValueError(f"Unsupported training time scope: {scope!r}")
    return validate_seconds(checkpoint["training_time_seconds"]), scope


class TrainingClock:
    """Paused by default; synchronize before starting and before stopping.

    The synchronize callback must wait on the training device. Waiting for
    training work happens before stopping the clock; waiting for excluded GPU
    work happens before restarting it. CPU/GPU overlap is counted only once.
    """

    def __init__(self, initial_seconds=0.0, *, now=time.perf_counter, synchronize=None):
        self._seconds = validate_seconds(initial_seconds)
        self._now = now
        self._synchronize = synchronize or (lambda: None)
        self._started = None
        self._training_depth = 0
        self._excluded_depth = 0
        self._pending_gpu = True  # Initialization may have queued device work.
        self._finished = False

    def _resume(self):
        if self._finished:
            raise RuntimeError("TrainingClock has already finished")
        if self._pending_gpu:
            self._synchronize()
            self._pending_gpu = False
        self._started = self._now()

    def _pause(self):
        if self._started is None:
            return
        try:
            self._synchronize()
        finally:
            # Clear the running state even if CUDA reports a failure.
            self._seconds += max(0.0, self._now() - self._started)
            self._started = None

    @contextmanager
    def training_region(self):
        if self._finished or self._excluded_depth:
            raise RuntimeError("Cannot start training inside an exclusion or after finish")
        if self._training_depth == 0:
            self._resume()
        self._training_depth += 1
        try:
            yield self
        finally:
            self._training_depth -= 1
            if self._training_depth == 0:
                self._pause()

    @contextmanager
    def excluded_region(self, *, gpu_work=True):
        if self._excluded_depth == 0:
            self._pause()
        self._excluded_depth += 1
        self._pending_gpu |= gpu_work
        try:
            yield self
        finally:
            self._excluded_depth -= 1
            if not self._excluded_depth and self._training_depth and not self._finished:
                self._resume()

    def snapshot_seconds(self):
        """Read a frozen value without synchronization or clock advancement."""
        if self._started is not None:
            raise RuntimeError("Read training time after leaving the training region")
        return self._seconds

    def finish(self):
        self._pause()
        self._finished = True
        return self._seconds
