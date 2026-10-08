"""One CSV row per complete training group, with stable per-image columns."""

import csv
import json
import math
import shutil
import statistics
import uuid
from pathlib import Path


class LossScaleLogger:
    def __init__(self, model_path, cameras, loss_mode, resume_from_iter=0):
        self.csv_path = Path(model_path) / "loss_scales.csv"
        self.loss_mode = loss_mode
        self.image_columns = [
            f"image_{index:04d}:{camera.image_name}:{getattr(camera, 'normalizing_method', 'all')}"
            for index, camera in enumerate(cameras)
        ]
        self.columns = [
            "group", "start_iteration", "end_iteration", "training_time_seconds",
            "loss_intensity_mode", "scale_min", "scale_max", "scale_mean",
            "scale_std", "scale_cv", "scale_max_min_ratio", "training_order",
            *self.image_columns,
        ]
        self.archive_path = None
        self.latest_batch_mean = None
        rows = []
        if self.csv_path.exists():
            with self.csv_path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                compatible = reader.fieldnames == self.columns
                history = list(reader)
            compatible = compatible and all(
                row["loss_intensity_mode"] == loss_mode for row in history
            )
            # Preserve the original table before restarting or rolling back.
            self.archive_path = self.csv_path.with_name(f"loss_scales.archive_{uuid.uuid4().hex}.csv")
            shutil.copy2(self.csv_path, self.archive_path)
            if compatible and resume_from_iter > 0:
                rows = [row for row in history if int(row["end_iteration"]) <= resume_from_iter]
                if rows and int(rows[-1]["end_iteration"]) == resume_from_iter:
                    self.latest_batch_mean = float(rows[-1]["scale_mean"]) if rows[-1]["scale_mean"] else None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(rows)

    def log(self, group, start_iteration, end_iteration, training_time_seconds, scales, training_order):
        if len(scales) != len(self.image_columns) or any(scale is None for scale in scales):
            raise ValueError("A loss scale row requires every image in a complete training group")
        values = [value for scale in scales for value in scale]
        row = dict(
            group=group, start_iteration=start_iteration, end_iteration=end_iteration,
            training_time_seconds=training_time_seconds, loss_intensity_mode=self.loss_mode,
            training_order=json.dumps(training_order),
        )
        # Keep non-finite coefficients visible, but do not report misleading statistics.
        if values and all(math.isfinite(value) for value in values):
            minimum, maximum = min(values), max(values)
            mean = statistics.fmean(values)
            std = statistics.pstdev(values)
            row.update(
                scale_min=minimum, scale_max=maximum, scale_mean=mean, scale_std=std,
                scale_cv=std / abs(mean) if mean else None,
                scale_max_min_ratio=maximum / minimum if minimum > 0 else None,
            )
        for column, scale in zip(self.image_columns, scales):
            row[column] = scale[0] if len(scale) == 1 else json.dumps(scale)
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.columns).writerow(row)
        self.latest_batch_mean = row.get("scale_mean")
