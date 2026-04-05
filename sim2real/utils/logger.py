from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Combined TensorBoard + CSV logger for training metrics."""

    def __init__(self, log_dir: str | Path, run_name: str = "ppo_biped"):
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        self._csv_path = self.log_dir / "progress.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fields: list[str] | None = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._tb_writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        row: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self._tb_writer.add_scalar(tag, v, step)
            row[tag] = v

        if self._csv_writer is None:
            self._csv_fields = ["step"] + sorted(row.keys() - {"step"})
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields, extrasaction="ignore")
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        self._tb_writer.close()
        self._csv_file.close()
