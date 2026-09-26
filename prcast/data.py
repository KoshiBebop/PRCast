"""Data loading for the multivariate forecasting benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch


DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Traffic",
    "Weather",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
)

LONG_HORIZONS = (96, 192, 336, 720)
PEMS_HORIZONS = (12, 24, 48, 96)


@dataclass(frozen=True)
class Split:
    series: torch.Tensor
    marks: torch.Tensor
    offset: int


@dataclass(frozen=True)
class DatasetBundle:
    train: Split
    validation: Split
    test: Split

    @property
    def channels(self) -> int:
        return int(self.train.series.shape[1])

    @property
    def mark_dim(self) -> int:
        return int(self.train.marks.shape[1])


def _calendar_features(dates, frequency: str) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    columns = []
    if frequency == "minute":
        columns.append(np.asarray(index.minute, dtype=np.float64) / 59.0 - 0.5)
    columns.extend(
        (
            np.asarray(index.hour, dtype=np.float64) / 23.0 - 0.5,
            np.asarray(index.dayofweek, dtype=np.float64) / 6.0 - 0.5,
            (np.asarray(index.day, dtype=np.float64) - 1.0) / 30.0 - 0.5,
            (np.asarray(index.dayofyear, dtype=np.float64) - 1.0) / 365.0 - 0.5,
        )
    )
    return np.stack(columns, axis=1).astype(np.float32)


def _read_csv(path: Path, frequency: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"dataset must contain a date column and values: {path}")
    raw = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    marks = _calendar_features(frame.iloc[:, 0], frequency)
    return raw, marks


def _read_pems(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    values = np.load(path, allow_pickle=True)["data"]
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError(f"expected [time, channel, feature] data: {path}")
    raw = values[:, :, 0].astype(np.float32)
    tick = np.arange(len(raw), dtype=np.float32)
    marks = np.stack(
        (
            np.sin(2 * np.pi * tick / 288.0),
            np.cos(2 * np.pi * tick / 288.0),
            np.sin(2 * np.pi * tick / 2016.0),
            np.cos(2 * np.pi * tick / 2016.0),
        ),
        axis=1,
    ).astype(np.float32)
    return raw, marks


def load_dataset(
    name: str,
    data_root: Path | str,
    phase_period: int,
    history: int = 96,
) -> DatasetBundle:
    """Load and normalize one dataset using chronological splits.

    Statistics are fitted only on the training interval.  Non-PEMS validation
    and test intervals include ``history`` preceding observations so their
    first window has the same context length as the training windows.
    """

    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    if phase_period <= 0:
        raise ValueError("phase period must be positive")
    if history <= 0:
        raise ValueError("history must be positive")

    root = Path(data_root)
    if name in ("ETTh1", "ETTh2"):
        raw, marks = _read_csv(root / "ETT-small" / f"{name}.csv", "hour")
        train_end, validation_end, test_end = 8640, 11520, 14400
        bounds = (
            (0, train_end),
            (train_end - history, validation_end),
            (validation_end - history, test_end),
        )
    elif name in ("ETTm1", "ETTm2"):
        raw, marks = _read_csv(root / "ETT-small" / f"{name}.csv", "minute")
        train_end, validation_end, test_end = 34560, 46080, 57600
        bounds = (
            (0, train_end),
            (train_end - history, validation_end),
            (validation_end - history, test_end),
        )
    elif name in ("Electricity", "Traffic", "Weather"):
        directory = {
            "Electricity": "electricity",
            "Traffic": "traffic",
            "Weather": "weather",
        }[name]
        filename = {
            "Electricity": "electricity.csv",
            "Traffic": "traffic.csv",
            "Weather": "weather.csv",
        }[name]
        frequency = "minute" if name == "Weather" else "hour"
        raw, marks = _read_csv(root / directory / filename, frequency)
        train_end = int(len(raw) * 0.7)
        test_length = int(len(raw) * 0.2)
        validation_end = len(raw) - test_length
        bounds = (
            (0, train_end),
            (train_end - history, validation_end),
            (validation_end - history, len(raw)),
        )
    else:
        raw, marks = _read_pems(root / "PEMS" / f"{name}.npz")
        train_end = int(len(raw) * 0.6)
        validation_end = int(len(raw) * 0.8)
        bounds = (
            (0, train_end),
            (train_end, validation_end),
            (validation_end, len(raw)),
        )

    if len(raw) != len(marks):
        raise ValueError(f"values and markers have different lengths for {name}")
    if any(start < 0 or end > len(raw) or start >= end for start, end in bounds):
        raise ValueError(f"invalid chronological split for {name}: {bounds}")

    train_start, train_stop = bounds[0]
    mean = np.nanmean(raw[train_start:train_stop], axis=0, keepdims=True)
    std = np.nanstd(raw[train_start:train_stop], axis=0, keepdims=True) + 1e-6
    normalized = np.nan_to_num((raw - mean) / std).astype(np.float32)

    phase = (
        np.arange(len(raw), dtype=np.float32) % float(phase_period)
    )[:, None] / float(phase_period)
    marks = np.concatenate((marks, phase), axis=1).astype(np.float32)

    def make_split(start: int, stop: int) -> Split:
        return Split(
            series=torch.from_numpy(normalized[start:stop]),
            marks=torch.from_numpy(marks[start:stop]),
            offset=start,
        )

    return DatasetBundle(
        train=make_split(*bounds[0]),
        validation=make_split(*bounds[1]),
        test=make_split(*bounds[2]),
    )


def window_starts(length: int, history: int, horizon: int) -> torch.Tensor:
    count = length - history - horizon + 1
    return torch.arange(max(0, count), dtype=torch.long)


def limit_starts(starts: torch.Tensor, limit: int = 0) -> torch.Tensor:
    if limit and len(starts) > limit:
        return torch.linspace(0, len(starts) - 1, limit).long().unique()
    return starts


def batches(
    split: Split,
    starts: torch.Tensor,
    history: int,
    horizon: int,
    batch_size: int,
    device: torch.device,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if shuffle:
        starts = starts[torch.randperm(len(starts), generator=generator)]
    series = split.series.to(device)
    marks = split.marks.to(device)
    history_offset = torch.arange(history, device=device)
    future_offset = history + torch.arange(horizon, device=device)
    for begin in range(0, len(starts), batch_size):
        index = starts[begin : begin + batch_size].to(device)
        yield (
            series[index[:, None] + history_offset],
            series[index[:, None] + future_offset],
            marks[index[:, None] + history_offset],
            marks[index[:, None] + future_offset],
        )
