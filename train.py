#!/usr/bin/env python3
"""Train and evaluate PRCast."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from prcast.data import (
    DATASETS,
    LONG_HORIZONS,
    PEMS_HORIZONS,
    DatasetBundle,
    batches,
    limit_starts,
    load_dataset,
    window_starts,
)
from prcast.model import build_model


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "benchmark.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def phase_svd_initialization(
    train: torch.Tensor,
    period: int,
    rank: int,
    init_periods: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a low-rank phase initialization using training data only."""

    start = 0
    if init_periods:
        start = max(0, len(train) - init_periods * period)
    source = train[start:]
    phase = torch.arange(start, len(train)) % period
    template = torch.zeros(period, train.shape[1], dtype=train.dtype)
    counts = torch.zeros(period, 1, dtype=train.dtype)
    template.index_add_(0, phase, source)
    counts.index_add_(0, phase, torch.ones(len(source), 1, dtype=train.dtype))
    template = template / counts.clamp_min(1.0)

    u, singular_values, vh = torch.linalg.svd(template, full_matrices=False)
    retained = min(rank, len(singular_values))
    root = singular_values[:retained].sqrt()
    basis = u[:, :retained] * root
    coefficients = vh[:retained].T * root
    if retained < rank:
        basis = F.pad(basis, (0, rank - retained))
        coefficients = F.pad(coefficients, (0, rank - retained))
    return basis, coefficients


def make_model_config(
    settings: dict,
    bundle: DatasetBundle,
    horizon: int,
    history: int,
    phase_basis: torch.Tensor | None,
    phase_coefficients: torch.Tensor | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=history,
        pred_len=horizon,
        enc_in=bundle.channels,
        mark_dim=bundle.mark_dim,
        hidden=int(settings["hidden"]),
        layers=int(settings["layers"]),
        temporal_rank=int(settings["temporal_rank"]),
        phase_rank=int(settings["phase_rank"]),
        channel_rank=int(settings["channel_rank"]),
        channel_slots=int(settings.get("channel_slots", 8)),
        kernel=int(settings.get("kernel", 25)),
        expansion=float(settings["expansion"]),
        dropout=float(settings.get("dropout", 0.1)),
        phase_period=int(settings["phase_period"]),
        phase_ridge=float(settings.get("phase_ridge", 1.0)),
        phase_prior_strength=float(settings["phase_prior_strength"]),
        phase_scale_init=float(settings["phase_scale_init"]),
        phase_update_scale_init=float(settings["phase_update_scale_init"]),
        phase_residual_scale_init=float(settings["phase_residual_scale_init"]),
        phase_basis_init=phase_basis,
        phase_coefficient_init=phase_coefficients,
        channel_scale_init=float(settings.get("channel_scale_init", -2.0)),
        factor_rank=int(settings["factor_rank"]),
        factor_scale_init=float(settings["factor_scale_init"]),
        calendar_rank=int(settings["calendar_rank"]),
        calendar_init=float(settings["calendar_init"]),
        calendar_decay_init=float(settings["calendar_decay_init"]),
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    return device


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    split,
    starts: torch.Tensor,
    history: int,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    count = 0
    for x, y, x_mark, y_mark in batches(
        split, starts, history, horizon, batch_size, device
    ):
        output = model.forecast(x, x_mark, y_mark)
        squared += F.mse_loss(output, y, reduction="sum").item()
        absolute += F.l1_loss(output, y, reduction="sum").item()
        count += y.numel()
    if count == 0:
        raise ValueError("evaluation split has no complete windows")
    return squared / count, absolute / count


def validation_groups(
    starts: torch.Tensor, folds: int
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    if len(starts) == 0:
        raise ValueError("validation split has no complete windows")
    if folds <= 1:
        return starts, starts, [starts]
    groups = [part for part in torch.tensor_split(starts, folds) if len(part)]
    if len(groups) < 2:
        raise ValueError("validation split is too short for rolling validation")
    return groups[0], torch.cat(groups[1:]), groups


def train_one(
    bundle: DatasetBundle,
    settings: dict,
    seed: int,
    horizon: int,
    args,
) -> dict:
    set_seed(seed)
    device = resolve_device(args.device)
    history = int(args.history)

    phase_basis = phase_coefficients = None
    if settings.get("phase_svd_init", True):
        phase_basis, phase_coefficients = phase_svd_initialization(
            bundle.train.series,
            int(settings["phase_period"]),
            int(settings["phase_rank"]),
            int(settings.get("phase_init_periods", 0)),
        )
    config = make_model_config(
        settings,
        bundle,
        horizon,
        history,
        phase_basis,
        phase_coefficients,
    )
    model = build_model(config).to(device)

    train_starts = limit_starts(
        window_starts(len(bundle.train.series), history, horizon),
        args.max_train_windows,
    )
    validation_starts = limit_starts(
        window_starts(len(bundle.validation.series), history, horizon),
        args.max_validation_windows,
    )
    test_starts = limit_starts(
        window_starts(len(bundle.test.series), history, horizon),
        args.max_test_windows,
    )
    stop_starts, selection_starts, groups = validation_groups(
        validation_starts, int(settings.get("rolling_folds", 3))
    )

    phase_parameters = list(model.phase_operator.parameters())
    phase_ids = {id(parameter) for parameter in phase_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in phase_ids
    ]
    phase_lr_scale = float(settings.get("phase_lr_scale", 1.0))
    if phase_lr_scale != 1.0:
        optimizer = torch.optim.AdamW(
            [
                {"params": other_parameters, "lr": float(settings["lr"])},
                {
                    "params": phase_parameters,
                    "lr": float(settings["lr"]) * phase_lr_scale,
                },
            ],
            weight_decay=float(settings["weight_decay"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
        )

    generator = torch.Generator().manual_seed(seed)
    best_metric = math.inf
    best_state = None
    stale = 0
    ema_decay = settings.get("ema_decay")
    ema_start = int(settings.get("ema_start", 1))
    ema_state = None
    max_epochs = int(settings["epochs"])
    patience = int(settings["patience"])
    diff_weight = float(settings.get("diff_weight", 0.0))

    for epoch in range(max_epochs):
        model.train()
        for x, y, x_mark, y_mark in batches(
            bundle.train,
            train_starts,
            history,
            horizon,
            int(settings["batch_size"]),
            device,
            shuffle=True,
            generator=generator,
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model.forecast(x, x_mark, y_mark)
            loss = F.mse_loss(output, y)
            if diff_weight:
                loss = loss + diff_weight * F.mse_loss(
                    output[:, 1:] - output[:, :-1], y[:, 1:] - y[:, :-1]
                )
            loss.backward()
            optimizer.step()

            if ema_decay and epoch + 1 >= ema_start:
                state = model.state_dict()
                if ema_state is None:
                    ema_state = {
                        key: value.detach().clone()
                        for key, value in state.items()
                    }
                else:
                    for key, value in state.items():
                        if value.is_floating_point():
                            ema_state[key].mul_(ema_decay).add_(
                                value.detach(), alpha=1.0 - ema_decay
                            )
                        else:
                            ema_state[key].copy_(value)

        fold_mse = [
            evaluate(
                model,
                bundle.validation,
                fold,
                history,
                horizon,
                int(settings.get("eval_batch_size", 256)),
                device,
            )[0]
            for fold in groups
        ]
        validation_metric = max(fold_mse)
        candidate_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if ema_state is not None:
            live_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            model.load_state_dict(ema_state)
            ema_fold_mse = [
                evaluate(
                    model,
                    bundle.validation,
                    fold,
                    history,
                    horizon,
                    int(settings.get("eval_batch_size", 256)),
                    device,
                )[0]
                for fold in groups
            ]
            ema_metric = max(ema_fold_mse)
            if ema_metric < validation_metric:
                validation_metric = ema_metric
                candidate_state = {
                    key: value.detach().cpu().clone()
                    for key, value in ema_state.items()
                }
            model.load_state_dict(live_state)
        if validation_metric < best_metric:
            best_metric = validation_metric
            best_state = candidate_state
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

        if args.lr_decay == "type1":
            learning_rate = float(settings["lr"]) * (0.5**epoch)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        elif args.lr_decay == "cosine":
            learning_rate = float(settings["lr"]) * 0.5 * (
                1.0 + math.cos((epoch + 1) / max_epochs * math.pi)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    selection_mse, selection_mae = evaluate(
        model,
        bundle.validation,
        selection_starts,
        history,
        horizon,
        int(settings.get("eval_batch_size", 256)),
        device,
    )
    test_mse, test_mae = evaluate(
        model,
        bundle.test,
        test_starts,
        history,
        horizon,
        int(settings.get("eval_batch_size", 256)),
        device,
    )
    result = {
        "dataset": args.dataset_name,
        "prediction_length": horizon,
        "seed": seed,
        "architecture_id": model.architecture_id,
        "mse": test_mse,
        "mae": test_mae,
        "best_validation_mse": best_metric,
        "selection_validation_mse": selection_mse,
        "selection_validation_mae": selection_mae,
        "epochs_ran": epoch + 1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_results_csv(output_dir: Path) -> None:
    fields = [
        "dataset",
        "prediction_length",
        "seed",
        "architecture_id",
        "mse",
        "mae",
        "best_validation_mse",
        "selection_validation_mse",
        "selection_validation_mae",
        "epochs_ran",
        "parameters",
    ]
    rows = []
    for path in sorted(output_dir.glob("*.json")):
        row = json.loads(path.read_text())
        if set(row) == set(fields):
            rows.append(row)
    if not rows:
        return
    rows.sort(
        key=lambda row: (
            row["dataset"],
            int(row["prediction_length"]),
            int(row["seed"]),
        )
    )
    temporary = output_dir / "results.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_dir / "results.csv")


def smoke_test() -> None:
    set_seed(7)
    config = SimpleNamespace(
        seq_len=16,
        pred_len=8,
        enc_in=3,
        mark_dim=5,
        hidden=8,
        layers=1,
        temporal_rank=4,
        phase_rank=4,
        channel_rank=2,
        channel_slots=2,
        kernel=5,
        expansion=1.5,
        dropout=0.0,
        phase_period=8,
        phase_ridge=1.0,
        phase_prior_strength=2.0,
        phase_scale_init=-2.0,
        phase_update_scale_init=-2.0,
        phase_residual_scale_init=0.0,
        phase_basis_init=None,
        phase_coefficient_init=None,
        channel_scale_init=-2.0,
        factor_rank=1,
        factor_scale_init=-2.0,
        calendar_rank=4,
        calendar_init=-2.0,
        calendar_decay_init=1.0,
    )
    x = torch.randn(2, 16, 3)
    history_mark = torch.zeros(2, 16, 5)
    future_mark = torch.zeros(2, 8, 5)
    history_phase = (torch.arange(16) % 8).float() / 8.0
    future_phase = ((torch.arange(8) + 16) % 8).float() / 8.0
    history_mark[:, :, -1] = history_phase
    future_mark[:, :, -1] = future_phase
    model = build_model(config)
    output = model.forecast(x, history_mark, future_mark)
    if tuple(output.shape) != (2, 8, 3):
        raise AssertionError(f"unexpected output shape: {output.shape}")
    print("smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRCast experiment runner")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--history", type=int, default=96)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-validation-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--lr-decay", choices=("none", "type1", "cosine"), default="none")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return 0
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if args.history != int(config.get("history", args.history)):
        raise ValueError("--history must match the frozen configuration")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for dataset in args.datasets:
        dataset_config = {
            key: value
            for key, value in config.items()
            if key not in ("datasets", "history")
        }
        dataset_config.update(config["datasets"][dataset])
        horizons = tuple(args.horizons or (
            PEMS_HORIZONS if dataset.startswith("PEMS") else LONG_HORIZONS
        ))
        valid_horizons = PEMS_HORIZONS if dataset.startswith("PEMS") else LONG_HORIZONS
        invalid = [horizon for horizon in horizons if horizon not in valid_horizons]
        if invalid:
            raise ValueError(f"invalid horizon(s) for {dataset}: {invalid}")
        for horizon in horizons:
            for seed in args.seeds:
                tasks.append((dataset, horizon, seed, dataset_config))

    if args.dry_run:
        for dataset, horizon, seed, _ in tasks:
            print(f"{dataset} horizon={horizon} seed={seed}")
        print(f"planned tasks: {len(tasks)}")
        return 0

    loaded = {}
    for dataset, horizon, seed, settings in tasks:
        key = (dataset, int(settings["phase_period"]))
        if key not in loaded:
            loaded[key] = load_dataset(
                dataset,
                args.data_root,
                phase_period=key[1],
                history=args.history,
            )
        artifact = output_dir / (
            f"{dataset}_h{horizon}_seed{seed}.json"
        )
        if args.resume and artifact.exists():
            print(f"SKIP {artifact.name}")
            continue
        args.dataset_name = dataset
        print(
            f"RUN {dataset} horizon={horizon} seed={seed}",
            flush=True,
        )
        result = train_one(
            loaded[key], settings, seed, horizon, args
        )
        write_json(artifact, result)
        print(
            f"DONE mse={result['mse']:.8f} mae={result['mae']:.8f} "
            f"epochs={result['epochs_ran']}",
            flush=True,
        )
    write_results_csv(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
