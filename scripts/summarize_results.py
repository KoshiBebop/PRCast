#!/usr/bin/env python3
"""Build compact mean ± population-std tables from the seed-level results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


SUMMARY_FIELDS = (
    "dataset",
    "prediction_length",
    "n_seeds",
    "mse_mean",
    "mse_std",
    "mae_mean",
    "mae_std",
    "best_val_mse_mean",
    "best_val_mse_std",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], int(row["prediction_length"]))
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for dataset, horizon in sorted(groups):
        selected = groups[(dataset, horizon)]
        seeds = {int(row["seed"]) for row in selected}
        if len(selected) != len(seeds):
            raise SystemExit(f"duplicate seed in {dataset} h={horizon}")
        metrics = {
            name: [float(row[name]) for row in selected]
            for name in ("mse", "mae", "best_val_mse")
        }
        output.append(
            {
                "dataset": dataset,
                "prediction_length": horizon,
                "n_seeds": len(selected),
                **{
                    f"{name}_mean": statistics.mean(values)
                    for name, values in metrics.items()
                },
                **{
                    f"{name}_std": statistics.pstdev(values)
                    for name, values in metrics.items()
                },
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# PRCast main experiment results",
        "",
        "Each row is the mean ± population standard deviation over the three",
        "configured training seeds (2024, 2025, 2026) for one dataset and horizon.",
        "The seed-level source remains in `results.csv`.",
        "",
        "| Dataset | Horizon | Seeds | MSE | MAE | Best validation MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def pair(name: str) -> str:
            return f"{float(row[name + '_mean']):.6f} ± {float(row[name + '_std']):.6f}"

        lines.append(
            f"| {row['dataset']} | {row['prediction_length']} | "
            f"{row['n_seeds']} | {pair('mse')} | {pair('mae')} | {pair('best_val_mse')} |"
        )
    lines.extend(
        [
            "",
            "Standard deviations are population std across seeds, not across",
            "time points or validation folds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results.csv"))
    parser.add_argument("--csv", type=Path, default=Path("results_summary.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("RESULTS.md"))
    args = parser.parse_args()
    rows = summarize(read_rows(args.source))
    if not rows:
        raise SystemExit("no rows found")
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    print(f"OK summary_rows={len(rows)}")


if __name__ == "__main__":
    main()
