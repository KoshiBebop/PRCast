# PRCast

## Main experiment

PRCast runs a multivariate forecasting model with one fixed
architecture. The benchmark covers 11 datasets, four prediction horizons per
dataset, and seeds 2024, 2025, and 2026. Validation selects the checkpoint;
the test split is evaluated once per run. The [main results](RESULTS.md) and
[seed-level data](results.csv) are included in this repository.

## Quick start

Install the pinned dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Place the benchmark datasets under `data/` (or pass another directory with
`--data-root`). CSV datasets use their first column as timestamps; PEMS files
contain a `data` array. Check the model and preview the run:

```bash
uv run train.py --smoke-test
uv run train.py --data-root data --seeds 2024 2025 2026 --dry-run
```

Run the main experiment, or restrict it to one dataset for a quick trial:

```bash
uv run train.py --data-root data --seeds 2024 2025 2026 --resume
uv run train.py --data-root data --datasets ETTh2 --horizons 96 --seeds 2024 --resume
```

Use `--device cuda:0` for a CUDA GPU. Run outputs are written to `outputs/`.
