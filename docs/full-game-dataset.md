# Small Full-Game Dataset Path

This path builds a small supervised dataset from local standard-chess PGN games.
It is meant for fast feedback on full-game behavior, not full archive-scale PGN
processing.

The builder streams games lazily from a local PGN file, extracts
position-to-played-move samples from mainline plies, computes legal moves for
each position, and stops once `max_samples` or `max_games` is reached.

## What It Filters

The default run:

- accepts standard chess only
- requires a normal game result
- skips games shorter than `--min-game-ply`
- starts sampling positions at `--min-sample-ply`
- optionally filters by `--min-rating`
- skips malformed games safely

The default target is `20,000` samples because the first goal is a quick
baseline experiment, not broad coverage.

## Build A Dataset

Pass a local PGN path explicitly. Plain `.pgn`, `.pgn.gz`, and `.pgn.bz2`
are supported. If your Lichess archive is `.pgn.zst`, decompress it first
rather than adding archive-management complexity to this small experiment.

```powershell
uv run prepare-full-game-dataset `
  --pgn C:\path\to\lichess_standard_sample.pgn `
  --output-dir datasets\full_games_20k `
  --name lichess_full_games_20k `
  --max-games 2000 `
  --max-samples 20000 `
  --min-game-ply 20 `
  --min-sample-ply 8 `
  --seed 42
```

The output uses the same dataset files as the existing trainer:

- `prepared_samples.parquet`
- `dataset_metadata.json`

## Train A Baseline

Use the existing move-scoring training pipeline:

```powershell
uv run train-model `
  --dataset-dir datasets\full_games_20k `
  --output-dir artifacts `
  --epochs 5 `
  --batch-size 256 `
  --hidden-size 256 `
  --num-hidden-layers 2 `
  --dropout 0.2 `
  --seed 42
```

This intentionally mirrors the first puzzle-trained baseline so the comparison
is mostly about training signal rather than architecture or tuning.

## Evaluate

Use the existing playing-strength harness:

```powershell
uv run evaluate-trained-model `
  --artifact-dir artifacts\<full-game-run-id> `
  --games 20 `
  --max-plies 300 `
  --seed 42
```

Compare against the puzzle-trained artifact using the same evaluation settings.

## Interpretation

A better full-game result would mean this supervised signal is more aligned with
complete-game move selection than puzzle-only labels. It still would not prove
general chess strength: random is a trivial baseline, the dataset is small, and
there is no engine or outcome-value target.
