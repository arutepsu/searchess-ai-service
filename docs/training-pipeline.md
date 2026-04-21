# Training Pipeline — Developer Guide

## Overview

The Python AI service is split into six internal modules, each with a single responsibility.
Training, evaluation, and serving are distinct operating modes — not co-located at runtime.

```
data_pipeline/    ingest PGN → filter games → extract positions → write parquet + metadata
training/         load dataset → split → train policy network → save versioned artifact
model_runtime/    load artifact → encode FEN/moves → score & select legal moves
evaluation/       compute offline metrics (top-1, top-5, legal-rate) → persist in artifact
artifacts/        versioned artifact structure: manifest + weights + all metadata
inference_api/    (existing api/ layer) request validation → runtime call → response
```

---

## Prerequisites

Install training extras:

```bash
uv sync --extra training
```

This adds: `python-chess`, `torch`, `numpy`, `pandas`, `pyarrow`.

---

## Step 1 — Prepare a dataset

```bash
uv run prepare-dataset \
  --pgn /path/to/games.pgn \
  --output-dir /data/datasets/baseline_v1 \
  --name baseline_v1 \
  --min-ply 10 \
  --max-games 50000
```

**Optional flags:**
- `--min-rating 1500` — skip games where either player is below 1500 ELO
- `--no-legal-moves` — skip legal-move computation (faster; disables masked training)
- `--seed 42` — reproducibility seed

**Outputs** in `--output-dir`:
- `prepared_samples.parquet` — one row per training position
- `dataset_metadata.json` — source file, SHA-256, filter config, rejection breakdown, counts

**Filtering policy** (configurable via CLI flags, recorded in metadata):
- Skip chess variants (Variant header present and non-standard)
- Skip games with non-standard starting positions (SetUp=1 + FEN header)
- Skip games without a result (1-0 / 0-1 / 1/2-1/2)
- Skip games shorter than `--min-ply` half-moves
- Optionally filter by minimum player rating

---

## Step 2 — Train a model

```bash
uv run train-model \
  --dataset-dir /data/datasets/baseline_v1 \
  --output-dir /data/artifacts \
  --epochs 10 \
  --batch-size 256 \
  --device cpu
```

**Optional flags:**
- `--hidden-size 512` — neurons per hidden layer
- `--num-hidden-layers 2` — depth of network
- `--learning-rate 1e-3`
- `--device cuda` — if GPU available
- `--seed 42`

**What happens:**
1. Loads `prepared_samples.parquet` from `--dataset-dir`
2. Splits games (not samples) 80/10/10 into train/val/test to prevent data leakage
3. Trains a `PolicyNetwork` (MLP) with masked cross-entropy loss over legal moves
4. Best checkpoint (lowest val loss) is restored before final evaluation
5. Computes offline metrics on the held-out test set
6. Saves a versioned artifact under `<output-dir>/<run_id>/`

**Artifact directory layout:**
```
<output-dir>/<run_id>/
  manifest.json        top-level manifest with all provenance
  model.pt             PyTorch state dict (best checkpoint)
  model_config.json    architecture parameters
  encoder_config.json  feature encoding parameters
  dataset_ref.json     dataset version and source info
  training_run.json    all hyperparameters + run ID
  evaluation.json      offline test metrics
```

---

## Step 3 — Inspect offline metrics

```bash
cat /data/artifacts/<run_id>/evaluation.json
```

Fields:
- `top1_accuracy` — fraction of test positions where model's best move = played move
- `top5_accuracy` — fraction where played move is in top-5 model predictions
- `avg_loss` — average masked cross-entropy on test set
- `legal_move_selection_rate` — should always be 1.0 (sanity check for masking)
- `dataset_sizes` — train/val/test sample counts

---

## Step 4 — Run inference with the trained model

Set environment variables and start the service:

```bash
INFERENCE_BACKEND=supervised \
MODEL_ARTIFACT_DIR=/data/artifacts/<run_id> \
uv run uvicorn searchess_ai.main:app --host 0.0.0.0 --port 8765
```

Or via Docker:
```dockerfile
ENV INFERENCE_BACKEND=supervised
ENV MODEL_ARTIFACT_DIR=/artifacts/<run_id>
```

The service will load the artifact at startup. If torch or the artifact is missing, it logs the
error and surfaces a runtime failure when inference is first called (service still starts).

---

## Dataset format

Each row in `prepared_samples.parquet`:

| Column | Type | Description |
|---|---|---|
| sample_id | string | `{game_id}_{ply_index}` |
| source_game_id | string | stable game identifier within dataset |
| position_fen | string | FEN before the move |
| side_to_move | string | "white" or "black" |
| played_move_uci | string | UCI of the move played |
| legal_moves_uci | string? | JSON array of legal UCIs, or null |
| ply_index | int32 | half-move number within the game |
| game_result | string | "1-0", "0-1", "1/2-1/2", or "*" |
| white_rating | int? | ELO if available |
| black_rating | int? | ELO if available |
| opening | string? | Opening name or ECO code |
| time_control | string? | From PGN header |
| termination | string? | From PGN header |

---

## Feature encoding

The shared encoder lives in `model_runtime/encoder.py`. Both training and inference import from
there — this is the critical consistency constraint.

- **Board features** (768): 12 piece-type planes × 64 squares
  - Plane order: white/black for each of PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
- **Castling rights** (4): WK, WQ, BK, BQ at indices 768–771
- **Side to move** (1): 1.0=white, 0.0=black at index 772
- **Total**: 773 float32 features

Move vocabulary: `from_square * 64 + to_square` → index in [0, 4095].
Promotion piece is **not** encoded. All promotions between the same square pair share one
vocabulary index. At inference the runtime prefers queen promotion. This is a known baseline
limitation to be addressed in a later iteration.

---

## Module responsibilities summary

| Module | Owns |
|---|---|
| `data_pipeline` | PGN ingestion, game filtering, sample extraction, dataset write |
| `training` | Dataset loading, splits, training loop, artifact saving |
| `model_runtime` | Shared encoding, artifact loading, move scoring and selection |
| `evaluation` | Offline metrics computation, report serialisation |
| `artifacts` | Manifest schema, save/load contracts |
| `inference_api` (existing `api/`) | HTTP validation, backend dispatch, response serialisation |

---

## Reproducibility checklist

To reproduce a training run:
1. `dataset_metadata.json` → identifies source PGN SHA-256 and filter config
2. `training_run.json` → all hyperparameters and random seed
3. `encoder_config.json` → encoding version and parameters
4. `manifest.json` → ties everything together with artifact_id

Two runs with the same PGN, filter config, and training config will produce
statistically equivalent models (subject to hardware floating-point determinism).
