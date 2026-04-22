# Lichess Puzzle Dataset Path

## What this is

A fast, self-contained path for producing a real trained artifact from the
Lichess puzzle CSV (`lichess_db_puzzle.csv`).  It is completely separate from
the PGN game-ingestion pipeline.  The output is identical in format — a
parquet + metadata pair — so the existing `train-model` CLI trains on it
without modification.

## What it is good for

- **Pipeline validation**: proving that dataset → training → artifact → inference
  works end-to-end with real chess data.
- **Fast iteration**: 20 k puzzles produce a dataset in seconds; training on CPU
  finishes in minutes.
- **Legal move coverage**: each puzzle position is mid-game with many legal moves,
  which exercises the move-encoder path thoroughly.

## What it is NOT good for

- **Gameplay training**: puzzles are tactical one-liners, not balanced positions.
  A model trained only on puzzles will bias heavily toward forcing moves and will
  play poorly in quiet positions.
- **Evaluating Elo strength**: puzzle accuracy (did the model find the tactic?)
  does not translate directly to game strength.
- **Replacing a full game corpus**: use this as a bootstrap only.

---

## How the puzzle CSV maps to training samples

```
Lichess CSV column layout:
  PuzzleId   FEN   Moves   Rating   Themes   …

FEN    = board position BEFORE the opponent's tactical setup move
Moves  = space-separated UCI moves
           moves[0] = opponent's forced move  ← applied to reach puzzle position
           moves[1] = first solution move     ← training target
           moves[2+]= continuation            ← ignored this milestone
```

**One TrainingSample per puzzle row:**

| TrainingSample field | Source |
|---|---|
| `position_fen` | FEN after applying `moves[0]` |
| `played_move_uci` | `moves[1]` (first solution move) |
| `legal_moves_uci` | Computed fresh with python-chess |
| `source_game_id` | `puzzle_<PuzzleId>` (each puzzle is its own "game") |
| `white_rating` | Puzzle `Rating` (Elo difficulty) |
| `opening` | Puzzle `Themes` (e.g. "fork pin middlegame") |
| `game_result` | `"*"` (not applicable for puzzles) |
| `ply_index` | `0` (single-step dataset) |

---

## Running the puzzle pipeline locally

### Step 1 — Prepare the dataset

```bash
uv run prepare-puzzle-dataset \
    --csv train_data/lichess_db_puzzle.csv \
    --output-dir datasets/puzzles_20k \
    --name lichess_puzzles_20k \
    --sample-size 20000 \
    --seed 42
```

This writes:
```
datasets/puzzles_20k/
├── prepared_samples.parquet
└── dataset_metadata.json
```

### Step 2 — Train the model

```bash
uv run train-model \
    --dataset-dir datasets/puzzles_20k \
    --output-dir artifacts \
    --epochs 5 \
    --batch-size 256 \
    --hidden-size 256 \
    --num-hidden-layers 2 \
    --dropout 0.2 \
    --learning-rate 1e-3 \
    --device cpu \
    --seed 42
```

**Baseline training config rationale:**
- `epochs=5`: fast feedback loop; val loss curve is informative after 3 epochs
- `hidden-size=256`: half the production default; trains in minutes on CPU
- `batch-size=256`: default; matches the PGN pipeline config
- `device=cpu`: no CUDA required for this milestone; swap to `cuda` if available

The artifact is written to `artifacts/run_<timestamp>_<hash>/`.

### Step 3 — Sanity-check inference

```bash
uv run python scripts/sanity_check_puzzle_inference.py \
    --artifact-dir artifacts/run_<id> \
    --dataset-dir  datasets/puzzles_20k \
    --n-positions  10
```

Expected output (all checks green):
```
=== 1. Artifact loading ===
  [PASS] load_artifact() succeeded
  [PASS] model_version present  (v...)
  [PASS] artifact_id present  (a1b2c3d4...)
  ...

=== 3. Inference on 10 positions ===
  [PASS] all decisions are MODEL mode  (10 model / 0 fallback)
  [PASS] no illegal moves returned  (0 issues)
  [PASS] confidence in (0, 1]  (avg=0.312)

=== 4. Decision transparency (spot check) ===
  FEN           : ...
  Selected      : e2e4
  Decision mode : model
  Confidence    : 0.312
  Fallback reason: None

========================================
[PASS] All checks passed.
========================================
```

### Step 4 — Serve with the supervised backend (optional)

```bash
INFERENCE_BACKEND=supervised \
MODEL_ARTIFACT_DIR=artifacts/run_<id> \
uv run uvicorn searchess_ai.main:app --port 8000
```

---

## Metadata recorded per dataset

`dataset_metadata.json` includes:

```json
{
  "dataset_id":              "<uuid hex>",
  "dataset_name":            "lichess_puzzles_20k",
  "dataset_type":            "lichess_puzzles",
  "schema_version":          "1.0",
  "created_at":              "2026-04-22T…",
  "source_csv":              "train_data/lichess_db_puzzle.csv",
  "source_csv_sha256":       "<sha256>",
  "puzzle_adapter_version":  "1.0",
  "extraction_version":      "1.0",
  "filter_config_hash":      "<16-char hash of sampling config>",
  "pipeline_config": {
    "source_csv":   "…",
    "sample_size":  20000,
    "random_seed":  42
  },
  "total_rows_in_source":    3149835,
  "rows_sampled":            20000,
  "rows_accepted":           19983,
  "rows_rejected":           17,
  "rejection_reasons": {
    "too_few_moves":        0,
    "invalid_fen":          0,
    "illegal_setup_move":   12,
    "illegal_target_move":  5
  }
}
```

The artifact's `dataset_ref` block is populated from these fields, giving
complete lineage in every trained model.

---

## Limitations of this dataset path

1. **Tactical bias** — puzzles are non-representative of normal game play.
   The model learns to prefer forcing moves even in quiet positions.

2. **Single-ply supervision** — only `moves[1]` is used.  The model does not
   learn continuations; it sees each position once with one labeled move.

3. **No game context** — puzzles lack result (`1-0`, `0-1`) and game-level
   metadata.  Features that depend on game outcome are not exercisable.

4. **Game-level split degenerates** — each puzzle is its own "game" so the
   train/val/test split is effectively per-sample random (no leakage risk, but
   also no game-coherence benefit).

5. **Not a replacement for a game corpus** — transition to a full game dataset
   (PGN pipeline) for any model intended for actual gameplay.
