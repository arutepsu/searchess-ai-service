# Architecture Note — Supervised Training Baseline

## What was implemented

This milestone establishes the first end-to-end supervised training path inside the Python AI
service, enabling the full pipeline: PGN → dataset → train → evaluate → artifact → infer.

Six internal modules were added:

**`data_pipeline/`**
Ingests a local PGN file, applies a configurable filtering policy, extracts position-level
training samples (FEN + played UCI + legal moves), and writes a reproducible parquet dataset
with a JSON metadata file tracking provenance, filter config, and rejection breakdowns.

**`training/`**
Loads the prepared dataset, performs game-level train/val/test splits (not sample-level, to
prevent data leakage), trains a baseline MLP policy network using masked cross-entropy loss,
restores the best checkpoint, evaluates on the held-out test set, and saves a fully versioned
artifact directory.

**`model_runtime/`**
Contains the shared encoding logic (`encoder.py`) used by both training and inference — this is
the critical consistency constraint. Also contains the artifact loader and `SupervisedModelRuntime`
which scores and selects among legal moves at inference time.

**`evaluation/`**
Computes offline test metrics (top-1 accuracy, top-5 accuracy, average loss, legal-move
selection sanity check) and returns them for embedding in the artifact.

**`artifacts/`**
Defines the `ArtifactManifest` dataclass and save/load contracts. An artifact is a
self-contained directory: weights + model config + encoder config + dataset provenance +
training run metadata + evaluation summary.

**`infrastructure/inference/supervised_inference_engine.py`**
Wires the trained model into the existing `InferenceEngine` port. Active when
`INFERENCE_BACKEND=supervised` and `MODEL_ARTIFACT_DIR=<path>` are set.

---

## Key tradeoffs

**Game-level splits, not sample-level**
Samples from the same game are kept together in one split. This is more expensive (fewer
effective training samples) but gives honest evaluation numbers — a sample-level random split
would let the model memorise game continuations and report inflated accuracy.

**Fixed move vocabulary (4096 indices)**
Moves are indexed by (from_square, to_sq) only; promotion piece is dropped. This keeps the
output layer fixed-size and simple at the cost of losing under-promotion specificity. At
inference, queen promotion is assumed. This is acceptable for a baseline; a future iteration can
add promotion planes (+64 indices) or a separate promotion head.

**Shared encoder, isolated from the API layer**
`model_runtime/encoder.py` is the single source of truth for feature encoding. The training
dataset module and the runtime module both import from there. This enforces consistency without
runtime versioning machinery. If the encoder is ever changed, the version string in
`encoder_config.json` will surface mismatches at artifact load time.

**Optional training extras**
`torch`, `python-chess`, `pandas`, and `pyarrow` are under `[project.optional-dependencies]`
`training`. The inference service can be deployed without training dependencies when using the
`fake` or `random` backends. The supervised engine handles `ImportError` gracefully so the
service starts even if extras are absent (it will fail at first inference call with a clear error).

**MLP, not a transformer or CNN**
The baseline model is the simplest architecture that can train, evaluate, and serve in one
session. Chess strength is explicitly not a goal of this milestone. The architecture is trivially
replaceable: swap out `training/model.py` and re-run the pipeline. The artifact format and all
surrounding machinery are independent of model architecture.

**No automatic retraining**
Dataset build, training, and serving are distinct operational modes invoked as CLI scripts.
The service does not train models on startup. This is intentional: training is a batch operation
that should be triggered explicitly with known inputs.

---

## What remains intentionally out of scope

- **Reinforcement learning / self-play** — no RL loop, MCTS, or OpenSpiel self-play
- **Distributed training** — single-process CPU/GPU only
- **Model registry / platform** — artifacts are filesystem directories; no registry server
- **Automatic retraining** — no triggers, schedulers, or online learning
- **Chess-strength optimisation** — architecture tuning, opening books, endgame tablebases
- **Promotion handling** — under-promotions default to queen; acceptable for baseline
- **Streaming dataset** — full parquet is loaded into memory; fine for moderate dataset sizes

---

## Future extension path

The module boundaries are designed to accommodate later phases cleanly:

- **Stronger model**: replace `training/model.py` with a CNN or transformer; everything else
  stays the same
- **RL integration**: add `rl_training/` beside `training/`; share `model_runtime/encoder.py`
- **Promotion planes**: extend `encoder.py` MOVE_VOCAB_SIZE to 4096+64 and bump encoder version
- **Model registry**: replace filesystem artifact paths with registry URLs in `artifacts/`
- **Streaming PGN**: swap `writer.py` to write incrementally; no API changes needed
