# Observability, Decision Transparency, and Artifact Compatibility

## Decision modes

Every inference result from the supervised engine carries an explicit decision mode.

### DecisionMode enum (`model_runtime/decision.py`)

| Value | Meaning |
|---|---|
| `model` | Model scored and ranked legal moves; returned highest-confidence choice |
| `fallback` | Model could not produce a usable output; first legal move returned |
| `error_recovered` | Unexpected exception; first legal move returned |

### FallbackReason enum

| Value | Meaning |
|---|---|
| `model_failure` | Feature encoding or forward pass raised an exception |
| `invalid_output` | Model logits could not be mapped to any legal move vocabulary index |
| `no_legal_moves` | Empty legal move list (programming error; Game Service should prevent this) |
| `runtime_exception` | Unhandled exception type not covered by the above |

### How to read a decision from logs

Every inference produces a JSON log line.  INFO for model decisions, WARNING for fallbacks:

```json
{"event": "inference_decision", "request_id": "abc-123", "model_version": "vrun_...",
 "decision_mode": "model", "fallback_reason": null, "confidence": 0.3421, "latency_ms": 12}

{"event": "inference_decision", "request_id": "abc-456", "model_version": "vrun_...",
 "decision_mode": "fallback", "fallback_reason": "model_failure", "confidence": null, "latency_ms": 3}
```

### Confidence interpretation

| confidence | Meaning |
|---|---|
| `0.0 – 1.0` | Softmax probability of selected move over legal moves (model decision) |
| `null` | Fallback decision — model did not produce this move |

---

## Counters and aggregate visibility

`SupervisedInferenceEngine.counters_snapshot()` returns the full breakdown:

```json
{
  "total_requests": 5000,
  "model_decision_count": 4920,
  "fallback_count": 75,
  "error_recovered_count": 5,
  "model_failure_count": 50,
  "invalid_output_count": 25,
  "runtime_exception_count": 5,
  "no_legal_moves_count": 0,
  "model_decision_rate": 0.984,
  "fallback_rate": 0.016,
  "invalid_output_rate": 0.005,
  "model_failure_rate": 0.01
}
```

`SupervisedInferenceEngine.serving_status()` returns counters plus model identity and candidate state.

### Automatic periodic snapshot log

A structured `counters_snapshot` log line is emitted automatically every
`_DecisionCounters.LOG_INTERVAL` requests (default 500).  This ensures aggregate
trends appear in logs without scanning individual inference lines:

```json
{"event": "counters_snapshot", "total_requests": 500, "fallback_rate": 0.006, ...}
```

Aggregate log search: `grep '"event": "counters_snapshot"'`

### Interpreting degradation signals

| Signal | Likely cause | Action |
|---|---|---|
| Rising `fallback_rate` | Any fallback trigger | Check `model_failure_rate` and `invalid_output_rate` first |
| Rising `model_failure_rate` | Bad FEN inputs or torch error | Check input FENs; check torch installation |
| Rising `invalid_output_rate` | Model weights corrupted or uninitialized | Reload artifact; check artifact integrity |
| `no_legal_moves_count > 0` | Bug in Game Service | Should never happen; investigate upstream |
| Stable `fallback_rate`, low `confidence` | Weak model | Expected for an untrained baseline; improve training data |

---

## Serving state and model lifecycle

`SupervisedInferenceEngine` delegates all model lifecycle to `ServingState`
(`model_runtime/serving_state.py`).

### At startup

```json
{"event": "supervised_engine_loaded", "model_version": "...", "artifact_id": "...", "encoder_version": "...", "move_encoder_version": "..."}
```

### Reload

```python
engine.reload()                                     # reload same artifact dir
engine.reload(new_artifact_dir=Path("/new/run/"))   # swap to new artifact atomically
```

Old model stays active until new one is fully loaded and validated.

### Candidate validation and promotion

```python
# Step 1: Validate — full load + all compatibility checks
status = engine.register_and_validate_candidate(Path("/artifacts/new_run/"))
# status: {"is_valid": True/False, "validation_error": null|"...", "metadata": {...}}

# Step 2: Inspect before promoting
if status["is_valid"]:
    engine.promote_candidate()    # atomic swap; old model released
else:
    print(f"Not promoting: {status['validation_error']}")
```

#### What validation checks
The candidate undergoes the same checks as any artifact at load time:
- Manifest completeness (all 11 required fields present)
- Encoder version exact match
- Encoder behavioral fingerprint match
- Model config completeness
- Weights load successfully

#### Promotion failure path
`promote_candidate()` raises `RuntimeError` if:
- No candidate was registered
- Candidate failed validation (`is_valid=False`)
- Candidate runtime was discarded (re-validate to recover)

An invalid candidate **cannot be promoted**.

---

## Artifact compatibility rules

An artifact is rejected at load time if any check fails:

| Check | Error type | When triggered |
|---|---|---|
| `manifest.json` absent | `FileNotFoundError` | Path wrong or partial write |
| Required manifest field missing | `ValueError` | Incomplete or old-format artifact |
| `move_encoder_config` absent | `ValueError` | Old fixed-vocabulary artifact — re-train required |
| `encoder_config.version` mismatch | `ValueError` | Position encoder version bumped without re-training |
| `encoder_config.fingerprint` mismatch | `ValueError` | `encode_fen()` changed without bumping version |
| `encoder_config.feature_size` mismatch | `ValueError` | Constants changed |
| `move_encoder_config.version` mismatch | `ValueError` | Move encoder version bumped without re-training |
| `move_encoder_config.fingerprint` mismatch | `ValueError` | `encode_move()` changed without bumping version |
| `move_encoder_config.move_feature_size` mismatch | `ValueError` | Constants changed |
| `model_config.architecture` != `MoveScoringNetwork` | `ValueError` | Old `PolicyNetwork` artifact — re-train required |
| `model_config` missing required keys | `ValueError` | Incomplete or old artifact |
| `model.pt` absent | `FileNotFoundError` | Partial write |

---

## Encoder versioning and drift protection

### Position encoder — two-layer protection

**Layer 1 — Version string (`ENCODER_VERSION` in `encoder.py`)**
Bumped when `encode_fen()` is intentionally changed.

**Layer 2 — Behavioral fingerprint (`compute_encoder_fingerprint`)**
SHA-256 of the sorted non-zero feature indices on the reference starting FEN.
Stored in every artifact's `encoder_config.fingerprint`. Re-verified at load time.
Catches drift even when the version string is not bumped.

### Move encoder — same two-layer protection

**Layer 1 — `MOVE_ENCODER_VERSION` in `move_encoder.py`**
Bumped when `encode_move()` is intentionally changed.

**Layer 2 — `compute_move_encoder_fingerprint`**
SHA-256 of the sorted non-zero indices on the reference move (`e2e4`).
Stored in every artifact's `move_encoder_config.fingerprint`. Re-verified at load time.

### How to update the position encoder intentionally

1. Change `encode_fen()`.
2. Run `python -m searchess_ai.model_runtime.encoder` to see the new fingerprint.
3. Update `_REFERENCE_NONZERO_INDICES` in `encoder.py`.
4. Bump `ENCODER_VERSION`.
5. Run `pytest tests/model_runtime/test_encoder_stability.py`.
6. Re-run dataset → train pipeline. All existing artifacts will be rejected until re-trained.

### How to update the move encoder intentionally

1. Change `encode_move()` in `move_encoder.py`.
2. Bump `MOVE_ENCODER_VERSION`.
3. Run `pytest tests/model_runtime/test_move_encoder.py`.
4. Re-run dataset → train pipeline. All existing artifacts will be rejected until re-trained.

---

## Dataset identity

Each `dataset_metadata.json` records:

| Field | Purpose |
|---|---|
| `dataset_id` | Unique ID for this preparation run (UUID hex) |
| `source_pgn_sha256` | Hash of the source PGN file |
| `extraction_version` | Version of the sample extraction code |
| `filter_config_hash` | SHA-256 of the serialized filter config |

Together, `source_pgn_sha256 + extraction_version + filter_config_hash` identify
the dataset content.  The artifact's `dataset_ref` section carries these fields,
tying every trained model back to its exact dataset.

---

## Evaluation provenance

`evaluation.json` in every artifact includes a `provenance` block:

```json
{"provenance": {"dataset_id": "...", "model_version": "...", "encoder_version": "..."}, ...}
```

This ties every metric to the exact (dataset, model, encoder) triple so comparisons
are always apples-to-apples.
