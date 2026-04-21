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
{"event": "supervised_engine_loaded", "model_version": "...", "artifact_id": "...", "encoder_version": "..."}
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
| Required manifest field missing | `ValueError` | Incomplete artifact |
| `encoder_config.version` mismatch | `ValueError` | Encoder version bumped without re-training |
| `encoder_config.fingerprint` mismatch | `ValueError` | `encode_fen()` changed without bumping version |
| `encoder_config.feature_size` mismatch | `ValueError` | Constants changed |
| `encoder_config.move_vocab_size` mismatch | `ValueError` | Constants changed |
| `model_config` missing required keys | `ValueError` | Incomplete or old artifact |
| `model.pt` absent | `FileNotFoundError` | Partial write |

---

## Encoder versioning and drift protection

### Two-layer protection

**Layer 1 — Version string (`ENCODER_VERSION`)**
A manual semver-style tag bumped when the encoder is intentionally changed.
Catches intentional changes but is only as reliable as the developer remembering to bump it.

**Layer 2 — Behavioral fingerprint (`compute_encoder_fingerprint`)**
A SHA-256 of the sorted non-zero feature indices when encoding the reference starting position.
Computed at training time and stored in every artifact's `encoder_config.fingerprint`.
Re-computed at artifact load time and compared.

If `encode_fen()` changes but `ENCODER_VERSION` is not bumped, the fingerprint check still
catches the drift at load time — before any wrong predictions can be made.

### Automated test coverage

`tests/model_runtime/test_encoder_stability.py` contains:

| Test | What it catches |
|---|---|
| `test_verify_encoder_stable_passes` | Any deviation from `_REFERENCE_NONZERO_INDICES` |
| `test_reference_nonzero_indices_match_encode_fen` | Specific added/removed indices (human-readable diff) |
| `test_nonzero_count_for_starting_position` | Wrong number of set features (layout shift) |
| `test_feature_values_are_binary` | Non-binary feature values introduced |
| `test_fingerprint_is_deterministic` | Nondeterminism in encoder |
| `test_different_positions_have_different_fingerprints` | Collapsed encoding |
| `TestEncoderPiecePlanes.*` | Wrong plane assignment per piece type/color |
| `test_side_to_move_index_772` | Side-to-move bit at wrong index or inverted |

### How to update the encoder intentionally

1. Make the change to `encode_fen()`.
2. Run `python -m searchess_ai.model_runtime.encoder` to see the new fingerprint and verify the stability check output.
3. Update `_REFERENCE_NONZERO_INDICES` in `encoder.py` to match the new output.
4. Bump `ENCODER_VERSION` (e.g. `"1.0"` → `"2.0"`).
5. Run `pytest tests/model_runtime/test_encoder_stability.py` — all tests must pass.
6. Re-run the full dataset → train pipeline.  All existing artifacts will be rejected until re-trained.

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
