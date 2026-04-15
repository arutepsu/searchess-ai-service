# OpenSpiel Notation Mapping

## Notation Authority

The **Scala searchess platform** is the authority for:

- FEN position semantics
- Move notation (UCI strings)
- Which moves are legal in any given position

The Python AI service **consumes** notation produced by the platform. It does
not redefine, extend, or validate the semantics of FEN or UCI moves. The
mapping module (`_openspiel_mapping.py`) makes this boundary explicit with
named constants and an explicit reconciliation policy.

## Position Contract: FEN (`POSITION_FORMAT = "FEN"`)

Positions arrive at the AI service boundary as FEN strings inside
`InferenceRequest.position.value`. FEN is the canonical format for
communicating chess positions between the Scala platform and this service.

The AI service:

- Accepts FEN from the platform request unchanged
- Performs a lightweight structural check (`validate_fen`) before passing to
  OpenSpiel — checking that the string has 7 rank separators and a valid
  side-to-move field
- Does **not** validate FEN chess-legality; that is the platform's
  responsibility

The structural check exists to surface clearly malformed input early, before a
cryptic native-extension error reaches the caller.

## Move Contract: UCI (`MOVE_FORMAT = "UCI"`)

Move strings at the boundary use UCI notation, e.g. `"e2e4"`, `"g1f3"`,
`"e7e8q"`. OpenSpiel's `action_to_string()` also produces UCI notation, so in
the common case the formats are identical.

Known divergence points:

| Case | Platform example | OpenSpiel example |
|------|-----------------|-------------------|
| Promotion | `e7e8=Q` | `e7e8q` |
| Castling | platform-specific | `e1g1` |

The reconciliation policy handles these mismatches safely.

## Platform Legal Moves Are Authoritative

`InferenceRequest.legal_moves` carries the platform-supplied `LegalMoveSet`.
**The AI service must never return a move outside this set.** OpenSpiel is
used to inform ordering among the platform-provided legal moves, not to widen
or override them.

## Reconciliation Policy

When OpenSpiel move strings and platform move strings are compared,
disagreement is handled by an explicit `ReconciliationPolicy` (defined in
`_openspiel_mapping.py`).

### Available Policies

| Policy | Behaviour on empty intersection |
|--------|---------------------------------|
| `intersect_then_platform_fallback` | Fall back to `platform_moves[0]`. Never raises. |
| `intersect_then_fail` | Raise `OpenSpielAdapterError`. Surfaces mismatches immediately. |
| `strict_fail` | Also raise if any OpenSpiel move is absent from the platform set. For notation alignment testing. |

### Current Default: `intersect_then_platform_fallback`

**Why:** The prototype needs to remain robust while notation alignment between
OpenSpiel and the Scala platform is still being hardened. The fallback ensures
the service always returns a valid move from the platform-authoritative set,
even in the presence of notation divergence.

**When fallback occurs** (`MoveReconciliationResult.fallback_used == True`):
OpenSpiel and platform move strings share no common elements. The service
returns `platform_moves[0]` — the first move from the platform-supplied list,
which is always a valid legal move.

Expected fallback triggers during the prototype phase:

1. Promotion notation mismatch (`e7e8q` vs `e7e8=Q`)
2. Special move encoding differences (castling)
3. Platform request contains a restricted subset that happens not to overlap
   with OpenSpiel's ordering

### Upgrading the Policy

When notation alignment between the Scala platform and OpenSpiel is confirmed,
switch to `intersect_then_fail` to surface any remaining divergence as an
explicit error rather than a silent fallback.

```python
# In api/dependencies.py or via environment configuration:
OpenSpielInferenceEngine(
    reconciliation_policy=ReconciliationPolicy.INTERSECT_THEN_FAIL
)
```

## Architecture Boundary

```
Scala searchess platform
  │  FEN position + UCI legal moves  (authoritative)
  ▼
Python AI service boundary
  │  validate_fen()      — structural guard only; no chess-legality check
  │  reconcile_moves()   — explicit named policy
  ▼
OpenSpiel chess  (infrastructure/inference/ only)
  │  new_initial_state(fen)          → game state
  │  action_to_string(player, action) → UCI move strings
  ▼
selected_move ∈ platform legal moves  (invariant, all policies)
```

OpenSpiel code lives exclusively in
`src/searchess_ai/infrastructure/inference/`. Domain, application, and API
layers have no OpenSpiel imports.

## Files

| File | Responsibility |
|------|---------------|
| `_openspiel_mapping.py` | Boundary constants, `validate_fen`, `ReconciliationPolicy`, `reconcile_moves` |
| `_openspiel_chess.py` | OpenSpiel state operations (`load_chess_state_from_fen`, `get_legal_move_strings`) |
| `openspiel_inference_engine.py` | `InferenceEngine` implementation; wires mapping + pyspiel |
| `tests/infrastructure/test_openspiel_mapping.py` | Unit tests for mapping policy |
| `tests/infrastructure/test_openspiel_inference_engine.py` | Adapter wiring tests (pyspiel mocked) + optional integration |
