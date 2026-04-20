# Integration Test Hook — `metadata.testMode`

## Purpose

The `testMode` field lets Game Service integration tests provoke controlled
bad-provider responses so they can verify that the Scala side correctly rejects
invalid AI output.

This is **not** part of the production inference contract. It exists solely to
exercise Game Service defensive code paths without mocking the entire AI service.

## Usage

Include the optional `metadata` object in the standard `POST /v1/move-suggestions`
request and set `testMode` to one of the supported values:

```json
{
  "requestId": "...",
  ...
  "metadata": {
    "testMode": "illegal_move"
  }
}
```

## Supported values

| `testMode` | HTTP status | What it returns |
|---|---|---|
| `"illegal_move"` | 200 | Structurally valid success body. The `move` is **not** present in the request's `legalMoves`. Game Service must reject it at legality validation. |
| `"malformed_response"` | 200 | Valid JSON, but the required `move` field is **absent**. Game Service decoder must treat this as a malformed-provider-response error. |

Any other value (or absent `metadata`) leaves the normal inference path completely
unchanged.

## What it does NOT do

- It does not affect inference engine selection, confidence, reconciliation, or timeout.
- It does not return error status codes — both modes return 200 so the failure
  is detectable only at the application/contract level on the Scala side.
- It does not validate that the returned illegal move is a board-legal move in
  the given position — only that it is outside the provided `legalMoves` list.

## Implementation

Hook logic lives entirely in `src/searchess_ai/api/routes/_test_hook.py`.
The route checks `dto.metadata.test_mode` in two explicit `if` statements before
the normal inference executor is invoked. No other module is affected.
