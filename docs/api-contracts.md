# API Contracts of searchess-ai-service

## Purpose

This document defines the external service contracts exposed by `searchess-ai-service`.

These contracts are intended for integration with the main Searchess platform.

Key principles:

- contracts must be stable
- contracts must not expose OpenSpiel-specific types
- contracts must remain usable across multiple model backends
- contracts should support future growth without leaking internal implementation details

---

## v1 Endpoint Surface

- GET /api/v1/health
- POST /api/v1/inference/move
- GET /api/v1/models
- GET /api/v1/models/{modelId}
- POST /api/v1/training/jobs
- GET /api/v1/training/jobs/{trainingJobId}
- POST /api/v1/evaluation/jobs
- GET /api/v1/evaluation/jobs/{evaluationJobId}

---

## Contract Design Principles

### 1. Platform-facing, not library-facing
The API must speak in terms of:

- board state
- legal moves
- model identity
- training jobs
- evaluation requests

It must NOT expose:

- OpenSpiel action ids
- internal simulator state objects
- framework-specific model structures

### 2. Explicit versionability
Contracts should be designed so they can evolve over time without breaking consumers.

### 3. Separation of concerns
Inference, training, and evaluation are distinct capabilities and should remain separate at the contract level.

### 4. Source of truth rule
The Searchess platform remains authoritative for official gameplay state and move legality.

The AI service provides decisions, not official rule enforcement.

---

## API Areas

The service exposes four initial contract areas:

1. Health and service status
2. Move inference
3. Model discovery
4. Training and evaluation control

---

# 1. Health and Service Status

## Purpose
Allows the platform or operators to verify that the AI service is running and able to serve requests.

## Example responsibilities
- liveness check
- readiness check
- service version visibility

## Response shape
Should include:

- service status
- service version
- current environment
- optionally active model summary

## Notes
This endpoint is operational, not domain-specific.

---

# 2. Move Inference Contract

## Purpose
Allows the platform to ask the AI service to choose a move for a given position.

This is the most important integration contract.

## Request

The inference request should contain:

### Match Context
- `matchId`
  - unique identifier of the match in the platform
- `gameId`
  - unique game identifier if separated from match identity
- `requestId`
  - correlation id for tracing and idempotency if needed

### Position Context
- `boardState`
  - canonical serialized board representation
  - should use a platform-defined stable format
  - FEN is a strong candidate
- `sideToMove`
  - white or black
- `legalMoves`
  - full list of legal moves allowed in the current position
  - represented in a stable move notation chosen by the platform
- `moveHistory`
  - optional compact move history for context if needed later

### Time / Decision Context
- `timeControl`
  - optional summary of match timing rules
- `remainingTimeMillis`
  - optional remaining decision time for the acting side
- `moveDeadline`
  - optional absolute or relative deadline
- `decisionMode`
  - optional policy selection hint such as:
    - safe
    - fast
    - balanced
    - experimental

### Model / Policy Context
- `modelId`
  - optional explicit requested model
- `modelVersion`
  - optional explicit requested version
- `policyProfile`
  - optional named behavior profile

## Response

The inference response should contain:

### Decision
- `selectedMove`
  - chosen move from the provided legal move list
- `decisionType`
  - for example:
    - move
    - resign
    - no_decision
  - initially, `move` is enough; other types can be added later

### Model Metadata
- `modelId`
- `modelVersion`
- `policyProfile`
  - if used

### Trace / Diagnostics
- `requestId`
- `decisionTimeMillis`
- `confidence`
  - optional
- `reasoningMetadata`
  - optional structured metadata
  - should remain machine-oriented, not freeform explanations

## Validation Rules

The platform should assume:

- the AI service may only return a move from the supplied `legalMoves`
- if the returned move is not in `legalMoves`, the platform rejects it
- the platform remains the final validator of official move application

## Important Design Rule

The AI service must not require platform callers to understand internal search, RL, or OpenSpiel concepts.

---

# 3. Model Discovery Contract

## Purpose
Allows the platform or operators to inspect available models and choose which one to use.

## Use cases
- select active model for inference
- inspect candidate models
- compare versions
- show operational metadata

## Model Summary Response

Each model summary should include:

- `modelId`
- `modelVersion`
- `status`
  - draft
  - candidate
  - approved
  - active
  - retired
- `createdAt`
- `description`
- `trainingRunId`
  - optional
- `tags`
  - optional
- `capabilities`
  - optional, such as:
    - inference
    - self_play
    - tournament_ready

## Single Model Detail
A more detailed model view may include:

- summary fields
- evaluation metrics
- supported policy profiles
- compatibility information
- artifact metadata
- rollout notes

## Important Rule

Model identity must be explicit and stable enough for:
- debugging
- reproducibility
- rollback
- tournament pinning

---

# 4. Training Control Contract

## Purpose
Allows the platform or operators to trigger or inspect training activity.

This should be treated as an internal or operator-facing capability, not necessarily a public client-facing API.

## Training Job Start Request

Should include:

- `trainingJobId`
  - optional if generated by service
- `trainingProfile`
  - named config profile or preset
- `baseModelId`
  - optional
- `baseModelVersion`
  - optional
- `selfPlayConfig`
  - structured settings for self-play generation
- `rewardConfig`
  - structured reward settings
- `maxIterations`
  - optional
- `notes`
  - optional

## Training Job Response

Should include:

- `trainingJobId`
- `status`
  - queued
  - running
  - completed
  - failed
  - cancelled
- `submittedAt`
- `estimatedOutputModel`
  - optional

## Training Job Status Response

Should include:

- `trainingJobId`
- `status`
- `startedAt`
- `updatedAt`
- `progress`
  - optional
- `producedModelVersion`
  - optional
- `failureReason`
  - optional

## Important Rule

Training contracts should describe orchestration intent, not low-level ML internals.

---

# 5. Evaluation Contract

## Purpose
Allows the service to evaluate one model against another.

This is important for promotion, regression testing, and AI-vs-AI comparison.

## Evaluation Request

Should include:

- `evaluationJobId`
  - optional if service-generated
- `candidateModelId`
- `candidateModelVersion`
- `baselineModelId`
- `baselineModelVersion`
- `numberOfGames`
- `startingConditions`
  - optional
- `timeControl`
  - optional
- `policyProfiles`
  - optional
- `notes`
  - optional

## Evaluation Response

Should include:

- `evaluationJobId`
- `status`
- `submittedAt`

## Evaluation Result

Should include:

- `evaluationJobId`
- `candidateModelId`
- `candidateModelVersion`
- `baselineModelId`
- `baselineModelVersion`
- `gamesPlayed`
- `candidateWins`
- `baselineWins`
- `draws`
- `scoreSummary`
- `status`
- `completedAt`

## Optional Future Metrics

Later, this may include:
- average move latency
- confidence calibration
- opening diversity
- error rates
- tournament-readiness indicators

---

# 6. Error Contract

## Purpose
Provides consistent error handling across all endpoints.

## Error Response Fields

Should include:

- `errorCode`
- `message`
- `requestId`
- `details`
  - optional structured details

## Error Categories

Initial categories may include:

- `INVALID_REQUEST`
- `MODEL_NOT_FOUND`
- `MODEL_NOT_ACTIVE`
- `UNSUPPORTED_POLICY_PROFILE`
- `TRAINING_JOB_NOT_FOUND`
- `EVALUATION_JOB_NOT_FOUND`
- `SERVICE_UNAVAILABLE`
- `INTERNAL_ERROR`

## Important Rule

Errors should be structured and machine-readable.

Avoid relying on freeform text for program flow.

---

# 7. Serialization Rules

## Board State
A single canonical board serialization must be chosen for service boundaries.

Recommendation:
- use a stable chess notation such as FEN for board position
- keep move representation consistent with platform conventions

## Moves
Move representation must be:
- explicit
- stable
- unambiguous
- validated by the platform

The AI service must return moves in the same format used in `legalMoves`.

## Timestamps
Use a single consistent timestamp format across all endpoints.

## Identifiers
All identifiers should be opaque strings from the caller’s point of view.

---

# 8. Versioning Strategy

The contracts should support evolution without tight coupling.

Recommended approach:
- begin with a versioned API namespace or explicit API version policy
- avoid breaking field renames without version bumps
- add optional fields in a backward-compatible way where possible

---

# 9. Security and Trust Assumptions

## Trust Boundary
The AI service is trusted as an internal system component, but its outputs are still validated by the platform.

## Important Rule
The platform must never blindly accept AI output as authoritative game truth.

The platform must always:
- verify move membership in `legalMoves`
- enforce official timing and match rules
- persist official match state independently

---

# 10. Near-Term Initial Endpoint Set

A minimal first endpoint set should conceptually support:

- health/status
- infer move
- list models
- get model detail
- submit training job
- get training job status
- submit evaluation job
- get evaluation job result

This is enough to support:
- internal AI gameplay
- controlled experiments
- future automation

---

# 11. Non-Goals of the Current Contract

The initial contract does NOT yet need to support:

- public external team bot APIs
- tournament registration
- spectator APIs
- full explainability payloads
- distributed training cluster coordination
- direct OpenSpiel environment control by external callers

These can be added later through separate contract areas if needed.

---

# Summary

The API contracts of `searchess-ai-service` must:

- remain platform-oriented
- hide OpenSpiel internals
- support inference, training, evaluation, and model discovery
- preserve the Scala platform as the authoritative chess system
- stay stable enough for long-term microservice evolution