# v1 DTO Shapes

This section defines the initial request and response DTOs for the v1 API surface.

The DTOs are intentionally platform-oriented and must remain independent from OpenSpiel internals.

---

## 1. MoveInferenceRequest

Used by:

- `POST /api/v1/inference/move`

### Fields

#### requestId
- type: string
- required: yes
- purpose:
  - correlation id
  - tracing
  - debugging

#### matchId
- type: string
- required: yes
- purpose:
  - identifies the match in the Searchess platform

#### gameId
- type: string
- required: no
- purpose:
  - identifies the game if distinct from match id

#### boardState
- type: string
- required: yes
- purpose:
  - canonical serialized board representation
- recommendation:
  - use FEN or another single stable platform-defined format

#### sideToMove
- type: string
- required: yes
- allowed values:
  - `white`
  - `black`

#### legalMoves
- type: array of string
- required: yes
- purpose:
  - full list of legal moves that the AI is allowed to choose from
- rule:
  - returned move must be one of these

#### moveHistory
- type: array of string
- required: no
- purpose:
  - optional compact history for context

#### timeControl
- type: object
- required: no
- purpose:
  - describes timing regime for decision logic

#### remainingTimeMillis
- type: integer
- required: no
- purpose:
  - remaining time for the acting side

#### moveDeadline
- type: string
- required: no
- purpose:
  - absolute deadline timestamp if needed

#### decisionMode
- type: string
- required: no
- examples:
  - `safe`
  - `fast`
  - `balanced`
  - `experimental`

#### modelId
- type: string
- required: no
- purpose:
  - explicit requested model identity

#### modelVersion
- type: string
- required: no
- purpose:
  - explicit requested model version

#### policyProfile
- type: string
- required: no
- purpose:
  - named behavioral profile

---

## 2. MoveInferenceResponse

Used by:

- `POST /api/v1/inference/move`

### Fields

#### requestId
- type: string
- required: yes
- purpose:
  - echoes the request correlation id

#### selectedMove
- type: string
- required: yes
- rule:
  - must be one of the input `legalMoves`

#### decisionType
- type: string
- required: yes
- initial allowed values:
  - `move`

Future values may include:
- `resign`
- `no_decision`

#### modelId
- type: string
- required: yes

#### modelVersion
- type: string
- required: yes

#### policyProfile
- type: string
- required: no

#### decisionTimeMillis
- type: integer
- required: yes

#### confidence
- type: number
- required: no
- notes:
  - optional
  - semantics must be documented if exposed

#### reasoningMetadata
- type: object
- required: no
- purpose:
  - structured diagnostics
- rule:
  - machine-readable only
  - not freeform natural-language explanations

---

## 3. TimeControlDto

Reusable optional object.

### Fields

#### type
- type: string
- required: yes
- examples:
  - `unlimited`
  - `blitz`
  - `rapid`
  - `classical`
  - `custom`

#### initialTimeMillis
- type: integer
- required: no

#### incrementMillis
- type: integer
- required: no

#### moveTimeMillis
- type: integer
- required: no

---

## 4. ModelSummary

Used by:

- `GET /api/v1/models`

### Fields

#### modelId
- type: string
- required: yes

#### modelVersion
- type: string
- required: yes

#### status
- type: string
- required: yes
- allowed values:
  - `draft`
  - `candidate`
  - `approved`
  - `active`
  - `retired`

#### createdAt
- type: string
- required: yes

#### description
- type: string
- required: no

#### trainingRunId
- type: string
- required: no

#### tags
- type: array of string
- required: no

#### capabilities
- type: array of string
- required: no
- examples:
  - `inference`
  - `self_play`
  - `evaluation`

---

## 5. ModelDetail

Used by:

- `GET /api/v1/models/{modelId}`

### Fields

Includes all `ModelSummary` fields plus:

#### supportedPolicyProfiles
- type: array of string
- required: no

#### evaluationSummary
- type: object
- required: no
- purpose:
  - compact metrics or summary of recent evaluations

#### notes
- type: string
- required: no

#### artifactMetadata
- type: object
- required: no
- purpose:
  - machine-readable artifact information

---

## 6. TrainingJobRequest

Used by:

- `POST /api/v1/training/jobs`

### Fields

#### trainingProfile
- type: string
- required: yes
- purpose:
  - named training preset

#### baseModelId
- type: string
- required: no

#### baseModelVersion
- type: string
- required: no

#### selfPlayConfig
- type: object
- required: no
- purpose:
  - structured self-play settings

#### rewardConfig
- type: object
- required: no
- purpose:
  - structured reward settings

#### maxIterations
- type: integer
- required: no

#### notes
- type: string
- required: no

---

## 7. TrainingJobAcceptedResponse

Used by:

- `POST /api/v1/training/jobs`

### Fields

#### trainingJobId
- type: string
- required: yes

#### status
- type: string
- required: yes
- initial values:
  - `queued`
  - `running`

#### submittedAt
- type: string
- required: yes

---

## 8. TrainingJobStatus

Used by:

- `GET /api/v1/training/jobs/{trainingJobId}`

### Fields

#### trainingJobId
- type: string
- required: yes

#### status
- type: string
- required: yes
- allowed values:
  - `queued`
  - `running`
  - `completed`
  - `failed`
  - `cancelled`

#### submittedAt
- type: string
- required: yes

#### startedAt
- type: string
- required: no

#### updatedAt
- type: string
- required: yes

#### progress
- type: object
- required: no
- purpose:
  - coarse progress information

#### producedModelId
- type: string
- required: no

#### producedModelVersion
- type: string
- required: no

#### failureReason
- type: string
- required: no

---

## 9. EvaluationJobRequest

Used by:

- `POST /api/v1/evaluation/jobs`

### Fields

#### candidateModelId
- type: string
- required: yes

#### candidateModelVersion
- type: string
- required: yes

#### baselineModelId
- type: string
- required: yes

#### baselineModelVersion
- type: string
- required: yes

#### numberOfGames
- type: integer
- required: yes

#### startingConditions
- type: object
- required: no

#### timeControl
- type: object
- required: no

#### policyProfiles
- type: object
- required: no

#### notes
- type: string
- required: no

---

## 10. EvaluationJobAcceptedResponse

Used by:

- `POST /api/v1/evaluation/jobs`

### Fields

#### evaluationJobId
- type: string
- required: yes

#### status
- type: string
- required: yes
- initial values:
  - `queued`
  - `running`

#### submittedAt
- type: string
- required: yes

---

## 11. EvaluationJobStatus

Used by:

- `GET /api/v1/evaluation/jobs/{evaluationJobId}`

### Fields

#### evaluationJobId
- type: string
- required: yes

#### status
- type: string
- required: yes
- allowed values:
  - `queued`
  - `running`
  - `completed`
  - `failed`
  - `cancelled`

#### submittedAt
- type: string
- required: yes

#### updatedAt
- type: string
- required: yes

#### completedAt
- type: string
- required: no

#### candidateModelId
- type: string
- required: yes

#### candidateModelVersion
- type: string
- required: yes

#### baselineModelId
- type: string
- required: yes

#### baselineModelVersion
- type: string
- required: yes

#### gamesPlayed
- type: integer
- required: no

#### candidateWins
- type: integer
- required: no

#### baselineWins
- type: integer
- required: no

#### draws
- type: integer
- required: no

#### scoreSummary
- type: object
- required: no

#### failureReason
- type: string
- required: no

---

## 12. ErrorResponse

Used by:

- all endpoints

### Fields

#### errorCode
- type: string
- required: yes

#### message
- type: string
- required: yes

#### requestId
- type: string
- required: no

#### details
- type: object
- required: no

### Initial error codes

- `INVALID_REQUEST`
- `MODEL_NOT_FOUND`
- `MODEL_NOT_ACTIVE`
- `UNSUPPORTED_POLICY_PROFILE`
- `TRAINING_JOB_NOT_FOUND`
- `EVALUATION_JOB_NOT_FOUND`
- `SERVICE_UNAVAILABLE`
- `INTERNAL_ERROR`

---

## 13. Serialization Rules

### Timestamps
- use one consistent timestamp format across all endpoints

### Identifiers
- identifiers are opaque strings

### Move format
- the move format returned by the AI service must match the move format used in `legalMoves`

### Board format
- the board state format must be stable and platform-defined
- FEN is a strong candidate

---

## 14. Minimal v1 DTO Set

The minimum DTOs required for first implementation are:

- `MoveInferenceRequest`
- `MoveInferenceResponse`
- `ModelSummary`
- `TrainingJobAcceptedResponse`
- `TrainingJobStatus`
- `EvaluationJobAcceptedResponse`
- `EvaluationJobStatus`
- `ErrorResponse`

Other DTOs may start as internal documentation-level shapes and become concrete as the service evolves.