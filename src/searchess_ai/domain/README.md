# Domain Layer

## Purpose

The domain layer defines the **core concepts of the AI service**.

It represents the service's internal business model for:

- move selection
- model identity and lifecycle
- training jobs
- evaluation jobs
- AI-facing game abstractions

It does **not** represent the full authoritative chess platform domain.

---

## What This Domain Is

This domain is the **AI service domain**.

It should model concepts such as:

- Position
- Move
- LegalMoveSet
- InferenceRequest
- InferenceDecision
- ModelId / ModelVersion
- TrainingJob
- EvaluationJob
- PolicyProfile

These concepts exist to support:
- inference
- training
- evaluation
- model management

---

## What This Domain Is Not

The domain layer is **not** the full chess engine domain.

It must NOT become the place for:

- authoritative chess rule enforcement
- official move legality computation
- tournament scheduling
- platform match lifecycle ownership
- public API protocol ownership

Those responsibilities belong elsewhere, especially in the Scala platform.

---

## Architectural Role

```text
API Layer
    ↓
Application Layer
    ↓
Domain Layer
    ↓
Infrastructure Layer
```

The domain layer provides the stable concepts that the application layer orchestrates.

It should be:

- framework-independent
- infrastructure-independent
- small
- explicit
- testable

---

## Core Domain Areas

The domain should be organized around a few clear concept groups.

1. Game Abstractions for AI

These are simplified representations of chess information needed by the AI service.

Examples:

* Position
a canonical board state representation used by the service
should wrap a stable serialized format rather than expose infrastructure types
* Move
a stable move representation understood by both platform and AI service
LegalMoveSet
the set of legal moves provided by the platform for a given position
* SideToMove
white or black

Important:
These are AI-facing abstractions, not the platform's full chess domain model.

2. Inference Concepts

These define the decision-making boundary of the service.

Examples:

- InferenceRequest
- InferenceDecision
- DecisionType
- DecisionMetadata
- PolicyProfile

These concepts should express:

- what the AI is asked to decide
- what it decided
- under which policy/model context
3. Model Concepts

These define model identity and lifecycle.

Examples:

- ModelId
- ModelVersion
- ModelStatus
- ModelSummary
- ModelDetail

Model lifecycle values should be explicit, for example:

- draft
- candidate
- approved
- active
- retired

This enables:

- reproducibility
- rollback
- evaluation
- promotion workflows

4. Training Concepts

These define the orchestration-level representation of training.

Examples:

- TrainingJobId
- TrainingProfile
- TrainingJob
- TrainingJobStatus
- SelfPlayConfig
- RewardConfig

The domain should model training as a job concept, not as a blocking operation.

5. Evaluation Concepts

These define model-vs-model comparison.

Examples:

- EvaluationJobId
- EvaluationJob
- EvaluationJobStatus
- EvaluationRequest
- EvaluationResult
- ScoreSummary

These concepts support:

- model comparison
- candidate promotion
- regression checks
- AI vs AI analysis

## Key Design Rules
1. Stable Platform-Oriented Representations

Domain concepts should use stable representations such as:

- canonical board state strings
- canonical move strings
- opaque ids

Do NOT expose:

- OpenSpiel state objects
- raw action ids
- framework-specific classes
2. No OpenSpiel Leakage

The domain must not depend on OpenSpiel.

OpenSpiel belongs to infrastructure.

This means:

- no OpenSpiel imports
- no OpenSpiel terminology in domain types
- no OpenSpiel-specific assumptions in domain modeling

3. No Authoritative Rule Ownership

The AI service domain may represent positions and legal moves, but it does not own official chess legality.

The platform remains authoritative.

The domain should assume:

- the platform provides a valid position
- the platform provides the legal move set
- the platform validates official move application

4. Explicit Value Concepts

Prefer explicit domain concepts over loose primitive usage.

Examples:

- ModelId instead of arbitrary string everywhere
- DecisionType instead of open text
- ModelStatus instead of unbounded status strings

This improves:

- clarity
- validation
- testability
- long-term consistency

5. Keep the Domain Small

This is critical.

The AI service domain should model only what the service truly owns.

Do not duplicate the platform's richer chess domain just because it seems convenient.

Duplication here would create:

- drift
- inconsistency
- coupling
- future migration pain
Recommended Domain Subareas
domain/
  game/
    # position, move, legal move set, side to move

  inference/
    # inference request, decision, policy profile

  model/
    # model identity, version, status, metadata

  training/
    # training job, training status, configs

  evaluation/
    # evaluation job, results, score summaries

  common/
    # shared ids, timestamps, metadata abstractions

This structure is optional, but the separation is useful.

## Canonical Concepts to Define Early

The following concepts should be stabilized early because many other parts depend on them.

#### Position

Represents a board state in a stable serialized form.

Recommendation:

- use a single canonical format at the service boundary
- FEN is a strong candidate

#### Move

Represents a move in the same stable format used by the platform in legalMoves.

Rule:

- the AI service must return moves in exactly this format

#### SideToMove

Simple but important explicit concept:

- white
- black

* Model Identity

Must include:

- model id
- model version
- status

* Job Identity

Training and evaluation jobs should each have explicit ids and statuses.

## Domain Invariants

The domain layer should make key assumptions and invariants explicit.

Examples:

- an inference decision of type move must contain a selected move
- a selected move must belong to the legal move set provided in the request
- a model must have a valid identity and status
- a completed evaluation job should have result data
- a failed job should include a failure reason where available

These invariants help the application layer remain clean.

## Relationship to Application Layer

The application layer orchestrates domain concepts.

Example:

ChooseMoveUseCase
  -> receives InferenceRequest
  -> selects model / policy
  -> calls InferenceEngine port
  -> returns InferenceDecision

The domain layer provides the vocabulary.
The application layer provides the workflow.

## Relationship to Infrastructure

Infrastructure adapts technical systems to domain concepts.

Examples:

- OpenSpiel adapter maps between domain Position / Move and OpenSpiel state/action representations
- storage adapters persist model metadata and job states
- execution adapters run training and evaluation jobs

The domain must not know how these are implemented.

## Testing Guidance

The domain layer should be easy to test in isolation.

Focus tests on:

- value rules
- invariants
- status transitions
- domain validation rules
- result consistency

These tests should not require:

- HTTP
- OpenSpiel
- databases
- background workers
