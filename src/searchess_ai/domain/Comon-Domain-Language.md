# Domain Common Language

## Purpose

This document defines the shared language of the AI service domain.

It ensures that:

* API layer
* application layer
* domain logic
* infrastructure adapters

all speak the same concepts consistently.

This is critical for:

* avoiding ambiguity
* preventing architectural drift
* enabling independent evolution of components

The AI service domain is intentionally **smaller than the platform domain** and focuses only on:

* inference
* model lifecycle
* training jobs
* evaluation jobs

It must not evolve into a second chess engine.

---

## Design Principles

### 1. Minimal but Explicit

The domain should:

* define clear concepts
* avoid unnecessary complexity
* avoid duplicating platform logic

---

### 2. Platform-Aligned

The domain must align with the Scala platform:

* same move format
* same board representation
* same game semantics

The platform remains the **source of truth**.

---

### 3. No Infrastructure Leakage

The domain must not depend on:

* OpenSpiel
* databases
* HTTP frameworks

All such concerns belong to the infrastructure layer.

---

### 4. Value-Oriented Design

Prefer explicit value concepts over raw primitives.

Examples:

* Position instead of string
* Move instead of string
* ModelId instead of string

---

## Core Concepts

### Position

Represents a board state.

* stored as a canonical serialized string
* recommended format: FEN

Purpose:

* provide a stable input for inference
* allow reproducibility and logging

Constraints:

* must not include logic
* must not depend on OpenSpiel
* must not become a full board model

---

### Move

Represents a single move.

* stored as a canonical string
* must match the platform’s move format

Critical rule:
The AI must return a move that exists in the provided legal move set.

---

### LegalMoveSet

Represents all legal moves for a position.

* provided by the platform
* contains a collection of Move values

Purpose:

* ensures the AI does not compute legality
* enforces correct decision boundaries

Invariant:
Selected move must belong to this set.

---

### SideToMove

Represents which player is acting.

Allowed values:

* white
* black

This concept is simple but important for:

* inference
* evaluation
* logging

---

## Inference Concepts

### InferenceRequest

Represents a decision problem.

Contains:

* Position
* SideToMove
* LegalMoveSet
* optional context (time, model, policy)

Purpose:
Defines what the AI must decide.

---

### InferenceDecision

Represents the result of inference.

Contains:

* decisionType
* selected Move (if applicable)
* modelId
* modelVersion
* timing metadata

Initial constraint:

* only decisionType = move is required

Invariant:
If decisionType is move → selectedMove must exist and be legal.

---

### PolicyProfile

Represents a high-level behavior mode.

Examples:

* safe
* fast
* balanced
* experimental

Purpose:
Allows tuning behavior without changing models or APIs.

Constraint:
Must remain high-level (not low-level engine tuning).

---

## Model Concepts

### ModelId

Represents a model family.

Purpose:
Groups related versions of a model.

---

### ModelVersion

Represents a specific model instance.

Purpose:

* reproducibility
* debugging
* evaluation comparison

---

### ModelStatus

Represents lifecycle state.

Allowed values:

* draft
* candidate
* approved
* active
* retired

Purpose:
Supports:

* promotion workflows
* safe deployment
* rollback

---

## Training Concepts

### TrainingJob

Represents a training process.

Contains:

* job id
* training profile
* optional base model
* status
* timestamps
* optional output model

Training is modeled as a **job**, not a blocking operation.

---

### TrainingJobStatus

Allowed values:

* queued
* running
* completed
* failed
* cancelled

Purpose:
Provides clear lifecycle tracking.

---

## Evaluation Concepts

### EvaluationJob

Represents a comparison between models.

Contains:

* candidate model
* baseline model
* number of games
* status
* optional results

---

### EvaluationResult

Represents completed evaluation output.

Contains:

* games played
* candidate wins
* baseline wins
* draws
* score summary

Purpose:
Supports:

* model comparison
* promotion decisions
* regression detection

---

## Domain Invariants

The following must always hold:

* Selected move must be in LegalMoveSet
* InferenceDecision must include model identity
* ModelVersion must belong to a ModelId
* Completed jobs must include results
* Failed jobs should include failure reason when available

---

## Boundary Definition

### The AI Service DOES:

* choose moves
* manage models
* run training
* evaluate models

### The AI Service DOES NOT:

* enforce chess rules
* compute legal moves
* manage matches
* own game state

These responsibilities belong to the platform.

---

## Summary

The domain language is:

* small
* explicit
* stable
* aligned with the platform
* independent from infrastructure

It defines the vocabulary for:

* inference
* model lifecycle
* training
* evaluation

Nothing more.
