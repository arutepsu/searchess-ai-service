# Architecture of searchess-ai-service

## Overview

The AI service follows a layered architecture:

```text
API -> Application -> Domain
                   -> Infrastructure
```
This ensures:

- separation of concerns
- testability
- replaceable infrastructure (e.g., OpenSpiel)
## Layers
### 1. API Layer (api/)

#### Responsibilities:

- HTTP endpoints
- request/response validation
- transport mapping

Does NOT:

- contain business logic
- interact directly with OpenSpiel
### 2. Application Layer (application/)

#### Responsibilities:

- orchestrate use cases:
- inference
- training
- evaluation
- coordinate domain + infrastructure

Examples of use cases:

- choose_move
- start_training_job
- evaluate_models

Depends on:

- domain concepts
- infrastructure ports (not implementations)
### 3. Domain Layer (domain/)

Represents the internal domain of the AI service.

#### Key concepts:

- ModelVersion
- TrainingJob
- EvaluationResult
- Policy
- InferenceRequest / InferenceResult

Important:
- This is NOT the chess domain model of the platform.
- It is the AI service domain.

### 4. Infrastructure Layer (infrastructure/)

#### Responsibilities:

- OpenSpiel integration
- model loading/saving
- artifact storage
- configuration
- logging/metrics

This is where external dependencies live.

#### OpenSpiel Integration

OpenSpiel is used as an internal backend:

Application
  -> OpenSpiel Adapter (Infrastructure)
      -> OpenSpiel

#### Key rule:

OpenSpiel must NOT leak into API or domain layers.

This ensures:

- flexibility
- replaceability
- clean architecture
- Inference vs Training Separation

Two distinct subsystems:

- Inference
- low latency
- stable model
- production-safe
- Training
- long-running jobs
- experimental
- generates new model versions

These must remain conceptually separate.

#### Interaction with Searchess Platform
Searchess Platform (Scala)
        |
        v
   AI Service (Python)
        |
   Inference / Training / Evaluation
        |
   OpenSpiel (internal)

#### Key rule:

The platform is the source of truth.
The AI service is a decision provider.

#### Future Evolution

The architecture supports:

- distributed training
- model registry systems
- tournament evaluation pipelines
- multiple AI strategies
- integration with external bot ecosystems (via platform)
#### Risks and Mitigations
* Risk: OpenSpiel leakage

Mitigation:

strict adapter boundary
* Risk: mixing training and inference logic

Mitigation:

separate application use cases
* Risk: model chaos

Mitigation:

explicit model lifecycle
* Risk: tight coupling to platform

Mitigation:

stable API contracts