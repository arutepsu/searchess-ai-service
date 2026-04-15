# Application Layer

## Purpose

The application layer defines the **use cases** of the AI service.

It orchestrates:
- domain concepts
- infrastructure dependencies
- external requests (via API)

It is the **core coordination layer** of the service.

---

## Responsibilities

The application layer is responsible for:

- executing use cases such as:
  - move inference
  - training job submission
  - evaluation job submission
  - model queries
- coordinating domain logic and infrastructure
- enforcing high-level business rules
- validating inputs beyond basic schema validation
- managing asynchronous workflows (training/evaluation jobs)

---

## Non-Responsibilities

The application layer must NOT:

- contain HTTP or framework logic (FastAPI, etc.)
- contain OpenSpiel-specific logic
- contain persistence implementations
- implement low-level algorithms
- define transport-level DTOs

---

## Architectural Position

```text
API Layer (FastAPI)
        ↓
Application Layer (Use Cases)
        ↓
Domain Layer
        ↓
Infrastructure Layer (OpenSpiel, storage, execution)
```
The application layer acts as the boundary between API and core logic.

--- 

## Use Cases

Each use case represents a single business capability.

### Core Use Cases (v1)
ChooseMoveUseCase
* selects a move for a given position
ListModelsUseCase
* lists available models
GetModelDetailUseCase
* retrieves detailed model information
SubmitTrainingJobUseCase
* creates and schedules a training job
GetTrainingJobStatusUseCase
* retrieves training job status
SubmitEvaluationJobUseCase
* creates and schedules an evaluation job
GetEvaluationJobStatusUseCase
* retrieves evaluation job status
### Design Principles
1. Use Case per Operation

Each operation is isolated into a dedicated use case.

Why:

- improves testability
- reduces coupling
- enables independent evolution
2. Dependency on Ports, Not Implementations

Use cases depend on ports (interfaces), not concrete implementations.

Examples:

- InferenceEngine
- ModelRepository
- TrainingScheduler
- EvaluationRepository

This ensures:

- testability
- replaceability
- clean architecture boundaries
3. No OpenSpiel Leakage

OpenSpiel is strictly an infrastructure concern.

The application layer:

- must not import OpenSpiel
- must not depend on OpenSpiel types
- must remain backend-agnostic
4. Asynchronous Job Handling

Training and evaluation are modeled as jobs, not direct execution.

Use cases:

- create jobs
- delegate execution to schedulers
- return immediately

This supports:

- scalability
- distributed execution
- future orchestration (e.g. Kubernetes)
5. Thin but Not Trivial

Use cases should be:

- thin in logic
- but responsible for orchestration and validation

They must NOT be:

- simple pass-through calls
- Internal Structure
application/
  usecase/
    # individual use cases

  port/
    # interfaces for infrastructure

  service/
    # internal orchestration helpers
Ports

Ports define the required capabilities from infrastructure.

#### Examples:
InferenceEngine
* executes move selection
ModelRepository
* provides model metadata
TrainingScheduler
* schedules training jobs
TrainingRepository
* persists job state
EvaluationScheduler
* runs evaluation jobs
EvaluationRepository
* stores evaluation results

#### Example Flow: Move Inference
API → ChooseMoveUseCase
        ↓
   ModelSelector
        ↓
   PolicyResolver
        ↓
   InferenceEngine (port)
        ↓
   DecisionValidator
        ↓
Response

#### Example Flow: Training Job
API → SubmitTrainingJobUseCase
        ↓
   Validation
        ↓
   Job Creation
        ↓
   TrainingScheduler (port)
        ↓
   Persist Job
        ↓
Response (job accepted)


### Testing Strategy

The application layer must be fully testable in isolation.

Approach:

- mock all ports
- test each use case independently
- simulate infrastructure behavior

Benefits:

- fast tests
- deterministic behavior
- no external dependencies
- Future Evolution

The application layer is designed to support:

- distributed training pipelines
- advanced evaluation workflows
- model promotion pipelines
- tournament integration
- multiple AI strategies

### Key Architectural Rule

The application layer defines the true service boundary.

It is not defined by:

- FastAPI
- OpenSpiel
- Python modules

But by:

- use cases
- ports
- orchestration logic

This ensures long-term scalability and clean microservice evolution.