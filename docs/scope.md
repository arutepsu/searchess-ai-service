
---

# 📄 `docs/scope.md`

```md
# Scope of searchess-ai-service

## Responsibilities (Owns)

The AI service is responsible for:

### 1. Inference
- Selecting moves for AI-controlled players
- Providing deterministic or stochastic policies
- Returning decision metadata (model version, confidence, etc.)

### 2. Training
- Running self-play simulations
- Generating training data
- Managing training jobs and configurations

### 3. Evaluation
- Running AI vs AI matches
- Comparing model performance
- Producing evaluation metrics

### 4. Model Lifecycle
- Managing model versions
- Tracking model status:
  - draft
  - candidate
  - approved
  - active
  - retired
- Supporting model promotion and rollback

### 5. Simulation Backend Integration
- Using OpenSpiel for:
  - game simulation
  - self-play
  - training environments

---

## Non-Responsibilities (Does NOT Own)

The AI service explicitly does NOT handle:

### 1. Authoritative Game State
- No official match state
- No rule enforcement for production matches
- No persistence of platform games

### 2. Tournament Logic
- No scheduling
- No ranking systems
- No official results

### 3. Public API Gateway
- No direct exposure to external teams
- No authentication or rate limiting for public access

### 4. UI / Client Logic
- No frontend concerns
- No user interaction handling

---

## Integration Boundaries

### With Searchess Platform

The AI service is called via:

- inference requests (get move)
- model queries
- (future) training/evaluation triggers

The platform:
- validates moves
- owns the game lifecycle
- acts as the referee

---

## Design Constraints

- Must remain replaceable (no tight coupling to OpenSpiel)
- Must expose stable contracts
- Must support multiple model versions
- Must separate training and inference concerns

---

## Long-Term Role

The AI service evolves into:

- a dedicated AI platform
- supporting:
  - competitive AI
  - tournaments
  - external bot interoperability (via platform)