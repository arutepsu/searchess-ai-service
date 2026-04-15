# searchess-ai-service

Python-based AI service for the Searchess chess platform.

## Purpose

This service is responsible for:

- AI move inference (choosing moves for AI players)
- Self-play training
- Model evaluation (AI vs AI)
- Model lifecycle management
- Integration with OpenSpiel for training and simulation

It is designed as a standalone microservice that integrates with the main Scala-based Searchess platform.

## Context

The overall system is split into two main parts:

- **Searchess Platform (Scala)**  
  - authoritative chess rules  
  - game and tournament orchestration  
  - persistence and APIs  

- **Searchess AI Service (Python)**  
  - training and inference  
  - model management  
  - simulation and evaluation  

This separation ensures:
- strong architectural boundaries
- independent evolution of AI and platform
- clean integration via service contracts

## Key Principles

- The AI service is **not the source of truth** for game state
- OpenSpiel is an **internal dependency**, not exposed externally
- Inference and training are **separate concerns**
- Model versions are **explicit and traceable**
- All integration happens through **stable contracts**

## Current Status

Initial service skeleton.  
Focus: structure, contracts, and architecture.

## Future Capabilities

- Self-play training pipelines
- Model evaluation and ranking
- Tournament integration (AI participants)
- External bot compatibility (via platform)

## Tech Stack (planned)

- Python 3.12+
- uv (project & dependency management)
- FastAPI (service layer)
- OpenSpiel (training & simulation backend)

## Repository Structure

```text
src/searchess_ai/
  api/             # HTTP layer
  application/     # use cases / orchestration
  domain/          # AI service domain model
  infrastructure/  # OpenSpiel + technical integrations
```

Relation to Platform

This service will be called by the Searchess platform via:

move inference requests
model queries
(later) training/evaluation triggers

The platform remains authoritative for all official matches.