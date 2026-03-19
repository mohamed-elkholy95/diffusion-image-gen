# 05 Diffusion Image Gen — Architecture Guide

## System Overview

This document provides a detailed architectural overview of the 05-diffusion-image-gen project,
explaining design decisions, data flow, and component interactions.

## Design Decisions

1. **Modular Architecture**: Each component is independently testable and replaceable.
2. **Configuration-Driven**: All parameters centralized in `config.py`.
3. **Type Hints**: Full type annotations for IDE support and maintainability.

## Data Flow

```
Input → Preprocessing → Model → Post-processing → Output
```

## Component Map

| Component | File | Purpose |
|-----------|------|---------|
| Config | `src/config.py` | Centralized parameters |
| Core | `src/` | Business logic and models |
| API | `src/api/main.py` | FastAPI REST endpoints |
| Dashboard | `streamlit_app/` | Interactive UI |
| Tests | `tests/` | Unit and integration tests |
