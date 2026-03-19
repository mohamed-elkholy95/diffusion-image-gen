# Development Guide — 05-diffusion-image-gen

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Running the Dashboard

```bash
streamlit run streamlit_app/app.py
```

## Running the API

```bash
python -m src.api.main
```

## Code Style

- Type hints on all functions
- Docstrings on all public classes and methods
- Use `logging` instead of `print()`
- `pathlib.Path` for file paths
- Constants in UPPER_CASE
