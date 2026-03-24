"""
Project Configuration for Diffusion Image Generation.

Centralizes all hyperparameters, paths, and settings. Values can be
overridden via environment variables for deployment flexibility without
modifying code (12-factor app principle).

Environment variable format: DIFFUSION_{SETTING_NAME}
Example: DIFFUSION_EPOCHS=100 python train_diffusion_cifar.py
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# Configure logging early so all modules inherit the format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    """Read an integer from environment, falling back to default."""
    val = os.environ.get(f"DIFFUSION_{key}")
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read a float from environment, falling back to default."""
    val = os.environ.get(f"DIFFUSION_{key}")
    return float(val) if val is not None else default


def _env_str(key: str, default: str) -> str:
    """Read a string from environment, falling back to default."""
    return os.environ.get(f"DIFFUSION_{key}", default)


# ---------------------------------------------------------------------------
# Paths — derived from project root, auto-created on import
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------

RANDOM_SEED: int = _env_int("SEED", 42)
IMAGE_SIZE: int = _env_int("IMAGE_SIZE", 64)
DIFFUSION_STEPS: int = _env_int("DIFFUSION_STEPS", 100)
LEARNING_RATE: float = _env_float("LEARNING_RATE", 1e-4)
BATCH_SIZE: int = _env_int("BATCH_SIZE", 16)
EPOCHS: int = _env_int("EPOCHS", 10)

# Noise schedule: "linear" or "cosine"
NOISE_SCHEDULE: str = _env_str("NOISE_SCHEDULE", "linear")
BETA_START: float = _env_float("BETA_START", 0.0001)
BETA_END: float = _env_float("BETA_END", 0.02)

# EMA (Exponential Moving Average) for inference smoothing
EMA_ENABLED: bool = os.environ.get("DIFFUSION_EMA_ENABLED", "true").lower() == "true"
EMA_DECAY: float = _env_float("EMA_DECAY", 0.9999)

# Gradient clipping for training stability
GRAD_CLIP_NORM: float = _env_float("GRAD_CLIP_NORM", 1.0)

# ---------------------------------------------------------------------------
# API Server
# ---------------------------------------------------------------------------

API_HOST: str = _env_str("API_HOST", "0.0.0.0")
API_PORT: int = _env_int("API_PORT", 8006)

# ---------------------------------------------------------------------------
# Logging summary
# ---------------------------------------------------------------------------

logger.debug(
    "Config loaded: image=%dx%d, steps=%d, schedule=%s, lr=%.0e, batch=%d, ema=%s",
    IMAGE_SIZE,
    IMAGE_SIZE,
    DIFFUSION_STEPS,
    NOISE_SCHEDULE,
    LEARNING_RATE,
    BATCH_SIZE,
    EMA_ENABLED,
)
