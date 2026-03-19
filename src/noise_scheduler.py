"""WORK IN PROGRESS — Adding methods and implementation details."""

"""Noise scheduler for diffusion models."""
import logging
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    logger.info("torch not available")

class NoiseScheduler:
    """Linear and cosine noise schedules."""

    def __init__(self, num_steps: int = 100, beta_start: float = 0.0001,
                 beta_end: float = 0.02, schedule: str = "linear", seed: int = 42) -> None:
        self.num_steps = num_steps
        self.schedule = schedule
        if schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_steps)
        else:
            steps = np.arange(num_steps, dtype=np.float64)
            s = 0.008 * (steps / num_steps + 1) ** 2
            f = (steps + 1) / num_steps
            self.betas = np.clip(1 - (f / s), 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
