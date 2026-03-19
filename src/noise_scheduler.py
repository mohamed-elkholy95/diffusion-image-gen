"""WORK IN PROGRESS — Core structure and imports."""

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

