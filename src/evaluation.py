"""Evaluation metrics for diffusion models."""
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


def compute_fid_score(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """Simplified FID score (mock)."""
    if real_features is None or generated_features is None:
        return 0.0
    mu1, mu2 = np.mean(real_features, axis=0), np.mean(generated_features, axis=0)
    sigma1, sigma2 = np.cov(real_features.T), np.cov(generated_features.T)
    diff = mu1 - mu2
    return float(np.sqrt(diff @ diff + np.trace(sigma1 + sigma2 - 2 * np.sqrt(sigma1 @ sigma2) if np.all(np.linalg.eigvals(sigma1 @ sigma2) >= 0) else np.abs(sigma1 - sigma2))))


def generate_report(training_history: Dict, n_images: int = 0) -> str:
    lines = ["# Diffusion Model — Evaluation Report", "",
             f"## Training\n- Steps: {len(training_history.get('loss', []))}",
             f"- Final Loss: {training_history['loss'][-1]:.4f}" if training_history.get("loss") else "",
             f"\n## Generation\n- Images generated: {n_images}"]
    return "\n".join(lines)
