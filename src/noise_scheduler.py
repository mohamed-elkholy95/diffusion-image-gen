"""
Noise Scheduler for Denoising Diffusion Probabilistic Models (DDPM).

This module implements the forward diffusion process — the core mechanism that
gradually corrupts clean images into Gaussian noise over a series of timesteps.

Key Concepts:
    - **Forward process (q)**: Adds noise incrementally: q(x_t | x_{t-1}) = N(√(1-β_t) * x_{t-1}, β_t * I)
    - **Beta schedule**: Controls HOW MUCH noise is added at each step
    - **Alpha bar (ᾱ_t)**: Cumulative product of (1 - β_t), enables jumping directly
      to any timestep without iterating: q(x_t | x_0) = N(√ᾱ_t * x_0, (1-ᾱ_t) * I)
    - **Posterior variance**: Used in the reverse process to remove noise step-by-step

Schedule Types:
    - **Linear**: β increases linearly from β_start to β_end. Simple but can destroy
      information too quickly at high timesteps.
    - **Cosine**: Follows a cosine curve (Nichol & Dhariwal, 2021). Preserves more signal
      at high timesteps, leading to better image quality. Preferred for most applications.

References:
    - Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    logger.info("torch not available — scheduler will operate in numpy-only mode")


class NoiseScheduler:
    """Noise schedule manager for DDPM forward and reverse diffusion.

    The scheduler precomputes all the constants needed for the diffusion process
    at initialization time. This avoids redundant computation during training
    and sampling, where these values are accessed thousands of times.

    Precomputed values:
        - betas: Noise variance at each timestep
        - alphas: Signal retention at each timestep (1 - beta)
        - alphas_cumprod (ᾱ_t): Cumulative signal retention — the key quantity
          that lets us jump to any timestep in one step
        - sqrt_alphas_cumprod: Used to scale the clean image in the forward process
        - sqrt_one_minus_alphas_cumprod: Used to scale the noise component
        - posterior_variance: Variance for the reverse (denoising) process

    Args:
        num_steps: Total number of diffusion timesteps (T). More steps = finer
            noise granularity but slower sampling. Typical: 200-1000.
        beta_start: Starting noise variance. Keep small (1e-4) to preserve
            the image at early timesteps.
        beta_end: Ending noise variance. Controls how quickly we approach
            pure noise. Typical: 0.02 for linear schedule.
        schedule: Noise schedule type — 'linear' or 'cosine'.
        seed: Random seed for reproducible noise generation.
    """

    # Valid schedule types for input validation
    VALID_SCHEDULES = ("linear", "cosine")

    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        seed: int = 42,
    ) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError(f"beta values must be positive, got start={beta_start}, end={beta_end}")
        if beta_start >= beta_end:
            raise ValueError(f"beta_start ({beta_start}) must be less than beta_end ({beta_end})")
        if schedule not in self.VALID_SCHEDULES:
            raise ValueError(f"schedule must be one of {self.VALID_SCHEDULES}, got '{schedule}'")

        self.num_steps = num_steps
        self.schedule = schedule
        self._rng = np.random.default_rng(seed)

        # Compute the beta schedule — this defines the noise curve
        if schedule == "linear":
            # Simple linear interpolation between beta_start and beta_end
            self.betas = np.linspace(beta_start, beta_end, num_steps)
        else:
            # Cosine schedule (Nichol & Dhariwal, 2021)
            # Derives betas from a cosine-shaped cumulative alpha curve.
            # f(t) = cos((t/T + s) / (1+s) * π/2)^2, where s is a small offset
            # to prevent β from being too small near t=0
            s = 0.008  # Offset to prevent singularity at t=0
            steps = np.arange(num_steps + 1, dtype=np.float64)
            f_t = np.cos(((steps / num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = f_t / f_t[0]
            self.betas = np.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 1e-4, 0.9999)

        # Precompute all derived quantities for efficiency
        # α_t = 1 - β_t — how much of the signal is retained at step t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏_{i=1}^{t} α_i — cumulative signal retention
        # This is the KEY quantity: it lets us go from x_0 directly to x_t
        self.alphas_cumprod = np.cumprod(self.alphas)

        # ᾱ_{t-1} with ᾱ_0 = 1 — needed for posterior variance calculation
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])

        # Square roots used in the forward process reparameterization:
        # x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε, where ε ~ N(0, I)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance: used in the reverse process
        # σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        # This is the variance of q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        logger.info(
            "Initialized %s schedule: %d steps, β=[%.5f, %.5f], final ᾱ=%.6f",
            schedule,
            num_steps,
            self.betas[0],
            self.betas[-1],
            self.alphas_cumprod[-1],
        )

    def add_noise(
        self, x: np.ndarray, t: int, noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the forward diffusion process at timestep t.

        Uses the reparameterization trick to jump directly from x_0 to x_t:
            x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

        This avoids iterating through all intermediate steps, making training
        efficient — we can sample ANY timestep directly.

        Args:
            x: Clean input data (any shape). During training, this is a batch
               of images normalized to [-1, 1].
            t: Diffusion timestep (0 to num_steps-1). Higher t = more noise.
            noise: Optional pre-generated noise. If None, standard Gaussian
                noise is sampled. Providing noise is useful for deterministic
                testing and reproducibility.

        Returns:
            Tuple of (noisy_data, noise) — both same shape as input x.
            Returns the noise so the training loop can use it as the target.

        Raises:
            IndexError: If t is outside [0, num_steps-1].
        """
        if t < 0 or t >= self.num_steps:
            raise IndexError(f"Timestep t={t} out of range [0, {self.num_steps - 1}]")

        if noise is None:
            noise = self._rng.standard_normal(x.shape).astype(x.dtype)

        # Apply the forward diffusion formula
        sqrt_a = self.sqrt_alphas_cumprod[t]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t]
        noisy = sqrt_a * x + sqrt_1ma * noise
        return noisy, noise

    def reverse_step(
        self, x_t: np.ndarray, predicted_noise: np.ndarray, t: int
    ) -> np.ndarray:
        """Perform one step of the reverse (denoising) diffusion process.

        Given x_t and the model's noise prediction, compute x_{t-1} using
        the DDPM reverse process formula:

            μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t / √(1-ᾱ_t)) * ε_θ(x_t, t))
            x_{t-1} = μ_θ + σ_t * z,  where z ~ N(0, I) and σ_t = √(posterior_variance)

        Args:
            x_t: Noisy data at timestep t.
            predicted_noise: Model's prediction of the noise component.
            t: Current timestep (must be > 0 for stochastic step, at t=0 returns mean).

        Returns:
            Denoised data at timestep t-1.
        """
        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Compute the predicted mean of x_{t-1}
        coef1 = 1.0 / np.sqrt(alpha)
        coef2 = beta / np.sqrt(1.0 - alpha_bar)
        mean = coef1 * (x_t - coef2 * predicted_noise)

        if t > 0:
            # Add stochastic noise (except at the final step t=0)
            sigma = np.sqrt(self.posterior_variance[t])
            noise = self._rng.standard_normal(x_t.shape).astype(x_t.dtype)
            return mean + sigma * noise
        else:
            return mean

    def get_schedule_stats(self) -> dict:
        """Return summary statistics of the noise schedule for logging/display.

        Useful for verifying the schedule behaves as expected:
        - For linear: ᾱ_T should be near 0 (almost pure noise at the end)
        - For cosine: ᾱ_T is typically higher than linear (preserves more signal)
        """
        return {
            "schedule": self.schedule,
            "num_steps": self.num_steps,
            "beta_range": f"{self.betas[0]:.5f} → {self.betas[-1]:.5f}",
            "final_alpha_bar": round(float(self.alphas_cumprod[-1]), 6),
            "midpoint_alpha_bar": round(float(self.alphas_cumprod[self.num_steps // 2]), 6),
        }

    def __repr__(self) -> str:
        return (
            f"NoiseScheduler(schedule='{self.schedule}', num_steps={self.num_steps}, "
            f"final_ᾱ={self.alphas_cumprod[-1]:.6f})"
        )
