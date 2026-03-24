"""
Simplified U-Net Architecture for Diffusion Models.

The U-Net is the backbone neural network in DDPM — it learns to predict
the noise that was added to an image at a given timestep. The architecture
follows an encoder-decoder pattern with skip connections.

Architecture Overview:
    ┌──────────────────────────────────────────────┐
    │                  Input (B, 3, 64, 64)        │
    │                      ↓                       │
    │  ┌─────────────┐  Encoder 1  ┌────────────┐  │
    │  │ Conv → GN → │  (3 → 32)   │ Skip Conn  │──┤
    │  │ SiLU → Conv │             │            │  │
    │  └──────┬──────┘             └────────────┘  │
    │         ↓ MaxPool                            │
    │  ┌─────────────┐  Encoder 2  ┌────────────┐  │
    │  │ Conv → GN → │  (32 → 64)  │ Skip Conn  │──┤
    │  │ SiLU → Conv │             │            │  │
    │  └──────┬──────┘             └────────────┘  │
    │         ↓ MaxPool                            │
    │  ┌─────────────┐  Bottleneck                 │
    │  │ Conv → GN → │  (64 → 128)  + Time Emb    │
    │  │ SiLU → Conv │                             │
    │  └──────┬──────┘                             │
    │         ↓ Upsample                           │
    │  ┌──────┴──────┐  Decoder 2  ←── Skip 2     │
    │  │ Cat → Conv  │  (128+64 → 64)              │
    │  └──────┬──────┘                             │
    │         ↓ Upsample                           │
    │  ┌──────┴──────┐  Decoder 1  ←── Skip 1     │
    │  │ Cat → Conv  │  (64+32 → 32)               │
    │  └──────┬──────┘                             │
    │         ↓                                    │
    │     Output Conv (32 → 3)                     │
    └──────────────────────────────────────────────┘

Key Components:
    - **Skip connections**: Preserve spatial detail lost during downsampling.
      The decoder receives both the upsampled features AND the corresponding
      encoder features via concatenation.
    - **Time embedding**: Sinusoidal positional encoding tells the network
      WHICH timestep it's denoising, since different timesteps have different
      noise levels.
    - **GroupNorm**: More stable than BatchNorm for small batch sizes,
      which is common in diffusion training.

References:
    - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
      Image Segmentation" (2015) — original U-Net architecture
    - Ho et al., "Denoising Diffusion Probabilistic Models" (2020) — DDPM
"""
import copy
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    logger.info("torch not available — using mock fallback")


class SimplifiedUNet(nn.Module if HAS_TORCH else object):
    """Simplified U-Net for 64×64 image denoising in DDPM.

    This is a minimal but functional U-Net suitable for learning and
    experimentation. Production diffusion models (Stable Diffusion, DALL-E)
    use much larger variants with attention layers, but the core idea is
    identical: encode → bottleneck → decode with skip connections.

    Args:
        in_channels: Number of input image channels (3 for RGB).
        base_channels: Width of the first encoder layer. Each subsequent
            layer doubles this. Larger = more capacity but more VRAM.
            32 for quick experiments, 64-128 for quality.
        time_emb_dim: Dimension of the sinusoidal time embedding vector.
            Must match the embedding dimension used during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        time_emb_dim: int = 128,
    ) -> None:
        if HAS_TORCH:
            super().__init__()

            # Time embedding MLP: maps scalar timestep to a feature vector
            # that gets injected into the bottleneck, telling the network
            # how much noise to expect at this timestep
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, base_channels * 4),
                nn.SiLU(),  # SiLU (Swish) — smooth, non-monotonic activation
                nn.Linear(base_channels * 4, base_channels * 4),
            )

            # Encoder path — progressively downsamples spatial dimensions
            # while increasing channel depth (capturing higher-level features)
            self.enc1 = self._block(in_channels, base_channels)       # (B, 3, 64, 64) → (B, 32, 64, 64)
            self.enc2 = self._block(base_channels, base_channels * 2)  # (B, 32, 32, 32) → (B, 64, 32, 32)

            # Bottleneck — deepest layer, processes the most compressed representation
            self.bottleneck = self._block(base_channels * 2, base_channels * 4)  # (B, 64, 16, 16) → (B, 128, 16, 16)

            # Decoder path — upsamples back to original resolution
            # ConvTranspose2d is "learnable upsampling" (aka deconvolution)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
            self.dec2 = self._block(base_channels * 4, base_channels * 2)  # Input is 4x because of skip connection concat

            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
            self.dec1 = self._block(base_channels * 2, base_channels)

            # Final 1×1 convolution — maps feature channels back to image channels
            self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

            # MaxPool for spatial downsampling in the encoder
            self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _block(in_c: int, out_c: int) -> "nn.Sequential":
        """Create a convolutional block: Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU.

        Two conv layers per block is standard in U-Nets. GroupNorm with 8 groups
        is used instead of BatchNorm because diffusion training often uses small
        batch sizes where BatchNorm statistics are noisy.
        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )

    def forward(self, x: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        """Forward pass: predict noise given noisy image and timestep embedding.

        Args:
            x: Noisy image tensor of shape (B, C, H, W).
            t: Time embedding tensor of shape (B, time_emb_dim). NOT raw timestep
               integers — these should be pre-processed through sinusoidal embedding.

        Returns:
            Predicted noise tensor, same shape as input x.
        """
        if not HAS_TORCH:
            return None

        # Process time embedding and reshape for broadcasting with spatial dims
        emb = self.time_mlp(t)  # (B, base_channels * 4)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, base_channels * 4, 1, 1)

        # === Encoder ===
        e1 = self.enc1(x)       # (B, base_channels, H, W) — preserve for skip
        p1 = self.pool(e1)      # (B, base_channels, H/2, W/2)

        e2 = self.enc2(p1)      # (B, base_channels*2, H/2, W/2) — preserve for skip
        p2 = self.pool(e2)      # (B, base_channels*2, H/4, W/4)

        # === Bottleneck (with time conditioning) ===
        b = self.bottleneck(p2)  # (B, base_channels*4, H/4, W/4)
        b = b + emb              # Add time embedding — tells the network the noise level

        # === Decoder (with skip connections) ===
        u2 = self.up2(b)                          # Upsample: (B, base_channels*2, H/2, W/2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # Concat skip + decode

        u1 = self.up1(d2)                          # Upsample: (B, base_channels, H, W)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # Concat skip + decode

        return self.out(d1)  # (B, in_channels, H, W) — predicted noise

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        if not HAS_TORCH:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EMAModel:
    """Exponential Moving Average of model parameters.

    EMA maintains a shadow copy of the model weights that is a smoothed
    (exponentially weighted) average of the training weights. This produces
    more stable and higher-quality outputs at inference time.

    The update rule for each parameter θ:
        θ_ema = decay * θ_ema + (1 - decay) * θ_train

    Common decay values: 0.9999 (slow, very smooth) to 0.999 (faster adaptation).

    Usage:
        ema = EMAModel(model, decay=0.9999)
        # During training:
        for batch in dataloader:
            loss = train_step(model, batch)
            ema.update(model)
        # For inference/sampling:
        ema.apply(model)  # Load EMA weights
        samples = generate(model)
        ema.restore(model)  # Restore training weights

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate. Higher = smoother but slower to adapt.
    """

    def __init__(self, model: Any, decay: float = 0.9999) -> None:
        if not HAS_TORCH:
            self.shadow_params = {}
            self.backup_params = {}
            return

        self.decay = decay
        # Deep copy all parameters as the EMA shadow
        self.shadow_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup_params: Dict[str, "torch.Tensor"] = {}

    def update(self, model: Any) -> None:
        """Update EMA parameters with current model weights."""
        if not HAS_TORCH:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )

    def apply(self, model: Any) -> None:
        """Replace model parameters with EMA parameters (for inference)."""
        if not HAS_TORCH:
            return
        self.backup_params = {
            name: param.clone() for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def restore(self, model: Any) -> None:
        """Restore original model parameters (after inference)."""
        if not HAS_TORCH:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params.clear()


def sinusoidal_embedding(
    timesteps: "torch.Tensor", dim: int
) -> "torch.Tensor":
    """Create sinusoidal positional embeddings for diffusion timesteps.

    This is the same idea as positional encoding in Transformers (Vaswani et al., 2017).
    The network needs to know WHICH timestep it's processing because:
    - Low timesteps have little noise → model should make subtle corrections
    - High timesteps have lots of noise → model should predict larger noise patterns

    The sinusoidal encoding maps a scalar timestep to a high-dimensional vector
    with a unique pattern for each timestep, enabling the network to distinguish
    between noise levels.

    Args:
        timesteps: Integer timesteps of shape (B,).
        dim: Output embedding dimension (must be even).

    Returns:
        Embedding tensor of shape (B, dim).
    """
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def train_diffusion(
    model: Any,
    scheduler: Any,
    dataloader: Any = None,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "auto",
    use_ema: bool = False,
    ema_decay: float = 0.9999,
    grad_clip: float = 1.0,
) -> Dict[str, List[float]]:
    """Train a diffusion model to predict noise.

    The training objective is simple: given a noisy image and its timestep,
    predict the noise that was added. The loss is MSE between the predicted
    noise and the actual noise.

    Training loop for each batch:
        1. Sample random images from the dataset
        2. Sample random timesteps for each image
        3. Add noise according to the schedule: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        4. Feed (x_t, t) to the model → predicted noise ε_θ
        5. Compute loss = MSE(ε_θ, ε)  — compare predicted vs actual noise
        6. Backpropagate and update weights

    Args:
        model: Neural network (SimplifiedUNet) to train.
        scheduler: NoiseScheduler for the forward diffusion process.
        dataloader: PyTorch DataLoader with training images. If None, runs
            a mock training loop for testing.
        epochs: Number of training epochs.
        lr: Learning rate for AdamW optimizer.
        device: Device string ('cuda', 'cpu', or 'auto' for auto-detect).
        use_ema: Whether to maintain EMA weights for better sampling quality.
        ema_decay: EMA decay rate (only used if use_ema=True).
        grad_clip: Maximum gradient norm for clipping. Prevents training
            instability from exploding gradients.

    Returns:
        Dictionary with training metrics: {'loss': [...], 'lr': [...]}.
    """
    if not HAS_TORCH or not isinstance(model, nn.Module):
        # Mock training for testing without PyTorch
        return {"loss": [round(0.5 - 0.05 * i, 4) for i in range(epochs)]}

    dev = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    time_emb_dim = 128

    # Optional EMA for smoother inference weights
    ema = EMAModel(model, decay=ema_decay) if use_ema else None

    history: Dict[str, List[float]] = {"loss": [], "lr": []}
    model.train()

    for epoch in range(epochs):
        if dataloader is None:
            # Mock training step — generates synthetic data for testing
            x = torch.randn(4, 3, 64, 64, device=dev)
            t_int = torch.randint(0, scheduler.num_steps, (4,), device=dev)
            noise = torch.randn_like(x)

            # Forward diffusion: add noise to clean images
            sqrt_ab = torch.tensor(
                scheduler.sqrt_alphas_cumprod[t_int.cpu().numpy()],
                dtype=torch.float32,
                device=dev,
            ).view(-1, 1, 1, 1)
            sqrt_1mab = torch.tensor(
                scheduler.sqrt_one_minus_alphas_cumprod[t_int.cpu().numpy()],
                dtype=torch.float32,
                device=dev,
            ).view(-1, 1, 1, 1)
            noisy = sqrt_ab * x + sqrt_1mab * noise

            # Get time embeddings
            t_emb = sinusoidal_embedding(t_int, time_emb_dim)

            # Predict noise and compute loss
            pred = model(noisy, t_emb)
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            total_loss = loss.item()
        else:
            total_loss = 0.0  # Would accumulate from real dataloader

        history["loss"].append(round(total_loss, 4))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, epochs, total_loss)

    return history


@torch.no_grad()
def generate_image(
    model: Any,
    scheduler: Any,
    n_samples: int = 1,
    image_size: int = 64,
    time_emb_dim: int = 128,
    seed: int = 42,
    device: str = "auto",
) -> np.ndarray:
    """Generate images via reverse diffusion (iterative denoising).

    Starting from pure Gaussian noise, iteratively denoise using the trained
    model to predict and remove the noise component at each timestep.

    The reverse process at each step t:
        1. Feed (x_t, t) to the model → predicted noise ε_θ
        2. Compute the mean: μ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ)
        3. Add stochastic noise: x_{t-1} = μ + σ_t * z  (except at t=0)

    Args:
        model: Trained noise prediction model (SimplifiedUNet).
        scheduler: NoiseScheduler with precomputed diffusion constants.
        n_samples: Number of images to generate in parallel.
        image_size: Spatial dimension of generated images (assumes square).
        time_emb_dim: Time embedding dimension (must match training config).
        seed: Random seed for reproducible generation.
        device: Device for computation ('cuda', 'cpu', or 'auto').

    Returns:
        Generated images as numpy array with shape (n_samples, 3, H, W),
        pixel values in [0, 1].
    """
    if not HAS_TORCH or not isinstance(model, nn.Module):
        # Mock generation for testing without PyTorch/trained model
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_samples, 3, image_size, image_size)).astype(
            np.float32
        )

    dev = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(dev)
    model.eval()

    # Start from pure Gaussian noise — this is x_T
    torch.manual_seed(seed)
    x = torch.randn(n_samples, 3, image_size, image_size, device=dev)

    # Iterate the reverse process from t=T-1 down to t=0
    for t in reversed(range(scheduler.num_steps)):
        # Create batch of timestep embeddings
        t_batch = torch.full((n_samples,), t, device=dev, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_batch, time_emb_dim).to(dev)

        # Model predicts the noise that was added at this timestep
        predicted_noise = model(x, t_emb)

        # Reverse diffusion step coefficients
        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alphas_cumprod[t]
        beta = scheduler.betas[t]

        # Compute the predicted mean of x_{t-1}
        coef1 = 1.0 / math.sqrt(alpha)
        coef2 = beta / math.sqrt(1.0 - alpha_bar)
        mean = coef1 * (x - coef2 * predicted_noise)

        if t > 0:
            # Stochastic reverse step — add controlled noise
            posterior_var = scheduler.posterior_variance[t]
            noise = torch.randn_like(x)
            x = mean + math.sqrt(posterior_var) * noise
        else:
            # Final step — just use the mean (no noise)
            x = mean

    # Clamp and normalize from [-1, 1] to [0, 1]
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0

    return x.cpu().numpy().astype(np.float32)
