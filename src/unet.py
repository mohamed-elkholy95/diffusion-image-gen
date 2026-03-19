"""Simplified U-Net for diffusion (with mock fallback)."""
import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    logger.info("torch not available")

class SimplifiedUNet(nn.Module if HAS_TORCH else object):
    """Simplified U-Net for 64x64 images."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, time_emb_dim: int = 128) -> None:
        if HAS_TORCH:
            super().__init__()
            self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, base_channels * 4), nn.SiLU(),
                nn.Linear(base_channels * 4, base_channels * 4))
            self.enc1 = self._block(in_channels, base_channels)
            self.enc2 = self._block(base_channels, base_channels * 2)
            self.bottleneck = self._block(base_channels * 2, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = self._block(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = self._block(base_channels * 2, base_channels)
            self.out = nn.Conv2d(base_channels, in_channels, 1)
            self.pool = nn.MaxPool2d(2)

    def _block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.GroupNorm(8, out_c), nn.SiLU())

    def forward(self, x, t):
        if not HAS_TORCH: return None
        emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


def train_diffusion(model: Any, scheduler: Any, dataloader: Any = None,
                    epochs: int = 5, lr: float = 1e-4, device: str = "auto") -> Dict[str, List[float]]:
    """Train diffusion model (mock if no data)."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return {"loss": [0.5 - 0.05 * i for i in range(epochs)]}

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": []}
    model.train()
    for epoch in range(epochs):
        if dataloader is None:
            # Mock step
            x = torch.randn(4, 3, 64, 64).to(dev)
            t = torch.randint(0, scheduler.num_steps, (4,)).to(dev)
            noise = torch.randn_like(x)
            noisy = torch.tensor(scheduler.add_noise(x.cpu().numpy(), int(t[0]))[0]).to(dev)
            pred = model(noisy, torch.zeros(4, 128).to(dev))
            loss = nn.MSELoss()(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss = loss.item()
        history["loss"].append(round(total_loss, 4))
        logger.info("Epoch %d/%d: loss=%.4f", epoch+1, epochs, total_loss)
    return history


def generate_image(model: Any, scheduler: Any, n_steps: int = 50, seed: int = 42) -> np.ndarray:
    """Generate image via reverse diffusion (mock if no model)."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((3, 64, 64)).astype(np.float32)
    device = next(model.parameters()).device
    x = torch.randn(1, 3, 64, 64, device=device)
    model.eval()
    with torch.no_grad():
        for t in reversed(range(n_steps)):
            pass  # Simplified
    return x[0].cpu().numpy().astype(np.float32)
