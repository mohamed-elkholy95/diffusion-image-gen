"""Tests for SimplifiedUNet, EMA, training, and image generation."""
import numpy as np
import pytest

from src.unet import SimplifiedUNet, EMAModel, train_diffusion, generate_image
from src.noise_scheduler import NoiseScheduler


class TestSimplifiedUNet:
    """Test U-Net architecture initialization and forward pass."""

    def test_init_default(self):
        model = SimplifiedUNet(in_channels=3, base_channels=16)
        assert model is not None

    def test_count_parameters(self):
        """Model should have a nonzero parameter count."""
        model = SimplifiedUNet(in_channels=3, base_channels=16)
        count = model.count_parameters()
        assert count > 0, "Model should have trainable parameters"

    def test_larger_model_more_params(self):
        """Increasing base_channels should increase parameter count."""
        small = SimplifiedUNet(base_channels=16)
        large = SimplifiedUNet(base_channels=64)
        assert large.count_parameters() > small.count_parameters()


class TestEMAModel:
    """Test Exponential Moving Average weight tracking."""

    def test_ema_init(self):
        """EMA should initialize shadow params from model."""
        model = SimplifiedUNet(base_channels=8)
        ema = EMAModel(model, decay=0.999)
        assert len(ema.shadow_params) > 0

    def test_ema_update_changes_shadow(self):
        """After update, shadow params should differ from initial copy."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        model = SimplifiedUNet(base_channels=8)
        ema = EMAModel(model, decay=0.5)  # Low decay for visible change

        # Get initial shadow state
        initial_shadow = {
            k: v.clone() for k, v in ema.shadow_params.items()
        }

        # Modify model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        ema.update(model)

        # Shadow should have changed
        changed = False
        for name in initial_shadow:
            if not torch.equal(initial_shadow[name], ema.shadow_params[name]):
                changed = True
                break
        assert changed, "EMA shadow should update after model changes"

    def test_ema_apply_and_restore(self):
        """Apply should swap weights, restore should bring them back."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        model = SimplifiedUNet(base_channels=8)
        ema = EMAModel(model, decay=0.999)

        # Save original weights
        original = {
            name: p.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # Modify model (simulating training)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        # Apply EMA → model gets shadow weights (original-ish)
        ema.apply(model)

        # Restore → model gets back the modified weights
        ema.restore(model)

        for name, p in model.named_parameters():
            if p.requires_grad and name in original:
                # After modify + apply + restore, params should be original + 1.0
                expected = original[name] + 1.0
                assert torch.allclose(p, expected, atol=1e-6)


class TestTrainDiffusion:
    """Test the training loop (mock mode)."""

    def test_mock_training(self):
        """Mock training should return loss history."""
        history = train_diffusion(None, None, epochs=3)
        assert "loss" in history
        assert len(history["loss"]) == 3

    def test_mock_loss_decreasing(self):
        """Mock loss should decrease over epochs."""
        history = train_diffusion(None, None, epochs=5)
        assert history["loss"][0] > history["loss"][-1]

    def test_train_with_scheduler(self):
        """Training with a real scheduler (mock data) should work."""
        model = SimplifiedUNet(base_channels=8)
        scheduler = NoiseScheduler(num_steps=50)
        history = train_diffusion(model, scheduler, epochs=2, lr=1e-3)
        assert len(history["loss"]) == 2
        assert all(isinstance(v, float) for v in history["loss"])

    def test_train_with_ema(self):
        """Training with EMA enabled should not crash."""
        model = SimplifiedUNet(base_channels=8)
        scheduler = NoiseScheduler(num_steps=50)
        history = train_diffusion(
            model, scheduler, epochs=1, use_ema=True, ema_decay=0.99
        )
        assert len(history["loss"]) == 1


class TestGenerateImage:
    """Test image generation via reverse diffusion."""

    def test_mock_generation(self):
        """Mock generation should return correct shape."""
        img = generate_image(None, None, seed=42)
        assert img.shape == (1, 3, 64, 64)

    def test_mock_generation_multiple(self):
        """Mock generation should support n_samples > 1."""
        imgs = generate_image(None, None, n_samples=4, seed=42)
        assert imgs.shape == (4, 3, 64, 64)

    def test_mock_reproducible(self):
        """Same seed should produce identical mock output."""
        img1 = generate_image(None, None, seed=99)
        img2 = generate_image(None, None, seed=99)
        np.testing.assert_array_equal(img1, img2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different images."""
        img1 = generate_image(None, None, seed=1)
        img2 = generate_image(None, None, seed=2)
        assert not np.array_equal(img1, img2)
