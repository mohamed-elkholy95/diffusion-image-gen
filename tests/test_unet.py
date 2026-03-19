import pytest
from src.unet import SimplifiedUNet, train_diffusion, generate_image

class TestSimplifiedUNet:
    def test_init(self):
        model = SimplifiedUNet(in_channels=3, base_channels=16)
        assert model is not None

    def test_train_mock(self):
        history = train_diffusion(None, None, epochs=3)
        assert "loss" in history
        assert len(history["loss"]) == 3

    def test_generate_mock(self):
        img = generate_image(None, None, seed=42)
        assert img.shape == (3, 64, 64)
