import pytest
import numpy as np
from src.noise_scheduler import NoiseScheduler

class TestNoiseScheduler:
    def test_linear(self):
        s = NoiseScheduler(num_steps=100, schedule="linear")
        assert len(s.betas) == 100
        assert s.betas[-1] > s.betas[0]

    def test_cosine(self):
        s = NoiseScheduler(num_steps=100, schedule="cosine")
        assert len(s.betas) == 100

    def test_add_noise(self):
        s = NoiseScheduler(num_steps=50)
        x = np.zeros((2, 3, 8, 8))
        noisy, noise = s.add_noise(x, t=25)
        assert noisy.shape == x.shape

    def test_get_stats(self):
        s = NoiseScheduler()
        stats = s.get_schedule_stats()
        assert "beta_range" in stats
