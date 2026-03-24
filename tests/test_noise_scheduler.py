"""Tests for the NoiseScheduler — validates forward diffusion behavior."""
import numpy as np
import pytest

from src.noise_scheduler import NoiseScheduler


class TestNoiseSchedulerInit:
    """Test scheduler initialization and validation."""

    def test_linear_schedule_shape(self):
        """Linear schedule should produce monotonically increasing betas."""
        s = NoiseScheduler(num_steps=100, schedule="linear")
        assert len(s.betas) == 100
        assert s.betas[-1] > s.betas[0]

    def test_cosine_schedule_shape(self):
        """Cosine schedule should produce valid betas within clipping bounds."""
        s = NoiseScheduler(num_steps=100, schedule="cosine")
        assert len(s.betas) == 100
        assert np.all(s.betas >= 1e-4)
        assert np.all(s.betas <= 0.9999)

    def test_cosine_preserves_more_signal(self):
        """Cosine schedule should retain more signal at the final timestep than linear."""
        linear = NoiseScheduler(num_steps=200, schedule="linear")
        cosine = NoiseScheduler(num_steps=200, schedule="cosine")
        # Cosine schedule is designed to preserve more signal at high timesteps
        assert cosine.alphas_cumprod[-1] > linear.alphas_cumprod[-1]

    def test_alphas_cumprod_decreasing(self):
        """Cumulative alpha product must be monotonically decreasing."""
        for sched in ["linear", "cosine"]:
            s = NoiseScheduler(num_steps=50, schedule=sched)
            diffs = np.diff(s.alphas_cumprod)
            assert np.all(diffs <= 0), f"{sched}: alphas_cumprod not monotonically decreasing"

    def test_alphas_cumprod_range(self):
        """ᾱ_t should start near 1.0 and end near 0.0 for sufficient steps."""
        s = NoiseScheduler(num_steps=200, schedule="linear")
        assert s.alphas_cumprod[0] > 0.99, "First ᾱ should be near 1.0"
        assert s.alphas_cumprod[-1] < 0.1, "Final ᾱ should be near 0.0"

    def test_posterior_variance_computed(self):
        """Posterior variance should be precomputed for reverse diffusion."""
        s = NoiseScheduler(num_steps=50)
        assert hasattr(s, "posterior_variance")
        assert len(s.posterior_variance) == 50

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="positive"):
            NoiseScheduler(num_steps=0)

    def test_invalid_beta_values(self):
        with pytest.raises(ValueError, match="positive"):
            NoiseScheduler(beta_start=-0.01)

    def test_invalid_beta_order(self):
        with pytest.raises(ValueError, match="less than"):
            NoiseScheduler(beta_start=0.02, beta_end=0.0001)

    def test_invalid_schedule_type(self):
        with pytest.raises(ValueError, match="schedule must be"):
            NoiseScheduler(schedule="exponential")


class TestAddNoise:
    """Test the forward diffusion process."""

    def test_add_noise_shape_preserved(self):
        """Output shape must match input shape."""
        s = NoiseScheduler(num_steps=50)
        x = np.zeros((2, 3, 8, 8))
        noisy, noise = s.add_noise(x, t=25)
        assert noisy.shape == x.shape
        assert noise.shape == x.shape

    def test_add_noise_at_t0_preserves_signal(self):
        """At t=0, almost no noise should be added (ᾱ_0 ≈ 1)."""
        s = NoiseScheduler(num_steps=100)
        x = np.ones((1, 3, 8, 8))
        noisy, _ = s.add_noise(x, t=0)
        # At t=0, noisy ≈ x since √ᾱ_0 ≈ 1 and √(1-ᾱ_0) ≈ 0
        assert np.allclose(noisy, x, atol=0.02)

    def test_add_noise_at_high_t_mostly_noise(self):
        """At high t, output should be dominated by noise."""
        s = NoiseScheduler(num_steps=200)
        x = np.ones((1, 3, 8, 8)) * 10.0  # Large signal
        noisy, _ = s.add_noise(x, t=199)
        # At t=199, √ᾱ is very small so signal should be heavily degraded
        assert np.abs(np.mean(noisy)) < np.abs(np.mean(x))

    def test_custom_noise(self):
        """Custom noise input should be used instead of random."""
        s = NoiseScheduler(num_steps=50)
        x = np.zeros((1, 3, 4, 4))
        custom_noise = np.ones((1, 3, 4, 4))
        noisy, returned_noise = s.add_noise(x, t=25, noise=custom_noise)
        np.testing.assert_array_equal(returned_noise, custom_noise)

    def test_timestep_out_of_range(self):
        """Out-of-range timesteps should raise IndexError."""
        s = NoiseScheduler(num_steps=50)
        x = np.zeros((1, 3, 4, 4))
        with pytest.raises(IndexError):
            s.add_noise(x, t=50)
        with pytest.raises(IndexError):
            s.add_noise(x, t=-1)

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical noise."""
        s1 = NoiseScheduler(num_steps=50, seed=123)
        s2 = NoiseScheduler(num_steps=50, seed=123)
        x = np.zeros((2, 3, 8, 8))
        noisy1, noise1 = s1.add_noise(x, t=10)
        noisy2, noise2 = s2.add_noise(x, t=10)
        np.testing.assert_array_equal(noise1, noise2)


class TestReverseStep:
    """Test the reverse diffusion (denoising) step."""

    def test_reverse_step_shape(self):
        """Reverse step output must have same shape as input."""
        s = NoiseScheduler(num_steps=50, seed=42)
        x_t = np.random.randn(1, 3, 8, 8)
        pred_noise = np.random.randn(1, 3, 8, 8)
        result = s.reverse_step(x_t, pred_noise, t=25)
        assert result.shape == x_t.shape

    def test_reverse_step_t0_no_noise(self):
        """At t=0, reverse step should return just the mean (no added noise)."""
        s = NoiseScheduler(num_steps=50, seed=42)
        x_t = np.ones((1, 3, 4, 4))
        pred_noise = np.zeros((1, 3, 4, 4))
        result = s.reverse_step(x_t, pred_noise, t=0)
        # At t=0 with zero predicted noise, result should be x_t / √α_0
        expected_coef = 1.0 / np.sqrt(s.alphas[0])
        np.testing.assert_allclose(result, x_t * expected_coef, atol=1e-6)


class TestScheduleStats:
    """Test diagnostic output."""

    def test_stats_keys(self):
        s = NoiseScheduler()
        stats = s.get_schedule_stats()
        assert "beta_range" in stats
        assert "schedule" in stats
        assert "midpoint_alpha_bar" in stats

    def test_repr(self):
        s = NoiseScheduler(num_steps=200, schedule="cosine")
        r = repr(s)
        assert "cosine" in r
        assert "200" in r
