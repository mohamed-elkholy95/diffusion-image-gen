"""Tests for evaluation metrics — FID, PSNR, SSIM, diversity, and reporting."""
import numpy as np
import pytest

from src.evaluation import (
    compute_fid_score,
    compute_pixel_diversity,
    compute_psnr,
    generate_report,
)


class TestFIDScore:
    """Test Fréchet Inception Distance computation."""

    def test_identical_distributions(self):
        """FID should be near zero for identical feature distributions."""
        features = np.random.randn(100, 64)
        fid = compute_fid_score(features, features.copy())
        assert fid < 1.0, f"FID for identical distributions should be ~0, got {fid}"

    def test_different_distributions(self):
        """FID should be positive for different distributions."""
        real = np.random.randn(100, 32)
        generated = np.random.randn(100, 32) + 5.0  # Shifted mean
        fid = compute_fid_score(real, generated)
        assert fid > 0.0

    def test_none_inputs(self):
        """Should return 0.0 for None inputs."""
        assert compute_fid_score(None, None) == 0.0
        assert compute_fid_score(np.array([[1]]), None) == 0.0

    def test_invalid_dimensions(self):
        """Should raise ValueError for non-2D inputs."""
        with pytest.raises(ValueError, match="2D"):
            compute_fid_score(np.array([1, 2, 3]), np.array([1, 2, 3]))


class TestPSNR:
    """Test Peak Signal-to-Noise Ratio."""

    def test_identical_images(self):
        """PSNR should be infinity for identical images."""
        img = np.random.rand(64, 64, 3)
        assert compute_psnr(img, img) == float("inf")

    def test_noisy_image(self):
        """PSNR should decrease as noise increases."""
        img = np.random.rand(64, 64)
        noisy_small = img + np.random.randn(*img.shape) * 0.01
        noisy_large = img + np.random.randn(*img.shape) * 0.1
        psnr_small = compute_psnr(img, noisy_small)
        psnr_large = compute_psnr(img, noisy_large)
        assert psnr_small > psnr_large

    def test_positive_value(self):
        """PSNR should be positive for reasonable noise levels."""
        img = np.ones((32, 32)) * 0.5
        noisy = img + np.random.randn(32, 32) * 0.05
        psnr = compute_psnr(img, noisy)
        assert psnr > 0


class TestPixelDiversity:
    """Test diversity measurement for generated image batches."""

    def test_identical_images_zero_diversity(self):
        """Identical images should have zero diversity."""
        imgs = np.stack([np.ones((3, 8, 8))] * 5)
        assert compute_pixel_diversity(imgs) == 0.0

    def test_diverse_images_positive(self):
        """Random images should have positive diversity."""
        imgs = np.random.randn(10, 3, 8, 8)
        assert compute_pixel_diversity(imgs) > 0.0

    def test_single_image(self):
        """Single image should return 0.0 diversity."""
        assert compute_pixel_diversity(np.random.randn(1, 3, 8, 8)) == 0.0


class TestReport:
    """Test evaluation report generation."""

    def test_report_contains_metrics(self):
        report = generate_report({"loss": [0.5, 0.4, 0.3]}, n_images=10)
        assert "# Diffusion" in report
        assert "10" in report
        assert "Final loss" in report
        assert "Best loss" in report

    def test_report_with_fid(self):
        report = generate_report({"loss": [0.5]}, fid_score=25.3)
        assert "25.30" in report
        assert "good" in report

    def test_report_with_diversity(self):
        report = generate_report({"loss": [0.5]}, diversity_score=1.234)
        assert "1.2340" in report

    def test_empty_loss(self):
        """Report should handle empty training history gracefully."""
        report = generate_report({"loss": []}, n_images=0)
        assert "0" in report

    def test_report_has_recommendations(self):
        report = generate_report({"loss": [0.5, 0.3]})
        assert "Recommendations" in report
        assert "EMA" in report
