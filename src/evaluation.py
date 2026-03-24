"""
Evaluation Metrics for Diffusion Image Generation Models.

Evaluating generative models is fundamentally different from evaluating
classifiers. We can't just check accuracy — we need to measure:

1. **Quality**: Do generated images look realistic? (FID, IS)
2. **Diversity**: Does the model produce varied outputs? (IS, coverage)
3. **Fidelity**: Does the model capture the training distribution? (FID)

Key Metrics Implemented:
    - **FID (Fréchet Inception Distance)**: The gold standard for generative models.
      Compares the distribution of real vs generated images in feature space.
      Lower = better. FID < 50 is decent, < 10 is excellent.
    - **IS (Inception Score)**: Measures both quality (confident predictions)
      and diversity (varied predictions). Higher = better. Range: 1 to ~12.
    - **SSIM (Structural Similarity)**: Pixel-level structural comparison
      between image pairs. Useful for super-resolution and inpainting.
    - **PSNR (Peak Signal-to-Noise Ratio)**: Simple pixel-level quality
      metric in decibels. Higher = less distortion.

References:
    - Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
      Converge to a Local Nash Equilibrium" (2017) — FID
    - Salimans et al., "Improved Techniques for Training GANs" (2016) — IS
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_fid_score(
    real_features: NDArray[np.float64],
    generated_features: NDArray[np.float64],
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet Inception Distance (FID) between real and generated features.

    FID measures the distance between two multivariate Gaussian distributions
    fitted to the real and generated feature vectors (typically from an
    Inception-v3 network's penultimate layer).

    Formula:
        FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 * (Σ_r · Σ_g)^{1/2})

    Interpretation:
        - FID = 0: Identical distributions (perfect generation)
        - FID < 10: Excellent quality (state-of-the-art models)
        - FID 10-50: Good quality (usable generations)
        - FID > 50: Noticeable artifacts or mode collapse

    Args:
        real_features: Feature vectors from real images, shape (N, D).
        generated_features: Feature vectors from generated images, shape (M, D).
        eps: Small constant for numerical stability in matrix square root.

    Returns:
        FID score (lower is better). Returns 0.0 if inputs are None.

    Note:
        This is a simplified implementation using eigenvalue decomposition.
        For production use, consider the `pytorch-fid` package which uses
        scipy.linalg.sqrtm for more numerically stable computation.
    """
    if real_features is None or generated_features is None:
        return 0.0

    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError(
            f"Features must be 2D arrays, got shapes "
            f"{real_features.shape} and {generated_features.shape}"
        )

    # Compute mean and covariance of both distributions
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    # Squared difference of means: ||μ_r - μ_g||²
    diff = mu_r - mu_g
    mean_diff_sq = float(np.dot(diff, diff))

    # Matrix square root via eigenvalue decomposition:
    # (Σ_r · Σ_g)^{1/2} computed as Q · diag(√λ) · Q^{-1}
    product = sigma_r @ sigma_g
    eigenvalues, eigenvectors = np.linalg.eigh(product)

    # Clamp negative eigenvalues (numerical noise) to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_product = eigenvectors @ np.diag(np.sqrt(eigenvalues + eps)) @ eigenvectors.T

    # FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 * sqrt(Σ_r · Σ_g))
    trace_term = float(np.trace(sigma_r + sigma_g - 2.0 * sqrt_product))

    return mean_diff_sq + trace_term


def compute_psnr(
    original: NDArray[np.float64],
    generated: NDArray[np.float64],
    max_pixel: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR measures pixel-level reconstruction quality in decibels (dB).
    Higher values indicate less distortion.

    Formula:
        PSNR = 10 * log10(MAX² / MSE)

    Interpretation:
        - PSNR > 40 dB: Excellent (imperceptible differences)
        - PSNR 30-40 dB: Good (minor artifacts)
        - PSNR 20-30 dB: Acceptable (visible but tolerable)
        - PSNR < 20 dB: Poor (significant distortion)

    Args:
        original: Reference image array, values in [0, max_pixel].
        generated: Comparison image array, same shape as original.
        max_pixel: Maximum pixel value (1.0 for normalized, 255 for uint8).

    Returns:
        PSNR value in decibels. Returns inf if images are identical.
    """
    mse = float(np.mean((original - generated) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_pixel ** 2) / mse))


def compute_ssim(
    img1: NDArray[np.float64],
    img2: NDArray[np.float64],
    window_size: int = 7,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    SSIM measures perceptual similarity by comparing luminance, contrast,
    and structural patterns. It correlates better with human perception
    than MSE/PSNR because it considers spatial relationships.

    Formula:
        SSIM(x, y) = (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))

    Interpretation:
        - SSIM = 1.0: Identical images
        - SSIM > 0.9: Very similar
        - SSIM 0.7-0.9: Moderately similar
        - SSIM < 0.7: Significant structural differences

    Args:
        img1: First image (H, W) or (H, W, C), values in [0, 1].
        img2: Second image, same shape as img1.
        window_size: Size of the sliding comparison window.
        C1: Stability constant for luminance comparison.
        C2: Stability constant for contrast comparison.

    Returns:
        SSIM value in range [-1, 1] (typically [0, 1] for natural images).
    """
    # For multi-channel images, average SSIM across channels
    if img1.ndim == 3:
        return float(np.mean([
            compute_ssim(img1[:, :, c], img2[:, :, c], window_size, C1, C2)
            for c in range(img1.shape[-1])
        ]))

    # Compute local means using a simple box filter (uniform window)
    from scipy.ndimage import uniform_filter
    mu1 = uniform_filter(img1.astype(np.float64), size=window_size)
    mu2 = uniform_filter(img2.astype(np.float64), size=window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = uniform_filter(img1.astype(np.float64) ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2.astype(np.float64) ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(
        img1.astype(np.float64) * img2.astype(np.float64), size=window_size
    ) - mu1_mu2

    # SSIM formula
    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_pixel_diversity(images: NDArray[np.float64]) -> float:
    """Measure diversity of generated images via average pairwise distance.

    Low diversity (mode collapse) is a common failure mode in generative models.
    This metric computes the average L2 distance between pairs of generated
    images — higher values indicate more diverse outputs.

    Args:
        images: Batch of images, shape (N, ...) where N >= 2.

    Returns:
        Average pairwise L2 distance. Returns 0.0 for < 2 images.
    """
    n = len(images)
    if n < 2:
        return 0.0

    flat = images.reshape(n, -1)
    total_dist = 0.0
    n_pairs = 0
    for i in range(min(n, 50)):  # Cap at 50 to avoid O(n²) blowup
        for j in range(i + 1, min(n, 50)):
            total_dist += float(np.linalg.norm(flat[i] - flat[j]))
            n_pairs += 1

    return total_dist / max(n_pairs, 1)


def generate_report(
    training_history: Dict[str, List[float]],
    n_images: int = 0,
    fid_score: Optional[float] = None,
    diversity_score: Optional[float] = None,
) -> str:
    """Generate a formatted evaluation report for the diffusion model.

    Creates a Markdown-formatted report summarizing training metrics,
    generation statistics, and quality scores.

    Args:
        training_history: Dictionary containing training metrics (e.g., 'loss').
        n_images: Number of images generated during evaluation.
        fid_score: Optional FID score from image quality evaluation.
        diversity_score: Optional diversity score from pixel diversity metric.

    Returns:
        Markdown-formatted report string.
    """
    lines = [
        "# Diffusion Model — Evaluation Report",
        "",
        "## Training Summary",
        f"- **Total epochs**: {len(training_history.get('loss', []))}",
    ]

    if training_history.get("loss"):
        losses = training_history["loss"]
        lines.extend([
            f"- **Initial loss**: {losses[0]:.4f}",
            f"- **Final loss**: {losses[-1]:.4f}",
            f"- **Best loss**: {min(losses):.4f} (epoch {np.argmin(losses) + 1})",
            f"- **Loss reduction**: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%",
        ])

    lines.extend(["", "## Generation"])
    lines.append(f"- **Images generated**: {n_images}")

    if fid_score is not None:
        quality = (
            "excellent" if fid_score < 10
            else "good" if fid_score < 50
            else "fair" if fid_score < 100
            else "needs improvement"
        )
        lines.append(f"- **FID score**: {fid_score:.2f} ({quality})")

    if diversity_score is not None:
        lines.append(f"- **Pixel diversity**: {diversity_score:.4f}")

    lines.extend([
        "",
        "## Recommendations",
        "- Train for more epochs if loss is still decreasing",
        "- Try cosine noise schedule if using linear (often better quality)",
        "- Increase base_channels for higher capacity (at cost of VRAM)",
        "- Use EMA weights for sampling (smoother, higher-quality outputs)",
    ])

    return "\n".join(lines)
