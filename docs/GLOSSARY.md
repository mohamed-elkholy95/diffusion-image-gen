# 📚 Diffusion Models — Glossary of Key Concepts

A reference guide for understanding the terminology and mathematics behind
Denoising Diffusion Probabilistic Models (DDPM).

---

## Core Concepts

### Forward Process (Diffusion)
The process of gradually adding Gaussian noise to a clean image over T timesteps
until it becomes pure noise. Each step adds a small amount of noise controlled by
the beta schedule. Mathematically:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

The key insight is that we can jump directly to any timestep t without iterating:

```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
```

### Reverse Process (Denoising)
The learned process of removing noise step-by-step to recover a clean image.
A neural network (U-Net) learns to predict the noise at each step:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t · I)
```

### Noise Schedule
Defines how much noise (β_t) is added at each timestep. Two common types:

| Schedule | Formula | Pros | Cons |
|----------|---------|------|------|
| **Linear** | β_t = β_start + t/(T-1) · (β_end - β_start) | Simple, predictable | Destroys info quickly |
| **Cosine** | Derived from cos((t/T + s)/(1+s) · π/2)² | Better quality, preserves signal | Slightly more complex |

---

## Key Variables

| Symbol | Name | What It Means |
|--------|------|---------------|
| **β_t** | Beta | Noise variance at timestep t. Small (~0.0001 to 0.02) |
| **α_t** | Alpha | Signal retention = 1 - β_t |
| **ᾱ_t** | Alpha bar | Cumulative signal retention = ∏ α_i from i=1 to t |
| **ε** | Epsilon | Gaussian noise ~ N(0, I) added during forward process |
| **ε_θ** | Epsilon theta | The noise predicted by the neural network |
| **x_0** | Clean image | The original, noise-free image |
| **x_t** | Noisy image | Image at timestep t (partially corrupted) |
| **x_T** | Pure noise | Image at final timestep (≈ Gaussian noise) |
| **T** | Total steps | Number of diffusion timesteps (typically 200-1000) |
| **σ²_t** | Posterior variance | Variance of the reverse step at timestep t |

---

## Architecture Terms

### U-Net
An encoder-decoder neural network with skip connections. The encoder compresses
the input into a low-dimensional representation, and the decoder reconstructs it.
Skip connections pass detailed information from encoder layers directly to decoder
layers, preserving spatial detail.

### Skip Connections
Direct connections from encoder layers to corresponding decoder layers. Without them,
fine details (edges, textures) would be lost during compression.

### Time Embedding
A sinusoidal positional encoding that tells the network which timestep it's
processing. Different timesteps have different noise levels, so the network
needs to adjust its predictions accordingly.

### GroupNorm
A normalization technique that divides channels into groups and normalizes within
each group. Preferred over BatchNorm in diffusion models because training often
uses small batch sizes where batch statistics are noisy.

### SiLU (Swish)
Activation function: SiLU(x) = x · σ(x). Smooth and non-monotonic, which helps
with gradient flow. Used in most modern diffusion architectures.

---

## Training Concepts

### Noise Prediction Objective
Instead of directly predicting x_0, the model predicts ε (the noise that was added).
This is equivalent but trains more stably. Loss = MSE(ε_θ, ε).

### EMA (Exponential Moving Average)
A shadow copy of model weights that is a smoothed average of training weights:
`θ_ema = decay · θ_ema + (1-decay) · θ_train`. Produces higher-quality samples
at inference time. Typical decay: 0.9999.

### Gradient Clipping
Limits the maximum gradient norm during backpropagation. Prevents training
instability from sudden large gradient updates. Typical max norm: 1.0.

---

## Evaluation Metrics

### FID (Fréchet Inception Distance)
Measures distance between distributions of real and generated images in feature
space. **Lower is better.** Uses Inception-v3 features.

| FID Range | Quality |
|-----------|---------|
| < 10 | Excellent (state-of-the-art) |
| 10-50 | Good (usable) |
| 50-100 | Fair (visible artifacts) |
| > 100 | Poor (mode collapse or artifacts) |

### IS (Inception Score)
Measures quality (confident class predictions) and diversity (varied predictions).
**Higher is better.** Range: 1 to ~12.

### PSNR (Peak Signal-to-Noise Ratio)
Pixel-level quality in decibels. **Higher is better.** 40+ dB is excellent.

### SSIM (Structural Similarity)
Perceptual similarity considering luminance, contrast, and structure.
**Range: [-1, 1]**, with 1.0 = identical. Better correlated with human
perception than MSE/PSNR.

---

## Sampling Methods

### DDPM Sampling
The original sampling algorithm. Uses all T timesteps sequentially (slow but
high quality). Each step adds stochastic noise for diversity.

### DDIM Sampling
Deterministic variant (Song et al., 2020). Can skip timesteps for faster
generation (e.g., 50 steps instead of 1000). Same quality with fewer steps.

### Classifier-Free Guidance (CFG)
Steers generation toward a text prompt by combining conditional and unconditional
predictions. Scale parameter (typically 7.5) controls adherence to the prompt.
Not implemented in this project (unconditional generation only).

---

## References

1. **Ho et al.** — "Denoising Diffusion Probabilistic Models" (2020)
   *The foundational DDPM paper*

2. **Nichol & Dhariwal** — "Improved Denoising Diffusion Probabilistic Models" (2021)
   *Cosine schedule, learned variance*

3. **Song et al.** — "Denoising Diffusion Implicit Models" (2020)
   *DDIM — faster deterministic sampling*

4. **Ronneberger et al.** — "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
   *The original U-Net architecture*

5. **Vaswani et al.** — "Attention Is All You Need" (2017)
   *Sinusoidal positional encoding (adapted for time embedding)*
