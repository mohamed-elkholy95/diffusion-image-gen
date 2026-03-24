<div align="center">

# 🎨 Diffusion Image Generation

**Denoising Diffusion Probabilistic Model** with U-Net architecture and CUDA acceleration

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-success?style=flat-square)](#)

</div>

## Overview

A **diffusion model implementation** following the DDPM framework (Ho et al., 2020) with a simplified U-Net backbone, configurable noise schedules (linear/cosine), and GPU-accelerated training via PyTorch CUDA. Features a complete forward/reverse diffusion pipeline, EMA weight averaging, and comprehensive evaluation metrics.

## How Diffusion Models Work

Diffusion models generate images through a two-phase process:

```
Forward Process (Training):
  Clean Image → Add noise → Add noise → ... → Pure Noise
  x_0          x_1          x_2                x_T

Reverse Process (Generation):
  Pure Noise → Remove noise → Remove noise → ... → Generated Image
  x_T          x_{T-1}       x_{T-2}               x_0
```

**Training**: The model learns to predict the noise added at each step. Given a noisy image x_t and its timestep t, the U-Net predicts ε (the noise), and we minimize MSE(ε_predicted, ε_actual).

**Generation**: Starting from pure Gaussian noise, we iteratively denoise using the trained model, stepping backwards from T to 0.

## Features

- 🏗️ **Simplified U-Net** — Encoder-decoder with skip connections, GroupNorm, and sinusoidal time embeddings
- 📊 **Noise Scheduling** — Linear and cosine schedules (Nichol & Dhariwal, 2021) with configurable beta ranges
- 🎲 **Forward/Reverse Process** — Full DDPM pipeline with posterior variance computation
- 🚀 **CUDA Acceleration** — GPU-accelerated training on NVIDIA RTX (tested on RTX 3050 Ti)
- 🖼️ **Image Generation** — Iterative denoising with configurable sampling steps and seed control
- 📈 **Training Dashboard** — Streamlit app with loss curves and interactive generation
- 🔄 **EMA Weights** — Exponential Moving Average for smoother, higher-quality inference
- 📐 **Evaluation Suite** — FID, PSNR, SSIM, and pixel diversity metrics
- 🌐 **REST API** — FastAPI service with model info, schedule comparison, and generation endpoints

## Architecture

```
Input Image (B, 3, 64, 64)
    │
    ├── Encoder 1: Conv→GN→SiLU→Conv (3 → 32 channels) ──── Skip ──┐
    │   └── MaxPool (64→32)                                          │
    ├── Encoder 2: Conv→GN→SiLU→Conv (32 → 64 channels) ─── Skip ─┐│
    │   └── MaxPool (32→16)                                         ││
    ├── Bottleneck: Conv→GN→SiLU→Conv (64 → 128) + Time Embedding  ││
    │   └── Upsample (16→32)                                        ││
    ├── Decoder 2: Cat(skip) → Conv→GN→SiLU→Conv (192 → 64) ──────┘│
    │   └── Upsample (32→64)                                        │
    ├── Decoder 1: Cat(skip) → Conv→GN→SiLU→Conv (96 → 32) ───────┘
    │
    └── Output Conv 1×1 (32 → 3) → Predicted Noise (B, 3, 64, 64)
```

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/diffusion-image-gen.git
cd diffusion-image-gen
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Train on CIFAR-10

```bash
# Quick test (10 epochs, ~5 min on GPU)
python train_diffusion_cifar.py --quick

# Full training (50 epochs, ~4-8 hours on RTX 3050 Ti)
python train_diffusion_cifar.py

# With cosine schedule (often better quality)
python train_diffusion_cifar.py --schedule cosine --epochs 100
```

### Launch Dashboard

```bash
streamlit run streamlit_app/app.py
```

### Start API Server

```bash
python -m src.api.main
# API docs at http://localhost:8006/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with model status |
| `GET` | `/model/info` | Architecture and config details |
| `POST` | `/generate` | Generate an image from a prompt |
| `GET` | `/schedules` | Compare linear vs cosine schedules |

## Project Structure

```
├── src/
│   ├── unet.py              # U-Net architecture + EMA + training loop
│   ├── noise_scheduler.py   # Forward/reverse diffusion with linear/cosine schedules
│   ├── evaluation.py        # FID, PSNR, SSIM, pixel diversity metrics
│   ├── config.py            # Project configuration and paths
│   └── api/main.py          # FastAPI REST service
├── tests/                   # 45+ tests with pytest
├── train_diffusion_cifar.py # Full CIFAR-10 training script
├── streamlit_app/           # Interactive dashboard
└── docs/
    ├── ARCHITECTURE.md      # System design decisions
    ├── GLOSSARY.md          # Diffusion model terminology guide
    ├── DEVELOPMENT.md       # Setup and contribution guide
    └── CONTRIBUTING.md      # Issue and PR guidelines
```

## Key Concepts Covered

This project demonstrates understanding of:

- **Diffusion theory**: Forward process, reverse process, noise schedules, ELBO derivation
- **Neural architecture**: U-Net encoder-decoder, skip connections, time conditioning
- **Training techniques**: AdamW optimizer, cosine LR scheduling, gradient clipping, EMA
- **Evaluation**: FID (distribution distance), SSIM (perceptual similarity), PSNR, diversity
- **Engineering**: FastAPI serving, Streamlit dashboards, comprehensive testing

## References

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Nichol & Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (2021)
- Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2020)

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
