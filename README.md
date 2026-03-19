<div align="center">

# 🎨 Diffusion Image Generation

**Denoising Diffusion Probabilistic Model** with U-Net architecture and CUDA acceleration

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda)
[![Tests](https://img.shields.io/badge/Tests-10%20passed-success?style=flat-square)](#)

</div>

## Overview

A **diffusion model implementation** following the DDPM framework with a simplified U-Net backbone, configurable noise schedules (linear/cosine), and GPU-accelerated training via PyTorch CUDA. Includes noise scheduling visualization and sampling pipelines.

## Features

- 🏗️ **Simplified U-Net** — Encoder-decoder with skip connections and time embeddings
- 📊 **Noise Scheduling** — Linear and cosine schedules with configurable beta ranges
- 🎲 **Forward/Reverse Process** — Full DDPM training and image generation pipeline
- 🚀 **CUDA Acceleration** — GPU-accelerated training on NVIDIA RTX (tested on RTX 3050 Ti)
- 🖼️ **Image Generation** — Iterative denoising with configurable sampling steps
- 📈 **Training Dashboard** — Loss curves and generation visualization

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/diffusion-image-gen.git
cd diffusion-image-gen
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
