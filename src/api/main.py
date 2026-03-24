"""
FastAPI service for diffusion image generation.

Provides REST endpoints for generating images and inspecting model status.
Designed as a portfolio demonstration of how to wrap an ML model in a
production-style API with proper validation, error handling, and docs.

Endpoints:
    GET  /health         — Health check with model status
    GET  /model/info     — Model architecture and configuration details
    POST /generate       — Generate an image from a text prompt
    GET  /schedules      — Compare available noise schedules
"""
import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.noise_scheduler import NoiseScheduler
from src import config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diffusion Image Gen API",
    description=(
        "REST API for DDPM-based image generation. Supports configurable noise "
        "schedules, sampling steps, and seed-based reproducibility."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Parameters for image generation."""

    prompt: str = Field(
        default="a photo of a cat",
        min_length=1,
        max_length=500,
        description="Text description of the desired image.",
    )
    steps: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of reverse diffusion (denoising) steps. More steps = higher quality but slower.",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducible generation. None = random.",
    )
    schedule: str = Field(
        default="linear",
        description="Noise schedule type: 'linear' or 'cosine'.",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "prompt": "a sunset over mountains",
                    "steps": 100,
                    "seed": 42,
                    "schedule": "linear",
                }
            ]
        }


class GenerateResponse(BaseModel):
    """Result of image generation."""

    status: str
    message: str
    image_size: str
    steps: int
    schedule: str
    seed: Optional[int]
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model architecture and configuration details."""

    architecture: str
    image_size: int
    diffusion_steps: int
    noise_schedule: str
    learning_rate: float
    batch_size: int
    base_channels: int
    device: str


class ScheduleComparisonResponse(BaseModel):
    """Comparison of available noise schedules."""

    schedules: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check API health and model loading status.

    Returns the current server status. In production, this would also verify
    GPU availability and model weights are loaded in memory.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=False,
        version="1.1.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model architecture and training configuration.

    Returns details about the U-Net architecture, noise schedule,
    and training hyperparameters. Useful for debugging and documentation.
    """
    return ModelInfoResponse(
        architecture="SimplifiedUNet (encoder-decoder with skip connections)",
        image_size=config.IMAGE_SIZE,
        diffusion_steps=config.DIFFUSION_STEPS,
        noise_schedule=config.NOISE_SCHEDULE,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        base_channels=32,
        device="cuda (if available)",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an image using reverse diffusion sampling.

    Takes a text prompt and sampling parameters, then runs the reverse
    diffusion process to generate a 64×64 RGB image.

    In this demo version, actual model inference is simulated. In production,
    this would load the trained U-Net weights and run the full reverse
    diffusion loop.
    """
    if req.schedule not in ("linear", "cosine"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schedule '{req.schedule}'. Must be 'linear' or 'cosine'.",
        )

    start = time.monotonic()

    # In production: load model, run reverse diffusion, return base64 image
    # For demo: simulate the generation with timing
    logger.info("Generating image: prompt='%s', steps=%d, schedule=%s", req.prompt, req.steps, req.schedule)

    elapsed_ms = round((time.monotonic() - start) * 1000, 2)

    return GenerateResponse(
        status="generated",
        message=f"Image for: {req.prompt}",
        image_size=f"{config.IMAGE_SIZE}x{config.IMAGE_SIZE}x3",
        steps=req.steps,
        schedule=req.schedule,
        seed=req.seed,
        elapsed_ms=elapsed_ms,
    )


@app.get("/schedules", response_model=ScheduleComparisonResponse)
async def compare_schedules():
    """Compare linear and cosine noise schedules side by side.

    Shows key statistics for each schedule type to help users understand
    the tradeoffs. Cosine generally preserves more signal at high timesteps,
    leading to better image quality.
    """
    linear = NoiseScheduler(num_steps=config.DIFFUSION_STEPS, schedule="linear")
    cosine = NoiseScheduler(num_steps=config.DIFFUSION_STEPS, schedule="cosine")

    return ScheduleComparisonResponse(
        schedules={
            "linear": linear.get_schedule_stats(),
            "cosine": cosine.get_schedule_stats(),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
