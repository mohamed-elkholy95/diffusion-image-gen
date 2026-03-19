"""FastAPI for diffusion."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
logger = logging.getLogger(__name__)
app = FastAPI(title="Diffusion Image Gen API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str = Field(default="a photo of a cat", min_length=1, max_length=500)
    steps: int = Field(default=50, ge=1, le=1000)

class GenerateResponse(BaseModel):
    status: str; message: str; image_size: str; steps: int

@app.get("/health")
async def health(): return {"status": "healthy", "model_loaded": False}

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    return GenerateResponse(status="generated", message=f"Image for: {req.prompt}",
        image_size="64x64x3", steps=req.steps)

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8006)
