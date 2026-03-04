"""FastAPI application for VLM Temporal Operation Intelligence."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from app.model import VLMPredictor
from app.schemas import PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: VLMPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
    adapter_path = os.getenv("ADAPTER_PATH", None)
    quantize = os.getenv("QUANTIZE_4BIT", "true").lower() == "true"

    predictor = VLMPredictor(
        model_id=model_id,
        adapter_path=adapter_path,
        quantize_4bit=quantize,
    )
    predictor.load()
    logger.info("Model ready for inference")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="VLM Temporal Operation Intelligence",
    description="Video-based temporal operation recognition and anticipation for logistics",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.model is not None,
        model_name=predictor.model_id if predictor else "",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Video clip file (mp4, avi)"),
    clip_id: str = Query(default="unknown", description="Unique clip identifier"),
    n_frames: int = Query(default=8, ge=1, le=32, description="Number of frames to sample"),
):
    """
    Analyze a video clip of warehouse packaging operations.

    Returns the dominant operation, temporal boundaries, and anticipated next operation.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if file.content_type and not file.content_type.startswith("video/"):
        # Also accept octet-stream as some clients don't set content type
        if file.content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400,
                detail=f"Expected video file, got {file.content_type}",
            )

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predictor.predict(
            video_path=tmp_path, clip_id=clip_id, n_frames=n_frames
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/predict/frames", response_model=PredictionResponse)
async def predict_from_frames(
    files: list[UploadFile] = File(..., description="Ordered frame images"),
    clip_id: str = Query(default="unknown", description="Unique clip identifier"),
    total_frames: int = Query(default=125, description="Total frames in original clip"),
):
    """
    Analyze pre-extracted frames from a warehouse packaging video clip.

    Accepts multiple image files representing ordered frames from the clip.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from PIL import Image
    import io

    frames = []
    for f in files:
        content = await f.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        frames.append(img)

    try:
        result = predictor.predict_from_frames(
            frames=frames, clip_id=clip_id, total_frames=total_frames
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
