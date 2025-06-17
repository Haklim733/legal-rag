import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from instructor import patch
from models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    QueryRequest,
    FOLIOExtraction,
)

app = FastAPI(
    title="Query/Embedding Service",
    description="Hugging Face model service for LightRAG",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# Global instances
model = None
patched_model = None

# Cache configuration
response_cache: Dict[str, FOLIOExtraction] = {}
CACHE_TTL = 3600  # 1 hour


@app.on_event("startup")
async def load_model():
    """Load the model at startup"""
    global model, patched_model
    try:
        start_time = time.time()

        # Create model directory if it doesn't exist
        model_path = MODEL_DIR / MODEL_NAME.replace("/", "--")
        model_path.mkdir(parents=True, exist_ok=True)

        # Load the model - SentenceTransformer handles downloading automatically
        model = SentenceTransformer(
            MODEL_NAME, device=MODEL_DEVICE, cache_folder=str(model_path)
        )

        # Warm up
        model.encode(["warmup"])

        # Patch for instructor
        patched_model = patch(model)

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    except Exception as e:
        error_msg = f"Error loading model {MODEL_NAME}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": MODEL_NAME,
        "device": MODEL_DEVICE,
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return ModelInfo(
        model_id=MODEL_NAME,
        model_type="sentence_transformer",
        max_seq_length=model.max_seq_length,
        embedding_dimension=model.get_sentence_embedding_dimension(),
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for a list of texts"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        embeddings = model.encode(
            request.inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=False,
        )

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return {
            "embeddings": embeddings,
            "model": MODEL_NAME,
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}",
        )


@app.post("/query", response_model=FOLIOExtraction)
async def query_text(
    request: QueryRequest, background_tasks: BackgroundTasks
) -> FOLIOExtraction:
    """Extract FOLIO types from text using instructor-patched model"""
    if not patched_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        # Check cache first
        cache_key = f"{request.text}:{','.join(request.branches or [])}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        # Get extraction
        extraction = await patched_model.complete(
            request.text,
            response_model=FOLIOExtraction,
            system_prompt=f"""
            You are a FOLIO type extraction expert. Analyze the text and identify all relevant FOLIO types.
            For each type, provide:
            1. The exact type name from the available types
            2. The branch it belongs to
            3. A confidence score (0-1)
            4. The relevant context from the text
            
            Available types: {FOLIO_TYPES}
            """,
        )

        # Cache the result
        response_cache[cache_key] = extraction

        # Schedule cache cleanup
        background_tasks.add_task(cleanup_cache)

        return extraction

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


async def cleanup_cache():
    """Clean up expired cache entries"""
    current_time = time.time()
    expired_keys = [
        key
        for key, value in response_cache.items()
        if current_time - value.timestamp > CACHE_TTL
    ]
    for key in expired_keys:
        del response_cache[key]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        workers=int(os.getenv("WEB_CONCURRENCY", 1)),
    )
