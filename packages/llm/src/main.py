import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

app = FastAPI(
    title="LightRAG Embedding Service",
    description="Hugging Face model service for LightRAG",
    version="0.1.0"
)

# CORS middleware to allow cross-origin requests
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
EMBEDDING_DIM = 384  # Default for all-MiniLM-L6-v2

# Global model instance
model = None

class EmbeddingRequest(BaseModel):
    inputs: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_length=1,
        max_length=100,  # Prevent abuse with too many inputs
        example=["This is a sample text"]
    )

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(
        ...,
        description="List of embeddings, one for each input text"
    )
    model: str = Field(
        ...,
        description="Name of the model used for generating embeddings"
    )
    dimensions: int = Field(
        ...,
        description="Dimensionality of the embedding vectors"
    )

class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Identifier of the model")
    model_type: str = Field(..., description="Type of the model")
    max_seq_length: int = Field(
        512,
        description="Maximum sequence length the model can handle"
    )
    embedding_dimension: int = Field(
        384,
        description="Dimensionality of the embedding vectors"
    )

@app.on_event("startup")
async def load_model():
    """Load the model at startup using huggingface_hub for better model management."""
    global model
    try:
        start_time = time.time()
        
        # Create model directory if it doesn't exist
        model_path = MODEL_DIR / MODEL_NAME.replace("/", "--")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Check if model is already downloaded, if not download it
        if not any(model_path.iterdir()):
            print(f"Downloading model {MODEL_NAME}...")
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
            )
        
        # Load the model
        model = SentenceTransformer(
            str(model_path),
            device=MODEL_DEVICE,
        )
        
        # Warm up
        model.encode(["warmup"])
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        error_msg = f"Error loading model {MODEL_NAME}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": MODEL_NAME,
        "device": MODEL_DEVICE
    }

@app.get("/info")
async def model_info():
    """Get model information."""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        model_id=MODEL_NAME,
        model_type="sentence_transformer",
        max_seq_length=model.max_seq_length,
        embedding_dimension=model.get_sentence_embedding_dimension()
    )

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for a list of texts."""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Generate embeddings
        embeddings = model.encode(
            request.inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        
        # Convert numpy arrays to lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return {
            "embeddings": embeddings,
            "model": MODEL_NAME,
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}"
        )

# For backward compatibility
@app.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings_compat(request: EmbeddingRequest):
    return await embed_texts(request)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        workers=int(os.getenv("WEB_CONCURRENCY", 1))
    )