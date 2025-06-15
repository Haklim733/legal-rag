import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List 

app = FastAPI(title="Embedding Service")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = os.getenv("MODEL_DIR", "models")  # Default to 'models' if not set

# Load the model at startup
model = None

class EmbeddingRequest(BaseModel):
    texts: List[str]
    convert_to_numpy: bool = True
    normalize_embeddings: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int

def get_model_path() -> Path:
    """Get the path to the model directory."""
    model_path = Path(MODEL_DIR) / MODEL_NAME
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = SentenceTransformer(str(get_model_path()), device='cuda')
        # Warm up the model with a small input
        model.encode("warmup")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        embeddings = model.encode(
            request.texts,
            convert_to_numpy=request.convert_to_numpy,
            normalize_embeddings=request.normalize_embeddings
        )
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
            
        return {
            "embeddings": embeddings,
            "model": "all-MiniLM-L6-v2",
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)