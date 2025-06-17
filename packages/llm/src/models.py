from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    inputs: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_length=1,
        max_length=100,  # Prevent abuse with too many inputs
        example=["This is a sample text"],
    )


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(
        ..., description="List of embeddings, one for each input text"
    )
    model: str = Field(
        ..., description="Name of the model used for generating embeddings"
    )
    dimensions: int = Field(..., description="Dimensionality of the embedding vectors")


class SearchRequest(BaseModel):
    query: str = Field(..., description="The query to search for")
    search_set: List[Dict[str, str]] = Field(
        ..., description="List of items with IRI and label"
    )
    limit: int = Field(10, description="Maximum number of results to return")
    scale: int = Field(10, description="Scale for relevancy scoring")
    include_reason: bool = Field(False, description="Whether to include explanation")


class SearchResult(BaseModel):
    iri: str = Field(..., description="The IRI of the matched item")
    relevance: int = Field(..., description="Relevance score")
    explanation: Optional[str] = Field(None, description="Explanation for the match")


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="Search results")
