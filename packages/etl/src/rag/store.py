"""
Knowledge Graph Store implementation using LightRAG for efficient storage and retrieval.

This module provides a generic KnowledgeGraphStore class that can store and query
entities in a knowledge graph using LightRAG as the underlying storage engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, overload, Callable
import httpx

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from pydantic import BaseModel
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Default configuration
DEFAULT_RAG_CONFIG = {
    "working_dir": "./rag_storage",
    "enable_kg": True,
    "enable_kg_entity_merging": True,
    "embedding_endpoint": "http://localhost:8000/embed",  # Default endpoint for the container
    "batch_size": 32,
    "timeout": 30.0
}

class EmbeddingClient:
    """Client for interacting with a containerized embedding service."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000/embed",
        batch_size: int = 32,
        timeout: float = 30.0
    ):
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            response = await self.client.post(
                self.endpoint,
                json={"inputs": texts},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

def get_embedding_func(
    endpoint: str = None,
    batch_size: int = 32,
    timeout: float = 30.0
) -> Callable[[List[str]], List[List[float]]]:
    """Create an embedding function that uses the containerized service.
    
    Args:
        endpoint: The HTTP endpoint of the embedding service
        batch_size: Number of texts to process in each batch
        timeout: Request timeout in seconds
        
    Returns:
        A function that takes a list of texts and returns their embeddings
    """
    endpoint = endpoint or DEFAULT_RAG_CONFIG["embedding_endpoint"]
    batch_size = batch_size or DEFAULT_RAG_CONFIG["batch_size"]
    timeout = timeout or DEFAULT_RAG_CONFIG["timeout"]
    
    client = EmbeddingClient(endpoint=endpoint, batch_size=batch_size, timeout=timeout)
    
    async def embed_func(texts: List[str]) -> List[List[float]]:
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await client.embed_texts(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    return embed_func, client.close

class KnowledgeGraphStore(Generic[T]):
    """A generic knowledge graph store using LightRAG for storage and retrieval.
    
    This class provides a type-safe interface for storing and querying entities
    in a knowledge graph. It supports CRUD operations, semantic search, and
    relationship management using LightRAG as the backend.
    
    Args:
        entity_type: The Pydantic model class for entities
        db_path: Path to store the knowledge graph
        id_field: Name of the field to use as unique identifier
        text_fields: List of fields to index for full-text search
        relationship_types: Dict of relationship types and their properties
        embedding_endpoint: URL of the embedding service (default: http://localhost:8000/embed)
        batch_size: Number of texts to process in each batch (default: 32)
        timeout: Request timeout in seconds (default: 30.0)
        **kwargs: Additional arguments to pass to LightRAG
    """
    
    def __init__(
        self,
        entity_type: Type[T],
        db_path: Union[str, Path] = "knowledge_graph",
        id_field: str = "id",
        text_fields: Optional[List[str]] = None,
        relationship_types: Optional[Dict[str, List[str]]] = None,
        embedding_endpoint: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        if not is_dataclass(entity_type) and not issubclass(entity_type, BaseModel):
            raise ValueError("entity_type must be a dataclass or Pydantic model")
            
        self.entity_type = entity_type
        self.id_field = id_field
        self.text_fields = text_fields or []
        self.relationship_types = relationship_types or {}
        self.entity_type_name = entity_type.__name__.lower()
        self._rag = None
        self._initialized = False
        self._close_embedding_func = None
        
        # Initialize LightRAG configuration with containerized embeddings
        self.rag_config = DEFAULT_RAG_CONFIG.copy()
        
        # Create embedding function
        embedding_func, self._close_embedding_func = get_embedding_func(
            endpoint=embedding_endpoint,
            batch_size=batch_size,
            timeout=timeout
        )
        
        self.rag_config.update({
            "working_dir": str(db_path),
            "embedding_func": embedding_func,
            **{k: v for k, v in kwargs.items() if not k.startswith('embedding_')}
        })
    
    async def initialize(self) -> None:
        """Initialize the LightRAG instance and required storages."""
        if self._initialized:
            return
            
        # Create working directory if it doesn't exist
        os.makedirs(self.rag_config["working_dir"], exist_ok=True)
        
        # Initialize LightRAG
        self._rag = LightRAG(**self.rag_config)
        
        # Initialize required storages and pipeline status
        await self._rag.initialize_storages()
        await initialize_pipeline_status()
        
        self._initialized = True
    
    async def finalize(self) -> None:
        """Finalize and clean up resources."""
        if self._rag and self._initialized:
            await self._rag.finalize_storages()
            self._initialized = False
        
        # Close the embedding client
        if self._close_embedding_func:
            await self._close_embedding_func()
            self._close_embedding_func = None
    
    async def add_entity(self, entity: T) -> str:
        """Add a single entity to the knowledge graph.
        
        Args:
            entity: The entity to add
            
        Returns:
            The ID of the added entity
        """
        if not self._initialized:
            await self.initialize()
            
        entity_data = entity.model_dump() if isinstance(entity, BaseModel) else asdict(entity)
        entity_id = str(entity_data.pop(self.id_field))
        
        # Prepare text for embedding
        text = ' '.join(str(entity_data.get(field, '')) for field in self.text_fields)
        
        # Add to LightRAG
        await self._rag.insert(
            text=text,
            metadata={
                "entity_type": self.entity_type_name,
                "entity_id": entity_id,
                **entity_data
            },
            doc_id=f"{self.entity_type_name}_{entity_id}",
            doc_metadata={"entity_type": self.entity_type_name}
        )
        
        return entity_id
    
    async def add_entities(
        self,
        entities: List[T],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[str]:
        """Add multiple entities to the knowledge graph.
        
        Args:
            entities: List of entities to add
            batch_size: Number of entities to process in each batch
            show_progress: Whether to show a progress bar
            
        Returns:
            List of IDs of the added entities
        """
        if not entities:
            return []
            
        if not self._initialized:
            await self.initialize()
        
        entity_ids = []
        
        for i in tqdm(
            range(0, len(entities), batch_size),
            desc=f"Adding {self.entity_type_name} entities",
            disable=not show_progress
        ):
            batch = entities[i:i + batch_size]
            batch_ids = []
            
            for entity in batch:
                entity_data = entity.model_dump() if isinstance(entity, BaseModel) else asdict(entity)
                entity_id = str(entity_data.pop(self.id_field))
                
                # Prepare text for embedding
                text = ' '.join(str(entity_data.get(field, '')) for field in self.text_fields)
                
                # Add to batch
                batch_ids.append((entity_id, text, entity_data))
            
            # Process batch
            for entity_id, text, entity_data in batch_ids:
                await self._rag.insert(
                    text=text,
                    metadata={
                        "entity_type": self.entity_type_name,
                        "entity_id": entity_id,
                        **entity_data
                    },
                    doc_id=f"{self.entity_type_name}_{entity_id}",
                    doc_metadata={"entity_type": self.entity_type_name}
                )
                entity_ids.append(entity_id)
        
        return entity_ids
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to retrieve
            
        Returns:
            The entity data as a dictionary, or None if not found
        """
        if not self._initialized:
            await self.initialize()
            
        doc_id = f"{self.entity_type_name}_{entity_id}"
        
        # Search for the document by ID
        results = await self._rag.query(
            query=f"id:{doc_id}",
            param=QueryParam(
                mode="hybrid",
                top_k=1,
                filter_conditions={"doc_id": doc_id}
            )
        )
        
        if not results or not results.get("documents"):
            return None
            
        # Get the first result
        doc = results["documents"][0]
        metadata = doc.get("metadata", {})
        
        # Reconstruct the entity
        entity_data = {
            self.id_field: entity_id,
            **{k: v for k, v in metadata.items() if k not in {"entity_type", "entity_id"}}
        }
        
        return entity_data
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        return_properties: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for entities using natural language query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            return_properties: List of properties to include in results
            **kwargs: Additional arguments to pass to LightRAG search
            
        Returns:
            List of matching entities with scores
        """
        if not self._initialized:
            await self.initialize()
            
        if return_properties is None:
            return_properties = self.text_fields
            
        # Build filter conditions
        filter_conditions = {"metadata.entity_type": self.entity_type_name}
        
        # Execute search
        results = await self._rag.query(
            query=query,
            param=QueryParam(
                mode="hybrid",
                top_k=limit,
                filter_conditions=filter_conditions,
                **kwargs
            )
        )
        
        # Process results
        entities = []
        for doc, score in zip(results.get("documents", []), results.get("scores", [])):
            metadata = doc.get("metadata", {})
            
            # Filter return properties if specified
            if return_properties:
                metadata = {k: v for k, v in metadata.items() if k in return_properties}
            
            entities.append({
                "id": metadata.get("entity_id", ""),
                "score": float(score),
                **metadata
            })
        
        return entities
        results = self.rag.search(
            query=query,
            node_types=[self.entity_type_name],
            limit=limit,
            return_properties=return_properties,
            **kwargs
        )
        
        return [{
            'id': result.node.node_id,
            'score': result.score,
            **result.node.properties
        } for result in results]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the knowledge graph to disk.
        
        Args:
            path: Path to save the knowledge graph to
        """
        self.rag.save(str(path))
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        entity_type: Type[T],
        **kwargs
    ) -> 'KnowledgeGraphStore[T]':
        """Load a knowledge graph from disk.
        
        Args:
            path: Path to the saved knowledge graph
            entity_type: The entity type class
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            A new KnowledgeGraphStore instance
        """
        rag = LightRAG.load(str(path))
        store = cls(entity_type, **kwargs)
        store.rag = rag
        return store

async def create_kg_store(
    entities: List[T],
    entity_type: Type[T],
    db_path: Union[str, Path] = "knowledge_graph",
    **kwargs
) -> KnowledgeGraphStore[T]:
    """Create a new knowledge graph store with the given entities.
    
    This is a convenience function that creates a new KnowledgeGraphStore,
    initializes it with the given entities, and returns it.
    
    Args:
        entities: List of entities to add to the store
        entity_type: The entity type class
        db_path: Path to store the knowledge graph
        **kwargs: Additional arguments to pass to KnowledgeGraphStore
        
    Returns:
        A new KnowledgeGraphStore instance with the given entities
    """
    store = KnowledgeGraphStore(entity_type=entity_type, db_path=db_path, **kwargs)
    await store.initialize()
    await store.add_entities(entities)
    return store
