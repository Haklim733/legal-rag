import asyncio
import logging
import nest_asyncio
import os
from pathlib import Path
import time

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

from .embed import load_ontology_for_rag
from .models import QueryParam, SAMPLE_KG, SYSTEM_PROMPT

nest_asyncio.apply()

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


WORKING_DIR = Path(__file__).parent / "rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

def clear_cache(working_dir: Path = WORKING_DIR):
    os.remove(working_dir / "kv_store_llm_response_cache.json")

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="phi3:mini",
        llm_model_max_async=4,
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {
                "num_ctx": 4096,
                "temperature": 0.7,  # Add some randomness
                "seed": int(time.time()) % 1000000,  # Random seed to avoid caching
            },
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=256,
            func=lambda texts: ollama_embed(
                texts, embed_model="all-minilm", host="http://localhost:11434"
            ),
        ),
        graph_storage="Neo4JStorage",
        addon_params={"insert_batch_size": 20},  # Process 20 documents per batch
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def main_query():

    # Initialize RAG
    rag = asyncio.run(initialize_rag())
    logger.info(f"Initialized RAG object: {rag}")
    clear_cache()
    

    # Load FOLIO ontology data
    logger.info("Loading FOLIO ontology data...")
    # documents_to_embed, folio_instance = load_ontology_for_rag(
    #     limit=20, max_depth=1
    # )  # Start with a small limit
    # logger.info(f"Loaded {len(documents_to_embed)} FOLIO documents")

    # Create custom knowledge graph from FOLIO triples
    logger.info("Creating custom knowledge graph from FOLIO triples...")
    # custom_kg = create_custom_kg(folio_instance)
    # logger.info(
    #     f"Created custom KG with {len(custom_kg.chunks)} chunks, {len(custom_kg.entities)} entities, {len(custom_kg.relationships)} relationships"
    # )

    # Insert custom knowledge graph into LightRAG
    logger.info("Inserting custom knowledge graph into LightRAG...")
    rag.insert_custom_kg(SAMPLE_KG)
    logger.info("Successfully inserted custom knowledge graph")

    query_param = QueryParam()
    logger.info(f"QueryParam instance: {query_param}")

    # Execute query with a more specific legal question
    logger.info("Calling rag.query...")
    query_text = "John, a professional at the Legal Aid of Los Angeles, spoke on behalf of Jane, a recent evictee of her apartment. Jane is a tenant of a rental property in Los Angeles, California. She received a notice to vacate the property, but she disputes the eviction. She is seeking legal representation and court proceedings to defend her case."
    logger.info(f"Query text: {query_text}")
    response = rag.query(query_text, param=query_param, system_prompt=SYSTEM_PROMPT)
    logger.info(f"Response: {response}")

    # Finalize
    asyncio.run(rag.finalize_storages())
    logger.info("-------------------------")


if __name__ == "__main__":
    main_query()
