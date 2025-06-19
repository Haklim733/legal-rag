import os
import asyncio
import nest_asyncio
import logging

import torch
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc


from lightrag.utils import setup_logger

from .embed import load_ontology_for_rag
from .query_param import QueryParam

nest_asyncio.apply()

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ollama_embedding_model_name = "all-minilm"  # Use all-MiniLM-L6-v2 via Ollama
try:
    ollama_embedder = OllamaEmbedder(model=ollama_embedding_model_name, base_url="http://localhost:11434")
    logger.info(f"Successfully initialized OllamaEmbedder with model: {ollama_embedding_model_name}")
except Exception as e:
    logger.error(f"Failed to initialize OllamaEmbedder: {e}")

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="phi3:mini",
        llm_model_max_async=4,
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 4096},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=256,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="all-minilm",
                host="http://localhost:11434"
            )
        ),
        addon_params={
            "insert_batch_size": 20  # Process 20 documents per batch
        }
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def main_query():
    logger.info("--- Testing RAG Query ---")

    # Initialize RAG
    logger.info("Initializing RAG...")
    rag = asyncio.run(initialize_rag())
    logger.info(f"Initialized RAG object: {rag}")

    # Load FOLIO ontology data
    logger.info("Loading FOLIO ontology data...")
    documents_to_embed = load_ontology_for_rag(
        limit=20, max_depth=1
    )  # Start with a small limit
    logger.info(f"Loaded {len(documents_to_embed)} FOLIO documents")

    # Insert FOLIO documents
    rag.insert([x.text for x in documents_to_embed[0:5]])  # This is synchronous

    query_param = QueryParam()
    logger.info(f"QueryParam instance: {query_param}")

    # Execute query with a more specific legal question
    logger.info("Calling rag.query...")
    query_text = (
        "What is legal representation and how does it work in court proceedings?"
    )
    logger.info(f"Query text: {query_text}")
    response = rag.query(query_text, param=query_param)
    logger.info(f"Response: {response}")

    # Finalize
    asyncio.run(rag.finalize_storages())
    logger.info("-------------------------")


if __name__ == "__main__":
    main_query()
