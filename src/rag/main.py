import os
import asyncio
import nest_asyncio
import logging

import torch
from lightrag import LightRAG
from lightrag.llm.hf import hf_embed, hf_model_complete
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status

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

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                embed_model=AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2",
                ).to(device),
            ),
        ),
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
    for i, doc in enumerate(documents_to_embed[0:20]):
        logger.info(doc)
        logger.info(
            f"Inserting FOLIO document {i+1}/{len(documents_to_embed)}: {doc.id}"
        )
        rag.insert(doc.text)  # This is synchronous
        logger.info(f"Successfully inserted document {doc.id}")

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
