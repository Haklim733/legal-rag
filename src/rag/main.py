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
from .models import (
    QueryParam,
    SAMPLE_KG,
    SYSTEM_PROMPT,
    validate_rag_response,
    RAGResponse,
)

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
    if (working_dir / "kv_store_llm_response_cache.json").exists():
        os.remove(working_dir / "kv_store_llm_response_cache.json")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="phi3:mini",
        llm_model_max_async=4,
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": "localhost:11434",  # Use Ollama service hostname in Docker network
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
                texts,
                embed_model="all-minilm",
                host="localhost:11434",  # Use Ollama service hostname in Docker network
            ),
        ),
        graph_storage="Neo4JStorage",
        addon_params={"insert_batch_size": 20},  # Process 20 documents per batch
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def main(query_text: str = None) -> RAGResponse:

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

    logger.info("Inserting custom knowledge graph into LightRAG...")
    rag.insert_custom_kg(SAMPLE_KG)
    logger.info("Successfully inserted custom knowledge graph")

    query_param = QueryParam()
    logger.info(f"QueryParam instance: {query_param}")

    # Execute query with a more specific legal question
    logger.info("Calling rag.query...")
    logger.info(f"Query text: {query_text}")

    # Use ontological system prompt for concept extraction
    response = rag.query(query_text, param=query_param, system_prompt=SYSTEM_PROMPT)
    logger.info(f"Raw response: {response}")

    # Validate response structure
    try:
        validated_response = validate_rag_response(response, query_text)
        logger.info("✅ Response validation successful!")
        logger.info(f"📊 Validation Summary:")
        logger.info(f"   - Total concepts found: {validated_response.total_concepts}")
        logger.info(f"   - Entities: {len(validated_response.entities_found)}")
        logger.info(
            f"   - Relationships: {len(validated_response.relationships_found)}"
        )
        logger.info(f"   - Classes: {len(validated_response.classes_found)}")
        logger.info(f"   - Properties: {len(validated_response.properties_found)}")
        logger.info(
            f"   - Overall confidence: {validated_response.confidence_summary['overall']:.3f}"
        )

        # Log detailed concept information
        if validated_response.entities_found:
            logger.info("🏢 Entities found:")
            for entity in validated_response.entities_found:
                logger.info(
                    f"   - {entity.concept_name} (confidence: {entity.confidence_score:.3f})"
                )
                logger.info(f"     Description: {entity.description}")

        if validated_response.relationships_found:
            logger.info("🔗 Relationships found:")
            for rel in validated_response.relationships_found:
                logger.info(
                    f"   - {rel.concept_name} (confidence: {rel.confidence_score:.3f})"
                )
                logger.info(f"     Description: {rel.description}")

        if validated_response.classes_found:
            logger.info("📚 Classes found:")
            for cls in validated_response.classes_found:
                logger.info(
                    f"   - {cls.concept_name} (confidence: {cls.confidence_score:.3f})"
                )
                logger.info(f"     Description: {cls.description}")

        if validated_response.properties_found:
            logger.info("🔧 Properties found:")
            for prop in validated_response.properties_found:
                logger.info(
                    f"   - {prop.concept_name} (confidence: {prop.confidence_score:.3f})"
                )
                logger.info(f"     Description: {prop.description}")

        # Return validated response for further processing
        return validated_response

    except ValueError as e:
        logger.error(f"❌ Response validation failed: {e}")
        logger.error(
            "Response does not contain valid JSON structure with ontological concepts"
        )
        logger.error(f"Raw response: {response}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during response validation: {e}")
        logger.error(f"Raw response: {response}")
        raise

    # Finalize
    asyncio.run(rag.finalize_storages())
    logger.info("-------------------------")


if __name__ == "__main__":
    query_text = "John, a professional lawyer at the Legal Aid of Los Angeles, spoke on behalf of Jane, a recent evictee of her apartment. Jane is a tenant of a rental property in Los Angeles, California. She received a notice to vacate the property, but she disputes the eviction. She is seeking legal representation and court proceedings to defend her case. The lawyer provides legal advice and represents clients in eviction cases."
    result = main(query_text)
    print(f"\n🎯 Final Result Summary:")
    print(f"Query: {result.query_text}")
    print(f"Total ontological concepts found: {result.total_concepts}")
    print(f"Overall confidence: {result.confidence_summary['overall']:.3f}")
    print(f"Response validation: ✅ SUCCESS")
