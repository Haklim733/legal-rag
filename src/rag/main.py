import asyncio
import logging
import nest_asyncio
import os
from pathlib import Path
import time
import sys
import argparse

from folio import FOLIO
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc
from src.docs.process_pdfs import extract_pdf
from src.docs.models import ExtractionMethod

from .embed import create_custom_kg
from .models import (
    QueryParam,
    SYSTEM_PROMPT,
    validate_rag_response,
    RAGResponse,
)

nest_asyncio.apply()

# Configure logging to handle ascii_colors errors gracefully
logging.getLogger("ascii_colors").setLevel(logging.ERROR)


# Suppress ascii_colors errors by redirecting stderr for that specific error
class SuppressAsciiColorsError:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr


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
    for f in [
        "kv_store_llm_response_cache.json",
        "graph_chunk_entity_relation.graphml",
    ]:
        if (working_dir / f).exists():
            os.remove(working_dir / f)


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
                "num_predict": 4096,  # Allow for a much longer response
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


def main(query_text: str, entities: list[str]) -> RAGResponse:

    # Initialize RAG
    rag = asyncio.run(initialize_rag())
    logger.info(f"Initialized RAG object: {rag}")
    clear_cache()

    # Load FOLIO ontology data
    logger.info("Loading FOLIO ontology data...")
    folio_instance = FOLIO("github", llm=None)

    # Create filtered knowledge graph directly
    logger.info(f"Creating filtered knowledge graph with entities: {entities}")
    custom_kg = create_custom_kg(
        folio_instance,
        entities=entities,
        subclasses=True,
    ).to_dict()

    logger.info("Inserting filtered custom knowledge graph into LightRAG...")
    rag.insert_custom_kg(custom_kg)
    logger.info("Successfully inserted filtered custom knowledge graph")

    query_param = QueryParam()
    logger.info("Calling rag.query...")
    logger.info(f"Query text: {query_text}")

    # Use ontological system prompt for concept extraction
    response = rag.query(query_text, param=query_param, system_prompt=SYSTEM_PROMPT)
    logger.info(f"Raw response: {response}")

    # Validate response structure
    try:
        validated_response = validate_rag_response(response, query_text)
        logger.info("✅ Response validation successful!")
        print(validated_response)
        logger.info(f"📊 Validation Summary:")
        logger.info(
            f"   - Total concepts found: {len(validated_response.entities) + len(validated_response.relationships)}"
        )
        logger.info(f"   - Entities: {len(validated_response.entities)}")
        logger.info(f"   - Relationships: {len(validated_response.relationships)}")

        # Log detailed concept information
        if validated_response.entities:
            logger.info("🏢 Entities found:")
            for entity in validated_response.entities:
                logger.info(f"   - {entity.entity_name} (type: {entity.entity_type})")
                logger.info(f"     Description: {entity.description}")

        if validated_response.relationships:
            logger.info("🔗 Relationships found:")
            for rel in validated_response.relationships:
                logger.info(
                    f"   - {rel.src_id} -> {rel.tgt_id} (weight: {rel.weight:.3f})"
                )
                logger.info(f"     Description: {rel.description}")

        # Return validated response for further processing
        # Finalize before returning
        try:
            asyncio.run(rag.finalize_storages())
        except Exception as e:
            logger.warning(f"Error during finalize (non-critical): {e}")
        return validated_response

    except ValueError as e:
        logger.error(f"Response validation failed: {e}")
        logger.error(
            "Response does not contain valid JSON structure with ontological concepts"
        )
        logger.error(f"Raw response: {response}")
        # Finalize before raising exception
        try:
            asyncio.run(rag.finalize_storages())
        except Exception as finalize_error:
            logger.warning(f"Error during finalize (non-critical): {finalize_error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during response validation: {e}")
        logger.error(f"Raw response: {response}")
        # Finalize before raising exception
        try:
            asyncio.run(rag.finalize_storages())
        except Exception as finalize_error:
            logger.warning(f"Error during finalize (non-critical): {finalize_error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG process on a PDF file or a raw text query."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to a PDF file to extract text from.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default=None,
        help="A raw text query to process. If not provided, and --file-path is not used, a default query is run.",
    )
    # PDF extraction arguments
    parser.add_argument(
        "--hierarchy",
        action="store_false",
        default=False,
        dest="hierarchy",
        help="Disable hierarchy preservation in PDF extraction. Default is to preserve it.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of pages to extract from the PDF. Default is 1.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fast",
        choices=["fast", "hi_res"],
        help="PDF extraction strategy. 'fast' is quicker, 'hi_res' is more detailed. Default is 'fast'.",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        default=[
            "Lawyer",
            "Litigant",
            "Entity",
            "Government Representative",
            "U.S. Federal Courts",
        ],
        help="List of entities to filter the knowledge graph.",
    )

    args = parser.parse_args()

    query_text = ""
    DEFAULT_QUERY = "John, a professional lawyer at the Legal Aid of Los Angeles, spoke on behalf of Jane, a recent evictee of her apartment. Jane is a tenant of a rental property in Los Angeles, California. She received a notice to vacate the property, but she disputes the eviction. She is seeking legal representation and court proceedings to defend her case. The lawyer provides legal advice and represents clients in eviction cases."

    if args.file_path:
        input_path = Path(args.file_path)
        if input_path.is_file():
            if input_path.suffix == ".pdf":
                print(f"Extracting text from PDF: {input_path}")
                print(f"  - Strategy: {args.strategy}")
                print(f"  - Hierarchy: {'enabled' if args.hierarchy else 'disabled'}")
                print(f"  - Max pages: {args.max_pages}")
                result = extract_pdf(
                    input_path,
                    method=ExtractionMethod.UNSTRUCTURED,
                    strategy=args.strategy,
                    preserve_hierarchy=args.hierarchy,
                    extract_metadata=False,
                    max_pages=args.max_pages,
                )
                if result.success:
                    query_text = result.text_result.full_text
                    print(f"Extracted {len(query_text)} characters.")
                else:
                    print(f"❌ Failed to extract text from {input_path}")
                    sys.exit(1)
            else:
                with open(input_path, "r") as file:
                    query_text = file.read()
        else:
            print(f"❌ Invalid file path: {args.file_path}")
            sys.exit(1)

    elif args.query:
        print(f"Using raw text query: '{args.query}'")
        query_text = args.query
    else:
        print("No file path or query provided. Using default standard text.")
        query_text = DEFAULT_QUERY
        print(f"Default query: '{query_text}'")

    # Run the main RAG function
    if query_text:
        response = main(query_text, entities=args.entities)
        print("\n✅ RAG processing complete.")
        print("\n--- Summary ---")
        print(
            f"Found {len(response.entities)} entities and {len(response.relationships)} relationships."
        )
        print(f"Overall confidence: {response.overall['overall']:.2f}")
        print("\n--- Details ---")
        print("Response Text from LLM:")
        print(response.response_text)
    else:
        print("No query text to process.")
        sys.exit(1)
