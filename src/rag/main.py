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
    extract_json_from_text,
    RAGResponse,
    Entity,
    Relationship,
    ExtractionMode,
    SYSTEM_PROMPT_SINGLE,
)

nest_asyncio.apply()

# Configure logging to handle ascii_colors errors gracefully
logging.getLogger("ascii_colors").setLevel(logging.ERROR)


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


async def initialize_rag(llm_model_name: str, embed_model_name: str):
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model_name,
        llm_model_max_async=4,
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": "localhost:11434",  # Use Ollama service hostname in Docker network
            "options": {
                "num_ctx": 8192,
                "temperature": 0.0,  # Set to 0 for deterministic output
                "seed": 42,  # Use a fixed seed for reproducible results
                "num_predict": 8192,  # Allow for a much longer response
            },
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=256,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embed_model_name,
                host="localhost:11434",  # Use Ollama service hostname in Docker network
            ),
        ),
        graph_storage="Neo4JStorage",
        addon_params={"insert_batch_size": 20},  # Process 20 documents per batch
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def main(
    query_text: str,
    pdf_path: str = None,
    entities: list[str] = None,
    llm_model_name: str = "llama3.1:8b",
    embed_model_name: str = "llama3.1:8b",
) -> RAGResponse:
    # Initialize RAG
    rag = asyncio.run(initialize_rag(llm_model_name, embed_model_name))
    logger.info(f"Initialized RAG object: {rag}")
    logger.info(f"Using LLM model: {llm_model_name}")
    logger.info(f"Using embedding model: {embed_model_name}")
    clear_cache()

    # Load FOLIO ontology data
    logger.info("Loading FOLIO ontology data...")
    folio_instance = FOLIO("github", llm=None)

    logger.info(f"Creating filtered knowledge graph with entities: {entities}")
    custom_kg = create_custom_kg(
        folio_instance,
        entities=entities,
        subclasses=True,
    ).to_dict()

    logger.info("Inserting custom knowledge graph into LightRAG...")
    rag.insert_custom_kg(custom_kg)
    logger.info("Successfully inserted custom knowledge graph")

    query_param = QueryParam()
    logger.info("Calling rag.query...")
    logger.info(f"Query text: {query_text}")

    # Single pass: Extract entities and relationships
    logger.info("🔄 Single pass: Extracting entities and relationships...")
    response = rag.query(
        query_text, param=query_param, system_prompt=SYSTEM_PROMPT_SINGLE
    )

    try:
        response_data = extract_json_from_text(response, ExtractionMode.AGGRESSIVE)
        extracted_entities = []
        relationships = []

        # Process entities
        for entity_data_item in response_data.get("entities", []):
            entity = Entity(
                entity_name=entity_data_item.get("entity_name", ""),
                entity_type=entity_data_item.get("entity_type", "entity"),
                description=entity_data_item.get("description", ""),
                # weight defaults to 1.0
            )
            extracted_entities.append(entity)

        # Get entity names for exact matching
        entity_names = {entity.entity_name for entity in extracted_entities}
        logger.info(f"Found entities: {entity_names}")

        # Process relationships with exact name matching
        for rel_data in response_data.get("relationships", []):
            src_id = rel_data.get("src_id", "")
            tgt_id = rel_data.get("tgt_id", "")

            # Only create relationships if both entities exist exactly
            if src_id in entity_names and tgt_id in entity_names:
                relationship = Relationship(
                    src_id=src_id,
                    tgt_id=tgt_id,
                    description=rel_data.get("description", ""),
                    keywords=rel_data.get("keywords", ""),
                    # weight defaults to 1.0
                )
                relationships.append(relationship)
                logger.info(f"✅ Created relationship: {src_id} -> {tgt_id}")
            else:
                logger.warning(
                    f"❌ Skipped relationship: {src_id} -> {tgt_id} (entities not found)"
                )

        logger.info(
            f"✅ Single pass successful: {len(extracted_entities)} entities, {len(relationships)} relationships"
        )

    except Exception as e:
        logger.error(f"❌ Single pass failed: {e}")
        logger.error(f"Response: {response}")
        extracted_entities = []
        relationships = []

    return RAGResponse(
        query_text=query_text,
        response_text=response,
        entities=extracted_entities,
        relationships=relationships,
    )


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
    # Model configuration arguments
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.1:8b",
        help="LLM model name to use (e.g., 'llama3.1:8b', 'mistral:7b-instruct', 'codellama:7b-instruct').",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-minilm",
        help="Embedding model name to use (e.g., 'all-minilm', 'nomic-embed-text').",
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
        response = main(
            query_text,
            entities=args.entities,
            llm_model_name=args.llm_model,
            embed_model_name=args.embed_model,
        )
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
