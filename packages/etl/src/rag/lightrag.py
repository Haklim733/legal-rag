import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    # Initialize LightRAG with Hugging Face model
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
        llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
        # Use Hugging Face embedding function
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            )
        ),
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag

async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        rag.insert("Your text")

        # Perform hybrid search
        mode="hybrid"
        print(
          await rag.query(
              "What are the top themes in this story?",
              param=QueryParam(mode=mode)
          )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())