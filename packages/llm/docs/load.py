import os
import time
from functools import wraps
from typing import Callable, Any, List, Dict
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document


def time_execution(metric_name: str) -> Callable:
    """Decorator to measure and print execution time of a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"⏱️  [{metric_name}] Execution time: {execution_time:.4f} seconds")
            return result
        return wrapper
    return decorator

def log_document_metrics(func: Callable) -> Callable:
    """Decorator to log document processing metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        result = func(*args, **kwargs)
        if hasattr(result, '__len__'):
            print(f"📊  Processed {len(result)} documents")
        return result
    return wrapper

@time_execution("Document Loading")
@log_document_metrics
def load_documents(directory: Path) -> List[Document]:
    """Load documents from the specified directory."""
    loader = DirectoryLoader(
        str(directory),
        glob='**/*.pdf',
        show_progress=True
    )
    return loader.load()

@time_execution("Text Splitting")
@log_document_metrics
def split_docs(docs: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

def interactive_search(vector_store: Chroma):
    """Run an interactive search loop."""
    print("\n🔍  Interactive search started. Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\n❓ Query: ").strip()
            
            if query.lower() in ('exit', 'quit', 'q'):
                print("👋  Exiting interactive search.")
                break
                
            if not query:
                print("⚠️  Please enter a query.")
                continue
                
            print("\n🔎  Searching...")
            try:
                results = vector_store.similarity_search_with_score(query, k=3)
                
                print(f"\n✅  Found {len(results)} results:")
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\n📄 Result {i} (Score: {score:.4f}):")
                    print("-" * 50)
                    content = doc.page_content
                    print(content[:300] + "..." if len(content) > 300 else content)
                    print("-" * 50)
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Page: {doc.metadata.get('page', 'N/A')}")
                    
            except Exception as e:
                print(f"❌  Error during search: {str(e)}")
                
        except KeyboardInterrupt:
            print("\n👋  Exiting interactive search.")
            break
        except Exception as e:
            print(f"⚠️  An error occurred: {str(e)}")
            continue

def main():
    # Initialize paths
    file_dir = Path(__file__).parents[3].resolve()
    print(f"📂  Loading documents from: {file_dir}/assets/")
    
    # Load and process documents
    docs = load_documents(file_dir.joinpath("assets/"))
    split_text = split_docs(docs)
    
    # Initialize embeddings and ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create in-memory ChromaDB instance
    vector_store = Chroma(
        collection_name="irs_docs",
        embedding_function=embeddings,
        persist_directory=None  # In-memory mode
    )
    
    # Prepare documents for ChromaDB
    documents = [doc.page_content for doc in split_text]
    metadatas = [doc.metadata for doc in split_text]
    ids = [f"doc_{i}" for i in range(len(split_text))]
    
    # Add documents to the collection
    vector_store.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅  Added {len(split_text)} document chunks to ChromaDB")
    
    # Start interactive search
    interactive_search(vector_store)

if __name__ == "__main__":
    main()
