# Overview

This is sample repo to extract text from pdfs and extract ontological concepts using FOLIO ontology and LightRAG. This is a proof of concept to show how to use FOLIO ontology to extract ontological concepts and use LightRAG to query the document store.

## Setup

ollama on localhost machine required.
gpu will make the embedding much faster.

## Usage

```bash
uv run src.rag.main
```

## Details
the main.py is the entry point. it will load the ontology and embed it into the document store.

the embed.py is the main function to load the ontology and embed it into the document store.

the query_param.py is the main function to query the document store.

# issues
1. the ontology may not be complete.
2. some subclasses inherit properties from their parents that should be excluded (e.g. Year of Law Degree Graduation)
3. not all predicates are of interest or have been vetted (only folio: predicates are used here). 
4. there are 660 labels with more than 1 classes 

## TODO
1. checkout Langchain LLM Transformers
2. store in pgsql vector db 
3. use llm api key
4. examine folio:operators
5. de-dup

# Cases RAG

A knowledge graph project using the FOLIO ontology and LightRAG framework.

## Docker Setup with Ollama Models

This project includes Docker support with Ollama models for local LLM inference.

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (optional, for GPU acceleration)

### Quick Start

1. **Build and start the services:**
   ```bash
   docker-compose up --build
   ```

   This will start:
   - **LightRAG service** on port 9621
   - **Ollama service** on port 11434 (with phi3:mini and all-minilm models)
   - **Neo4j database** on ports 7474 (HTTP) and 7687 (Bolt)

2. **Wait for models to download:**
   The first startup will download the required Ollama models:
   - `phi3:mini` - LLM model for text generation
   - `all-minilm` - Embedding model for vector search

3. **Access the services:**
   - LightRAG API: http://localhost:9621
   - Neo4j Browser: http://localhost:7474
   - Ollama API: http://localhost:11434

### Manual Model Pulling

If you need to pull the models manually:

```bash
# Pull the LLM model
ollama pull phi3:mini

# Pull the embedding model  
ollama pull all-minilm
```

### Configuration

The models used are configured in `src/rag/main.py`:
- **LLM Model**: `phi3:mini` for text generation
- **Embedding Model**: `all-minilm` for vector embeddings
- **Host**: `http://ollama:11434` (Docker service name)

### GPU Support

For GPU acceleration, ensure you have:
- NVIDIA Docker runtime installed
- Compatible GPU drivers

The docker-compose file includes GPU configuration for the LightRAG service.

## Development

For local development without Docker:

1. Install Ollama locally
2. Pull the required models
3. Update the host URLs in `src/rag/main.py` to use `localhost:11434`
4. Run the services locally

## Troubleshooting

- **Models not found**: Ensure the Ollama service has finished downloading models
- **Connection errors**: Check that all services are running and healthy
- **GPU issues**: Verify NVIDIA Docker runtime is properly configured
