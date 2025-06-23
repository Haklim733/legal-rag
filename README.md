# Overview

This is sample repo to extract text from pdfs and extract ontological concepts using FOLIO ontology and LightRAG. This is a proof of concept that stores FOLIO ontology in a knowledge graph and use LightRAG to query the document store.

# Execution  
1. download case from scotus website
2. extract text from pdf
3. ask llm to extract ontological concepts using knowledge graph 
4. validate the response structure as JSON format

**Note**: this is a proof of concept and not a production ready solution. For quick testing, only 1 page of the case document is extracted at low resolution without preserving hierarchical structure. Moreover, only a subset of the FOLIO ontology is inserted into the knowledge graph. Embedding of text into a vector database is not currently implemented due to resource constraints. Lastly, validation of returned ontology concepts against the FOLIO ontology is not currently implemented.

```bash
./scripts/run.sh
```

### Prerequisites

- ollama on localhost machine
- NVIDIA GPU (optional, for GPU acceleration)
- uv python package manager

- **LLM Model**: `phi3:mini` for text generation
- **Embedding Model**: `all-minilm` for vector embeddings
- **Host**: `http://localhost:11434` (local ollama service)

```bash
# Pull the LLM model
ollama pull phi3:mini

# Pull the embedding model  
ollama pull all-minilm
```

## Usage
```bash
uv run -m src.rag.main
```

## Details
the main.py is the entry point. it will load the ontology and embed it into the document store.

6. the embed.py is the main function to load the ontology and embed it into the document store.

the models.py contains the pydantic models for validating the response from the LLM and contains the system prompt for the LLM.

# issues
1. the ontology may not be complete.
2. some subclasses inherit properties from their parents that should be excluded (e.g. Year of Law Degree Graduation)
3. not all predicates are of interest or have been vetted (only folio: predicates and oasis: predicates are used here). 
4. there are 660 labels with more than 1 classes 
5. Lightrag requires explicit relationships between entities.


## TODO
1. validate Entity and Relationships in the response
2. examine folio:operators
3. de-dup
4. vet the ontology
5. checkout Langchain LLM Transformers
6. store in pgsql vector db 
7. use llm api key

## Docker Setup with Ollama Models
Coming soon

## Troubleshooting

- **Models not found**: Ensure the Ollama service has finished downloading models
- **Connection errors**: Check that all services are running and healthy
- **GPU issues**: Verify NVIDIA Docker runtime is properly configured

- **cannot run the script**: try: `chmod +x scripts/run.sh`

# Resources
https://python.plainenglish.io/generating-perfectly-structured-json-using-llms-all-the-time-13b7eb504240