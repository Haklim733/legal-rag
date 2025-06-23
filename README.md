# Overview

This is sample repo to extract text from legal pdfs like the supreme court cases and opinions and extract ontological concepts using the [Federated Open Legal Information Ontology (FOLIO) ontology](https://openlegalstandard.org/resources/folio-python-library/) and [LightRAG](https://github.com/HKUDS/LightRAG/). This is a proof of concept that stores FOLIO ontology in a knowledge graph that is utilized by LightRAG and llms to extract ontological concepts from the document.

# Execution  

1. download case from scotus website (default is DONALD J. TRUMP, PRESIDENT OF THE UNITED STATES, ET AL. v. GWYNNE A. WILCOX, ET AL. )
2. extract text from pdf
3. ask llm to extract ontological concepts using knowledge graph 
4. validate the response structure as JSON format

**IMPORTANT**: the neo4j docker container needs to be running. `docker compose up -d`

**Note**: this is a proof of concept and not a production ready solution. For quick testing, only 1 page of the case document is extracted at low resolution without preserving hierarchical structure. Moreover, only a subset of the FOLIO ontology is inserted into the knowledge graph. Embedding of text into a vector database is not currently implemented due to resource constraints. Lastly, validation of returned ontology concepts against the FOLIO ontology is not currently implemented.

```bash
./scripts/run.sh
```

### Prerequisites

- ollama on localhost machine
- NVIDIA GPU (optional, for GPU acceleration)
- docker compose (Neo4J needs to be running)
- uv python package manager

- **LLM Model**: `llama3.1:8b` for text generation
- **Embedding Model**: `all-minilm` for vector embeddings
- **Host**: `http://localhost:11434` (local ollama service)

```bash
# Pull the LLM model
ollama pull llama3.1:8b

# Pull the embedding model  
ollama pull all-minilm
```

## Usage
```bash
uv run -m src.rag.main
```

## Details
The main.py is the entry point. it will load the ontology and embed it into the document store.

The embed.py is the main function to load the ontology and embed it into the document store.
the models.py contains the pydantic models for validating the response structure from the LLM and contains the system prompt for the LLM.

The tests are in the tests folder.

### pdf extraction 
the main.py module will use the unstructured library to extract the text from the pdf. The default is 1 page and using the fast extraction strategy. you can change the page limit and extraction strategy in the main.py module.

### LLM models and system prompts
the llm models were chosen based on resource constraints of 8gb vram.
Initially, the phi3:mini model was used, but the response returned text in malformed JSON that could not be parsed correctly. The ollama 3.1b

### structured response
the response is structured in line with the structure required by lightrag when inserting a custom knowledge graph.

The weight is set to a default of 1.0 and should be disregarded for now.

### configuration
the parameters for the main.py can be set in the command line. Refer to the help message for the available options. 

```bash
uv run -m src.rag.main --help
```


# issues
1. the ontology may not be complete.
2. some subclasses inherit properties from their parents that should be excluded (e.g. Year of Law Degree Graduation)
3. not all predicates are of interest or have been vetted (only folio: predicates and oasis: predicates are used here). 
4. there are 660 labels with more than 1 classes 
5. Lightrag requires explicit relationships between entities.
6. JSON parsing not uniform across documents.


## TODO
1. validate Entity and Relationships in the response
2. examine folio:operators
4. vet the ontology (i.e. de-dup, vet inherited properties, etc)
5. checkout Langchain LLM Transformers
6. use llm api key
7. embed entity and relationship descriptions into vector db
8. create a data pipeline to extract text from pdfs and save to tables in data lake

## Docker Setup with Ollama Models
Coming soon

## Troubleshooting

- **Models not found**: Ensure the Ollama service has finished downloading models
- **Connection errors**: Check that all services are running and healthy
- **GPU issues**: Verify NVIDIA Docker runtime is properly configured

- **cannot run the script**: try: `chmod +x scripts/run.sh`