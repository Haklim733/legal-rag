# Overview

This is sample repo to extract text from pdfs and extract ontological concepts using FOLIO ontology and LightRAG.

## Setup

ollama is required.
gpu will make the embedding much faster.

## Usage

```bash
uv run src.rag.main
```

## Details

the main.py is the entry point. it will load the ontology and embed it into the document store.

the embed.py is the main function to load the ontology and embed it into the document store.

the query_param.py is the main function to query the document store.
