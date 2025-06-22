# Overview

This is sample repo to extract text from pdfs and extract ontological concepts using FOLIO ontology and LightRAG. This is a proof of concept to show how to use FOLIO ontology to extract ontological concepts and use LightRAG to query the document store.

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
