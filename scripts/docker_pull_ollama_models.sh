#!/bin/bash

# Script to pull required Ollama models for the LightRAG service

echo "Waiting for Ollama service to be ready..."
sleep 10

echo "Pulling required Ollama models..."

# Pull the LLM model used in main.py
echo "Pulling llama3.1:8b model..."
docker exec ollama ollama pull llama3.1:8b

# Pull the embedding model used in main.py
echo "Pulling all-minilm model..."
docker exec ollama ollama pull all-minilm

echo "All required models pulled successfully!"
echo "Available models:"
docker exec ollama ollama list 