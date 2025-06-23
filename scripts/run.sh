#!/bin/bash
set -e

# Create a directory to store downloaded assets
ASSET_DIR="assets/supreme-court/downloaded"
mkdir -p "$ASSET_DIR"

# Define the PDF URL and local file path
PDF_URL='https://www.supremecourt.gov/opinions/24pdf/24a966_1b8e.pdf'
PDF_FILENAME=$(basename "$PDF_URL")
PDF_PATH="$ASSET_DIR/$PDF_FILENAME"

# Download the file if it doesn't already exist
if [ ! -f "$PDF_PATH" ]; then
    echo "Downloading $PDF_URL..."
    # Use wget or curl. Let's use wget for simplicity.
    wget -q -O "$PDF_PATH" "$PDF_URL"
    echo "Download complete."
else
    echo "$PDF_FILENAME already exists. Skipping download."
fi

docker compose up -d

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to initialize..."
timeout=180 # 3 minutes
interval=5
elapsed=0

# Loop until the container's health status is "healthy"
while [ "$(docker inspect -f '{{.State.Health.Status}}' neo4j-rag)" != "healthy" ]; do
    if [ $elapsed -ge $timeout ]; then
        echo "Timed out waiting for Neo4j to become healthy."
        echo "Dumping Neo4j logs:"
        docker logs neo4j-rag
        exit 1
    fi
    sleep $interval
    elapsed=$((elapsed + interval))
    current_status=$(docker inspect -f '{{.State.Health.Status}}' neo4j-rag)
    echo "Still waiting for Neo4j... current status: $current_status"
done

echo "✅ Neo4j is healthy and ready."

# Run the RAG main script
echo "Running RAG on $PDF_PATH..."
uv run python -m src.rag.main --file-path "$PDF_PATH"
