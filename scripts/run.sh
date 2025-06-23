#!/bin/bash
set -e

# Create a directory to store downloaded assets
ASSET_DIR="assets/supreme-court/downloaded"
mkdir -p "$ASSET_DIR"

# Define the PDF URL and local file path
PDF_URL="https://www.supremecourt.gov/opinions/24pdf/24a1007_g2bh.pdf"
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

# Run the RAG main script
echo "Running RAG on $PDF_PATH..."
uv run python -m src.rag.main --file-path "$PDF_PATH"
