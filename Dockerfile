# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Rust and required build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env

COPY requirements.txt .
COPY lightrag/api/requirements.txt ./lightrag/api/

# Install dependencies
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir -r lightrag/api/requirements.txt

# Install depndencies for default storage
RUN pip install --user --no-cache-dir nano-vectordb networkx
# Install depndencies for default LLM
RUN pip install --user --no-cache-dir openai ollama tiktoken
# Install depndencies for default document loader
RUN pip install --user --no-cache-dir pypdf2 python-docx python-pptx openpyxl

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY ./lightrag ./lightrag
COPY setup.py .

RUN pip install ".[api]"
# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Docker data directories
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs


# 5. Copy the dependency files and install dependencies
# This layer is cached by Docker, so dependencies are only re-installed when they change.
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev --extras api

# 6. Copy the application source code into the container
COPY ./src ./src

COPY ./.env ./

EXPOSE 9621

CMD ["lightrag-server", "--host", "0.0.0.0", "--port", "9621"]

