"""Configuration settings for the RAG application."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Path configurations
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"

# Model configurations
AVAILABLE_MODELS = {
    "Llama 2": "llama2",
    "Llama 3.2": "llama3.2",
    "Mistral": "mistral",
    "CodeLlama": "codellama",
    "Gemma": "gemma",
}
DEFAULT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "simple-rag")

# Document splitting configurations
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
