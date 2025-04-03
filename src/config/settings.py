"""Configuration settings for the RAG application."""

import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

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
VECTOR_STORE_NAME = "simple-rag"

# Document splitting configurations
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True) 