"""Vector store module for document embeddings."""

import logging
from pathlib import Path
from typing import List, Optional

import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from src.config.settings import EMBEDDING_MODEL, PERSIST_DIRECTORY, VECTOR_STORE_NAME

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and vector storage."""

    def __init__(self):
        # Pull the embedding model if not already present
        ollama.pull(EMBEDDING_MODEL)
        self.embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    def get_store(self, documents: Optional[List] = None):
        """Get or create a vector store."""
        try:
            persist_dir = Path(PERSIST_DIRECTORY)
            if documents:
                store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding,
                    collection_name=VECTOR_STORE_NAME,
                    persist_directory=str(persist_dir),
                )
                store.persist()
                logger.info("New vector store created successfully.")
                return store

            if persist_dir.exists():
                store = Chroma(
                    embedding_function=self.embedding,
                    collection_name=VECTOR_STORE_NAME,
                    persist_directory=str(persist_dir),
                )
                logger.info("Existing vector store loaded successfully.")
                return store

            raise ValueError("No existing store found and no documents provided to create one.")

        except Exception as e:  # pragma: no cover - integration path
            logger.error(f"Error in vector store operations: {str(e)}")
            raise
