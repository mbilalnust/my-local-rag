"""Vector store module for document embeddings."""

import os
import logging
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from src.config.settings import (
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    VECTOR_STORE_NAME
)

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages document embeddings and vector storage."""
    
    def __init__(self):
        """Initialize the vector store with Ollama embeddings."""
        # Pull the embedding model if not already present
        ollama.pull(EMBEDDING_MODEL)
        self.embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    def get_store(self, documents=None):
        """Get or create a vector store.
        
        Args:
            documents: Optional list of documents to create a new store
            
        Returns:
            Chroma: Vector store instance
        """
        # If store exists, load it
        if os.path.exists(PERSIST_DIRECTORY):
            return Chroma(
                embedding_function=self.embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
        
        # Create new store if documents provided
        if documents:
            store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            store.persist()
            return store
            
        raise ValueError("No existing store found and no documents provided to create one.") 