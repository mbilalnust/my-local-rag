"""Document splitter module."""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentSplitter:
    """Class for splitting documents into chunks."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize the document splitter.
        
        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents):
        """Split documents into smaller chunks.
        
        Args:
            documents (list): List of documents to split
            
        Returns:
            list: List of document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Documents split into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return None 