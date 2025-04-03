"""Document loader module."""

import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Class for loading documents."""
    
    @staticmethod
    def load_pdf(doc_path: str):
        """Load PDF documents.
        
        Args:
            doc_path (str): Path to the PDF document
            
        Returns:
            list: List of loaded documents or None if loading fails
        """
        if os.path.exists(doc_path):
            try:
                loader = UnstructuredPDFLoader(file_path=doc_path)
                data = loader.load()
                logger.info("PDF loaded successfully.")
                return data
            except Exception as e:
                logger.error(f"Error loading PDF: {str(e)}")
                return None
        else:
            logger.error(f"PDF file not found at path: {doc_path}")
            return None 