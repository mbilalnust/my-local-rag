"""Retriever module for querying the vector store."""

import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Class for retrieving relevant documents."""
    
    def __init__(self, vector_store, llm):
        """Initialize the retriever.
        
        Args:
            vector_store: Vector store instance
            llm: Language model instance
        """
        self.vector_store = vector_store
        self.llm = llm
        
    def create_retriever(self):
        """Create a multi-query retriever.
        
        Returns:
            MultiQueryRetriever: Configured retriever
        """
        try:
            query_prompt = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
            )

            retriever = MultiQueryRetriever.from_llm(
                self.vector_store.as_retriever(),
                self.llm,
                prompt=query_prompt
            )
            logger.info("Multi-query retriever created successfully.")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise 