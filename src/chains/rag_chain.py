"""RAG chain module for question answering."""

import logging
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

class RAGChain:
    """Class for creating and managing the RAG chain."""
    
    def __init__(self, retriever, llm):
        """Initialize the RAG chain.
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
        self.chain = self._create_chain()
        
    def _create_chain(self):
        """Create the RAG chain.
        
        Returns:
            Chain: Configured chain
        """
        try:
            template = """Answer the question based ONLY on the following context: {context}
            Question: {question}
            Answer the question and ONLY use information from the provided context. If you cannot answer the question based on the context, say so. 
            Make sure to be precise and concise in your answer.
        """

            prompt = ChatPromptTemplate.from_template(template)
            
            # Create the chain
            chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("RAG chain created successfully.")
            return chain
            
        except Exception as e:
            logger.error(f"Error creating chain: {str(e)}")
            raise
            
    def run(self, question: str) -> str:
        """Run the chain with a question.
        
        Args:
            question (str): User's question
            
        Returns:
            str: Generated answer
        """
        try:
            return self.chain.invoke(question)
        except Exception as e:
            logger.error(f"Error running chain: {str(e)}")
            raise 