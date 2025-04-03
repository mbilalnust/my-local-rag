"""Streamlit frontend for the RAG application."""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Third-party imports
from langchain_ollama import ChatOllama

# Local application imports
from src.config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    UPLOAD_DIR,
    PERSIST_DIRECTORY
)
from src.utils.logging_config import setup_logging
from src.document_processing.loader import DocumentLoader
from src.document_processing.splitter import DocumentSplitter
from src.database.vector_store import VectorStore
from src.retrieval.retriever import DocumentRetriever
from src.chains.rag_chain import RAGChain

# Setup logging
logger = setup_logging()

def handle_pdf_uploads():
    """Handle multiple PDF file uploads and return the paths to the PDFs.
    
    Returns:
        list: List of paths to the uploaded PDF files
    """
    st.write("### Upload Your Documents")
    st.write("Upload one or more PDF documents to ask questions about.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="You can select multiple PDF files by holding Ctrl/Cmd while selecting"
    )
    
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF document to continue.")
        return None
        
    pdf_paths = []
    for uploaded_file in uploaded_files:
        # Save uploaded file
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(save_path)
        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
    
    return pdf_paths

def select_model():
    """Let user select the model to use.
    
    Returns:
        str: Selected model name
    """
    st.write("### Model Selection")
    selected_model = st.selectbox(
        "Choose an LLM model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.values()).index(DEFAULT_MODEL),
        help="Select the AI model you want to use for answering questions."
    )
    return AVAILABLE_MODELS[selected_model]

@st.cache_resource
def initialize_rag_components(_pdf_paths, model_name):
    """Initialize all RAG components.
    
    Args:
        _pdf_paths (list): List of paths to the PDF documents
        model_name (str): Name of the LLM model to use
    
    Returns:
        tuple: (llm, chain) or (None, None) if initialization fails
    """
    try:
        # Initialize language model
        llm = ChatOllama(model=model_name)
        
        # Load and process all documents
        all_documents = []
        loader = DocumentLoader()
        for pdf_path in _pdf_paths:
            documents = loader.load_pdf(pdf_path)
            if documents is None:
                st.error(f"Failed to load PDF document: {os.path.basename(pdf_path)}")
                continue
            all_documents.extend(documents)
        
        if not all_documents:
            st.error("No documents were successfully loaded.")
            return None, None
            
        # Split documents
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(all_documents)
        if chunks is None:
            st.error("Failed to split the documents.")
            return None, None
            
        # Initialize vector store
        vector_store = VectorStore()
        vector_db = vector_store.get_store(documents=chunks)
        
        # Create retriever
        doc_retriever = DocumentRetriever(vector_db, llm)
        retriever = doc_retriever.create_retriever()
        
        # Create RAG chain
        chain = RAGChain(retriever, llm)
        
        return llm, chain
        
    except Exception as e:
        logger.error(f"Error initializing RAG components: {str(e)}")
        st.error(f"An error occurred during initialization: {str(e)}")
        return None, None

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document Assistant")
    st.write("Ask questions about your PDF documents using AI.")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Handle PDF uploads
        pdf_paths = handle_pdf_uploads()
        if pdf_paths is None:
            return
            
    with col2:
        # Model selection
        model_name = select_model()
        
        # Add a button to clear the vector store
        st.write("### Actions")
        if st.button("🗑️ Clear Vector Store"):
            try:
                import shutil
                if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY)
                    st.success("✨ Vector store cleared successfully!")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing vector store: {str(e)}")
    
    # Add a separator
    st.divider()
    
    # Initialize components
    with st.spinner("🔄 Initializing AI components..."):
        llm, chain = initialize_rag_components(pdf_paths, model_name)
        if llm is None or chain is None:
            st.error("Failed to initialize the application components.")
            return
    
    # Question and Answer section
    st.write("### Ask Questions")
    st.write("Enter your question about the documents below:")
    
    # User input with a more descriptive placeholder
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main topics discussed in these documents?",
        key="question_input"
    )
    
    if user_input:
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from chain
                response = chain.run(user_input)
                
                # Display answer
                st.write("### Answer:")
                st.write(response)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.info("👋 Please enter a question about your documents to get started.")

if __name__ == "__main__":
    main() 