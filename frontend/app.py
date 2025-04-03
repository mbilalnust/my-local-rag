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

# Define available workspaces
WORKSPACES = {
    "Marketing": "marketing",
    "Taxation": "taxation",
    "Product": "product",
    "Data Team": "data_team"
}

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'current_workspace' not in st.session_state:
        st.session_state.current_workspace = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL

def select_workspace():
    """Let user select the workspace."""
    return st.sidebar.selectbox(
        "Select Workspace",
        options=list(WORKSPACES.keys()),
        key="workspace_selector"
    )

def select_model():
    """Let user select the model to use."""
    return st.sidebar.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.values()).index(DEFAULT_MODEL),
        key="model_selector"
    )

def handle_file_upload():
    """Handle file uploads (PDF and URL)."""
    st.sidebar.markdown("### Add Documents 📄")
    
    # Create tabs for PDF and URL upload
    pdf_tab, url_tab = st.sidebar.tabs(["PDF", "URL"])
    
    with pdf_tab:
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files (limit 200MB per file)",
            key="pdf_uploader"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.uploaded_files.append(uploaded_file)
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with url_tab:
        url = st.text_input(
            "Add URL",
            placeholder="https://example.com",
            help="Enter a URL to add to your knowledge base"
        )
        if st.button("Add URL", key="add_url"):
            if url:
                st.success(f"✅ URL added: {url}")
                return url
    return None

def clear_knowledge_base():
    """Clear the knowledge base."""
    if st.sidebar.button("Clear Knowledge Base"):
        try:
            import shutil
            if os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY)
                st.success("✨ Knowledge base cleared!")
                st.session_state.chain = None
                st.session_state.messages = []
                st.session_state.uploaded_files = []
                st.rerun()
        except Exception as e:
            st.error(f"Error clearing knowledge base: {str(e)}")

def initialize_rag_components(pdf_path, model_name):
    """Initialize RAG components."""
    try:
        # Initialize language model
        llm = ChatOllama(model=model_name)
        
        # Load and process document
        loader = DocumentLoader()
        documents = loader.load_pdf(pdf_path)
        if documents is None:
            st.error(f"Failed to load PDF document")
            return None
            
        # Split documents
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(documents)
        if chunks is None:
            st.error("Failed to split the document")
            return None
            
        # Initialize vector store
        vector_store = VectorStore()
        vector_db = vector_store.get_store(documents=chunks)
        
        # Create retriever
        doc_retriever = DocumentRetriever(vector_db, llm)
        retriever = doc_retriever.create_retriever()
        
        # Create RAG chain
        chain = RAGChain(retriever, llm)
        return chain
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None

def display_chat_messages():
    """Display chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handle user input in chat."""
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.chain:
                try:
                    response = st.session_state.chain.run(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                st.warning("Please upload a document first.")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Local RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Set dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        .stSidebar {
            background-color: #262730;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Local RAG")
        
        # Workspace selection
        selected_workspace = select_workspace()
        st.session_state.current_workspace = WORKSPACES[selected_workspace]
        
        # Model selection
        selected_model = select_model()
        st.session_state.selected_model = AVAILABLE_MODELS[selected_model]
        
        # Clear knowledge base option
        clear_knowledge_base()
        
        # File upload section
        url = handle_file_upload()
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.sidebar.markdown("### Uploaded Files")
            for file in st.session_state.uploaded_files:
                st.sidebar.text(f"📄 {file.name}")

    # Main chat interface
    if (st.session_state.uploaded_files or url) and not st.session_state.chain:
        with st.spinner("🔄 Processing documents..."):
            # Process all uploaded files
            for file in st.session_state.uploaded_files:
                pdf_path = os.path.join(UPLOAD_DIR, file.name)
                chain = initialize_rag_components(pdf_path, st.session_state.selected_model)
                if chain:
                    st.session_state.chain = chain
                    st.rerun()
                    break

    # Display chat interface
    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main() 