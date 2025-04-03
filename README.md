# Local RAG Application

A modular Retrieval-Augmented Generation (RAG) application that uses local models through Ollama for document question-answering.

![RAG Application Architecture]([images/pic.png](https://github.com/mbilalnust/my-local-rag))

## Features

- PDF document processing and chunking
- Vector storage with ChromaDB
- Multi-query retrieval for better context matching
- Local LLM integration through Ollama
- Streamlit web interface
- Modular and scalable architecture

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required models pulled in Ollama:
  - llama3.2 (or your preferred model)
  - nomic-embed-text (for embeddings)

## Project Structure

```
my-local-rag/
├── src/                    # Core application code
│   ├── config/            # Configuration settings
│   ├── database/          # Vector store management
│   ├── document_processing/# Document loading and splitting
│   ├── retrieval/         # Document retrieval logic
│   ├── chains/            # RAG chain implementation
│   └── utils/             # Utility functions
├── frontend/              # Streamlit web interface
├── notebooks/             # Jupyter notebooks for development
├── uploads/               # Temporary file upload directory
└── chroma_db/            # Vector database storage
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd my-local-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running and pull required models:
```bash
ollama pull llama2
ollama pull nomic-embed-text
```

## Usage

2. Start the Streamlit application:
```bash
cd frontend
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown by Streamlit (typically http://localhost:8501).

4. Enter your questions about the document in the text input field.

## Configuration

You can modify the following settings in `src/config/settings.py`:

- `MODEL_NAME`: The Ollama model to use for question answering (default: llama2)
- `EMBEDDING_MODEL`: The model to use for text embeddings (default: nomic-embed-text)
- `CHUNK_SIZE`: Size of document chunks
- `CHUNK_OVERLAP`: Overlap between chunks

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

# References used
1. https://www.youtube.com/watch?v=GWB9ApTPTv4
2. https://github.com/tonykipkemboi/ollama_pdf_rag
