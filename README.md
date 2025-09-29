# Enhanced RAG-Enhanced Chat with PDF Support

**Note: This application only accepts PDF documents. Other file formats are not supported.**

-------------------------

### Simplified File Structure

```
RAG-Enhanced-Chat-with-pdf/
│
├── main.py                     # Streamlit app entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env.example               # Environment configuration template
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code directory
│   ├── __init__.py            # Package initialization
│   ├── agent.py               # Agentic RAG core logic
│   ├── document_processor.py   # Multi-category document processing
│   ├── llm_client.py          # AI/ML API client (replaces ollama_client)
│   ├── vector_store.py        # Chroma DB Cloud operations
│   └── utils.py               # Utility functions and helpers
│
├── documentation/              # Project documentation
│   ├── agentic_rag_idea.md    # Multi-category RAG architecture
│   ├── vector_search_and_storage.md # Vector store configuration
│   └── special_features.md    # Advanced features (future)
│
└── data/                      # Document storage and cache
    ├── uploads/               # Uploaded PDF documents
    ├── processed/             # Processed document chunks
    └── chroma_config/         # Chroma DB Cloud connection configuration
```