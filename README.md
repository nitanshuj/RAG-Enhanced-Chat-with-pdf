# Enhanced RAG-Enhanced Chat with PDF Support

-------------------------

### File Structure

```
RAG-Enhanced-Chat-with-pdf/
│
├── main.py                     # Main application entry point
├── pyproject.toml              # Project configuration and dependencies
├── requirements.txt            # Python package requirements
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── .python-version             # Python version specification
│
├── src/                        # Source code directory
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Core application logic
│   ├── agent.py               # Agentic RAG implementation
│   ├── document_processor.py   # Document processing and parsing
│   ├── llm_configuration.py   # LLM configuration and settings
│   ├── memory.py              # Chat memory and session management
│   ├── ollama_client.py       # Ollama client integration
│   ├── utils.py               # Utility functions
│   └── vector_store.py        # Vector store operations
│
├── documentation/              # Project documentation
│   ├── agentic_rag_idea.md    # Agentic RAG architecture overview
│   └── special_features.md    # Advanced features and improvements
│
├── archived/                   # Archived files (currently empty)
│
├── .venv/                     # Virtual environment (if present)
└── .git/                      # Git repository data
```