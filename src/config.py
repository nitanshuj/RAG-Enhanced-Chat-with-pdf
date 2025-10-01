"""
Configuration file for RAG-Enhanced Chat with PDF.

This file centralizes all configuration constants for easy modification.
Change model names, API settings, and application parameters here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# API CONFIGURATION
# ==============================================================================

# AI/ML API Configuration
AIML_API_BASE_URL = "https://api.aimlapi.com/v1"
AIML_API_KEY = os.getenv("AIMLAPI_KEY")

# ChromaDB Cloud Configuration
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Chat/LLM Models
CHAT_MODEL = "gpt-4o-mini"  # Primary chat model for responses
FALLBACK_CHAT_MODEL = "gpt-4"  # Fallback if primary fails (optional)

# Embedding Models
EMBEDDING_MODEL = "text-embedding-3-small"  # Primary embedding model
EMBEDDING_DIMENSION = 1536  # Dimension of embedding vectors

# Model Parameters
DEFAULT_CHAT_TEMPERATURE = 0.1  # Lower = more focused, Higher = more creative
DEFAULT_MAX_TOKENS = 1000  # Maximum tokens in LLM response
DEFAULT_EMBEDDING_MAX_TOKENS = 8000  # Maximum tokens for embedding input

# ==============================================================================
# HTTP REQUEST CONFIGURATION
# ==============================================================================

# Retry and Rate Limiting
MAX_RETRIES = 3  # Number of retries for failed requests
BASE_DELAY = 1  # Base delay in seconds for exponential backoff
MAX_DELAY = 60  # Maximum delay in seconds between retries
REQUEST_TIMEOUT = 30  # Request timeout in seconds

# ==============================================================================
# RAG SYSTEM CONFIGURATION
# ==============================================================================

# Document Processing
DEFAULT_CHUNK_SIZE = 500  # Default chunk size for simple chunking
RESEARCH_PAPER_CHUNK_SIZE = 2000  # Chunk size for research papers
DEFAULT_CHUNK_OVERLAP = 50  # Character overlap between chunks
HEADING_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for heading detection

# Vector Store
DEFAULT_COLLECTION_NAME = "documents"  # ChromaDB collection name
DEFAULT_TOP_K = 5  # Number of results to retrieve in similarity search

# RAG Agent
MAX_CONTEXT_LENGTH = 2500  # Maximum context length for LLM
MAX_CONVERSATION_HISTORY = 10  # Maximum conversation history entries
RECENT_HISTORY_COUNT = 3  # Recent history entries to include in queries

# ==============================================================================
# DOCUMENT CATEGORIES
# ==============================================================================

DOCUMENT_CATEGORIES = [
    "Research Paper",
    "Article",
    "Book",
    "Other",
    "Receipts",
    "Terms & Conditions"
]

# ==============================================================================
# UI CONFIGURATION
# ==============================================================================

# Streamlit Settings
APP_TITLE = "PDF Document Q&A Chatbot"
APP_ICON = "🤖"
APP_LAYOUT = "wide"

# File Upload Settings
ALLOWED_FILE_TYPES = ['pdf']
MAX_FILE_SIZE_MB = 50  # Maximum file size in megabytes

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

LOG_LEVEL = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_config():
    """
    Validate that all required configuration is present.

    Raises:
        ValueError: If required configuration is missing
    """
    required_vars = {
        "AIMLAPI_KEY": AIML_API_KEY,
        "CHROMA_API_KEY": CHROMA_API_KEY,
        "CHROMA_TENANT": CHROMA_TENANT,
        "CHROMA_DATABASE": CHROMA_DATABASE,
    }

    missing = [key for key, value in required_vars.items() if not value]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please check your .env file and ensure all required variables are set."
        )


def get_model_info():
    """
    Get current model configuration information.

    Returns:
        Dictionary with model configuration details
    """
    return {
        "chat_model": CHAT_MODEL,
        "fallback_chat_model": FALLBACK_CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "base_url": AIML_API_BASE_URL,
        "max_retries": MAX_RETRIES,
        "request_timeout": REQUEST_TIMEOUT,
    }


def get_rag_config():
    """
    Get RAG system configuration information.

    Returns:
        Dictionary with RAG configuration details
    """
    return {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "research_chunk_size": RESEARCH_PAPER_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": DEFAULT_TOP_K,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "collection_name": DEFAULT_COLLECTION_NAME,
    }


# Validate configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"Configuration Warning: {e}")
