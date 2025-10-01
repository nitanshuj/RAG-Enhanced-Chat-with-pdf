"""
Embedding Generation Module for RAG-Enhanced Chat with PDF.

This module handles text embedding generation using Langchain with AI/ML API.
"""

from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    AIML_API_BASE_URL,
    AIML_API_KEY
)
from src.logger import get_logger
from src.exception import EmbeddingException
import sys

logger = get_logger(__name__)


# Initialize Langchain OpenAI Embeddings with AI/ML API
def get_embeddings_model():
    """Get configured Langchain embeddings model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=AIML_API_BASE_URL,
        openai_api_key=AIML_API_KEY,
        check_embedding_ctx_length=False  # Disable token counting that may cause issues
    )


def generate_embeddings(
    text: str,
    model: str = None
) -> Tuple[bool, List[float], str]:
    """
    Generate embedding vector for text using Langchain.

    Args:
        text: Text to embed
        model: Embedding model to use (defaults to EMBEDDING_MODEL from config)

    Returns:
        Tuple of (success, embedding_vector, error_message)

    Example:
        >>> success, embedding, error = generate_embeddings("Hello world")
        >>> if success:
        ...     print(f"Embedding dimension: {len(embedding)}")
        ... else:
        ...     print(f"Error: {error}")
    """
    if not text or not text.strip():
        return False, None, "Empty text provided for embedding"

    try:
        embeddings_model = get_embeddings_model()
        embedding = embeddings_model.embed_query(text.strip())
        logger.debug(f"Generated embedding of dimension {len(embedding)}")
        return True, embedding, None
    except Exception as e:
        error_msg = f"Failed to generate embedding: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def generate_embeddings_batch(
    texts: List[str],
    model: str = None
) -> Tuple[bool, List[List[float]], List[str]]:
    """
    Generate embeddings for multiple text chunks using Langchain.

    Args:
        texts: List of text strings to embed
        model: Embedding model to use (optional)

    Returns:
        Tuple of (success, list_of_embeddings, list_of_errors)

    Example:
        >>> texts = ["Hello world", "Goodbye world"]
        >>> success, embeddings, errors = generate_embeddings_batch(texts)
        >>> successful_count = sum(1 for e in embeddings if e is not None)
        >>> print(f"Successfully embedded {successful_count}/{len(texts)} texts")
    """
    embeddings = []
    errors = []

    try:
        embeddings_model = get_embeddings_model()
        embeddings = embeddings_model.embed_documents([t.strip() for t in texts if t and t.strip()])
        errors = [None] * len(embeddings)
        logger.info(f"Generated {len(embeddings)} embeddings in batch")
        return True, embeddings, errors
    except Exception as e:
        logger.warning(f"Batch embedding failed, falling back to individual: {str(e)}")
        # Fallback to individual embedding
        for i, text in enumerate(texts):
            success, embedding, error = generate_embeddings(text, model)
            if success:
                embeddings.append(embedding)
                errors.append(None)
            else:
                embeddings.append(None)
                errors.append(f"Text {i+1}: {error}")
                logger.error(f"Failed to embed text {i+1}/{len(texts)}: {error}")

        successful_count = len([e for e in embeddings if e is not None])
        overall_success = successful_count > 0
        return overall_success, embeddings, errors


def test_embedding_connection() -> Tuple[bool, str, str]:
    """
    Test embedding API connection and functionality.

    Returns:
        Tuple of (success, status_message, error_message)

    Example:
        >>> success, message, error = test_embedding_connection()
        >>> if success:
        ...     print(f"✓ {message}")
        ... else:
        ...     print(f"✗ {error}")
    """
    try:
        success, embedding, error = generate_embeddings("test connection")

        if success:
            return True, f"✓ Embedding connection successful (dimension: {len(embedding)})", None
        else:
            return False, None, f"✗ Embedding connection failed: {error}"

    except Exception as e:
        return False, None, f"✗ Embedding test error: {str(e)}"


def get_embedding_model_info():
    """
    Get information about the embedding model being used.

    Returns:
        Dictionary with embedding model configuration
    """
    return {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
    }
