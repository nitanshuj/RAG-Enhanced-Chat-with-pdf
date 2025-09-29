"""
Embedding generation module for RAG-Enhanced Chat with PDF.

This module handles all embedding generation using the AI/ML API,
including single text embeddings and batch processing for multiple chunks.
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
BASE_URL = "https://api.aimlapi.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
BASE_DELAY = 1
MAX_DELAY = 60


def get_api_key() -> str:
    """Get API key from environment or raise error if not found"""
    api_key = os.getenv("AIMLAPI_KEY")
    if not api_key:
        raise ValueError("AIMLAPI_KEY environment variable not found")
    return api_key


def make_embedding_request(endpoint: str, payload: Dict, operation: str = "Embedding request") -> Tuple[bool, Any, str]:
    """
    Make HTTP request with retry logic specifically for embedding operations

    Args:
        endpoint: API endpoint (e.g., "/embeddings")
        payload: Request payload
        operation: Operation name for error messages

    Returns:
        Tuple of (success, data, error_message)
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code in [200, 201]:  # Accept both 200 OK and 201 Created
                try:
                    data = response.json()
                    return True, data, None
                except json.JSONDecodeError:
                    return False, None, f"Invalid JSON response from {operation}"

            elif response.status_code == 429:  # Rate limited
                if attempt < MAX_RETRIES:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"Rate limited. Retrying {operation} in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return False, None, "Rate limit exceeded. Max retries reached."

            elif 500 <= response.status_code < 600:  # Server error
                if attempt < MAX_RETRIES:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"Server error {response.status_code}. Retrying {operation} in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return False, None, f"Server error {response.status_code}: {response.text}"

            else:  # Client error
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', response.text)
                except:
                    error_message = response.text
                return False, None, f"API error {response.status_code}: {error_message}"

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                print(f"Request timeout. Retrying {operation} in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                return False, None, "Request timeout. Max retries reached."

        except requests.exceptions.RequestException as e:
            return False, None, f"Request failed for {operation}: {str(e)}"

    return False, None, f"Max retries exceeded for {operation}"


def generate_embeddings(text: str, model: str = None) -> Tuple[bool, List[float], str]:
    """
    Generate embeddings for text using AI/ML API

    Args:
        text: Text to embed
        model: Embedding model to use (defaults to text-embedding-3-small)

    Returns:
        Tuple of (success, embedding_vector, error_message)
    """
    if not text or not text.strip():
        return False, None, "Empty text provided for embedding"

    model = model or EMBEDDING_MODEL

    # Truncate text if too long
    max_tokens = 8000
    if len(text.split()) > max_tokens:
        words = text.split()
        text = ' '.join(words[:max_tokens])

    payload = {
        "model": model,
        "input": text.strip(),
        "encoding_format": "float"
    }

    success, data, error = make_embedding_request("/embeddings", payload, "generate_embeddings")

    if success:
        try:
            embedding = data['data'][0]['embedding']
            return True, embedding, None
        except (KeyError, IndexError, TypeError) as e:
            return False, None, f"Failed to extract embedding from response: {str(e)}"
    else:
        return False, None, error


def get_embedding_for_chunks(chunks: List[str]) -> Tuple[bool, List[List[float]], List[str]]:
    """
    Generate embeddings for multiple text chunks

    Args:
        chunks: List of text chunks to embed

    Returns:
        Tuple of (success, list_of_embeddings, list_of_errors)
    """
    embeddings = []
    errors = []

    for i, chunk in enumerate(chunks):
        success, embedding, error = generate_embeddings(chunk)

        if success:
            embeddings.append(embedding)
            errors.append(None)
        else:
            embeddings.append(None)
            errors.append(f"Chunk {i+1}: {error}")
            print(f"Failed to embed chunk {i+1}: {error}")

    # Check if any embeddings were successful
    successful_count = len([e for e in embeddings if e is not None])
    overall_success = successful_count > 0

    return overall_success, embeddings, errors


def test_embedding_connection() -> Tuple[bool, str, str]:
    """
    Test embedding API connection and authentication

    Returns:
        Tuple of (success, status_message, error_message)
    """
    try:
        success, _, error = generate_embeddings("test connection")
        if success:
            return True, "Embedding connection successful", None
        else:
            return False, None, f"Embedding connection failed: {error}"
    except Exception as e:
        return False, None, f"Embedding connection test failed: {str(e)}"


def get_embedding_model_info() -> Dict[str, str]:
    """
    Get information about the embedding model being used

    Returns:
        Dictionary with embedding model information
    """
    return {
        "embedding_model": EMBEDDING_MODEL,
        "base_url": BASE_URL,
        "max_retries": MAX_RETRIES
    }


# Utility functions for embedding validation and processing

def validate_embedding(embedding: List[float]) -> bool:
    """
    Validate that an embedding is properly formatted

    Args:
        embedding: Embedding vector to validate

    Returns:
        True if valid, False otherwise
    """
    if not embedding:
        return False

    if not isinstance(embedding, list):
        return False

    if len(embedding) == 0:
        return False

    # Check if all elements are numbers
    for value in embedding:
        if not isinstance(value, (int, float)):
            return False

    return True


def get_embedding_dimension(embedding: List[float]) -> int:
    """
    Get the dimension of an embedding vector

    Args:
        embedding: Embedding vector

    Returns:
        Dimension of the embedding
    """
    return len(embedding) if embedding else 0


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize an embedding vector to unit length

    Args:
        embedding: Embedding vector to normalize

    Returns:
        Normalized embedding vector
    """
    if not embedding:
        return embedding

    # Calculate magnitude
    magnitude = sum(x * x for x in embedding) ** 0.5

    if magnitude == 0:
        return embedding

    # Normalize
    return [x / magnitude for x in embedding]