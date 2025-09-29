import requests
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple


# Configuration constants
BASE_URL = "https://api.aimlapi.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
FALLBACK_CHAT_MODEL = "openai/gpt-5-nano-2025-08-07"
MAX_RETRIES = 3
BASE_DELAY = 1
MAX_DELAY = 60


def get_api_key() -> str:
    """Get API key from environment or raise error if not found"""
    api_key = os.getenv("AIMLAPI_KEY")
    if not api_key:
        raise ValueError("AIMLAPI_KEY environment variable not found")
    return api_key


def make_request(endpoint: str, payload: Dict, operation: str = "API request") -> Tuple[bool, Any, str]:
    """
    Make HTTP request with retry logic

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

            if response.status_code == 200:
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



def generate_response(prompt: str, context: str = None, model: str = None,
                     max_tokens: int = 1000, temperature: float = 0.1) -> Tuple[bool, str, str]:
    """
    Generate chat completion using GPT-4 models

    Args:
        prompt: User prompt/question
        context: Additional context from retrieved documents
        model: Chat model to use (defaults to gpt-4o-mini)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        Tuple of (success, response_text, error_message)
    """
    if not prompt or not prompt.strip():
        return False, None, "Empty prompt provided"

    model = model or CHAT_MODEL

    # Construct messages
    messages = []

    # System message
    system_message = (
        "You are an expert document analysis assistant. You provide accurate, "
        "helpful answers based on the provided context. If information is not "
        "available in the context, clearly state that. Be concise but comprehensive."
    )
    messages.append({"role": "system", "content": system_message})

    # Add context if provided
    if context and context.strip():
        context_message = f"Document Context:\n{context.strip()}\n\nPlease answer the following question based on this context:"
        messages.append({"role": "user", "content": context_message})

    # Add user prompt
    messages.append({"role": "user", "content": prompt.strip()})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    success, data, error = make_request("/chat/completions", payload, "generate_response")

    if success:
        try:
            response_text = data['choices'][0]['message']['content']
            return True, response_text, None
        except (KeyError, IndexError, TypeError) as e:
            return False, None, f"Failed to extract response from API: {str(e)}"
    else:
        return False, None, error


def classify_document(text_sample: str, categories: List[str] = None) -> Tuple[bool, str, str]:
    """
    Auto-classify document category using GPT-4

    Args:
        text_sample: Sample text from document (first few paragraphs)
        categories: List of possible categories

    Returns:
        Tuple of (success, predicted_category, error_message)
    """
    if not categories:
        categories = [
            "Research Paper",
            "Article",
            "Book",
            "Other",
            "Receipts",
            "Terms & Conditions"
        ]

    if not text_sample or not text_sample.strip():
        return False, None, "Empty text sample provided"

    # Truncate sample to reasonable length
    max_chars = 2000
    if len(text_sample) > max_chars:
        text_sample = text_sample[:max_chars] + "..."

    prompt = f"""
Analyze this document sample and classify it into one of these categories:
{', '.join(categories)}

Document sample:
{text_sample}

Respond with only the category name that best fits this document.
"""

    success, response_text, error = generate_response(
        prompt=prompt,
        model=CHAT_MODEL,
        max_tokens=50,
        temperature=0.0
    )

    if not success:
        return False, None, error

    # Find matching category
    response_text = response_text.strip()
    for category in categories:
        if category.lower() in response_text.lower():
            return True, category, None

    # If no exact match, return the response text
    return True, response_text, None


def test_connection() -> Tuple[bool, str, str]:
    """
    Test API connection and authentication using a simple chat completion

    Returns:
        Tuple of (success, status_message, error_message)
    """
    try:
        success, _, error = generate_response("test", max_tokens=10)
        if success:
            return True, "Connection successful", None
        else:
            return False, None, f"Connection failed: {error}"
    except Exception as e:
        return False, None, f"Connection test failed: {str(e)}"




def generate_contextual_response(prompt: str, retrieved_chunks: List[str],
                                chunk_metadata: List[Dict] = None) -> Tuple[bool, str, str]:
    """
    Generate response using retrieved document chunks as context

    Args:
        prompt: User question
        retrieved_chunks: List of relevant text chunks
        chunk_metadata: Optional metadata for chunks (section names, etc.)

    Returns:
        Tuple of (success, response_text, error_message)
    """
    if not retrieved_chunks:
        return generate_response(prompt, context="No relevant context found.")

    # Combine chunks into context
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        if chunk_metadata and i < len(chunk_metadata):
            section = chunk_metadata[i].get('section', f'Section {i+1}')
            context_parts.append(f"From {section}:\n{chunk}")
        else:
            context_parts.append(f"Context {i+1}:\n{chunk}")

    combined_context = "\n\n---\n\n".join(context_parts)

    # Truncate context if too long
    max_context_length = 3000  # Conservative limit
    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "\n\n[Context truncated...]"

    return generate_response(prompt, context=combined_context)


def get_model_info() -> Dict[str, str]:
    """
    Get information about the models being used

    Returns:
        Dictionary with model information
    """
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "fallback_chat_model": FALLBACK_CHAT_MODEL,
        "base_url": BASE_URL
    }