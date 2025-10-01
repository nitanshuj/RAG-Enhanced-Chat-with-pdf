"""
Shared HTTP API Client for AI/ML API.

This module consolidates all HTTP communication logic with the AI/ML API,
including retry logic, rate limiting, error handling, and request formatting.

Used by both embeddings.py and llm_client.py to avoid code duplication.
"""

import requests
import time
import json
from typing import Any, Dict, Tuple
from src.config import (
    AIML_API_BASE_URL,
    AIML_API_KEY,
    MAX_RETRIES,
    BASE_DELAY,
    MAX_DELAY,
    REQUEST_TIMEOUT
)


def get_api_key() -> str:
    """
    Get API key from configuration.

    Returns:
        API key string

    Raises:
        ValueError: If API key is not configured
    """
    if not AIML_API_KEY:
        raise ValueError(
            "AIMLAPI_KEY not found in configuration. "
            "Please check your .env file and ensure AIMLAPI_KEY is set."
        )
    return AIML_API_KEY


def make_api_request(
    endpoint: str,
    payload: Dict[str, Any],
    operation: str = "API request"
) -> Tuple[bool, Any, str]:
    """
    Make HTTP POST request to AI/ML API with automatic retry logic.

    This is the core HTTP client used by all API interactions (embeddings, chat, etc.).
    Handles rate limiting, server errors, timeouts, and connection issues automatically.

    Args:
        endpoint: API endpoint path (e.g., "/embeddings", "/chat/completions")
        payload: Request payload dictionary
        operation: Operation name for logging/error messages

    Returns:
        Tuple of (success, response_data, error_message)
        - success: True if request succeeded, False otherwise
        - response_data: Parsed JSON response (if success=True) or None
        - error_message: Error description (if success=False) or None

    Example:
        >>> payload = {"model": "gpt-4o-mini", "messages": [...]}
        >>> success, data, error = make_api_request("/chat/completions", payload)
        >>> if success:
        ...     print(data['choices'][0]['message']['content'])
    """
    url = f"{AIML_API_BASE_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )

            # Success cases
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    return True, data, None
                except json.JSONDecodeError:
                    return False, None, f"Invalid JSON response from {operation}"

            # Rate limiting - retry with exponential backoff
            elif response.status_code == 429:
                if attempt < MAX_RETRIES:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"Rate limited. Retrying {operation} in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                else:
                    return False, None, "Rate limit exceeded. Maximum retries reached."

            # Server errors - retry with exponential backoff
            elif 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"Server error {response.status_code}. Retrying {operation} in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                else:
                    return False, None, f"Server error {response.status_code}: {response.text}"

            # Client errors (4xx) - don't retry, return error immediately
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', response.text)
                except:
                    error_message = response.text
                return False, None, f"API error {response.status_code}: {error_message}"

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                print(f"Request timeout. Retrying {operation} in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            else:
                return False, None, f"Request timeout after {REQUEST_TIMEOUT}s. Maximum retries reached."

        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                print(f"Connection error. Retrying {operation} in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            else:
                return False, None, "Connection error. Please check your internet connection."

        except requests.exceptions.RequestException as e:
            return False, None, f"Request failed for {operation}: {str(e)}"

    return False, None, f"Maximum retries ({MAX_RETRIES}) exceeded for {operation}"


def test_api_connection() -> Tuple[bool, str, str]:
    """
    Test API connectivity and authentication.

    Makes a minimal API request to verify that:
    - API key is valid
    - API endpoint is reachable
    - Authentication is working

    Returns:
        Tuple of (success, status_message, error_message)

    Example:
        >>> success, message, error = test_api_connection()
        >>> if success:
        ...     print(f"✓ {message}")
        ... else:
        ...     print(f"✗ {error}")
    """
    try:
        # Test with a minimal chat completion request
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }

        success, _, error = make_api_request("/chat/completions", payload, "connection test")

        if success:
            return True, "✓ API connection successful", None
        else:
            return False, None, f"✗ API connection failed: {error}"

    except Exception as e:
        return False, None, f"✗ Connection test error: {str(e)}"


def get_api_info() -> Dict[str, Any]:
    """
    Get API client configuration information.

    Returns:
        Dictionary with API client settings
    """
    return {
        "base_url": AIML_API_BASE_URL,
        "max_retries": MAX_RETRIES,
        "request_timeout": REQUEST_TIMEOUT,
        "base_delay": BASE_DELAY,
        "max_delay": MAX_DELAY,
        "api_key_configured": bool(AIML_API_KEY),
    }
