"""
LLM Client Module for RAG-Enhanced Chat with PDF.

This module handles chat completion generation using Langchain with AI/ML API.
"""

from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import (
    CHAT_MODEL,
    AIML_API_BASE_URL,
    AIML_API_KEY,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_MAX_TOKENS
)
from src.logger import get_logger
from src.exception import LLMException
import sys

logger = get_logger(__name__)


def get_llm_model(temperature: float = None, max_tokens: int = None):
    """Get configured Langchain ChatOpenAI model."""
    return ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_base=AIML_API_BASE_URL,
        openai_api_key=AIML_API_KEY,
        temperature=temperature if temperature is not None else DEFAULT_CHAT_TEMPERATURE,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS
    )


def generate_response(
    prompt: str,
    context: str = None,
    model: str = None,
    max_tokens: int = None,
    temperature: float = None
) -> Tuple[bool, str, str]:
    """
    Generate chat completion using Langchain ChatOpenAI.

    Args:
        prompt: User prompt/question
        context: Additional context from retrieved documents
        model: Chat model to use (defaults to CHAT_MODEL from config)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature 0.0-1.0

    Returns:
        Tuple of (success, response_text, error_message)

    Example:
        >>> success, response, error = generate_response(
        ...     prompt="What is RAG?",
        ...     context="RAG stands for Retrieval-Augmented Generation..."
        ... )
        >>> if success:
        ...     print(response)
        ... else:
        ...     print(f"Error: {error}")
    """
    if not prompt or not prompt.strip():
        return False, None, "Empty prompt provided"

    try:
        llm = get_llm_model(temperature, max_tokens)
        messages = _build_messages(prompt, context)
        logger.debug(f"Generating LLM response for prompt: {prompt[:50]}...")
        response = llm.invoke(messages)
        logger.info(f"Successfully generated LLM response ({len(response.content)} chars)")
        return True, response.content, None
    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def _build_messages(prompt: str, context: str = None) -> List:
    """
    Build message list for Langchain ChatOpenAI.

    Args:
        prompt: User prompt/question
        context: Optional context from retrieved documents

    Returns:
        List of Langchain message objects
    """
    messages = []

    # System message
    system_message = (
        "You are an expert document analysis assistant. You provide accurate, "
        "helpful answers based on the provided context. If information is not "
        "available in the context, clearly state that. Be concise but comprehensive."
    )
    messages.append(SystemMessage(content=system_message))

    # Add context if provided
    if context and context.strip():
        context_message = (
            f"Document Context:\n{context.strip()}\n\n"
            f"Please answer the following question based on this context:"
        )
        messages.append(HumanMessage(content=context_message))

    # Add user prompt
    messages.append(HumanMessage(content=prompt.strip()))

    return messages


def generate_contextual_response(
    prompt: str,
    retrieved_chunks: List[str],
    chunk_metadata: List[Dict] = None,
    model: str = None,
    max_tokens: int = None,
    temperature: float = None
) -> Tuple[bool, str, str]:
    """
    Generate response using retrieved document chunks as context.

    Args:
        prompt: User question
        retrieved_chunks: List of relevant text chunks
        chunk_metadata: Optional metadata for chunks
        model: Chat model to use (optional)
        max_tokens: Maximum tokens in response (optional)
        temperature: Sampling temperature (optional)

    Returns:
        Tuple of (success, response_text, error_message)
    """
    if not retrieved_chunks:
        return generate_response(
            prompt=prompt,
            context="No relevant context found.",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

    # Combine chunks into context
    context = _format_chunks_as_context(retrieved_chunks, chunk_metadata)

    return generate_response(
        prompt=prompt,
        context=context,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )


def _format_chunks_as_context(
    chunks: List[str],
    metadata: List[Dict] = None,
    max_context_length: int = 3000
) -> str:
    """Format retrieved chunks into a single context string."""
    context_parts = []

    for i, chunk in enumerate(chunks):
        if metadata and i < len(metadata):
            section = metadata[i].get('section', f'Section {i+1}')
            context_parts.append(f"From {section}:\n{chunk}")
        else:
            context_parts.append(f"Context {i+1}:\n{chunk}")

    combined_context = "\n\n---\n\n".join(context_parts)

    if len(combined_context) > max_context_length:
        combined_context = combined_context[:max_context_length] + "\n\n[Context truncated...]"

    return combined_context


def test_llm_connection() -> Tuple[bool, str, str]:
    """Test LLM API connection and functionality."""
    try:
        success, response, error = generate_response(
            prompt="test",
            max_tokens=10,
            temperature=0.0
        )

        if success:
            return True, "LLM connection successful", None
        else:
            return False, None, f"LLM connection failed: {error}"

    except Exception as e:
        return False, None, f"LLM test error: {str(e)}"


def get_llm_model_info() -> Dict[str, str]:
    """Get information about the LLM models being used."""
    return {
        "chat_model": CHAT_MODEL,
        "default_temperature": DEFAULT_CHAT_TEMPERATURE,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
    }
