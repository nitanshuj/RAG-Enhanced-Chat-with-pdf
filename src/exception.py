"""
Custom exception classes for RAG-Enhanced Chat with PDF.

This module defines custom exceptions for better error handling and debugging
across the application.
"""

import sys
from typing import Optional


class RAGException(Exception):
    """
    Base exception class for RAG application.

    Attributes:
        message: Error message
        error_detail: Detailed error information from sys.exc_info()
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        """
        Initialize RAG exception.

        Args:
            message: Error message
            error_detail: Optional detailed error information from sys
        """
        super().__init__(message)
        self.message = message

        if error_detail:
            _, _, exc_tb = error_detail
            if exc_tb:
                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.line_number = exc_tb.tb_lineno
            else:
                self.file_name = "Unknown"
                self.line_number = 0
        else:
            self.file_name = "Unknown"
            self.line_number = 0

    def __str__(self):
        """Return formatted error message."""
        return f"Error in [{self.file_name}] at line [{self.line_number}]: {self.message}"

    def __repr__(self):
        """Return detailed representation."""
        return f"RAGException(message='{self.message}', file='{self.file_name}', line={self.line_number})"


class DocumentProcessingException(RAGException):
    """
    Exception raised during document processing operations.

    This includes PDF parsing, text extraction, and chunking errors.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"Document Processing Error: {message}", error_detail)


class EmbeddingException(RAGException):
    """
    Exception raised during embedding generation.

    This includes API errors, model errors, and embedding failures.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"Embedding Error: {message}", error_detail)


class VectorStoreException(RAGException):
    """
    Exception raised during vector store operations.

    This includes ChromaDB connection, insertion, and query errors.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"Vector Store Error: {message}", error_detail)


class LLMException(RAGException):
    """
    Exception raised during LLM operations.

    This includes API errors, model errors, and generation failures.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"LLM Error: {message}", error_detail)


class AgentException(RAGException):
    """
    Exception raised during RAG agent operations.

    This includes context building, query processing, and response generation errors.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"Agent Error: {message}", error_detail)


class ConfigurationException(RAGException):
    """
    Exception raised for configuration-related errors.

    This includes missing environment variables, invalid settings, and configuration errors.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"Configuration Error: {message}", error_detail)


class ValidationException(RAGException):
    """
    Exception raised for validation errors.

    This includes invalid inputs, malformed data, and constraint violations.
    """

    def __init__(self, message: str, error_detail: Optional[sys.exc_info] = None):
        super().__init__(f"Validation Error: {message}", error_detail)


class APIException(RAGException):
    """
    Exception raised for external API errors.

    This includes HTTP errors, timeout errors, and API response errors.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_detail: Optional[sys.exc_info] = None
    ):
        self.status_code = status_code
        status_info = f" (Status: {status_code})" if status_code else ""
        super().__init__(f"API Error{status_info}: {message}", error_detail)


# Utility functions for exception handling

def handle_exception(exception: Exception, logger=None) -> str:
    """
    Handle an exception and return a formatted error message.

    Args:
        exception: The exception to handle
        logger: Optional logger to log the error

    Returns:
        Formatted error message string

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     error_msg = handle_exception(e, logger)
        ...     print(error_msg)
    """
    if isinstance(exception, RAGException):
        error_message = str(exception)
    else:
        error_message = f"Unexpected Error: {str(exception)}"

    if logger:
        logger.error(error_message, exc_info=True)

    return error_message


def create_detailed_error(
    exception: Exception,
    context: str = ""
) -> dict:
    """
    Create a detailed error dictionary from an exception.

    Args:
        exception: The exception to process
        context: Additional context about where the error occurred

    Returns:
        Dictionary with error details

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     error_info = create_detailed_error(e, "During document upload")
        ...     print(error_info)
    """
    error_dict = {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "context": context
    }

    if isinstance(exception, RAGException):
        error_dict["file"] = exception.file_name
        error_dict["line"] = exception.line_number

    if isinstance(exception, APIException) and hasattr(exception, 'status_code'):
        error_dict["status_code"] = exception.status_code

    return error_dict


if __name__ == "__main__":
    # Test custom exceptions
    import sys

    print("Testing custom exceptions:\n")

    # Test basic RAGException
    try:
        raise RAGException("This is a test error")
    except RAGException as e:
        print(f"1. {e}\n")

    # Test DocumentProcessingException with error detail
    try:
        try:
            1 / 0  # Cause an error
        except Exception:
            raise DocumentProcessingException("Failed to process document", sys.exc_info())
    except DocumentProcessingException as e:
        print(f"2. {e}\n")

    # Test APIException with status code
    try:
        raise APIException("API request failed", status_code=404)
    except APIException as e:
        print(f"3. {e}\n")

    # Test create_detailed_error
    try:
        raise VectorStoreException("Connection timeout")
    except VectorStoreException as e:
        error_info = create_detailed_error(e, "During vector store initialization")
        print(f"4. Detailed error info: {error_info}\n")
