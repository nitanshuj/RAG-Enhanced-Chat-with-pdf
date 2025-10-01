"""
Custom logging configuration for RAG-Enhanced Chat with PDF.

This module provides centralized logging with file rotation, proper formatting,
and different log levels for different components.
"""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys


# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file configuration
LOG_FILE = os.path.join(LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
ERROR_LOG_FILE = os.path.join(LOGS_DIR, f"error_{datetime.now().strftime('%Y%m%d')}.log")

# Log format
LOG_FORMAT = "[ %(asctime)s ] %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Maximum log file size (10 MB) and backup count
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


def setup_logger(
    name: str,
    log_file: str = LOG_FILE,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Name of the logger (usually __name__ of the module)
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console

    Returns:
        Configured logger instance

    Example:
        >>> from src.logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Error file handler (only logs ERROR and CRITICAL)
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    Args:
        name: Name of the logger (usually __name__ of the module)

    Returns:
        Configured logger instance

    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
        >>> logger.critical("Critical message")
    """
    return setup_logger(name)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and results.

    Args:
        logger: Logger instance to use

    Example:
        >>> from src.logger import get_logger, log_function_call
        >>> logger = get_logger(__name__)
        >>>
        >>> @log_function_call(logger)
        ... def add(a, b):
        ...     return a + b
        >>>
        >>> result = add(2, 3)  # Will log the call and result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func_name} raised exception: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use

    Example:
        >>> from src.logger import get_logger, log_execution_time
        >>> import time
        >>> logger = get_logger(__name__)
        >>>
        >>> @log_execution_time(logger)
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "Done"
        >>>
        >>> result = slow_function()  # Will log execution time
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            func_name = func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func_name} executed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator


# Create a default logger for the application
app_logger = get_logger("rag_app")


if __name__ == "__main__":
    # Test the logger
    test_logger = get_logger("test")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")

    print(f"\nLogs written to: {LOG_FILE}")
    print(f"Error logs written to: {ERROR_LOG_FILE}")
