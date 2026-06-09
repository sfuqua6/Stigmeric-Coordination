"""Structured logging configuration for AI Swarm Mechanics.

Usage:
    from swarm.core.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Message")
    logger.debug("Debug details")
    logger.warning("Warning")
    logger.error("Error")

Environment Variables:
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Default: INFO
    LOG_FILE: Optional log file path. If not set, logs to console only.

Examples:
    # Quiet mode (errors only)
    export LOG_LEVEL=ERROR
    python run_task.py creative

    # Debug mode (verbose)
    export LOG_LEVEL=DEBUG
    python run_task.py creative

    # Log to file
    export LOG_FILE=swarm.log
    python run_task.py creative
"""

import logging
import os
import sys
from typing import Optional


# Global flag to track if logging is configured
_logging_configured = False


def setup_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """Configure logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, reads from LOG_LEVEL env var (default: INFO)
        log_file: Optional log file path
                  If None, reads from LOG_FILE env var (default: console only)
    """
    global _logging_configured

    if _logging_configured:
        return  # Already configured

    # Get log level from parameter or environment
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'INFO').upper()

    # Validate log level
    valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    if level not in valid_levels:
        print(f"[WARNING] Invalid LOG_LEVEL '{level}', using INFO")
        level = 'INFO'

    # Get log file from parameter or environment
    if log_file is None:
        log_file = os.environ.get('LOG_FILE')

    # Configure root logger
    log_level = getattr(logging, level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=_get_handlers(log_file)
    )

    # Silence noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

    logger = logging.getLogger('swarm.core.logging_config')
    logger.info(f"Logging configured: level={level}, file={log_file or 'console'}")

    _logging_configured = True


def _get_handlers(log_file: Optional[str] = None) -> list:
    """Get list of logging handlers.

    Args:
        log_file: Optional log file path

    Returns:
        List of logging handlers
    """
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            handlers.append(file_handler)
        except Exception as e:
            print(f"[WARNING] Could not create log file {log_file}: {e}")

    return handlers


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Automatically sets up logging on first call.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting process")
    """
    # Auto-configure on first use
    setup_logging()

    return logging.getLogger(name)


# Convenience function for quick setup in scripts
def configure(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Convenience function to configure logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    global _logging_configured
    _logging_configured = False  # Reset to allow reconfiguration
    setup_logging(level, log_file)
