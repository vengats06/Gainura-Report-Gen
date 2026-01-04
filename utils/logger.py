"""
Logging Configuration for StockPulse Analytics
==============================================

This module sets up structured logging for the entire application.
Logs are written to both console and file for debugging and monitoring.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.error("An error occurred")
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Log filename with date
LOG_FILE = os.path.join(LOG_DIR, f"stockpulse_{datetime.now().strftime('%Y%m%d')}.log")

# Configure log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name.
    
    Args:
        name (str): Usually __name__ from the calling module
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Starting data collection")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Console Handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File Handler (writes to file, rotates at 10MB)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5  # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Example usage (for testing this module)
if __name__ == "__main__":
    test_logger = get_logger("test")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
