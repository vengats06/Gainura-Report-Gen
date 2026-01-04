"""
Input Validation Utilities
==========================

This module contains functions to validate user inputs and data.
Prevents errors from invalid stock symbols, dates, etc.

Usage:
    from utils.validators import validate_stock_symbol
    if validate_stock_symbol("TCS"):
        # proceed with API call
"""

import re
from datetime import datetime, timedelta
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_stock_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Validate if the stock symbol is in correct format.
    
    Args:
        symbol (str): Stock symbol like "TCS", "RELIANCE"
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
        
    Example:
        is_valid, msg = validate_stock_symbol("TCS")
        if not is_valid:
            print(msg)
    """
    if not symbol:
        return False, "Stock symbol cannot be empty"
    
    # Convert to uppercase and strip whitespace
    symbol = symbol.upper().strip()
    
    # Check length (NSE symbols are typically 1-10 characters)
    if len(symbol) < 1 or len(symbol) > 10:
        return False, "Stock symbol must be 1-10 characters"
    
    # Check for valid characters (only letters and hyphens)
    if not re.match(r'^[A-Z\-]+$', symbol):
        return False, "Stock symbol can only contain letters and hyphens"
    
    logger.info(f"Validated stock symbol: {symbol}")
    return True, symbol


def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
    """
    Validate date range for historical data fetching.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if start is before end
        if start >= end:
            return False, "Start date must be before end date"
        
        # Check if dates are not in the future
        if end > datetime.now():
            return False, "End date cannot be in the future"
        
        # Check if range is not too large (max 5 years)
        if (end - start).days > 5 * 365:
            return False, "Date range cannot exceed 5 years"
        
        logger.info(f"Validated date range: {start_date} to {end_date}")
        return True, "Valid date range"
        
    except ValueError as e:
        return False, f"Invalid date format. Use YYYY-MM-DD: {str(e)}"


def validate_api_credentials() -> Tuple[bool, str]:
    """
    Validate if required API credentials are set in environment.
    
    Returns:
        Tuple[bool, str]: (all_valid, error_message)
    """
    import os
    
    required_vars = [
        'ANGEL_API_KEY',
        'ANGEL_CLIENT_ID',
        'ANGEL_PASSWORD'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    
    logger.info("All API credentials validated")
    return True, "All credentials present"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename safe for filesystem
        
    Example:
        safe_name = sanitize_filename("TCS Report 2024.pdf")
        # Returns: "TCS_Report_2024.pdf"
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    return filename


def validate_aws_config() -> Tuple[bool, str]:
    """
    Validate AWS configuration is present.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    import os
    
    aws_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'AWS_S3_BUCKET_RAW'
    ]
    
    missing = []
    for var in aws_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        return False, f"Missing AWS config: {', '.join(missing)}"
    
    return True, "AWS config validated"


# Common Indian stock symbols (for autocomplete/suggestions)
COMMON_STOCKS = [
    "TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "WIPRO", "ULTRACEMCO", "NESTLEIND", "BAJFINANCE", "HCLTECH"
]


def is_market_open() -> bool:
    """
    Check if Indian stock market is currently open.
    NSE/BSE trading hours: 9:15 AM - 3:30 PM IST, Monday-Friday
    
    Returns:
        bool: True if market is open, False otherwise
    """
    from datetime import datetime
    import pytz
    
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if it's a weekend
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Check trading hours
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    return market_open <= now <= market_close


# Example usage
if __name__ == "__main__":
    print("Testing validators...")
    
    # Test stock symbol
    valid, msg = validate_stock_symbol("TCS")
    print(f"TCS validation: {valid}, {msg}")
    
    # Test invalid symbol
    valid, msg = validate_stock_symbol("TCS@123")
    print(f"Invalid symbol: {valid}, {msg}")
    
    # Test market status
    print(f"Market open: {is_market_open()}")
