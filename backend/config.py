"""
Configuration Management
=======================

This module loads all environment variables and provides configuration
settings for the entire application.

Usage:
    from backend.config import Config
    api_key = Config.ANGEL_API_KEY
"""

import os
from dotenv import load_dotenv
from utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)


class Config:
    """Application Configuration Class"""
    
    # =============================================================================
    # Flask Configuration
    # =============================================================================
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    # =============================================================================
    # Angel One API Configuration
    # =============================================================================
    ANGEL_API_KEY = os.getenv('ANGEL_API_KEY')
    ANGEL_CLIENT_ID = os.getenv('ANGEL_CLIENT_ID')
    ANGEL_PASSWORD = os.getenv('ANGEL_PASSWORD')
    ANGEL_TOTP_SECRET = os.getenv('ANGEL_TOTP_SECRET')
    
    # =============================================================================
    # AWS Configuration
    # =============================================================================
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-south-1')
    AWS_S3_BUCKET_RAW = os.getenv('AWS_S3_BUCKET_RAW', 'stockpulse-raw-data')
    AWS_S3_BUCKET_PROCESSED = os.getenv('AWS_S3_BUCKET_PROCESSED', 'stockpulse-processed-data')
    
    # RDS PostgreSQL
    AWS_RDS_HOST = os.getenv('AWS_RDS_HOST')
    AWS_RDS_PORT = int(os.getenv('AWS_RDS_PORT', 5432))
    AWS_RDS_DATABASE = os.getenv('AWS_RDS_DATABASE', 'stockpulse_db')
    AWS_RDS_USER = os.getenv('AWS_RDS_USER', 'postgres')
    AWS_RDS_PASSWORD = os.getenv('AWS_RDS_PASSWORD')
    
    # =============================================================================
    # News API Configuration
    # =============================================================================
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    # =============================================================================
    # Application Settings
    # =============================================================================
    # Maximum historical data to fetch (in days)
    MAX_HISTORICAL_DAYS = 365 * 3  # 3 years
    
    # Default analysis period
    DEFAULT_ANALYSIS_DAYS = 365  # 1 year
    
    # Number of predictions days
    PREDICTION_DAYS = 30
    
    # Technical indicator periods
    MA_SHORT_PERIOD = 20  # 20-day moving average
    MA_MEDIUM_PERIOD = 50  # 50-day moving average
    MA_LONG_PERIOD = 200  # 200-day moving average
    RSI_PERIOD = 14  # 14-day RSI
    
    # =============================================================================
    # File Paths
    # =============================================================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    
    # Create directories if they don't exist
    for directory in [CHARTS_DIR, REPORTS_DIR, LOGS_DIR, TEMP_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # =============================================================================
    # Validation Methods
    # =============================================================================
    @classmethod
    def validate_angel_one_config(cls) -> bool:
        """Check if Angel One API credentials are configured"""
        required = [cls.ANGEL_API_KEY, cls.ANGEL_CLIENT_ID, cls.ANGEL_PASSWORD]
        is_valid = all(required)
        
        if not is_valid:
            logger.warning("Angel One API credentials not configured")
        else:
            logger.info("Angel One API credentials validated")
            
        return is_valid
    
    @classmethod
    def validate_aws_config(cls) -> bool:
        """Check if AWS credentials are configured"""
        required = [
            cls.AWS_ACCESS_KEY_ID,
            cls.AWS_SECRET_ACCESS_KEY,
            cls.AWS_S3_BUCKET_RAW
        ]
        is_valid = all(required)
        
        if not is_valid:
            logger.warning("AWS credentials not configured")
        else:
            logger.info("AWS credentials validated")
            
        return is_valid
    
    @classmethod
    def validate_news_api_config(cls) -> bool:
        """Check if News API is configured"""
        is_valid = bool(cls.NEWS_API_KEY)
        
        if not is_valid:
            logger.warning("News API key not configured")
        else:
            logger.info("News API key validated")
            
        return is_valid
    
    @classmethod
    def get_rds_connection_string(cls) -> str:
        """
        Get PostgreSQL connection string for SQLAlchemy
        
        Returns:
            str: Database connection URL
        """
        if not all([cls.AWS_RDS_HOST, cls.AWS_RDS_PASSWORD]):
            return None
        
        return (f"postgresql://{cls.AWS_RDS_USER}:{cls.AWS_RDS_PASSWORD}"
                f"@{cls.AWS_RDS_HOST}:{cls.AWS_RDS_PORT}/{cls.AWS_RDS_DATABASE}")
    
    @classmethod
    def print_config_status(cls):
        """Print configuration status for debugging"""
        print("\n" + "=" * 50)
        print("StockPulse Configuration Status")
        print("=" * 50)
        print(f"Environment: {cls.FLASK_ENV}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"\nAngel One API: {'✓ Configured' if cls.validate_angel_one_config() else '✗ Not Configured'}")
        print(f"AWS Services: {'✓ Configured' if cls.validate_aws_config() else '✗ Not Configured'}")
        print(f"News API: {'✓ Configured' if cls.validate_news_api_config() else '✗ Not Configured'}")
        print(f"\nCharts Directory: {cls.CHARTS_DIR}")
        print(f"Reports Directory: {cls.REPORTS_DIR}")
        print("=" * 50 + "\n")


# Test configuration on import (only in development)
if __name__ == "__main__":
    Config.print_config_status()
