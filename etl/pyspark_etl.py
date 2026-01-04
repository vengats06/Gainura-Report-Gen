"""
PySpark ETL Pipeline
===================

This is the MAIN ETL pipeline that processes stock data using PySpark.
Combines all components: data collection, cleaning, indicators, storage.

Pipeline Flow:
1. Extract: Read raw data from S3
2. Transform: Clean + Calculate indicators
3. Load: Save to S3 (Parquet) and RDS PostgreSQL

Why PySpark?
- Handles big data (millions of rows)
- Distributed processing (parallel computation)
- Resume-worthy skill for data engineers

Note: This can also run with pandas for small data (development mode)

Usage:
    from etl.pyspark_etl import StockETLPipeline
    
    pipeline = StockETLPipeline(use_spark=True)
    pipeline.run_etl('TCS', '2024-01-01', '2024-12-31')
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
import os

# Try to import PySpark (optional for development)
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    print("PySpark not available, will use pandas mode")

from data_collection.angel_one_api import AngelOneAPI
from data_collection.scraper import StockScraper
from data_collection.news_fetcher import NewsFetcher
from aws.s3_handler import S3Handler
from aws.rds_connection import RDSConnection
from etl.transformations import DataTransformer
from etl.technical_indicators import TechnicalIndicators
from utils.logger import get_logger

logger = get_logger(__name__)


class StockETLPipeline:
    """
    Complete ETL pipeline for stock data processing.
    
    Can operate in two modes:
    1. Pandas mode (development, small data)
    2. PySpark mode (production, big data)
    """
    
    def __init__(self, use_spark: bool = False):
        """
        Initialize ETL pipeline.
        
        Args:
            use_spark: Use PySpark (True) or Pandas (False)
        """
        self.use_spark = use_spark and PYSPARK_AVAILABLE
        
        if self.use_spark:
            logger.info("Initializing PySpark ETL Pipeline")
            self.spark = self._create_spark_session()
        else:
            logger.info("Initializing Pandas ETL Pipeline (development mode)")
            self.spark = None
        
        # Initialize components
        self.angel_api = AngelOneAPI()
        self.scraper = StockScraper()
        self.news_fetcher = NewsFetcher()
        self.s3 = S3Handler()
        self.db = RDSConnection()
        self.transformer = DataTransformer()
        
        logger.info("ETL Pipeline initialized successfully")
    
    
    def _create_spark_session(self) -> SparkSession:
        """
        Create and configure Spark session.
        
        Configuration:
        - AWS credentials for S3 access
        - Memory settings
        - Parallelism settings
        
        Returns:
            SparkSession
        """
        from backend.config import Config
        
        spark = SparkSession.builder \
            .appName("StockPulse ETL") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
            .config("spark.hadoop.fs.s3a.access.key", Config.AWS_ACCESS_KEY_ID) \
            .config("spark.hadoop.fs.s3a.secret.key", Config.AWS_SECRET_ACCESS_KEY) \
            .config("spark.hadoop.fs.s3a.endpoint", f"s3.{Config.AWS_REGION}.amazonaws.com") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        logger.info("Spark session created")
        return spark
    
    
    def extract_data(self, symbol: str, from_date: str, to_date: str) -> Dict:
        """
        Extract data from all sources.
        
        Sources:
        1. Angel One API - Price data
        2. Screener.in - Fundamentals
        3. News API - News articles
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict with all extracted data
        """
        logger.info(f"Extracting data for {symbol}")
        logger.info(f"Date range: {from_date} to {to_date}")
        
        extracted_data = {}
        
        # 1. Extract price data from Angel One
        logger.info("1/3: Fetching price data from Angel One...")
        if not self.angel_api.session_active:
            self.angel_api.login()
        
        price_df = self.angel_api.get_historical_data(symbol, from_date, to_date)
        if price_df is not None:
            extracted_data['prices'] = price_df
            logger.info(f"   Fetched {len(price_df)} price records")
        else:
            logger.error("   Failed to fetch price data")
            return None
        
        # 2. Extract fundamentals from screener.in
        logger.info("2/3: Scraping fundamentals from screener.in...")
        fundamentals = self.scraper.get_fundamentals(symbol)
        if fundamentals:
            extracted_data['fundamentals'] = fundamentals
            logger.info(f"   Scraped fundamentals: {fundamentals['company_name']}")
        else:
            logger.warning("   Failed to scrape fundamentals (continuing anyway)")
            extracted_data['fundamentals'] = None
        
        # 3. Extract news
        logger.info("3/3: Fetching news articles...")
        if self.news_fetcher.client:
            news_days = min((datetime.strptime(to_date, '%Y-%m-%d') - 
                           datetime.strptime(from_date, '%Y-%m-%d')).days, 30)
            news_articles = self.news_fetcher.get_stock_news(symbol, days=news_days)
            if news_articles:
                extracted_data['news'] = news_articles
                logger.info(f"   Fetched {len(news_articles)} news articles")
            else:
                logger.warning("   No news articles found")
                extracted_data['news'] = []
        else:
            logger.warning("   News API not configured, skipping news")
            extracted_data['news'] = []
        
        logger.info("Data extraction complete")
        return extracted_data
    
    
    def transform_data(self, extracted_data: Dict) -> pd.DataFrame:
        """
        Transform data: clean, calculate indicators, feature engineering.
        
        Transformation steps:
        1. Clean data (remove nulls, duplicates)
        2. Calculate technical indicators
        3. Add date features
        4. Calculate price ranges
        
        Args:
            extracted_data: Dict from extract_data()
            
        Returns:
            Transformed DataFrame ready for loading
        """
        logger.info("Starting data transformation")
        
        price_df = extracted_data['prices']
        
        # 1. Clean data
        logger.info("1/4: Cleaning data...")
        clean_df = self.transformer.clean_stock_data(price_df)
        
        # 2. Calculate technical indicators
        logger.info("2/4: Calculating technical indicators...")
        ti = TechnicalIndicators(clean_df)
        indicators_df = ti.calculate_all()
        
        # 3. Add date features
        logger.info("3/4: Adding date features...")
        date_col = 'timestamp' if 'timestamp' in indicators_df.columns else 'date'
        dated_df = self.transformer.add_date_features(indicators_df, date_col)
        
        # 4. Calculate price ranges
        logger.info("4/4: Calculating price ranges...")
        final_df = self.transformer.calculate_price_ranges(dated_df)
        
        logger.info(f"Transformation complete. Final columns: {len(final_df.columns)}")
        logger.info(f"Final rows: {len(final_df)}")
        
        return final_df
    
    
    def load_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Load data to S3 and RDS PostgreSQL.
        
        Loading destinations:
        1. S3 (Parquet format) - For archiving and analytics
        2. RDS PostgreSQL - For quick queries
        
        Args:
            df: Transformed DataFrame
            symbol: Stock symbol
            
        Returns:
            bool: Success status
        """
        logger.info(f"Loading data for {symbol}")
        
        success = True
        
        # 1. Save to S3 (Parquet format)
        logger.info("1/2: Saving to S3...")
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d')
            
            # Convert to JSON for S3 (Parquet needs PySpark)
            data_dict = df.to_dict(orient='records')
            s3_key = f"processed/technical_indicators/{symbol}_{timestamp}.json"
            
            if self.s3.upload_json(data_dict, s3_key, bucket_type='processed'):
                logger.info(f"   Saved to S3: {s3_key}")
            else:
                logger.error("   Failed to save to S3")
                success = False
        except Exception as e:
            logger.error(f"   S3 save error: {str(e)}")
            success = False
        
        # 2. Save to RDS PostgreSQL
        logger.info("2/2: Saving to RDS PostgreSQL...")
        try:
            # Prepare data for bulk insert
            stock_prices = []
            technical_indicators = []
            
            for _, row in df.iterrows():
                # Stock prices table
                date_col = 'timestamp' if 'timestamp' in row else 'date'
                date_val = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                
                stock_prices.append((
                    symbol,
                    date_val,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                ))
                
                # Technical indicators table (if we have MA and RSI)
                if 'ma_20' in row and 'rsi_14' in row:
                    technical_indicators.append((
                        symbol,
                        date_val,
                        float(row.get('ma_20', 0)) if pd.notna(row.get('ma_20')) else None,
                        float(row.get('ma_50', 0)) if pd.notna(row.get('ma_50')) else None,
                        float(row.get('ma_200', 0)) if pd.notna(row.get('ma_200')) else None,
                        float(row.get('rsi_14', 0)) if pd.notna(row.get('rsi_14')) else None,
                        float(row.get('macd', 0)) if pd.notna(row.get('macd')) else None,
                        float(row.get('macd_signal', 0)) if pd.notna(row.get('macd_signal')) else None,
                        float(row.get('macd_histogram', 0)) if pd.notna(row.get('macd_histogram')) else None,
                        float(row.get('bollinger_upper', 0)) if pd.notna(row.get('bollinger_upper')) else None,
                        float(row.get('bollinger_middle', 0)) if pd.notna(row.get('bollinger_middle')) else None,
                        float(row.get('bollinger_lower', 0)) if pd.notna(row.get('bollinger_lower')) else None,
                        float(row.get('daily_return', 0)) if pd.notna(row.get('daily_return')) else None,
                        float(row.get('volatility_30d', 0)) if pd.notna(row.get('volatility_30d')) else None
                    ))
            
            # Bulk insert to stock_prices
            if stock_prices:
                self.db.bulk_insert_stock_prices(stock_prices)
                logger.info(f"   Inserted {len(stock_prices)} price records")
            
            # Bulk insert to technical_indicators
            if technical_indicators:
                query = """
                INSERT INTO technical_indicators 
                (symbol, date, ma_20, ma_50, ma_200, rsi_14, macd, macd_signal, 
                 macd_histogram, bollinger_upper, bollinger_middle, bollinger_lower,
                 daily_return, volatility_30d)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    ma_20 = EXCLUDED.ma_20,
                    ma_50 = EXCLUDED.ma_50,
                    ma_200 = EXCLUDED.ma_200,
                    rsi_14 = EXCLUDED.rsi_14,
                    macd = EXCLUDED.macd;
                """
                
                from psycopg2.extras import execute_batch
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    execute_batch(cursor, query, technical_indicators, page_size=1000)
                    conn.commit()
                    cursor.close()
                
                logger.info(f"   Inserted {len(technical_indicators)} indicator records")
        
        except Exception as e:
            logger.error(f"   RDS save error: {str(e)}")
            success = False
        
        if success:
            logger.info("Data loading complete")
        else:
            logger.error("Data loading failed")
        
        return success
    
    
    def run_etl(self, symbol: str, from_date: str = None, to_date: str = None) -> bool:
        """
        Run complete ETL pipeline for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            from_date: Start date (default: 1 year ago)
            to_date: End date (default: today)
            
        Returns:
            bool: Success status
            
        Example:
            pipeline = StockETLPipeline()
            pipeline.run_etl('TCS', '2024-01-01', '2024-12-31')
        """
        logger.info("="*60)
        logger.info(f"Starting ETL Pipeline for {symbol}")
        logger.info("="*60)
        
        # Set default dates
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        try:
            # Extract
            logger.info("\nPHASE 1: EXTRACT")
            logger.info("-" * 40)
            extracted_data = self.extract_data(symbol, from_date, to_date)
            
            if not extracted_data:
                logger.error("Extraction failed, aborting pipeline")
                return False
            
            # Transform
            logger.info("\nPHASE 2: TRANSFORM")
            logger.info("-" * 40)
            transformed_df = self.transform_data(extracted_data)
            
            # Load
            logger.info("\nPHASE 3: LOAD")
            logger.info("-" * 40)
            success = self.load_data(transformed_df, symbol)
            
            # Summary
            logger.info("\n" + "="*60)
            if success:
                logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
                logger.info(f"Processed {len(transformed_df)} records for {symbol}")
            else:
                logger.info("ETL PIPELINE COMPLETED WITH ERRORS")
            logger.info("="*60 + "\n")
            
            return success
            
        except Exception as e:
            logger.error(f"\nETL Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            if hasattr(self, 'angel_api'):
                self.angel_api.logout()
            if hasattr(self, 'db'):
                self.db.close_all_connections()


# Example usage
if __name__ == "__main__":
    """
    Test ETL pipeline
    """
    print("\n" + "="*60)
    print("Testing PySpark ETL Pipeline")
    print("="*60 + "\n")
    
    try:
        # Create pipeline (pandas mode for testing)
        pipeline = StockETLPipeline(use_spark=False)
        
        # Run ETL for TCS (last 30 days)
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"Running ETL for TCS")
        print(f"Period: {from_date} to {to_date}\n")
        
        success = pipeline.run_etl('TCS', from_date, to_date)
        
        if success:
            print("\n" + "="*60)
            print("ETL PIPELINE TEST PASSED!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("ETL PIPELINE TEST FAILED!")
            print("="*60)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()