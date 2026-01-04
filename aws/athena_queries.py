"""
AWS Athena Query Handler
========================

Amazon Athena allows you to query data directly in S3 using SQL.
No need to load data into database - query files directly!

Why Athena?
- Query petabytes of data in S3
- Pay only for queries run (serverless)
- Standard SQL syntax
- No infrastructure to manage

How it works:
1. Create external table pointing to S3 location
2. Write SQL query
3. Athena scans S3 files and returns results
4. Results saved back to S3

Usage:
    from aws.athena_queries import AthenaQueryHandler
    
    athena = AthenaQueryHandler()
    athena.create_table()
    results = athena.query_stock_data('TCS', '2024-01-01', '2024-12-31')
"""

import boto3
import time
from typing import List, Dict, Optional
from backend.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class AthenaQueryHandler:
    """
    Execute SQL queries on S3 data using AWS Athena.
    
    Athena is serverless - no database to manage!
    Pay per query (approximately $5 per TB scanned).
    """
    
    def __init__(self):
        """
        Initialize Athena client.
        
        Requirements:
        - S3 bucket for query results
        - AWS credentials with Athena permissions
        """
        self.athena_client = boto3.client(
            'athena',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION
        )
        
        self.s3_output_location = f"s3://{Config.AWS_S3_BUCKET_PROCESSED}/athena-results/"
        self.database = 'stockpulse_db'
        
        logger.info("AthenaQueryHandler initialized")
        logger.info(f"Query results will be saved to: {self.s3_output_location}")
    
    
    def create_database(self) -> bool:
        """
        Create Athena database if it doesn't exist.
        
        Database is just a logical grouping of tables.
        Doesn't store data - data stays in S3.
        
        Returns:
            bool: Success status
        """
        query = f"""
        CREATE DATABASE IF NOT EXISTS {self.database}
        COMMENT 'StockPulse Analytics Database'
        """
        
        try:
            logger.info(f"Creating Athena database: {self.database}")
            execution_id = self._execute_query(query)
            
            if execution_id:
                logger.info(f"Database {self.database} created/verified")
                return True
            else:
                logger.error("Failed to create database")
                return False
                
        except Exception as e:
            logger.error(f"Database creation error: {str(e)}")
            return False
    
    
    def create_technical_indicators_table(self) -> bool:
        """
        Create external table for technical indicators data.
        
        External table = Metadata pointing to S3 location.
        Data stays in S3, Athena just reads it.
        
        Table schema matches our processed data structure.
        
        Returns:
            bool: Success status
        """
        # Drop existing table first
        drop_query = f"DROP TABLE IF EXISTS {self.database}.technical_indicators"
        self._execute_query(drop_query)
        
        # Create table pointing to S3 location
        create_query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {self.database}.technical_indicators (
            symbol STRING,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            ma_20 DOUBLE,
            ma_50 DOUBLE,
            ma_200 DOUBLE,
            ema_12 DOUBLE,
            ema_26 DOUBLE,
            rsi_14 DOUBLE,
            macd DOUBLE,
            macd_signal DOUBLE,
            macd_histogram DOUBLE,
            bollinger_middle DOUBLE,
            bollinger_upper DOUBLE,
            bollinger_lower DOUBLE,
            bollinger_width DOUBLE,
            daily_return DOUBLE,
            volatility_30d DOUBLE,
            volume_ma_20 DOUBLE,
            volume_ratio DOUBLE
        )
        ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
        LOCATION 's3://{Config.AWS_S3_BUCKET_PROCESSED}/processed/technical_indicators/'
        """
        
        try:
            logger.info("Creating technical_indicators table")
            execution_id = self._execute_query(create_query)
            
            if execution_id:
                logger.info("Table created successfully")
                return True
            else:
                logger.error("Failed to create table")
                return False
                
        except Exception as e:
            logger.error(f"Table creation error: {str(e)}")
            return False
    
    
    def _execute_query(self, query: str, wait: bool = True) -> Optional[str]:
        """
        Execute SQL query in Athena.
        
        Args:
            query: SQL query string
            wait: Wait for query to complete (default: True)
            
        Returns:
            str or None: Query execution ID
            
        How it works:
        1. Submit query to Athena
        2. Athena scans S3 files
        3. Processes data
        4. Returns results to S3
        """
        try:
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={'Database': self.database},
                ResultConfiguration={'OutputLocation': self.s3_output_location}
            )
            
            execution_id = response['QueryExecutionId']
            logger.info(f"Query submitted: {execution_id}")
            
            if wait:
                self._wait_for_query(execution_id)
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return None
    
    
    def _wait_for_query(self, execution_id: str, max_wait: int = 60) -> bool:
        """
        Wait for query to complete.
        
        Args:
            execution_id: Query execution ID
            max_wait: Maximum wait time in seconds
            
        Returns:
            bool: True if succeeded, False if failed/timeout
        """
        start_time = time.time()
        
        while True:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=execution_id
            )
            
            status = response['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                logger.info("Query completed successfully")
                return True
            elif status in ['FAILED', 'CANCELLED']:
                reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
                logger.error(f"Query failed: {reason}")
                return False
            
            # Check timeout
            if time.time() - start_time > max_wait:
                logger.error("Query timeout")
                return False
            
            time.sleep(1)
    
    
    def _get_query_results(self, execution_id: str) -> List[Dict]:
        """
        Get results from completed query.
        
        Args:
            execution_id: Query execution ID
            
        Returns:
            List of dictionaries (query results)
        """
        try:
            response = self.athena_client.get_query_results(
                QueryExecutionId=execution_id
            )
            
            # Extract column names
            columns = [col['Name'] for col in response['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            
            # Extract rows (skip header row)
            rows = []
            for row in response['ResultSet']['Rows'][1:]:
                values = [field.get('VarCharValue', None) for field in row['Data']]
                rows.append(dict(zip(columns, values)))
            
            return rows
            
        except Exception as e:
            logger.error(f"Error getting query results: {str(e)}")
            return []
    
    
    def query_stock_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Query stock data for a symbol and date range.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of dictionaries with stock data
            
        Example:
            data = athena.query_stock_data('TCS', '2024-01-01', '2024-12-31')
            for row in data:
                print(f"{row['date']}: Close = {row['close']}")
        """
        query = f"""
        SELECT 
            symbol,
            date,
            open,
            high,
            low,
            close,
            volume,
            ma_20,
            ma_50,
            rsi_14,
            macd
        FROM {self.database}.technical_indicators
        WHERE symbol = '{symbol}'
        AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        ORDER BY date DESC
        """
        
        logger.info(f"Querying data for {symbol} from {start_date} to {end_date}")
        
        execution_id = self._execute_query(query, wait=True)
        
        if execution_id:
            results = self._get_query_results(execution_id)
            logger.info(f"Retrieved {len(results)} records")
            return results
        else:
            return []
    
    
    def get_latest_indicators(self, symbol: str) -> Optional[Dict]:
        """
        Get latest technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with latest indicators or None
            
        Example:
            latest = athena.get_latest_indicators('TCS')
            print(f"Latest RSI: {latest['rsi_14']}")
        """
        query = f"""
        SELECT *
        FROM {self.database}.technical_indicators
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT 1
        """
        
        logger.info(f"Getting latest indicators for {symbol}")
        
        execution_id = self._execute_query(query, wait=True)
        
        if execution_id:
            results = self._get_query_results(execution_id)
            if results:
                return results[0]
        
        return None
    
    
    def calculate_average_metrics(self, symbol: str, days: int = 30) -> Optional[Dict]:
        """
        Calculate average metrics over specified period.
        
        Args:
            symbol: Stock symbol
            days: Number of days to average
            
        Returns:
            Dict with average metrics
            
        Example:
            avg = athena.calculate_average_metrics('TCS', days=30)
            print(f"30-day avg close: {avg['avg_close']}")
        """
        query = f"""
        SELECT 
            symbol,
            AVG(close) as avg_close,
            AVG(volume) as avg_volume,
            AVG(rsi_14) as avg_rsi,
            AVG(volatility_30d) as avg_volatility,
            COUNT(*) as trading_days
        FROM {self.database}.technical_indicators
        WHERE symbol = '{symbol}'
        AND date >= DATE_ADD('day', -{days}, CURRENT_DATE)
        GROUP BY symbol
        """
        
        logger.info(f"Calculating {days}-day averages for {symbol}")
        
        execution_id = self._execute_query(query, wait=True)
        
        if execution_id:
            results = self._get_query_results(execution_id)
            if results:
                return results[0]
        
        return None


# Example usage and testing
if __name__ == "__main__":
    """
    Test Athena queries
    """
    print("\n" + "="*60)
    print("Testing AWS Athena Queries")
    print("="*60 + "\n")
    
    try:
        athena = AthenaQueryHandler()
        
        # Test 1: Create database
        print("Test 1: Creating Athena database...")
        if athena.create_database():
            print("Database created/verified\n")
        else:
            print("Failed to create database\n")
            exit()
        
        # Test 2: Create table
        print("Test 2: Creating technical_indicators table...")
        if athena.create_technical_indicators_table():
            print("Table created successfully\n")
        else:
            print("Failed to create table\n")
            exit()
        
        # Test 3: Query stock data
        print("Test 3: Querying TCS data (last 30 days)...")
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        results = athena.query_stock_data('TCS', start_date, end_date)
        
        if results:
            print(f"Retrieved {len(results)} records")
            print("\nFirst record:")
            for key, value in list(results[0].items())[:5]:
                print(f"  {key}: {value}")
            print()
        else:
            print("No data found (might need to run ETL first)\n")
        
        # Test 4: Get latest indicators
        print("Test 4: Getting latest indicators...")
        latest = athena.get_latest_indicators('TCS')
        
        if latest:
            print(f"Latest data for TCS:")
            print(f"  Date: {latest.get('date')}")
            print(f"  Close: Rs.{latest.get('close')}")
            print(f"  RSI: {latest.get('rsi_14')}")
            print()
        else:
            print("No latest data found\n")
        
        print("="*60)
        print("Tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()