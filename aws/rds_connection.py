"""
AWS RDS PostgreSQL Connection Handler
=====================================

This module manages all database operations with AWS RDS PostgreSQL.
RDS (Relational Database Service) is a managed database in the cloud.

Key Features:
- Create database connection
- Create tables
- Insert data (single row or bulk)
- Query data
- Update data
- Delete data
- Execute custom SQL
- Connection pooling (reuse connections for performance)

Usage:
    from aws.rds_connection import RDSConnection
    db = RDSConnection()
    db.create_tables()
    db.insert_stock_price('TCS', '2024-12-17', 3800, 3850, 3790, 3842, 2500000)
"""

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, execute_batch
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from backend.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class RDSConnection:
    """
    AWS RDS PostgreSQL Connection Manager
    
    This class handles all database operations using connection pooling
    for better performance and resource management.
    """
    
    def __init__(self, min_conn: int = 1, max_conn: int = 10):
        """
        Initialize database connection pool.
        
        Connection pooling means we keep a few connections open and reuse them,
        rather than creating a new connection for every query (which is slow).
        
        Args:
            min_conn (int): Minimum connections to maintain
            max_conn (int): Maximum connections allowed
            
        Example:
            db = RDSConnection()  # Creates pool with 1-10 connections
        """
        try:
            # Create connection pool
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_conn,
                max_conn,
                host=Config.AWS_RDS_HOST,
                port=Config.AWS_RDS_PORT,
                database=Config.AWS_RDS_DATABASE,
                user=Config.AWS_RDS_USER,
                password=Config.AWS_RDS_PASSWORD
            )
            
            if self.connection_pool:
                logger.info(" RDS Connection pool created successfully")
                logger.info(f"Database: {Config.AWS_RDS_DATABASE}")
                logger.info(f"Host: {Config.AWS_RDS_HOST}")
            else:
                raise Exception("Failed to create connection pool")
                
        except Exception as e:
            logger.error(f" Failed to connect to RDS: {str(e)}")
            raise
    
    
    @contextmanager
    def get_connection(self):
        """
        Context manager to get a database connection from pool.
        
        This ensures connections are properly returned to the pool
        even if an error occurs (using try/finally pattern).
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM stock_prices")
        """
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    
    def create_tables(self):
        """
        Create all required database tables if they don't exist.
        
        Tables:
        1. stock_prices - Historical OHLC data
        2. technical_indicators - Calculated indicators (RSI, MACD, MA)
        3. fundamentals - Company financial data
        4. news_sentiment - News headlines with sentiment scores
        5. ml_predictions - AI model predictions
        
        This method is idempotent (safe to run multiple times).
        """
        
        # Table 1: Stock Prices (OHLC data from Angel One)
        create_stock_prices = """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10, 2) NOT NULL,
            high DECIMAL(10, 2) NOT NULL,
            low DECIMAL(10, 2) NOT NULL,
            close DECIMAL(10, 2) NOT NULL,
            volume BIGINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date 
        ON stock_prices(symbol, date DESC);
        """
        
        # Table 2: Technical Indicators
        create_technical_indicators = """
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            ma_20 DECIMAL(10, 2),
            ma_50 DECIMAL(10, 2),
            ma_200 DECIMAL(10, 2),
            rsi_14 DECIMAL(5, 2),
            macd DECIMAL(10, 2),
            macd_signal DECIMAL(10, 2),
            macd_histogram DECIMAL(10, 2),
            bollinger_upper DECIMAL(10, 2),
            bollinger_middle DECIMAL(10, 2),
            bollinger_lower DECIMAL(10, 2),
            daily_return DECIMAL(8, 4),
            volatility_30d DECIMAL(8, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date 
        ON technical_indicators(symbol, date DESC);
        """
        
        # Table 3: Fundamentals (P/E, Market Cap, etc.)
        create_fundamentals = """
        CREATE TABLE IF NOT EXISTS fundamentals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL UNIQUE,
            company_name VARCHAR(200),
            sector VARCHAR(100),
            industry VARCHAR(100),
            market_cap BIGINT,
            pe_ratio DECIMAL(8, 2),
            pb_ratio DECIMAL(8, 2),
            dividend_yield DECIMAL(5, 2),
            roe DECIMAL(8, 2),
            roce DECIMAL(8, 2),
            debt_to_equity DECIMAL(8, 2),
            current_ratio DECIMAL(8, 2),
            revenue BIGINT,
            net_profit BIGINT,
            operating_margin DECIMAL(5, 2),
            net_margin DECIMAL(5, 2),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Table 4: News Sentiment
        create_news_sentiment = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            headline TEXT NOT NULL,
            source VARCHAR(100),
            published_at TIMESTAMP,
            sentiment_score DECIMAL(5, 3),
            sentiment_label VARCHAR(20),
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_date 
        ON news_sentiment(symbol, published_at DESC);
        """
        
        # Table 5: ML Predictions
        create_ml_predictions = """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_price DECIMAL(10, 2),
            confidence_score DECIMAL(5, 2),
            model_type VARCHAR(50),
            features_used JSONB,
            trend_classification VARCHAR(20),
            recommendation VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, prediction_date, model_type)
        );
        """
        
        tables = [
            ("stock_prices", create_stock_prices),
            ("technical_indicators", create_technical_indicators),
            ("fundamentals", create_fundamentals),
            ("news_sentiment", create_news_sentiment),
            ("ml_predictions", create_ml_predictions)
        ]
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for table_name, create_query in tables:
                    logger.info(f"Creating table: {table_name}")
                    cursor.execute(create_query)
                    logger.info(f" Table {table_name} created/verified")
                
                conn.commit()
                cursor.close()
                
            logger.info("All tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    
    def insert_stock_price(self, symbol: str, date: str, open_price: float, 
                          high: float, low: float, close: float, volume: int):
        """
        Insert a single stock price record.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            date: Date in YYYY-MM-DD format
            open_price: Opening price
            high: Highest price
            low: Lowest price
            close: Closing price
            volume: Trading volume
            
        Example:
            db.insert_stock_price('TCS', '2024-12-17', 3800, 3850, 3790, 3842, 2500000)
        """
        query = """
        INSERT INTO stock_prices (symbol, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol, date, open_price, high, low, close, volume))
                conn.commit()
                cursor.close()
                
            logger.info(f"Inserted stock price for {symbol} on {date}")
            
        except Exception as e:
            logger.error(f"Failed to insert stock price: {str(e)}")
            raise
    
    
    def bulk_insert_stock_prices(self, data: List[Tuple]):
        """
        Insert multiple stock price records efficiently.
        
        This is MUCH faster than calling insert_stock_price() multiple times.
        Uses execute_batch() for bulk insertion.
        
        Args:
            data: List of tuples (symbol, date, open, high, low, close, volume)
            
        Example:
            data = [
                ('TCS', '2024-12-17', 3800, 3850, 3790, 3842, 2500000),
                ('TCS', '2024-12-16', 3750, 3800, 3740, 3795, 2400000),
                ...
            ]
            db.bulk_insert_stock_prices(data)
        """
        query = """
        INSERT INTO stock_prices (symbol, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                execute_batch(cursor, query, data, page_size=1000)
                conn.commit()
                cursor.close()
                
            logger.info(f"Bulk inserted {len(data)} stock price records")
            
        except Exception as e:
            logger.error(f" Failed to bulk insert: {str(e)}")
            raise
    
    
    def get_stock_prices(self, symbol: str, start_date: str = None, 
                        end_date: str = None) -> List[Dict]:
        """
        Retrieve stock prices for a symbol within date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            
        Returns:
            List of dictionaries with stock price data
            
        Example:
            prices = db.get_stock_prices('TCS', '2024-01-01', '2024-12-17')
            for price in prices:
                print(f"{price['date']}: Close = {price['close']}")
        """
        query = "SELECT * FROM stock_prices WHERE symbol = %s"
        params = [symbol]
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                
            logger.info(f"Retrieved {len(results)} stock prices for {symbol}")
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve stock prices: {str(e)}")
            return []
    
    
    def insert_fundamental(self, symbol: str, data: Dict):
        """
        Insert or update fundamental data for a stock.
        
        Args:
            symbol: Stock symbol
            data: Dictionary with fundamental metrics
            
        Example:
            fundamentals = {
                'company_name': 'Tata Consultancy Services',
                'sector': 'IT Services',
                'market_cap': 1403245000000,  # 14.03 lakh crore
                'pe_ratio': 28.45,
                'roe': 44.8
            }
            db.insert_fundamental('TCS', fundamentals)
        """
        # Build dynamic query based on data keys
        columns = ['symbol'] + list(data.keys())
        values = [symbol] + list(data.values())
        
        query = sql.SQL("""
            INSERT INTO fundamentals ({})
            VALUES ({})
            ON CONFLICT (symbol) DO UPDATE SET
                {},
                updated_at = CURRENT_TIMESTAMP
        """).format(
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            sql.SQL(', ').join(sql.Placeholder() * len(columns)),
            sql.SQL(', ').join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
                for col in data.keys()
            )
        )
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                cursor.close()
                
            logger.info(f"Inserted fundamentals for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to insert fundamentals: {str(e)}")
            raise
    
    
    def execute_query(self, query: str, params: tuple = None, 
                     fetch: bool = True) -> Optional[List[Dict]]:
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters (tuple)
            fetch: Whether to fetch results (False for INSERT/UPDATE/DELETE)
            
        Returns:
            List of dictionaries if fetch=True, None otherwise
            
        Example:
            # Custom aggregation query
            result = db.execute_query("""'''SELECT symbol, AVG(close) as avg_price
                FROM stock_prices
                WHERE date >= '2024-01-01'
                GROUP BY symbol'''""")
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch:
                    results = cursor.fetchall()
                    cursor.close()
                    return [dict(row) for row in results]
                else:
                    conn.commit()
                    cursor.close()
                    return None
                    
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return None
    
    
    def close_all_connections(self):
        """
        Close all connections in the pool.
        Call this when shutting down the application.
        """
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All database connections closed")


# Example usage and testing
if __name__ == "__main__":
    """
    Test RDS connection and operations
    """
    print("\n" + "="*60)
    print("Testing RDS Connection")
    print("="*60 + "\n")
    
    try:
        # Initialize database connection
        db = RDSConnection()
        
        # Test 1: Create tables
        print("Test 1: Creating tables...")
        db.create_tables()
        print(" Tables created\n")
        
        # Test 2: Insert single stock price
        print("Test 2: Inserting stock price...")
        db.insert_stock_price(
            symbol='TCS',
            date='2024-12-17',
            open_price=3800.00,
            high=3850.00,
            low=3790.00,
            close=3842.50,
            volume=2500000
        )
        print("Stock price inserted\n")
        
        # Test 3: Bulk insert
        print("Test 3: Bulk inserting stock prices...")
        bulk_data = [
            ('TCS', '2024-12-16', 3750, 3800, 3740, 3795, 2400000),
            ('TCS', '2024-12-15', 3700, 3760, 3690, 3755, 2300000),
            ('RELIANCE', '2024-12-17', 2850, 2880, 2840, 2875, 5000000)
        ]
        db.bulk_insert_stock_prices(bulk_data)
        print("Bulk insert completed\n")
        
        # Test 4: Retrieve stock prices
        print("Test 4: Retrieving stock prices...")
        prices = db.get_stock_prices('TCS', '2024-12-15', '2024-12-17')
        print(f"Retrieved {len(prices)} records:")
        for price in prices:
            print(f"  {price['date']}: Close = ₹{price['close']}")
        print()
        
        # Test 5: Insert fundamentals
        print("Test 5: Inserting fundamentals...")
        fundamentals = {
            'company_name': 'Tata Consultancy Services',
            'sector': 'IT Services',
            'market_cap': 1403245000000,
            'pe_ratio': 28.45,
            'roe': 44.8
        }
        db.insert_fundamental('TCS', fundamentals)
        print("Fundamentals inserted\n")
        
        # Test 6: Custom query
        print("Test 6: Custom query...")
        result = db.execute_query("""
            SELECT symbol, COUNT(*) as record_count, 
                   AVG(close) as avg_price
            FROM stock_prices
            GROUP BY symbol
        """)
        print("Query results:")
        for row in result:
            print(f"  {row['symbol']}: {row['record_count']} records, Avg price: ₹{row['avg_price']:.2f}")
        print()
        
        print("="*60)
        print("All tests completed successfully!")
        print("="*60)
        
        # Close connections
        db.close_all_connections()
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()