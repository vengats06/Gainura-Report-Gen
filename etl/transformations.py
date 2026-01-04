"""
Data Transformations for ETL Pipeline
=====================================

This module contains data cleaning and transformation functions.
Prepares raw data for analysis and machine learning.

Transformations:
1. Data cleaning (remove nulls, duplicates)
2. Data type conversions
3. Feature engineering
4. Data validation
5. Outlier detection and handling

Why Transformations?
- Raw data is messy (nulls, duplicates, wrong formats)
- ML models need clean, consistent data
- Proper transformations improve model accuracy

Usage:
    from etl.transformations import DataTransformer
    transformer = DataTransformer()
    clean_df = transformer.clean_data(raw_df)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class DataTransformer:
    """
    Data cleaning and transformation utilities.
    
    Handles common data quality issues:
    - Missing values
    - Duplicates
    - Data type mismatches
    - Outliers
    """
    
    def __init__(self):
        """Initialize data transformer."""
        logger.info("DataTransformer initialized")
    
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock price data (OHLCV format).
        
        Cleaning steps:
        1. Remove null values
        2. Remove duplicates
        3. Fix data types
        4. Validate price ranges
        5. Sort by date
        
        Args:
            df: Raw stock price DataFrame
            
        Returns:
            Cleaned DataFrame
            
        Example:
            raw_df = pd.DataFrame(...)  # Has nulls, duplicates
            clean_df = transformer.clean_stock_data(raw_df)
            # Returns: Clean data, ready for analysis
        """
        logger.info(f"Cleaning stock data. Initial rows: {len(df)}")
        
        original_count = len(df)
        
        # Make copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Remove rows with null values in critical columns
        critical_columns = ['date', 'close', 'volume']
        if 'timestamp' in df_clean.columns:
            critical_columns = ['timestamp', 'close', 'volume']
        
        null_count = df_clean[critical_columns].isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values, removing...")
            df_clean = df_clean.dropna(subset=critical_columns)
        
        # 2. Remove duplicate dates
        date_col = 'timestamp' if 'timestamp' in df_clean.columns else 'date'
        duplicates = df_clean.duplicated(subset=[date_col, 'symbol'], keep='first')
        if duplicates.sum() > 0:
            logger.warning(f"Found {duplicates.sum()} duplicates, removing...")
            df_clean = df_clean[~duplicates]
        
        # 3. Ensure proper data types
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 4. Validate price ranges (prices should be positive)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df_clean.columns:
                invalid = (df_clean[col] <= 0)
                if invalid.sum() > 0:
                    logger.warning(f"Found {invalid.sum()} invalid prices in {col}, removing...")
                    df_clean = df_clean[~invalid]
        
        # 5. Validate OHLC relationships (High >= Low, etc.)
        if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df_clean['high'] < df_clean['low']) |
                (df_clean['high'] < df_clean['close']) |
                (df_clean['low'] > df_clean['close'])
            )
            if invalid_ohlc.sum() > 0:
                logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC rows, removing...")
                df_clean = df_clean[~invalid_ohlc]
        
        # 6. Validate volume (should be positive integer)
        if 'volume' in df_clean.columns:
            invalid_volume = (df_clean['volume'] < 0)
            if invalid_volume.sum() > 0:
                logger.warning(f"Found {invalid_volume.sum()} invalid volumes, removing...")
                df_clean = df_clean[~invalid_volume]
            
            # Convert to integer
            df_clean['volume'] = df_clean['volume'].astype('int64')
        
        # 7. Sort by date
        df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
        
        removed = original_count - len(df_clean)
        logger.info(f"Cleaning complete. Removed {removed} rows. Final rows: {len(df_clean)}")
        
        return df_clean
    
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Methods:
        - 'forward_fill': Use previous value
        - 'interpolate': Linear interpolation
        - 'drop': Remove rows with nulls
        
        Args:
            df: DataFrame with missing values
            method: How to handle nulls
            
        Returns:
            DataFrame with nulls handled
            
        Example:
            Date       Close
            2024-01-01  100
            2024-01-02  NaN  <- Missing!
            2024-01-03  110
            
            forward_fill: 100, 100, 110
            interpolate: 100, 105, 110
        """
        null_count = df.isnull().sum().sum()
        if null_count == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Handling {null_count} missing values using {method}")
        
        df_filled = df.copy()
        
        if method == 'forward_fill':
            df_filled = df_filled.fillna(method='ffill')
        elif method == 'interpolate':
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear')
        elif method == 'drop':
            df_filled = df_filled.dropna()
        else:
            logger.warning(f"Unknown method: {method}, using forward_fill")
            df_filled = df_filled.fillna(method='ffill')
        
        return df_filled
    
    
    def detect_outliers(self, df: pd.DataFrame, column: str, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a column.
        
        Methods:
        - 'iqr': Interquartile Range (standard method)
        - 'zscore': Z-score (for normal distributions)
        
        Args:
            df: DataFrame
            column: Column to check for outliers
            method: Detection method
            threshold: Sensitivity (1.5 = standard, 3.0 = only extreme)
            
        Returns:
            Boolean Series (True = outlier)
            
        Example:
            Prices: [100, 105, 110, 500]  <- 500 is outlier!
            outliers = detect_outliers(df, 'close')
            # Returns: [False, False, False, True]
        """
        logger.info(f"Detecting outliers in {column} using {method}")
        
        if method == 'iqr':
            # IQR method (robust to outliers)
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean = df[column].mean()
            std = df[column].std()
            z_scores = np.abs((df[column] - mean) / std)
            
            outliers = z_scores > threshold
        else:
            logger.warning(f"Unknown method: {method}, using IQR")
            return self.detect_outliers(df, column, method='iqr', threshold=threshold)
        
        outlier_count = outliers.sum()
        logger.info(f"Found {outlier_count} outliers in {column}")
        
        return outliers
    
    
    def remove_outliers(self, df: pd.DataFrame, columns: list, 
                       method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            df: DataFrame
            columns: Columns to check
            method: Detection method
            
        Returns:
            DataFrame with outliers removed
            
        Warning: Use carefully! Outliers might be real data.
        """
        df_clean = df.copy()
        original_count = len(df_clean)
        
        for column in columns:
            if column in df_clean.columns:
                outliers = self.detect_outliers(df_clean, column, method=method)
                df_clean = df_clean[~outliers]
        
        removed = original_count - len(df_clean)
        logger.info(f"Removed {removed} outlier rows")
        
        return df_clean
    
    
    def normalize_column(self, df: pd.DataFrame, column: str, 
                        method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize a column (scale to 0-1 range or standardize).
        
        Methods:
        - 'minmax': Scale to [0, 1]
        - 'standard': Mean=0, StdDev=1 (z-score normalization)
        
        Args:
            df: DataFrame
            column: Column to normalize
            method: Normalization method
            
        Returns:
            DataFrame with normalized column
            
        Why normalize?
        - ML models work better with normalized data
        - Different features have different scales
        - Example: Volume (millions) vs Price (thousands)
        
        Example:
            Original: [100, 200, 300]
            MinMax: [0.0, 0.5, 1.0]
            Standard: [-1.22, 0.0, 1.22]
        """
        df_norm = df.copy()
        
        if method == 'minmax':
            min_val = df_norm[column].min()
            max_val = df_norm[column].max()
            df_norm[f'{column}_normalized'] = (df_norm[column] - min_val) / (max_val - min_val)
            
        elif method == 'standard':
            mean = df_norm[column].mean()
            std = df_norm[column].std()
            df_norm[f'{column}_normalized'] = (df_norm[column] - mean) / std
        
        logger.info(f"Normalized {column} using {method}")
        
        return df_norm
    
    
    def add_date_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Extract date features (day, month, year, day of week, etc.).
        
        Useful for ML models to detect seasonal patterns.
        
        Args:
            df: DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with date features added
            
        Features added:
        - year, month, day
        - day_of_week (0=Monday, 6=Sunday)
        - is_month_start, is_month_end
        - quarter
        
        Example:
            Date: 2024-01-15
            → year=2024, month=1, day=15, day_of_week=0 (Monday), quarter=1
        """
        df_dated = df.copy()
        
        # Ensure datetime
        df_dated[date_column] = pd.to_datetime(df_dated[date_column])
        
        # Extract features
        df_dated['year'] = df_dated[date_column].dt.year
        df_dated['month'] = df_dated[date_column].dt.month
        df_dated['day'] = df_dated[date_column].dt.day
        df_dated['day_of_week'] = df_dated[date_column].dt.dayofweek
        df_dated['quarter'] = df_dated[date_column].dt.quarter
        df_dated['is_month_start'] = df_dated[date_column].dt.is_month_start
        df_dated['is_month_end'] = df_dated[date_column].dt.is_month_end
        
        logger.info("Added date features")
        
        return df_dated
    
    
    def calculate_price_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price range metrics.
        
        Metrics:
        - Daily range (high - low)
        - Range percentage (range / close)
        - Body (close - open) for candlestick
        - Body percentage
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with range metrics
        """
        df_ranges = df.copy()
        
        # Daily range
        df_ranges['daily_range'] = df_ranges['high'] - df_ranges['low']
        df_ranges['daily_range_pct'] = (df_ranges['daily_range'] / df_ranges['close']) * 100
        
        # Candlestick body
        df_ranges['body'] = df_ranges['close'] - df_ranges['open']
        df_ranges['body_pct'] = (df_ranges['body'] / df_ranges['open']) * 100
        
        # Upper and lower shadows (wicks)
        df_ranges['upper_shadow'] = df_ranges['high'] - df_ranges[['open', 'close']].max(axis=1)
        df_ranges['lower_shadow'] = df_ranges[['open', 'close']].min(axis=1) - df_ranges['low']
        
        logger.info("Calculated price range metrics")
        
        return df_ranges


# Example usage and testing
if __name__ == "__main__":
    """
    Test data transformations
    """
    print("\n" + "="*60)
    print("Testing Data Transformations")
    print("="*60 + "\n")
    
    # Create sample data with issues
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df_messy = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'open': np.random.normal(3000, 100, 100),
        'high': np.random.normal(3100, 100, 100),
        'low': np.random.normal(2900, 100, 100),
        'close': np.random.normal(3000, 100, 100),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Add some issues
    df_messy.loc[10, 'close'] = None  # Missing value
    df_messy.loc[20, 'close'] = -100  # Invalid price
    df_messy.loc[30, 'high'] = df_messy.loc[30, 'low'] - 10  # Invalid OHLC
    df_messy.loc[5] = df_messy.loc[4]  # Duplicate
    df_messy.loc[50, 'close'] = 10000  # Outlier
    
    print(f"Created messy data: {len(df_messy)} rows")
    print(f"Issues added: nulls, negatives, invalid OHLC, duplicates, outliers\n")
    
    # Test cleaning
    transformer = DataTransformer()
    
    print("Test 1: Clean stock data...")
    df_clean = transformer.clean_stock_data(df_messy)
    print(f"After cleaning: {len(df_clean)} rows\n")
    
    print("Test 2: Detect outliers...")
    outliers = transformer.detect_outliers(df_clean, 'close', method='iqr')
    print(f"Found {outliers.sum()} outliers\n")
    
    print("Test 3: Add date features...")
    df_with_dates = transformer.add_date_features(df_clean)
    print(f"Added columns: {[c for c in df_with_dates.columns if c not in df_clean.columns]}\n")
    
    print("Test 4: Calculate price ranges...")
    df_with_ranges = transformer.calculate_price_ranges(df_clean)
    print(f"Added range metrics: daily_range, body, shadows\n")
    
    print("="*60)
    print("Tests completed!")
    print("="*60)