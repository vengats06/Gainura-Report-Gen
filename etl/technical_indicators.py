"""
Technical Indicators Calculator
===============================

This module calculates technical indicators used in stock analysis.
These indicators help predict stock price movements and identify trends.

Indicators Calculated:
1. Moving Averages (MA) - Trend identification
2. RSI (Relative Strength Index) - Overbought/oversold
3. MACD (Moving Average Convergence Divergence) - Momentum
4. Bollinger Bands - Volatility
5. Daily Returns - Percentage change
6. Volatility - Risk measurement

Why Technical Indicators?
- Traders use these to make buy/sell decisions
- ML models use these as features for prediction
- Shows market sentiment and trends

Usage:
    import pandas as pd
    from etl.technical_indicators import TechnicalIndicators
    
    df = pd.DataFrame(...)  # Stock price data
    ti = TechnicalIndicators(df)
    df_with_indicators = ti.calculate_all()
"""

import pandas as pd
import numpy as np
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for stock price data.
    
    Requires DataFrame with columns: date, open, high, low, close, volume
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with stock price DataFrame.
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            
        Required columns:
        - date or timestamp
        - open
        - high  
        - low
        - close
        - volume
        """
        self.df = df.copy()
        
        # Ensure date column is datetime
        if 'timestamp' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp'])
        elif 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"TechnicalIndicators initialized with {len(self.df)} records")
    
    
    def calculate_moving_averages(self, periods: list = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages (SMA).
        
        Moving Average = Average price over N days
        - Smooths out price fluctuations
        - Identifies trend direction
        
        Args:
            periods: List of periods (default: [20, 50, 200] days)
            
        Returns:
            DataFrame with MA columns added
            
        Interpretation:
        - Price > MA → Uptrend (bullish)
        - Price < MA → Downtrend (bearish)
        - MA_20 > MA_50 > MA_200 → Strong uptrend
        
        Example:
            df = ti.calculate_moving_averages([20, 50])
            # Adds columns: ma_20, ma_50
        """
        for period in periods:
            col_name = f'ma_{period}'
            self.df[col_name] = self.df['close'].rolling(window=period).mean()
            logger.info(f"Calculated {period}-day moving average")
        
        return self.df
    
    
    def calculate_exponential_moving_averages(self, periods: list = [12, 26]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages (EMA).
        
        EMA = Weighted moving average (recent prices have more weight)
        - More responsive to recent price changes than SMA
        - Used in MACD calculation
        
        Args:
            periods: List of periods (default: [12, 26])
            
        Returns:
            DataFrame with EMA columns added
        """
        for period in periods:
            col_name = f'ema_{period}'
            self.df[col_name] = self.df['close'].ewm(span=period, adjust=False).mean()
            logger.info(f"Calculated {period}-day exponential moving average")
        
        return self.df
    
    
    def calculate_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures speed and magnitude of price changes.
        Range: 0 to 100
        
        Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss over period
        
        Args:
            period: Lookback period (default: 14 days)
            
        Returns:
            DataFrame with RSI column added
            
        Interpretation:
        - RSI > 70 → Overbought (sell signal)
        - RSI < 30 → Oversold (buy signal)
        - RSI = 50 → Neutral
        
        Example:
            df = ti.calculate_rsi(14)
            # Adds column: rsi_14
        """
        # Calculate price changes
        delta = self.df['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.df[f'rsi_{period}'] = rsi
        logger.info(f"Calculated RSI-{period}")
        
        return self.df
    
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD shows relationship between two moving averages.
        
        Components:
        1. MACD Line = EMA(12) - EMA(26)
        2. Signal Line = EMA(9) of MACD Line
        3. Histogram = MACD Line - Signal Line
        
        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            DataFrame with MACD columns added
            
        Interpretation:
        - MACD > Signal → Bullish (buy signal)
        - MACD < Signal → Bearish (sell signal)
        - Histogram > 0 → Upward momentum
        - Histogram < 0 → Downward momentum
        
        Example:
            df = ti.calculate_macd()
            # Adds columns: macd, macd_signal, macd_histogram
        """
        # Calculate EMAs
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        self.df['macd'] = macd_line
        self.df['macd_signal'] = signal_line
        self.df['macd_histogram'] = histogram
        
        logger.info(f"Calculated MACD({fast},{slow},{signal})")
        
        return self.df
    
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands show price volatility around a moving average.
        
        Components:
        1. Middle Band = 20-day SMA
        2. Upper Band = Middle + (2 x Standard Deviation)
        3. Lower Band = Middle - (2 x Standard Deviation)
        
        Args:
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            DataFrame with Bollinger Band columns
            
        Interpretation:
        - Price touches upper band → Overbought
        - Price touches lower band → Oversold
        - Bands squeeze → Low volatility (breakout coming)
        - Bands widen → High volatility
        
        Example:
            df = ti.calculate_bollinger_bands()
            # Adds: bollinger_middle, bollinger_upper, bollinger_lower
        """
        # Middle band (SMA)
        middle_band = self.df['close'].rolling(window=period).mean()
        
        # Standard deviation
        std = self.df['close'].rolling(window=period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        self.df['bollinger_middle'] = middle_band
        self.df['bollinger_upper'] = upper_band
        self.df['bollinger_lower'] = lower_band
        self.df['bollinger_width'] = upper_band - lower_band
        
        logger.info(f"Calculated Bollinger Bands({period},{std_dev})")
        
        return self.df
    
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns (percentage change).
        
        Daily Return = (Today's Close - Yesterday's Close) / Yesterday's Close
        
        Returns:
            DataFrame with daily_return column
            
        Interpretation:
        - Positive return → Price increased
        - Negative return → Price decreased
        - Used to calculate volatility and risk
        
        Example:
            If close yesterday = 100, today = 105
            Daily return = (105 - 100) / 100 = 0.05 = 5%
        """
        self.df['daily_return'] = self.df['close'].pct_change()
        logger.info("Calculated daily returns")
        
        return self.df
    
    
    def calculate_volatility(self, period: int = 30) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Volatility = Standard deviation of daily returns over period
        
        Args:
            period: Rolling window period (default: 30 days)
            
        Returns:
            DataFrame with volatility column
            
        Interpretation:
        - High volatility → High risk, high potential reward
        - Low volatility → Low risk, stable returns
        - Volatility > 2% → Risky stock
        - Volatility < 1% → Stable stock
        
        Example:
            df = ti.calculate_volatility(30)
            # Adds column: volatility_30d
        """
        # Calculate daily returns if not already done
        if 'daily_return' not in self.df.columns:
            self.calculate_returns()
        
        # Calculate rolling standard deviation
        self.df[f'volatility_{period}d'] = self.df['daily_return'].rolling(window=period).std()
        
        logger.info(f"Calculated {period}-day volatility")
        
        return self.df
    
    
    def calculate_volume_metrics(self) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Metrics:
        1. Volume Moving Average
        2. Volume Ratio (current / average)
        
        Returns:
            DataFrame with volume metrics
            
        Interpretation:
        - Volume spike → Strong interest (breakout likely)
        - Low volume → Weak interest (trend may reverse)
        """
        # 20-day volume moving average
        self.df['volume_ma_20'] = self.df['volume'].rolling(window=20).mean()
        
        # Volume ratio
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_ma_20']
        
        logger.info("Calculated volume metrics")
        
        return self.df
    
    
    def calculate_all(self) -> pd.DataFrame:
        """
        Calculate all technical indicators at once.
        
        Returns:
            DataFrame with all indicators added
            
        Example:
            ti = TechnicalIndicators(df)
            df_complete = ti.calculate_all()
            print(df_complete.columns)
            # Shows all OHLCV columns + all indicator columns
        """
        logger.info("Calculating all technical indicators...")
        
        # Moving averages
        self.calculate_moving_averages([20, 50, 200])
        self.calculate_exponential_moving_averages([12, 26])
        
        # Momentum indicators
        self.calculate_rsi(14)
        self.calculate_macd()
        
        # Volatility indicators
        self.calculate_bollinger_bands()
        
        # Returns and volatility
        self.calculate_returns()
        self.calculate_volatility(30)
        
        # Volume metrics
        self.calculate_volume_metrics()
        
        logger.info(f"All indicators calculated. Total columns: {len(self.df.columns)}")
        
        return self.df


# Example usage and testing
if __name__ == "__main__":
    """
    Test technical indicators with sample data
    """
    print("\n" + "="*60)
    print("Testing Technical Indicators")
    print("="*60 + "\n")
    
    # Create sample stock data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate stock prices (random walk)
    price = 3000
    prices = [price]
    for _ in range(len(dates) - 1):
        change = np.random.randn() * 30
        price = max(price + change, 100)  # Price can't go negative
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'symbol': 'TEST'
    })
    
    print(f"Sample data created: {len(df)} days of stock prices")
    print(f"Price range: Rs.{df['close'].min():.2f} to Rs.{df['close'].max():.2f}\n")
    
    # Calculate indicators
    ti = TechnicalIndicators(df)
    df_with_indicators = ti.calculate_all()
    
    print("\nIndicators calculated:")
    print(f"Total columns: {len(df_with_indicators.columns)}")
    print(f"\nColumn names:")
    for col in df_with_indicators.columns:
        print(f"  - {col}")
    
    print("\n\nSample data (last 5 days):")
    print(df_with_indicators[['date', 'close', 'ma_20', 'ma_50', 'rsi_14', 'macd']].tail())
    
    print("\n\nLatest values:")
    latest = df_with_indicators.iloc[-1]
    print(f"  Close Price: Rs.{latest['close']:.2f}")
    print(f"  MA(20): Rs.{latest['ma_20']:.2f}")
    print(f"  MA(50): Rs.{latest['ma_50']:.2f}")
    print(f"  RSI(14): {latest['rsi_14']:.2f}")
    print(f"  MACD: {latest['macd']:.2f}")
    print(f"  Volatility (30d): {latest['volatility_30d']:.4f}")
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)