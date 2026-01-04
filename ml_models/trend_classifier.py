"""
Stock Trend Classifier
======================

Classifies stock trends as Bullish, Bearish, or Neutral.
Provides BUY/HOLD/SELL recommendations based on technical analysis.

Classification Rules:
- Bullish: Uptrend, buy signal
- Bearish: Downtrend, sell signal
- Neutral: Sideways, hold signal

Uses technical indicators to determine trend.

Usage:
    from ml_models.trend_classifier import TrendClassifier
    
    classifier = TrendClassifier()
    trend = classifier.classify(df)
    print(f"Trend: {trend['trend_label']}")
    print(f"Recommendation: {trend['recommendation']}")
"""

import pandas as pd
import numpy as np
from typing import Dict
from utils.logger import get_logger

logger = get_logger(__name__)


class TrendClassifier:
    """
    Classify stock trends using technical indicators.
    
    Uses:
    - Moving average crossovers
    - RSI levels
    - MACD signals
    - Price momentum
    """
    
    def __init__(self):
        """Initialize trend classifier."""
        logger.info("TrendClassifier initialized")
    
    
    def check_ma_crossover(self, df: pd.DataFrame) -> int:
        """
        Check moving average crossover signals.
        
        Signals:
        - MA(20) > MA(50) > MA(200) = Strong bullish (+3)
        - MA(20) > MA(50) = Bullish (+2)
        - MA(20) < MA(50) < MA(200) = Strong bearish (-3)
        - MA(20) < MA(50) = Bearish (-2)
        
        Args:
            df: DataFrame with MA columns
            
        Returns:
            int: Signal strength (-3 to +3)
        """
        if not all(col in df.columns for col in ['ma_20', 'ma_50', 'ma_200']):
            return 0
        
        latest = df.iloc[-1]
        
        ma_20 = latest['ma_20']
        ma_50 = latest['ma_50']
        ma_200 = latest['ma_200']
        
        # Strong bullish
        if ma_20 > ma_50 > ma_200:
            return 3
        # Bullish
        elif ma_20 > ma_50:
            return 2
        # Strong bearish
        elif ma_20 < ma_50 < ma_200:
            return -3
        # Bearish
        elif ma_20 < ma_50:
            return -2
        else:
            return 0
    
    
    def check_rsi(self, df: pd.DataFrame) -> int:
        """
        Check RSI levels.
        
        Signals:
        - RSI < 30 = Oversold, bullish (+2)
        - RSI 30-50 = Slight bullish (+1)
        - RSI 50-70 = Slight bearish (-1)
        - RSI > 70 = Overbought, bearish (-2)
        
        Args:
            df: DataFrame with RSI column
            
        Returns:
            int: Signal strength (-2 to +2)
        """
        if 'rsi_14' not in df.columns:
            return 0
        
        rsi = df.iloc[-1]['rsi_14']
        
        if pd.isna(rsi):
            return 0
        
        if rsi < 30:
            return 2  # Oversold, buy signal
        elif rsi < 50:
            return 1
        elif rsi < 70:
            return -1
        else:
            return -2  # Overbought, sell signal
    
    
    def check_macd(self, df: pd.DataFrame) -> int:
        """
        Check MACD signals.
        
        Signals:
        - MACD > Signal = Bullish (+2)
        - MACD < Signal = Bearish (-2)
        
        Args:
            df: DataFrame with MACD columns
            
        Returns:
            int: Signal strength (-2 to +2)
        """
        if not all(col in df.columns for col in ['macd', 'macd_signal']):
            return 0
        
        latest = df.iloc[-1]
        
        macd = latest['macd']
        signal = latest['macd_signal']
        
        if pd.isna(macd) or pd.isna(signal):
            return 0
        
        if macd > signal:
            return 2  # Bullish
        else:
            return -2  # Bearish
    
    
    def check_momentum(self, df: pd.DataFrame, period: int = 10) -> int:
        """
        Check price momentum.
        
        Momentum = % change over period
        
        Args:
            df: DataFrame with close prices
            period: Lookback period
            
        Returns:
            int: Signal strength (-2 to +2)
        """
        if len(df) < period:
            return 0
        
        recent = df.tail(period)
        momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        if momentum > 0.05:  # >5% gain
            return 2
        elif momentum > 0.02:  # 2-5% gain
            return 1
        elif momentum < -0.05:  # >5% loss
            return -2
        elif momentum < -0.02:  # 2-5% loss
            return -1
        else:
            return 0
    
    
    def classify(self, df: pd.DataFrame) -> Dict:
        """
        Classify overall trend.
        
        Combines all signals:
        - MA crossover (weight: 3)
        - RSI (weight: 2)
        - MACD (weight: 2)
        - Momentum (weight: 2)
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dict with trend classification and recommendation
            
        Example:
            result = classifier.classify(df)
            print(f"Trend: {result['trend_label']}")
            print(f"Action: {result['recommendation']}")
            print(f"Confidence: {result['confidence']:.0%}")
        """
        logger.info("Classifying stock trend...")
        
        # Collect signals
        ma_signal = self.check_ma_crossover(df)
        rsi_signal = self.check_rsi(df)
        macd_signal = self.check_macd(df)
        momentum_signal = self.check_momentum(df)
        
        # Weighted score
        total_score = (
            ma_signal * 3 +
            rsi_signal * 2 +
            macd_signal * 2 +
            momentum_signal * 2
        )
        
        max_possible = 3*3 + 2*2 + 2*2 + 2*2  # 21
        
        # Normalize to -1 to +1
        normalized_score = total_score / max_possible
        
        # Classify
        if normalized_score > 0.3:
            trend_label = "Bullish"
            recommendation = "BUY"
            confidence = min((normalized_score - 0.3) / 0.7, 1.0)
        elif normalized_score < -0.3:
            trend_label = "Bearish"
            recommendation = "SELL"
            confidence = min((-normalized_score - 0.3) / 0.7, 1.0)
        else:
            trend_label = "Neutral"
            recommendation = "HOLD"
            confidence = 1.0 - abs(normalized_score) / 0.3
        
        # Get latest price
        latest_price = df.iloc[-1]['close']
        
        # Signal breakdown
        signals = {
            'moving_average': ma_signal,
            'rsi': rsi_signal,
            'macd': macd_signal,
            'momentum': momentum_signal
        }
        
        logger.info(f"Trend Classification: {trend_label}")
        logger.info(f"Recommendation: {recommendation}")
        logger.info(f"Confidence: {confidence:.0%}")
        
        return {
            'trend_label': trend_label,
            'recommendation': recommendation,
            'confidence': confidence,
            'score': normalized_score,
            'signals': signals,
            'current_price': latest_price
        }


# Testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Trend Classifier")
    print("="*60 + "\n")
    
    # Create sample data with uptrend
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    price = 3000
    prices = []
    for i in range(len(dates)):
        change = np.random.randn() * 20 + 2  # Upward trend
        price = max(price + change, 100)
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # Add indicators
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    print(f"Sample data: {len(df)} days")
    print(f"Price: Rs.{df['close'].iloc[0]:.2f} -> Rs.{df['close'].iloc[-1]:.2f}\n")
    
    classifier = TrendClassifier()
    
    print("Test 1: Classify uptrend data...")
    result = classifier.classify(df)
    
    print(f"\nResults:")
    print(f"  Trend: {result['trend_label']}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Current Price: Rs.{result['current_price']:.2f}")
    
    print(f"\n  Signal Breakdown:")
    for signal, value in result['signals'].items():
        print(f"    {signal}: {value:+d}")
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)