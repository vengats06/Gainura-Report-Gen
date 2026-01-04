"""
Chart Generator for Stock Analysis
==================================

Creates professional charts for stock reports:
1. Candlestick chart with volume
2. Technical indicators (RSI, MACD, MA)
3. Price prediction forecast
4. Performance comparison

Charts are saved as PNG images for PDF reports.

Usage:
    from visualization.chart_generator import ChartGenerator
    
    gen = ChartGenerator()
    gen.create_candlestick_chart(df, 'TCS')
    gen.create_indicators_chart(df, 'TCS')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf
import seaborn as sns
from datetime import datetime
import os
from typing import Optional, Dict, List
from backend.config import Config
from visualization.chart_styles import ChartStyles
from utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


class ChartGenerator:
    """
    Generate professional stock charts for reports.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save charts (default: Config.CHARTS_DIR)
        """
        self.output_dir = output_dir or Config.CHARTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = ChartStyles()
        logger.info(f"ChartGenerator initialized. Output: {self.output_dir}")
    
    
    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str, 
                                days: int = 90) -> str:
        """
        Create candlestick chart with volume.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            days: Number of recent days to show
            
        Returns:
            str: Path to saved chart image
        """
        logger.info(f"Creating candlestick chart for {symbol}")
        
        # Prepare data
        df = df.copy()
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.tail(days)
        df = df.set_index(date_col)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)
        
        # Candlestick chart
        ax1 = axes[0]
        
        # Plot candlesticks manually for better control
        for idx, row in df.iterrows():
            color = self.styles.COLORS['bullish'] if row['close'] >= row['open'] else self.styles.COLORS['bearish']
            
            # Draw wick (high-low line)
            ax1.plot([idx, idx], [row['low'], row['high']], 
                    color=color, linewidth=1, alpha=0.8)
            
            # Draw body (open-close rectangle)
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            rect = Rectangle((mdates.date2num(idx) - 0.3, bottom), 
                           0.6, height, 
                           facecolor=color, edgecolor=color, alpha=0.9)
            ax1.add_patch(rect)
        
        # Add moving averages if available
        if 'ma_20' in df.columns:
            ax1.plot(df.index, df['ma_20'], label='MA(20)', 
                    color=self.styles.COLORS['ma_20'], linewidth=2, alpha=0.8)
        if 'ma_50' in df.columns:
            ax1.plot(df.index, df['ma_50'], label='MA(50)', 
                    color=self.styles.COLORS['ma_50'], linewidth=2, alpha=0.8)
        
        ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{symbol} - Price Chart (Last {days} Days)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        
        # Volume chart
        ax2 = axes[1]
        colors = [self.styles.COLORS['bullish'] if c >= o else self.styles.COLORS['bearish'] 
                 for o, c in zip(df['open'], df['close'])]
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Format volume axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
        ))
        
        plt.tight_layout()
        
        # Save
        filename = f"{symbol}_candlestick_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Candlestick chart saved: {filepath}")
        return filepath
    
    
    def create_indicators_chart(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Create technical indicators chart (RSI, MACD).
        
        Args:
            df: DataFrame with indicators
            symbol: Stock symbol
            
        Returns:
            str: Path to saved chart
        """
        logger.info(f"Creating indicators chart for {symbol}")
        
        df = df.copy()
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).tail(90)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # RSI Chart
        ax1 = axes[0]
        if 'rsi_14' in df.columns:
            ax1.plot(df[date_col], df['rsi_14'], 
                    color=self.styles.COLORS['primary'], linewidth=2)
            ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax1.fill_between(df[date_col], 30, 70, alpha=0.1, color='gray')
            ax1.set_ylabel('RSI', fontsize=12, fontweight='bold')
            ax1.set_title(f'{symbol} - Technical Indicators', fontsize=16, fontweight='bold', pad=20)
            ax1.legend(loc='upper left', framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
        
        # MACD Chart
        ax2 = axes[1]
        if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
            ax2.plot(df[date_col], df['macd'], 
                    label='MACD', color=self.styles.COLORS['macd'], linewidth=2)
            ax2.plot(df[date_col], df['macd_signal'], 
                    label='Signal', color=self.styles.COLORS['signal'], linewidth=2)
            
            # Histogram
            colors = [self.styles.COLORS['bullish'] if h >= 0 else self.styles.COLORS['bearish'] 
                     for h in df['macd_histogram']]
            ax2.bar(df[date_col], df['macd_histogram'], 
                   color=colors, alpha=0.4, width=0.8, label='Histogram')
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('MACD', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper left', framealpha=0.9)
            ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        
        plt.tight_layout()
        
        filename = f"{symbol}_indicators_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Indicators chart saved: {filepath}")
        return filepath
    
    
    def create_prediction_chart(self, historical_df: pd.DataFrame, 
                               predictions_df: pd.DataFrame, symbol: str) -> str:
        """
        Create price prediction forecast chart.
        
        Args:
            historical_df: Historical price data
            predictions_df: Predicted prices
            symbol: Stock symbol
            
        Returns:
            str: Path to saved chart
        """
        logger.info(f"Creating prediction chart for {symbol}")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot historical prices (last 90 days)
        hist = historical_df.tail(90).copy()
        date_col = 'timestamp' if 'timestamp' in hist.columns else 'date'
        hist[date_col] = pd.to_datetime(hist[date_col])
        
        ax.plot(hist[date_col], hist['close'], 
               label='Historical Price', color=self.styles.COLORS['primary'], 
               linewidth=2.5)
        
        # Plot predictions
        ax.plot(predictions_df['date'], predictions_df['predicted_price'], 
               label='Predicted Price', color=self.styles.COLORS['prediction'], 
               linewidth=2.5, linestyle='--')
        
        # Confidence interval
        if 'confidence_lower' in predictions_df.columns:
            ax.fill_between(predictions_df['date'], 
                          predictions_df['confidence_lower'],
                          predictions_df['confidence_upper'],
                          alpha=0.2, color=self.styles.COLORS['prediction'],
                          label='Confidence Interval (95%)')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax.set_title(f'{symbol} - 30-Day Price Prediction', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        
        plt.tight_layout()
        
        filename = f"{symbol}_prediction_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Prediction chart saved: {filepath}")
        return filepath
    
    
    def create_performance_chart(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Create performance summary chart.
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol
            
        Returns:
            str: Path to saved chart
        """
        logger.info(f"Creating performance chart for {symbol}")
        
        df = df.copy()
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Calculate returns
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['close'].pct_change()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        cumulative_returns = (1 + df['daily_return'].fillna(0)).cumprod() - 1
        ax1.plot(df[date_col], cumulative_returns * 100, 
                color=self.styles.COLORS['primary'], linewidth=2)
        ax1.fill_between(df[date_col], 0, cumulative_returns * 100, 
                        alpha=0.2, color=self.styles.COLORS['primary'])
        ax1.set_title('Cumulative Returns (%)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. Daily returns distribution
        ax2 = axes[0, 1]
        ax2.hist(df['daily_return'].dropna() * 100, bins=50, 
                color=self.styles.COLORS['primary'], alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Daily Returns Distribution', fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Volume trend
        ax3 = axes[1, 0]
        ax3.plot(df[date_col], df['volume'], 
                color=self.styles.COLORS['secondary'], linewidth=1.5)
        ax3.set_title('Trading Volume Trend', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volume')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
        ))
        
        # 4. Price range analysis
        ax4 = axes[1, 1]
        if 'daily_range_pct' in df.columns:
            ax4.plot(df[date_col], df['daily_range_pct'], 
                    color=self.styles.COLORS['warning'], linewidth=1.5)
        else:
            daily_range_pct = ((df['high'] - df['low']) / df['close']) * 100
            ax4.plot(df[date_col], daily_range_pct, 
                    color=self.styles.COLORS['warning'], linewidth=1.5)
        ax4.set_title('Daily Price Range (%)', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Range (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{symbol} - Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = f"{symbol}_performance_{datetime.now().strftime('%Y%m%d')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Performance chart saved: {filepath}")
        return filepath
    
    
    def create_all_charts(self, df: pd.DataFrame, symbol: str, 
                         predictions_df: pd.DataFrame = None) -> Dict[str, str]:
        """
        Create all charts for a stock.
        
        Args:
            df: DataFrame with all data
            symbol: Stock symbol
            predictions_df: Predictions DataFrame (optional)
            
        Returns:
            Dict: Paths to all created charts
        """
        logger.info(f"Creating all charts for {symbol}")
        
        charts = {}
        
        try:
            charts['candlestick'] = self.create_candlestick_chart(df, symbol)
            charts['indicators'] = self.create_indicators_chart(df, symbol)
            charts['performance'] = self.create_performance_chart(df, symbol)
            
            if predictions_df is not None and len(predictions_df) > 0:
                charts['prediction'] = self.create_prediction_chart(df, predictions_df, symbol)
            
            logger.info(f"All charts created successfully for {symbol}")
            
        except Exception as e:
            logger.error(f"Error creating charts: {str(e)}")
            raise
        
        return charts


# Testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Chart Generator")
    print("="*60 + "\n")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    price = 3000
    prices = [price]
    for _ in range(len(dates) - 1):
        change = np.random.randn() * 30
        price = max(price + change, 100)
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': [p * 0.995 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'symbol': 'TEST',
        'ma_20': pd.Series(prices).rolling(20).mean(),
        'ma_50': pd.Series(prices).rolling(50).mean(),
        'rsi_14': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.randn(len(dates)) * 10,
        'macd_signal': np.random.randn(len(dates)) * 8,
        'macd_histogram': np.random.randn(len(dates)) * 5
    })
    
    # Create predictions
    pred_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    predictions = pd.DataFrame({
        'date': pred_dates,
        'predicted_price': [prices[-1] + i * 5 + np.random.randn() * 10 for i in range(30)],
        'confidence_lower': [prices[-1] + i * 5 - 50 for i in range(30)],
        'confidence_upper': [prices[-1] + i * 5 + 50 for i in range(30)]
    })
    
    # Generate charts
    gen = ChartGenerator()
    charts = gen.create_all_charts(df, 'TEST', predictions)
    
    print("\nCharts created:")
    for chart_type, filepath in charts.items():
        print(f"  {chart_type}: {filepath}")
    
    print("\n" + "="*60)
    print("Charts saved successfully!")
    print("="*60)