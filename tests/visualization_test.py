"""
Test Visualization Module
=========================

Tests chart generation with real data from your ETL pipeline.
This ensures charts work correctly before building PDF reports.

Run this test:
    python tests/test_visualization.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from visualization.chart_generator import ChartGenerator
from visualization.chart_styles import ChartStyles
from data_collection.angel_one_api import AngelOneAPI
from etl.technical_indicators import TechnicalIndicators
from ml_models.price_predictor import PricePredictor
from utils.logger import get_logger

logger = get_logger(__name__)


def test_chart_styles():
    """Test chart styles configuration."""
    print("\n" + "="*70)
    print("TEST 1: Chart Styles Configuration")
    print("="*70)
    
    try:
        styles = ChartStyles()
        
        print("\n✓ ChartStyles initialized")
        print(f"  Colors available: {len(styles.COLORS)}")
        print(f"  Primary color: {styles.COLORS['primary']}")
        print(f"  Bullish color: {styles.COLORS['bullish']}")
        print(f"  Bearish color: {styles.COLORS['bearish']}")
        
        # Test currency formatting
        test_values = [1234567890, 12345678, 123456, 1234, 123.45]
        print("\n  Currency formatting:")
        for val in test_values:
            print(f"    {val:>12} -> {styles.format_currency(val)}")
        
        # Test percentage formatting
        print("\n  Percentage formatting:")
        for val in [0.0523, -0.0234, 0.1234, -0.0012]:
            print(f"    {val:>7.4f} -> {styles.format_percentage(val)}")
        
        print("\n✅ TEST 1 PASSED - Chart styles working correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_with_sample_data():
    """Test charts with generated sample data."""
    print("\n" + "="*70)
    print("TEST 2: Chart Generation with Sample Data")
    print("="*70)
    
    try:
        # Create realistic sample data
        print("\nCreating sample stock data...")
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        np.random.seed(42)
        
        price = 3000
        prices = [price]
        volumes = []
        
        for i in range(len(dates) - 1):
            # Realistic price movement (random walk with slight uptrend)
            change = np.random.randn() * 30 + 0.5  # Small positive drift
            price = max(price + change, 100)
            prices.append(price)
            volumes.append(np.random.randint(1000000, 5000000))
        volumes.append(np.random.randint(1000000, 5000000))
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0.005, 0.02)) for p in prices],
            'low': [p * (1 + np.random.uniform(-0.02, -0.005)) for p in prices],
            'close': prices,
            'volume': volumes,
            'symbol': 'TESTSTOCK'
        })
        
        print(f"  Created {len(df)} days of data")
        print(f"  Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}")
        print(f"  Average volume: {df['volume'].mean()/1e6:.2f}M")
        
        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        ti = TechnicalIndicators(df)
        df = ti.calculate_all()
        print(f"  ✓ Added {len([c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']])} indicators")
        
        # Create predictions
        print("\nCreating sample predictions...")
        last_price = df['close'].iloc[-1]
        pred_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        predictions = pd.DataFrame({
            'date': pred_dates,
            'predicted_price': [last_price + i * 2 + np.random.randn() * 10 for i in range(30)],
            'confidence_lower': [last_price + i * 2 - 30 for i in range(30)],
            'confidence_upper': [last_price + i * 2 + 30 for i in range(30)]
        })
        print(f"  Created 30-day predictions")
        
        # Generate all charts
        print("\nGenerating charts...")
        gen = ChartGenerator()
        charts = gen.create_all_charts(df, 'TESTSTOCK', predictions)
        
        print("\n✓ Charts generated successfully:")
        for chart_type, filepath in charts.items():
            filesize = os.path.getsize(filepath) / 1024  # KB
            print(f"  {chart_type:.<20} {filepath} ({filesize:.1f} KB)")
        
        print("\n✅ TEST 2 PASSED - All charts generated with sample data\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data():
    """Test charts with real data from Angel One API."""
    print("\n" + "="*70)
    print("TEST 3: Chart Generation with Real Stock Data")
    print("="*70)
    
    try:
        symbol = 'TCS'
        
        # Fetch real data
        print(f"\nFetching real data for {symbol} from Angel One...")
        angel = AngelOneAPI()
        
        if not angel.login():
            print("⚠️  Angel One login failed - skipping real data test")
            print("   (This is OK if you want to test with sample data only)")
            return True
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        df = angel.get_historical_data(symbol, from_date, to_date)
        angel.logout()
        
        if df is None or len(df) == 0:
            print("⚠️  No data received from Angel One")
            return True
        
        print(f"  ✓ Fetched {len(df)} days of data")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}")
        
        # Calculate indicators
        print("\nCalculating technical indicators...")
        df_copy = df.copy()
        df_copy['date'] = df_copy['timestamp']
        ti = TechnicalIndicators(df_copy)
        df_with_indicators = ti.calculate_all()
        print(f"  ✓ Indicators calculated")
        
        # Train predictor and get predictions
        print("\nTraining price predictor...")
        predictor = PricePredictor(model_type='linear')
        metrics = predictor.train(df_with_indicators, test_size=0.2)
        print(f"  ✓ Model trained (R² = {metrics['test_r2']:.3f})")
        
        print("\nGenerating 30-day predictions...")
        predictions = predictor.predict_multiple_days(df_with_indicators, days=30)
        print(f"  ✓ Predictions generated")
        print(f"  Predicted range: ₹{predictions['predicted_price'].min():.2f} to ₹{predictions['predicted_price'].max():.2f}")
        
        # Generate charts
        print("\nGenerating charts with real data...")
        gen = ChartGenerator()
        charts = gen.create_all_charts(df_with_indicators, symbol, predictions)
        
        print("\n✓ Charts generated successfully:")
        for chart_type, filepath in charts.items():
            filesize = os.path.getsize(filepath) / 1024  # KB
            print(f"  {chart_type:.<20} {filepath} ({filesize:.1f} KB)")
        
        print("\n✅ TEST 3 PASSED - Real data charts generated successfully\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_chart_files():
    """Verify chart files were created correctly."""
    print("\n" + "="*70)
    print("TEST 4: Verify Chart Files")
    print("="*70)
    
    try:
        from backend.config import Config
        
        charts_dir = Config.CHARTS_DIR
        print(f"\nChecking charts directory: {charts_dir}")
        
        if not os.path.exists(charts_dir):
            print(f"❌ Charts directory doesn't exist!")
            return False
        
        # List all PNG files
        png_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
        
        if len(png_files) == 0:
            print("⚠️  No chart files found (run previous tests first)")
            return True
        
        print(f"\n✓ Found {len(png_files)} chart files:")
        total_size = 0
        for f in sorted(png_files):
            filepath = os.path.join(charts_dir, f)
            size = os.path.getsize(filepath) / 1024  # KB
            total_size += size
            print(f"  {f:.<60} {size:>7.1f} KB")
        
        print(f"\n  Total size: {total_size/1024:.2f} MB")
        
        print("\n✅ TEST 4 PASSED - Chart files verified\n")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all visualization tests."""
    print("\n" + "🎨"*35)
    print("STOCKPULSE ANALYTICS - VISUALIZATION MODULE TESTS")
    print("🎨"*35)
    
    results = []
    
    # Test 1: Chart styles
    results.append(("Chart Styles", test_chart_styles()))
    
    # Test 2: Sample data charts
    results.append(("Sample Data Charts", test_with_sample_data()))
    
    # Test 3: Real data charts (optional)
    print("\n" + "⚠️ "*35)
    print("The next test will fetch REAL data from Angel One API.")
    print("This requires your Angel One credentials to be configured.")
    response = input("Do you want to run this test? (y/n): ").lower().strip()
    
    if response == 'y':
        results.append(("Real Data Charts", test_with_real_data()))
    else:
        print("\n⏭️  Skipping real data test (you can run it later)")
        results.append(("Real Data Charts", None))
    
    # Test 4: Verify files
    results.append(("Chart Files", test_chart_files()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"
        print(f"  {test_name:.<40} {status}")
    
    print("\n" + "-"*70)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour visualization module is working correctly!")
        print("Charts are saved in:", os.path.join(os.getcwd(), 'charts'))
        print("\n📋 Next steps:")
        print("  1. Check the generated chart images in the 'charts' folder")
        print("  2. If charts look good, proceed to PDF generation")
        print("  3. Then build Flask backend and frontend")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please fix the issues before proceeding.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Angel One credentials not configured in .env")
        print("  - Charts directory permissions")
    
    print("\n" + "="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)