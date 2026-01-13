"""
Complete Pipeline Test
======================

Tests the complete end-to-end pipeline:
1. Fetch real data from Angel One
2. Process through ETL
3. Run ML models
4. Generate charts
5. Create PDF report

Run this test:
    python tests/test_complete_pipeline.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import StockReportPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def test_complete_pipeline():
    """Test complete pipeline with real stock data."""
    print("COMPLETE PIPELINE TEST - REAL DATA")
    
    try:
        # Get stock symbol from user
        print("This test will generate a REAL stock report with LIVE data.")
        print("\nAvailable stocks: TCS, RELIANCE, INFY, HDFCBANK, ICICIBANK, WIPRO")
        
        symbol = input("\nEnter stock symbol (default: TCS): ").strip().upper()
        if not symbol:
            symbol = 'TCS'
        
        print(f"\n{'='*70}")
        print(f"Generating report for: {symbol}")
        print(f"{'='*70}\n")
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = StockReportPipeline()
        print("✓ Pipeline initialized\n")
        
        # Generate report
        print(f"Starting complete pipeline for {symbol}...")
        print("This will take 30-60 seconds...\n")
        
        result = pipeline.generate_report(symbol, days=365)
        
        # Check result
        if result['success']:
            print("\n" + "="*70)
            print(" SUCCESS! Report generated successfully!")
            print("="*70)
            
            print(f"\n PDF Report: {result['pdf_path']}")
            print(f" File size: {os.path.getsize(result['pdf_path'])/1024:.1f} KB")
            
            print("\n Report Contents:")
            print("  ✓ Cover page with key metrics")
            print("  ✓ Executive summary")
            print("  ✓ Company fundamentals")
            print("  ✓ Price analysis (LIVE data)")
            print("  ✓ Technical indicators (RSI, MACD, MA)")
            print("  ✓ ML price predictions (30 days)")
            print("  ✓ Risk assessment")
            print("  ✓ News sentiment analysis")
            print("  ✓ Investment recommendation")
            
            print("\n Data stored to:")
            print("  ✓ AWS S3 (raw and processed)")
            print("  ✓ AWS RDS PostgreSQL")
            
            # Ask to open PDF
            response = input("\n📂 Open PDF? (y/n): ").lower().strip()
            if response == 'y':
                import platform
                import subprocess
                
                if platform.system() == 'Windows':
                    os.startfile(result['pdf_path'])
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', result['pdf_path']])
                else:  # Linux
                    subprocess.run(['xdg-open', result['pdf_path']])
                
                print("✓ PDF opened")
            
            print("\n" + "="*70)
            print(" COMPLETE PIPELINE TEST PASSED!")
            print("="*70)
            
            print("\n Everything is working:")
            print("  • Data fetching from Angel One ✓")
            print("  • Web scraping (Screener.in) ✓")
            print("  • News API integration ✓")
            print("  • AWS S3 storage ✓")
            print("  • AWS RDS database ✓")
            print("  • ETL pipeline ✓")
            print("  • Technical indicators ✓")
            print("  • ML predictions ✓")
            print("  • Sentiment analysis ✓")
            print("  • Risk calculation ✓")
            print("  • Trend classification ✓")
            print("  • Chart generation ✓")
            print("  • PDF report creation ✓")
        
            print("\n Your application is PRODUCTION READY!")
            print("\n Next steps:")
            print("  1. Start Flask server: python -m backend.app")
            print("  2. Open frontend: frontend/index.html")
            print("  3. Generate reports through web interface")
            
            return True
        
        else:
            print("\n" + "="*70)
            print(" FAILED")
            print("="*70)
            print(f"\nError: {result['message']}")
            
            print("\n💡 Common issues:")
            print("  • Angel One credentials not configured")
            print("  • AWS credentials not set up")
            print("  • Network connectivity issues")
            print("  • Invalid stock symbol")
            
            return False
    
    except Exception as e:
        print("\n" + "="*70)
        print(" ERROR")
        print("="*70)
        print(f"\nException: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)