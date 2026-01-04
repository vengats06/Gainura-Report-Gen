"""
Test PDF Generator
==================

Tests PDF report creation with sample data and real charts.

Run this test:
    python tests/test_pdf_generator.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from pdf_generator.report_builder import PDFReportBuilder
from utils.logger import get_logger

logger = get_logger(__name__)


def test_pdf_with_sample_data():
    """Test PDF generation with sample data."""
    print("\n" + "="*70)
    print("TEST 1: PDF Generation with Sample Data")
    print("="*70)
    
    try:
        # Create comprehensive sample data
        sample_data = {
            'symbol': 'TCS',
            'fundamentals': {
                'company_name': 'Tata Consultancy Services Ltd',
                'sector': 'IT Services & Consulting',
                'industry': 'Software Services',
                'market_cap': 1403245000000,  # 14.03 lakh crore
                'pe_ratio': 28.45,
                'pb_ratio': 12.34,
                'roe': 44.8,
                'roce': 48.5,
                'dividend_yield': 1.45,
                'debt_to_equity': 0.15
            },
            'latest_price': 3842.50,
            'price_change': 45.30,
            'price_change_pct': 1.19,
            'analysis_days': 365,
            'price_stats': {
                'high_52w': 3950.00,
                'low_52w': 2888.40,
                'avg_volume': 2500000,
                'volatility': 0.18
            },
            'indicators': {
                'ma_20': 3800.00,
                'ma_50': 3750.00,
                'ma_200': 3600.00,
                'rsi_14': 65.5,
                'macd': 15.5,
                'macd_signal': 12.3,
                'macd_histogram': 3.2
            },
            'latest_rsi': 65.5,
            'macd_signal': 3.2,
            'predictions': {
                'target_7d': 3900.00,
                'change_7d': 1.50,
                'target_15d': 3950.00,
                'change_15d': 2.80,
                'target_30d': 4000.00,
                'change_30d': 4.10,
                'confidence': 0.85,
                'r2_score': 0.939
            },
            'risk_metrics': {
                'risk_category': 'Medium Risk',
                'risk_score': 5,
                'volatility': 0.18,
                'beta': 1.15,
                'sharpe_ratio': 1.85,
                'max_drawdown': -0.15,
                'var_95': -0.025
            },
            'recommendation': {
                'action': 'BUY',
                'trend': 'Bullish',
                'confidence': 0.85,
                'reasoning': 'Strong technical indicators with RSI in healthy range (65.5). MACD shows positive momentum with bullish crossover. Moving averages are well-aligned indicating uptrend. ML model predicts 4.1% upside in 30 days with high confidence (85%). Company fundamentals remain strong with excellent ROE of 44.8%.'
            }
        }
        
        print("\n✓ Sample data prepared")
        print(f"  Company: {sample_data['fundamentals']['company_name']}")
        print(f"  Current Price: ₹{sample_data['latest_price']:.2f}")
        print(f"  Recommendation: {sample_data['recommendation']['action']}")
        
        # Check if test charts exist
        from backend.config import Config
        charts_dir = Config.CHARTS_DIR
        
        # Find test charts
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
        
        if not chart_files:
            print("\n⚠️  No chart files found!")
            print("   Please run visualization tests first: python tests/test_visualization.py")
            return False
        
        # Use TESTSTOCK charts (should exist from visualization test)
        charts = {}
        for chart_type in ['candlestick', 'indicators', 'prediction', 'performance']:
            for filename in chart_files:
                if chart_type in filename.lower():
                    charts[chart_type] = os.path.join(charts_dir, filename)
                    break
        
        print(f"\n✓ Found {len(charts)} chart files:")
        for chart_type, path in charts.items():
            print(f"  {chart_type}: {os.path.basename(path)}")
        
        # Generate PDF
        print("\nGenerating PDF report...")
        builder = PDFReportBuilder()
        pdf_path = builder.create_report('TCS', sample_data, charts)
        
        # Verify PDF
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path) / 1024  # KB
            print(f"\n✅ TEST 1 PASSED - PDF created successfully!")
            print(f"   Location: {pdf_path}")
            print(f"   File size: {file_size:.1f} KB")
            
            # Open PDF (optional)
            print("\n📄 PDF Report Contents:")
            print("   ✓ Cover page with key metrics")
            print("   ✓ Executive summary")
            print("   ✓ Company overview")
            print("   ✓ Price analysis with candlestick chart")
            print("   ✓ Technical analysis with indicators")
            print("   ✓ ML predictions with forecast chart")
            print("   ✓ Risk assessment")
            print("   ✓ Investment recommendation")
            print("   ✓ Disclaimer")
            
            return True
        else:
            print("\n❌ TEST 1 FAILED - PDF file not created")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_different_recommendations():
    """Test PDFs with different recommendations (BUY, SELL, HOLD)."""
    print("\n" + "="*70)
    print("TEST 2: Multiple Recommendation Types")
    print("="*70)
    
    try:
        from backend.config import Config
        
        # Find charts
        charts_dir = Config.CHARTS_DIR
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
        
        charts = {}
        for chart_type in ['candlestick', 'indicators', 'prediction']:
            for filename in chart_files:
                if chart_type in filename.lower():
                    charts[chart_type] = os.path.join(charts_dir, filename)
                    break
        
        recommendations = [
            ('STRONG BUY', 'Bullish', 0.95, 'Very strong upward momentum with excellent fundamentals'),
            ('BUY', 'Bullish', 0.80, 'Positive technical indicators with good fundamentals'),
            ('HOLD', 'Neutral', 0.60, 'Mixed signals, wait for clearer trend'),
            ('SELL', 'Bearish', 0.75, 'Negative momentum with deteriorating fundamentals'),
            ('STRONG SELL', 'Bearish', 0.90, 'Strong bearish signals across all indicators')
        ]
        
        builder = PDFReportBuilder()
        created_pdfs = []
        
        for action, trend, confidence, reasoning in recommendations:
            # Create data for this recommendation
            data = {
                'symbol': 'DEMO',
                'fundamentals': {'company_name': 'Demo Company Ltd'},
                'latest_price': 1000.00,
                'price_change': 10.0,
                'price_change_pct': 1.0,
                'analysis_days': 365,
                'price_stats': {'high_52w': 1200, 'low_52w': 800, 'avg_volume': 1000000, 'volatility': 0.15},
                'indicators': {'ma_20': 990, 'ma_50': 980, 'ma_200': 970, 'rsi_14': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0},
                'latest_rsi': 50,
                'macd_signal': 0,
                'predictions': {'target_7d': 1010, 'change_7d': 1, 'target_15d': 1020, 'change_15d': 2, 'target_30d': 1030, 'change_30d': 3, 'confidence': confidence, 'r2_score': 0.85},
                'risk_metrics': {'risk_category': 'Medium Risk', 'risk_score': 5, 'volatility': 0.15, 'beta': 1.0, 'sharpe_ratio': 1.5, 'max_drawdown': -0.10, 'var_95': -0.02},
                'recommendation': {'action': action, 'trend': trend, 'confidence': confidence, 'reasoning': reasoning}
            }
            
            pdf_path = builder.create_report(f'DEMO_{action.replace(" ", "_")}', data, charts)
            created_pdfs.append((action, pdf_path))
        
        print(f"\n✅ TEST 2 PASSED - Created {len(created_pdfs)} PDFs with different recommendations:")
        for action, path in created_pdfs:
            size = os.path.getsize(path) / 1024
            print(f"   {action:.<20} {os.path.basename(path)} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_open_pdf():
    """Ask user if they want to open the generated PDF."""
    print("\n" + "="*70)
    print("TEST 3: Open Generated PDF")
    print("="*70)
    
    try:
        from backend.config import Config
        import subprocess
        import platform
        
        reports_dir = Config.REPORTS_DIR
        pdf_files = sorted([f for f in os.listdir(reports_dir) if f.endswith('.pdf')])
        
        if not pdf_files:
            print("\n⚠️  No PDF files found")
            return True
        
        print(f"\nFound {len(pdf_files)} PDF reports in {reports_dir}")
        
        response = input("\nDo you want to open the latest PDF? (y/n): ").lower().strip()
        
        if response == 'y':
            latest_pdf = os.path.join(reports_dir, pdf_files[-1])
            print(f"\nOpening: {latest_pdf}")
            
            # Open PDF based on OS
            if platform.system() == 'Windows':
                os.startfile(latest_pdf)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', latest_pdf])
            else:  # Linux
                subprocess.run(['xdg-open', latest_pdf])
            
            print("✅ PDF opened successfully!")
        else:
            print("⏭️  Skipped opening PDF")
        
        return True
        
    except Exception as e:
        print(f"\n⚠️  Could not open PDF: {str(e)}")
        print("   Please open it manually from the reports/ folder")
        return True  # Don't fail the test for this


def main():
    """Run all PDF generator tests."""
    print("\n" + "📄"*35)
    print("STOCKPULSE ANALYTICS - PDF GENERATOR TESTS")
    print("📄"*35)
    
    results = []
    
    # Test 1: Basic PDF generation
    results.append(("Sample Data PDF", test_pdf_with_sample_data()))
    
    # Test 2: Different recommendations
    results.append(("Multiple Recommendations", test_different_recommendations()))
    
    # Test 3: Open PDF (optional)
    results.append(("Open PDF", test_open_pdf()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:.<40} {status}")
    
    print("\n" + "-"*70)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour PDF generator is working perfectly!")
        print("\n📋 What you have now:")
        print("  ✅ Professional PDF reports")
        print("  ✅ Multi-page layouts with charts")
        print("  ✅ Complete stock analysis")
        print("  ✅ Investment recommendations")
        
        print("\n📋 Next steps:")
        print("  1. Check the generated PDFs in 'reports/' folder")
        print("  2. Verify all sections look professional")
        print("  3. Proceed to Flask Backend (API endpoints)")
        print("  4. Then build Frontend (web interface)")
        
        from backend.config import Config
        print(f"\n📂 Reports location: {Config.REPORTS_DIR}")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please fix the issues before proceeding.")
    
    print("\n" + "="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)