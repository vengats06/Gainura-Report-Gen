"""
Flask Application - StockPulse Analytics
========================================

Main Flask application that orchestrates the complete pipeline:
1. Accept stock symbol from user
2. Fetch real data from all sources
3. Run ETL pipeline
4. Generate ML predictions
5. Create charts
6. Build PDF report
7. Return for download

Usage:
    python -m backend.app
    
    Then visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import pandas as pd
import os
import traceback

# Import all components
from backend.config import Config
from data_collection.angel_one_api import AngelOneAPI
from data_collection.scraper import StockScraper
from data_collection.news_fetcher import NewsFetcher
from aws.s3_handler import S3Handler
from aws.rds_connection import RDSConnection
from etl.transformations import DataTransformer
from etl.technical_indicators import TechnicalIndicators
from ml_models.price_predictor import PricePredictor
from ml_models.sentiment_analyzer import SentimentAnalyzer
from ml_models.risk_calculator import RiskCalculator
from ml_models.trend_classifier import TrendClassifier
from visualization.chart_generator import ChartGenerator
from pdf_generator.report_builder import PDFReportBuilder
from utils.logger import get_logger
from utils.validators import validate_stock_symbol

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
app.config['SECRET_KEY'] = Config.SECRET_KEY

logger = get_logger(__name__)


class StockReportPipeline:
    """
    Complete pipeline to generate stock report from live data.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.angel = AngelOneAPI()
        self.scraper = StockScraper()
        self.news_fetcher = NewsFetcher()
        self.s3 = S3Handler()
        self.db = RDSConnection()
        self.transformer = DataTransformer()
        self.chart_gen = ChartGenerator()
        self.pdf_builder = PDFReportBuilder()
        
        logger.info("StockReportPipeline initialized")
    
    
    def generate_report(self, symbol: str, days: int = 365) -> dict:
        """
        Generate complete stock report with live data.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            days: Historical data period (default: 365 days)
            
        Returns:
            dict: {
                'success': bool,
                'pdf_path': str,
                'message': str,
                'data': dict
            }
        """
        try:
            logger.info(f"="*70)
            logger.info(f"Starting report generation for {symbol}")
            logger.info(f"="*70)
            
            # Validate symbol
            is_valid, validated_symbol = validate_stock_symbol(symbol)
            if not is_valid:
                return {'success': False, 'message': f'Invalid symbol: {validated_symbol}'}
            
            symbol = validated_symbol
            
            # STEP 1: Fetch all data
            logger.info("\n STEP 1: FETCHING LIVE DATA")
            data = self._fetch_all_data(symbol, days)
            
            # FIX: Proper DataFrame empty check
            if data['price_data'] is None or (isinstance(data['price_data'], pd.DataFrame) and data['price_data'].empty):
                return {'success': False, 'message': 'No price data available for this symbol'}
            
            # STEP 2: Process data (ETL)
            logger.info("\n STEP 2: PROCESSING DATA (ETL)")
            processed_data = self._process_data(data['price_data'])
            
            # STEP 3: Store to AWS
            logger.info("\n STEP 3: STORING TO AWS")
            self._store_to_aws(symbol, data, processed_data)
            
            # STEP 4: Run ML models
            logger.info("\n STEP 4: RUNNING ML MODELS")
            ml_results = self._run_ml_models(processed_data, data.get('news', []))
            
            # STEP 5: Generate charts
            logger.info("\n STEP 5: GENERATING CHARTS")
            charts = self._generate_charts(symbol, processed_data, ml_results.get('predictions'))
            
            # STEP 6: Build report data
            logger.info("\nSTEP 6: PREPARING REPORT DATA")
            report_data = self._build_report_data(symbol, data, processed_data, ml_results)
            
            # STEP 7: Generate PDF
            logger.info("\n STEP 7: GENERATING PDF REPORT")
            pdf_path = self.pdf_builder.create_report(symbol, report_data, charts)
            
            # Cleanup
            self.angel.logout()
            self.db.close_all_connections()
            
            logger.info(f"\n REPORT GENERATED SUCCESSFULLY!")
            logger.info(f"PDF: {pdf_path}")
            logger.info(f"="*70)
            
            return {
                'success': True,
                'pdf_path': pdf_path,
                'message': f'Report generated successfully for {symbol}',
                'data': report_data
            }
            
        except Exception as e:
            logger.error(f"\n ERROR: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Cleanup on error
            try:
                self.angel.logout()
                self.db.close_all_connections()
            except:
                pass
            
            return {
                'success': False,
                'message': f'Error generating report: {str(e)}'
            }
    
    
    def _fetch_all_data(self, symbol: str, days: int) -> dict:
        """Fetch data from all sources."""
        logger.info(f"   Fetching data for {symbol} ({days} days)")
        
        data = {}
        
        # 1. Angel One - Price data
        logger.info("  Logging into Angel One...")
        if not self.angel.login():
            raise Exception("Angel One login failed")
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        logger.info(f"  Fetching price data from {from_date} to {to_date}")
        df = self.angel.get_historical_data(symbol, from_date, to_date)
        
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise Exception(f"No price data available for {symbol}")
        
        data['price_data'] = df
        logger.info(f"  Fetched {len(df)} days of price data")
        
        # 2. Screener.in - Fundamentals
        logger.info("  Scraping fundamentals from Screener.in...")
        fundamentals = self.scraper.get_fundamentals(symbol)
        data['fundamentals'] = fundamentals
        
        if fundamentals:
            logger.info(f"   Fetched fundamentals for {fundamentals.get('company_name', symbol)}")
        else:
            logger.warning("   Fundamentals not available")
        
        # 3. News API - Latest news
        logger.info("   Fetching latest news...")
        news = []
        if self.news_fetcher.client:
            news = self.news_fetcher.get_stock_news(symbol, days=30)
            data['news'] = news
            logger.info(f"   Fetched {len(news)} news articles")
        else:
            data['news'] = []
            logger.warning("  News API not configured")
        
        return data
    
    
    def _process_data(self, price_df):
        """Process data through ETL pipeline."""
        logger.info("  Cleaning data...")
        
        # Prepare DataFrame
        df = price_df.copy()
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp']
        
        # Clean data
        clean_df = self.transformer.clean_stock_data(df)
        logger.info(f"   Data cleaned: {len(clean_df)} records")
        
        # Calculate technical indicators
        logger.info("  Calculating technical indicators...")
        ti = TechnicalIndicators(clean_df)
        indicators_df = ti.calculate_all()
        logger.info(f"   Calculated {len(indicators_df.columns)} columns")
        
        # Add date features
        date_col = 'timestamp' if 'timestamp' in indicators_df.columns else 'date'
        final_df = self.transformer.add_date_features(indicators_df, date_col)
        
        return final_df
    
    
    def _store_to_aws(self, symbol: str, raw_data: dict, processed_data):
        """Store data to S3 and RDS."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        
        # 1. Store raw data to S3
        logger.info("  Uploading raw data to S3...")
        
        # Price data
        price_json = raw_data['price_data'].to_dict(orient='records')
        s3_key = f"angel_one/{symbol}_{timestamp}.json"
        self.s3.upload_json(price_json, s3_key, bucket_type='raw')
        logger.info(f"   Uploaded to s3://raw/{s3_key}")
        
        # Fundamentals
        if raw_data.get('fundamentals'):
            fund_key = f"fundamentals/{symbol}_{timestamp}.json"
            self.s3.upload_json(raw_data['fundamentals'], fund_key, bucket_type='raw')
        
        # News
        if raw_data.get('news'):
            news_key = f"news/{symbol}_{timestamp}.json"
            self.s3.upload_json(raw_data['news'], news_key, bucket_type='raw')
        
        # 2. Store processed data to S3
        logger.info("  Uploading processed data to S3...")
        processed_json = processed_data.to_dict(orient='records')
        proc_key = f"technical_indicators/{symbol}_{timestamp}.json"
        self.s3.upload_json(processed_json, proc_key, bucket_type='processed')
        logger.info(f"  Uploaded to s3://processed/{proc_key}")
        
        # 3. Store to RDS PostgreSQL
        logger.info("   Storing to RDS PostgreSQL...")
        
        # Ensure tables exist
        self.db.create_tables()
        
        # Insert price data
        price_records = []
        for _, row in processed_data.iterrows():
            date_col = 'timestamp' if 'timestamp' in row.index else 'date'
            date_val = row[date_col]
            if hasattr(date_val, 'strftime'):
                date_val = date_val.strftime('%Y-%m-%d')
            
            price_records.append((
                symbol,
                str(date_val),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ))
        
        self.db.bulk_insert_stock_prices(price_records)
        logger.info(f"  Inserted {len(price_records)} price records to RDS")
        
        # Insert fundamentals
        if raw_data.get('fundamentals'):
            self.db.insert_fundamental(symbol, raw_data['fundamentals'])
            logger.info(f"  Inserted fundamentals to RDS")
    
    
    def _run_ml_models(self, processed_data, news_articles):
        """Run all ML models."""
        results = {}
        
        # 1. Price Prediction
        logger.info(" Training price prediction model...")
        try:
            predictor = PricePredictor(model_type='linear')
            
            # Check if we have enough data
            if len(processed_data) >= 60:  # Need at least 60 days
                metrics = predictor.train(processed_data, test_size=0.2)
                predictions = predictor.predict_multiple_days(processed_data, days=30)
                
                results['predictions'] = predictions
                results['model_metrics'] = metrics
                logger.info(f"   Model trained (R² = {metrics['test_r2']:.3f})")
            else:
                logger.warning(f"   Not enough data for ML prediction ({len(processed_data)} days)")
                results['predictions'] = None
                results['model_metrics'] = None
        except Exception as e:
            logger.error(f"   Prediction failed: {str(e)}")
            results['predictions'] = None
            results['model_metrics'] = None
        
        # 2. Sentiment Analysis
        logger.info("   Analyzing news sentiment...")
        if news_articles and len(news_articles) > 0:
            analyzer = SentimentAnalyzer()
            headlines = [article['title'] for article in news_articles[:20]]
            sentiment_results = analyzer.get_overall_sentiment(headlines)
            results['sentiment'] = sentiment_results
            logger.info(f"  Sentiment: {sentiment_results['overall_label']} ({sentiment_results['average_score']:.2f})")
        else:
            results['sentiment'] = None
            logger.warning("   No news for sentiment analysis")
        
        # 3. Risk Calculation
        logger.info("   Calculating risk metrics...")
        risk_calc = RiskCalculator()
        risk_metrics = risk_calc.calculate_all_risks(processed_data)
        results['risk'] = risk_metrics
        logger.info(f"   Risk Score: {risk_metrics['risk_score']}/10 ({risk_metrics['risk_category']})")
        
        # 4. Trend Classification
        logger.info("  Classifying trend...")
        classifier = TrendClassifier()
        trend = classifier.classify(processed_data)
        results['trend'] = trend
        logger.info(f"   Trend: {trend['trend_label']} - {trend['recommendation']} ({trend['confidence']:.0%})")
        
        return results
    
    
    def _generate_charts(self, symbol: str, processed_data, predictions_df):
        """Generate all charts."""
        logger.info(f"  Generating charts for {symbol}...")
        
        charts = self.chart_gen.create_all_charts(
            processed_data, 
            symbol, 
            predictions_df
        )
        
        logger.info(f"  Generated {len(charts)} charts")
        return charts
    
    
    def _build_report_data(self, symbol: str, raw_data: dict, 
                          processed_data, ml_results: dict) -> dict:
        """Build complete report data structure."""
        logger.info("  Building report data...")
        
        latest = processed_data.iloc[-1]
        first = processed_data.iloc[0]
        
        # Calculate price change
        latest_price = float(latest['close'])
        first_price = float(first['close'])
        price_change = latest_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        # Get predictions
        predictions = ml_results.get('predictions')
        if predictions is not None and len(predictions) > 0:
            target_7d = float(predictions.iloc[6]['predicted_price']) if len(predictions) > 6 else latest_price
            target_15d = float(predictions.iloc[14]['predicted_price']) if len(predictions) > 14 else latest_price
            target_30d = float(predictions.iloc[-1]['predicted_price'])
            
            change_7d = ((target_7d - latest_price) / latest_price) * 100
            change_15d = ((target_15d - latest_price) / latest_price) * 100
            change_30d = ((target_30d - latest_price) / latest_price) * 100
            
            pred_dict = {
                'target_7d': target_7d,
                'change_7d': change_7d,
                'target_15d': target_15d,
                'change_15d': change_15d,
                'target_30d': target_30d,
                'change_30d': change_30d,
                'confidence': ml_results['trend']['confidence'],
                'r2_score': ml_results['model_metrics']['test_r2'] if ml_results.get('model_metrics') else 0.85
            }
        else:
            # No ML predictions available
            pred_dict = {
                'target_7d': latest_price,
                'change_7d': 0,
                'target_15d': latest_price,
                'change_15d': 0,
                'target_30d': latest_price,
                'change_30d': 0,
                'confidence': 0.5,
                'r2_score': 0
            }
        
        # Build recommendation reasoning
        trend = ml_results['trend']
        risk = ml_results['risk']
        sentiment = ml_results.get('sentiment')
        
        reasoning_parts = []
        reasoning_parts.append(f"Technical analysis shows {trend['trend_label'].lower()} trend with {trend['confidence']*100:.0f}% confidence.")
        
        if 'rsi_14' in latest.index and not pd.isna(latest['rsi_14']):
            rsi_val = float(latest['rsi_14'])
            if rsi_val > 70:
                reasoning_parts.append(f"RSI at {rsi_val:.1f} indicates overbought conditions.")
            elif rsi_val < 30:
                reasoning_parts.append(f"RSI at {rsi_val:.1f} indicates oversold conditions.")
            else:
                reasoning_parts.append(f"RSI at {rsi_val:.1f} is in healthy range.")
        
        if sentiment:
            reasoning_parts.append(f"News sentiment is {sentiment['overall_label'].lower()} based on {sentiment['total_articles']} recent articles.")
        
        reasoning_parts.append(f"Risk assessment: {risk['risk_category']} (Score: {risk['risk_score']}/10).")
        
        if predictions is not None:
            reasoning_parts.append(f"ML model predicts {change_30d:+.1f}% movement in 30 days.")
        
        reasoning = " ".join(reasoning_parts)
        
        # Build complete report data
        report_data = {
            'symbol': symbol,
            'fundamentals': raw_data.get('fundamentals', {}),
            'latest_price': latest_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'analysis_days': len(processed_data),
            'price_stats': {
                'high_52w': float(processed_data['high'].max()),
                'low_52w': float(processed_data['low'].min()),
                'avg_volume': float(processed_data['volume'].mean()),
                'volatility': float(processed_data['volatility_30d'].iloc[-1]) if 'volatility_30d' in processed_data.columns else 0.15
            },
            'indicators': {
                'ma_20': float(latest['ma_20']) if 'ma_20' in latest.index and not pd.isna(latest['ma_20']) else 0,
                'ma_50': float(latest['ma_50']) if 'ma_50' in latest.index and not pd.isna(latest['ma_50']) else 0,
                'ma_200': float(latest['ma_200']) if 'ma_200' in latest.index and not pd.isna(latest['ma_200']) else 0,
                'rsi_14': float(latest['rsi_14']) if 'rsi_14' in latest.index and not pd.isna(latest['rsi_14']) else 50,
                'macd': float(latest['macd']) if 'macd' in latest.index and not pd.isna(latest['macd']) else 0,
                'macd_signal': float(latest['macd_signal']) if 'macd_signal' in latest.index and not pd.isna(latest['macd_signal']) else 0,
                'macd_histogram': float(latest['macd_histogram']) if 'macd_histogram' in latest.index and not pd.isna(latest['macd_histogram']) else 0
            },
            'latest_rsi': float(latest['rsi_14']) if 'rsi_14' in latest.index and not pd.isna(latest['rsi_14']) else 50,
            'macd_signal': float(latest['macd_histogram']) if 'macd_histogram' in latest.index and not pd.isna(latest['macd_histogram']) else 0,
            'predictions': pred_dict,
            'risk_metrics': risk,
            'recommendation': {
                'action': trend['recommendation'],
                'trend': trend['trend_label'],
                'confidence': trend['confidence'],
                'reasoning': reasoning
            }
        }
        
        logger.info("  ✓ Report data compiled")
        return report_data


# Initialize pipeline
pipeline = StockReportPipeline()


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page."""
    return jsonify({
        'service': 'StockPulse Analytics API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'generate_report': 'POST /api/generate-report',
            'health': 'GET /api/health',
            'download': 'GET /api/download/<filename>'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'angel_one': 'configured',
            'aws_s3': 'configured',
            'aws_rds': 'configured',
            'news_api': 'configured' if Config.NEWS_API_KEY else 'not configured'
        }
    })


@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """
    Generate stock report.
    
    Request body:
    {
        "symbol": "TCS",
        "days": 365  (optional, default: 365)
    }
    
    Response:
    {
        "success": true,
        "message": "Report generated successfully",
        "pdf_url": "/api/download/TCS_Report_20260104_214308.pdf",
        "filename": "TCS_Report_20260104_214308.pdf"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'symbol' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required field: symbol'
            }), 400
        
        symbol = data['symbol'].strip().upper()
        days = data.get('days', 365)
        
        logger.info(f"API request: Generate report for {symbol}")
        
        # Generate report
        result = pipeline.generate_report(symbol, days)
        
        if result['success']:
            filename = os.path.basename(result['pdf_path'])
            return jsonify({
                'success': True,
                'message': result['message'],
                'pdf_url': f'/api/download/{filename}',
                'filename': filename
            })
        else:
            return jsonify({
                'success': False,
                'message': result['message']
            }), 500
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_report(filename):
    """
    Download generated PDF report.
    
    Args:
        filename: PDF filename
        
    Returns:
        PDF file for download
    """
    try:
        filepath = os.path.join(Config.REPORTS_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': 'File not found'
            }), 404
        
        return send_file(
            filepath,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Download failed: {str(e)}'
        }), 500


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("STOCKPULSE ANALYTICS - Starting Server")
    
    Config.print_config_status()
    
    print("\n Server starting on http://localhost:5000")
    print("\n Available endpoints:")
    print("  GET  /                      - API info")
    print("  GET  /api/health            - Health check")
    print("  POST /api/generate-report   - Generate report")
    print("  GET  /api/download/<file>   - Download PDF")
    print("\n" + "="*70 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG
    )