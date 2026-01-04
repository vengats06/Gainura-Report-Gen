"""
PDF Report Builder
==================

Creates professional PDF stock analysis reports with:
- Company overview
- Price charts
- Technical analysis
- ML predictions
- Risk assessment
- Buy/Hold/Sell recommendation

Usage:
    from pdf_generator.report_builder import PDFReportBuilder
    
    builder = PDFReportBuilder()
    pdf_path = builder.create_report(symbol='TCS', data=data_dict)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from datetime import datetime
import os
from typing import Dict, Optional
from backend.config import Config
from pdf_generator.templates.report_template import ReportTemplate
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFReportBuilder:
    """
    Build professional PDF reports for stock analysis.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize PDF builder.
        
        Args:
            output_dir: Directory to save reports (default: Config.REPORTS_DIR)
        """
        self.output_dir = output_dir or Config.REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.template = ReportTemplate()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        logger.info(f"PDFReportBuilder initialized. Output: {self.output_dir}")
    
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        
        # Check if styles already exist (prevents duplicate error)
        existing_styles = [s.name for s in self.styles.byName.values()]
        
        # Title style
        if 'ReportTitle' not in existing_styles:
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Title'],
                fontSize=24,
                textColor=colors.HexColor('#2E86DE'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        # Section heading
        if 'SectionHeading' not in existing_styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeading',
                parent=self.styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#2C3E50'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold',
                borderWidth=2,
                borderColor=colors.HexColor('#2E86DE'),
                borderPadding=5,
                backColor=colors.HexColor('#F8F9FA')
            ))
        
        # Subsection
        if 'Subsection' not in existing_styles:
            self.styles.add(ParagraphStyle(
                name='Subsection',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#34495E'),
                spaceAfter=8,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            ))
        
        # Body text - use custom name to avoid conflict
        if 'ReportBody' not in existing_styles:
            self.styles.add(ParagraphStyle(
                name='ReportBody',
                parent=self.styles['Normal'],
                fontSize=11,
                leading=16,
                alignment=TA_JUSTIFY,
                spaceAfter=10
            ))
        
        # Recommendation box
        if 'Recommendation' not in existing_styles:
            self.styles.add(ParagraphStyle(
                name='Recommendation',
                parent=self.styles['Normal'],
                fontSize=18,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER,
                spaceAfter=15,
                spaceBefore=15
            ))
    
    
    def create_report(self, symbol: str, data: Dict, charts: Dict) -> str:
        """
        Create complete PDF report.
        
        Args:
            symbol: Stock symbol
            data: Dictionary with all analysis data
            charts: Dictionary with chart file paths
            
        Returns:
            str: Path to generated PDF
        """
        logger.info(f"Creating PDF report for {symbol}")
        
        # Create PDF filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_Report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        story = []
        
        # Cover page
        story.extend(self._create_cover_page(symbol, data))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_summary(symbol, data))
        story.append(PageBreak())
        
        # Company overview
        story.extend(self._create_company_overview(data))
        
        # Price analysis with chart
        story.extend(self._create_price_analysis(symbol, data, charts))
        story.append(PageBreak())
        
        # Technical analysis with charts
        story.extend(self._create_technical_analysis(data, charts))
        story.append(PageBreak())
        
        # ML predictions
        story.extend(self._create_predictions(symbol, data, charts))
        story.append(PageBreak())
        
        # Risk assessment
        story.extend(self._create_risk_assessment(data))
        
        # Final recommendation
        story.extend(self._create_recommendation(data))
        story.append(PageBreak())
        
        # Disclaimer
        story.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(story, onFirstPage=self.template.first_page,
                 onLaterPages=self.template.later_pages)
        
        logger.info(f"PDF report created: {filepath}")
        return filepath
    
    
    def _create_cover_page(self, symbol: str, data: Dict) -> list:
        """Create cover page."""
        elements = []
        
        # Title
        title = Paragraph(
            f"<b>{symbol} Stock Analysis Report</b>",
            self.styles['ReportTitle']
        )
        elements.append(Spacer(1, 1.5*inch))
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))
        
        # Company name
        company_name = data.get('fundamentals', {}).get('company_name', symbol)
        company_text = Paragraph(
            f"<font size=16>{company_name}</font>",
            self.styles['Normal']
        )
        elements.append(company_text)
        elements.append(Spacer(1, 1*inch))
        
        # Key metrics table
        current_price = data.get('latest_price', 0)
        change = data.get('price_change', 0)
        change_pct = data.get('price_change_pct', 0)
        
        metrics_data = [
            ['Current Price', f"₹{current_price:.2f}"],
            ['Change', f"₹{change:+.2f} ({change_pct:+.2f}%)"],
            ['Report Date', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Period', f"{data.get('analysis_days', 365)} days"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6'))
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 1*inch))
        
        # Recommendation preview
        recommendation = data.get('recommendation', {}).get('action', 'HOLD')
        rec_color = self._get_recommendation_color(recommendation)
        
        rec_text = Paragraph(
            f"<font color='{rec_color}'><b>{recommendation}</b></font>",
            self.styles['Recommendation']
        )
        elements.append(rec_text)
        
        # Footer
        elements.append(Spacer(1, 1*inch))
        footer_text = Paragraph(
            "<i>Generated by StockPulse Analytics</i>",
            self.styles['Normal']
        )
        elements.append(footer_text)
        
        return elements
    
    
    def _create_summary(self, symbol: str, data: Dict) -> list:
        """Create executive summary."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary text
        recommendation = data.get('recommendation', {})
        risk = data.get('risk_metrics', {})
        
        summary = f"""
        This report provides a comprehensive analysis of {symbol} based on 
        {data.get('analysis_days', 365)} days of historical data, technical indicators, 
        and machine learning predictions.
        <br/><br/>
        <b>Key Findings:</b><br/>
        • Current trend is <b>{recommendation.get('trend', 'Neutral')}</b> with 
        {recommendation.get('confidence', 0)*100:.0f}% confidence<br/>
        • Risk level: <b>{risk.get('risk_category', 'Medium Risk')}</b> 
        (Score: {risk.get('risk_score', 5)}/10)<br/>
        • RSI indicates the stock is <b>{self._interpret_rsi(data.get('latest_rsi', 50))}</b><br/>
        • MACD shows <b>{self._interpret_macd(data.get('macd_signal', 0))}</b> momentum<br/>
        <br/>
        <b>Recommendation:</b> <b>{recommendation.get('action', 'HOLD')}</b>
        """
        
        elements.append(Paragraph(summary, self.styles['ReportBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    
    def _create_company_overview(self, data: Dict) -> list:
        """Create company overview section."""
        elements = []
        
        fundamentals = data.get('fundamentals', {})
        if not fundamentals:
            return elements
        
        elements.append(Paragraph("Company Overview", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Company details table
        details = [
            ['Company Name', fundamentals.get('company_name', 'N/A')],
            ['Sector', fundamentals.get('sector', 'N/A')],
            ['Industry', fundamentals.get('industry', 'N/A')],
        ]
        
        # Financial metrics
        if fundamentals.get('market_cap'):
            details.append(['Market Cap', f"₹{fundamentals['market_cap']/10000000:.2f} Cr"])
        if fundamentals.get('pe_ratio'):
            details.append(['P/E Ratio', f"{fundamentals['pe_ratio']:.2f}"])
        if fundamentals.get('pb_ratio'):
            details.append(['P/B Ratio', f"{fundamentals['pb_ratio']:.2f}"])
        if fundamentals.get('roe'):
            details.append(['ROE', f"{fundamentals['roe']:.2f}%"])
        if fundamentals.get('dividend_yield'):
            details.append(['Dividend Yield', f"{fundamentals['dividend_yield']:.2f}%"])
        
        table = Table(details, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    
    def _create_price_analysis(self, symbol: str, data: Dict, charts: Dict) -> list:
        """Create price analysis section with chart."""
        elements = []
        
        elements.append(Paragraph("Price Analysis", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add candlestick chart
        if 'candlestick' in charts:
            img = Image(charts['candlestick'], width=6.5*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Price statistics
        stats = data.get('price_stats', {})
        
        analysis_text = f"""
        <b>Current Price:</b> ₹{data.get('latest_price', 0):.2f}<br/>
        <b>52-Week High:</b> ₹{stats.get('high_52w', 0):.2f}<br/>
        <b>52-Week Low:</b> ₹{stats.get('low_52w', 0):.2f}<br/>
        <b>Average Volume:</b> {stats.get('avg_volume', 0)/1000000:.2f}M shares<br/>
        <b>Volatility (30d):</b> {stats.get('volatility', 0)*100:.2f}%
        """
        
        elements.append(Paragraph(analysis_text, self.styles['ReportBody']))
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    
    def _create_technical_analysis(self, data: Dict, charts: Dict) -> list:
        """Create technical analysis section."""
        elements = []
        
        elements.append(Paragraph("Technical Analysis", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Technical indicators chart
        if 'indicators' in charts:
            img = Image(charts['indicators'], width=6.5*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Indicator interpretation
        indicators = data.get('indicators', {})
        
        interpretation = f"""
        <b>Moving Averages:</b><br/>
        • MA(20): ₹{indicators.get('ma_20', 0):.2f}<br/>
        • MA(50): ₹{indicators.get('ma_50', 0):.2f}<br/>
        • MA(200): ₹{indicators.get('ma_200', 0):.2f}<br/>
        <br/>
        <b>RSI (14):</b> {indicators.get('rsi_14', 50):.2f} - 
        {self._interpret_rsi(indicators.get('rsi_14', 50))}<br/>
        <br/>
        <b>MACD:</b><br/>
        • MACD Line: {indicators.get('macd', 0):.2f}<br/>
        • Signal Line: {indicators.get('macd_signal', 0):.2f}<br/>
        • Histogram: {indicators.get('macd_histogram', 0):.2f} - 
        {self._interpret_macd(indicators.get('macd_histogram', 0))}<br/>
        """
        
        elements.append(Paragraph(interpretation, self.styles['ReportBody']))
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    
    def _create_predictions(self, symbol: str, data: Dict, charts: Dict) -> list:
        """Create ML predictions section."""
        elements = []
        
        elements.append(Paragraph("Price Predictions (AI)", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Prediction chart
        if 'prediction' in charts:
            img = Image(charts['prediction'], width=6.5*inch, height=3.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Prediction details
        predictions = data.get('predictions', {})
        
        pred_text = f"""
        Our machine learning model predicts the following price targets for the next 30 days:
        <br/><br/>
        <b>7-Day Target:</b> ₹{predictions.get('target_7d', 0):.2f} 
        ({predictions.get('change_7d', 0):+.2f}%)<br/>
        <b>15-Day Target:</b> ₹{predictions.get('target_15d', 0):.2f} 
        ({predictions.get('change_15d', 0):+.2f}%)<br/>
        <b>30-Day Target:</b> ₹{predictions.get('target_30d', 0):.2f} 
        ({predictions.get('change_30d', 0):+.2f}%)<br/>
        <br/>
        <b>Model Confidence:</b> {predictions.get('confidence', 0)*100:.1f}%<br/>
        <b>Model Accuracy:</b> R² Score = {predictions.get('r2_score', 0):.3f}
        """
        
        elements.append(Paragraph(pred_text, self.styles['ReportBody']))
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    
    def _create_risk_assessment(self, data: Dict) -> list:
        """Create risk assessment section."""
        elements = []
        
        elements.append(Paragraph("Risk Assessment", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        risk = data.get('risk_metrics', {})
        
        # Risk metrics table
        risk_data = [
            ['Risk Category', risk.get('risk_category', 'Medium Risk')],
            ['Risk Score', f"{risk.get('risk_score', 5)}/10"],
            ['Volatility (30d)', f"{risk.get('volatility', 0)*100:.2f}%"],
            ['Beta', f"{risk.get('beta', 1.0):.2f}"],
            ['Sharpe Ratio', f"{risk.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{risk.get('max_drawdown', 0)*100:.2f}%"],
            ['VaR (95%)', f"{risk.get('var_95', 0)*100:.2f}%"]
        ]
        
        table = Table(risk_data, colWidths=[2.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Risk interpretation
        risk_text = self._interpret_risk(risk.get('risk_score', 5))
        elements.append(Paragraph(risk_text, self.styles['ReportBody']))
        
        return elements
    
    
    def _create_recommendation(self, data: Dict) -> list:
        """Create final recommendation section."""
        elements = []
        
        elements.append(Paragraph("Investment Recommendation", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.3*inch))
        
        recommendation = data.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        
        # Big recommendation box
        rec_color = self._get_recommendation_color(action)
        rec_box = Paragraph(
            f"<font color='{rec_color}' size=24><b>{action}</b></font>",
            self.styles['Recommendation']
        )
        elements.append(rec_box)
        elements.append(Spacer(1, 0.2*inch))
        
        # Reasoning
        reasoning = f"""
        <b>Confidence:</b> {recommendation.get('confidence', 0)*100:.0f}%<br/>
        <b>Trend:</b> {recommendation.get('trend', 'Neutral')}<br/>
        <br/>
        <b>Rationale:</b><br/>
        {recommendation.get('reasoning', 'Based on comprehensive technical and fundamental analysis.')}
        """
        
        elements.append(Paragraph(reasoning, self.styles['ReportBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    
    def _create_disclaimer(self) -> list:
        """Create disclaimer section."""
        elements = []
        
        elements.append(Paragraph("Disclaimer", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        disclaimer = """
        This report is generated by StockPulse Analytics using automated algorithms and machine learning models. 
        It is intended for informational purposes only and should not be considered as financial advice.
        <br/><br/>
        <b>Important Notes:</b><br/>
        • Past performance does not guarantee future results<br/>
        • All investments carry risk, including potential loss of principal<br/>
        • Please consult with a qualified financial advisor before making investment decisions<br/>
        • The predictions and recommendations in this report are based on historical data and may not reflect future market conditions<br/>
        • StockPulse Analytics and its creators are not liable for any financial losses incurred based on this report<br/>
        <br/>
        <i>Report generated on {}</i>
        """.format(datetime.now().strftime('%B %d, %Y at %I:%M %p'))
        
        elements.append(Paragraph(disclaimer, self.styles['ReportBody']))
        
        return elements
    
    
    def _get_recommendation_color(self, action: str) -> str:
        """Get color for recommendation."""
        colors_map = {
            'BUY': '#26A69A',
            'STRONG BUY': '#00796B',
            'SELL': '#EF5350',
            'STRONG SELL': '#C62828',
            'HOLD': '#F79F1F'
        }
        return colors_map.get(action, '#95A5A6')
    
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value."""
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"
    
    
    def _interpret_macd(self, value: float) -> str:
        """Interpret MACD signal."""
        if value > 0:
            return "Bullish"
        elif value < 0:
            return "Bearish"
        else:
            return "Neutral"
    
    
    def _interpret_risk(self, risk_score: int) -> str:
        """Interpret risk score."""
        if risk_score <= 3:
            return """
            <b>Low Risk Investment:</b> This stock shows stable price movements with low volatility. 
            Suitable for conservative investors seeking steady returns with minimal risk.
            """
        elif risk_score <= 6:
            return """
            <b>Medium Risk Investment:</b> This stock shows moderate volatility with balanced risk-reward profile. 
            Suitable for investors with moderate risk appetite seeking reasonable returns.
            """
        else:
            return """
            <b>High Risk Investment:</b> This stock shows high volatility and significant price fluctuations. 
            Suitable only for aggressive investors with high risk tolerance seeking potentially high returns.
            """


# Testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing PDF Report Builder")
    print("="*60 + "\n")
    
    # Sample data
    sample_data = {
        'symbol': 'TCS',
        'fundamentals': {
            'company_name': 'Tata Consultancy Services Ltd',
            'sector': 'IT Services',
            'industry': 'Software Services',
            'market_cap': 1403245000000,
            'pe_ratio': 28.45,
            'pb_ratio': 12.34,
            'roe': 44.8,
            'dividend_yield': 1.45
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
            'reasoning': 'Strong technical indicators with RSI in healthy range. MACD shows positive momentum. ML model predicts 4.1% upside in 30 days with high confidence.'
        }
    }
    
    # Create dummy chart paths (use existing test charts)
    charts = {
        'candlestick': 'charts/TESTSTOCK_candlestick_20260104.png',
        'indicators': 'charts/TESTSTOCK_indicators_20260104.png',
        'prediction': 'charts/TESTSTOCK_prediction_20260104.png'
    }
    
    try:
        builder = PDFReportBuilder()
        pdf_path = builder.create_report('TCS', sample_data, charts)
        
        print(f"\n✅ PDF Report created successfully!")
        print(f"   Location: {pdf_path}")
        print(f"   File size: {os.path.getsize(pdf_path)/1024:.1f} KB")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Failed to create PDF: {str(e)}")
        import traceback
        traceback.print_exc()