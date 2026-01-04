"""
PDF Report Template
==================

Provides header/footer templates for PDF reports.
Ensures consistent branding across all pages.

Usage:
    from pdf_generator.templates.report_template import ReportTemplate
    
    template = ReportTemplate()
    doc.build(story, onFirstPage=template.first_page,
             onLaterPages=template.later_pages)
"""

from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime


class ReportTemplate:
    """
    PDF page templates with headers and footers.
    """
    
    def __init__(self):
        """Initialize template."""
        self.company_name = "StockPulse Analytics"
        self.tagline = "AI-Powered Stock Intelligence"
    
    
    def first_page(self, canvas_obj, doc):
        """
        Template for first page (cover page).
        
        Args:
            canvas_obj: ReportLab canvas
            doc: Document object
        """
        canvas_obj.saveState()
        
        # No header on first page, just footer
        self._draw_footer(canvas_obj, doc, page_num=1)
        
        canvas_obj.restoreState()
    
    
    def later_pages(self, canvas_obj, doc):
        """
        Template for all pages after first.
        
        Args:
            canvas_obj: ReportLab canvas
            doc: Document object
        """
        canvas_obj.saveState()
        
        # Header
        self._draw_header(canvas_obj, doc)
        
        # Footer
        self._draw_footer(canvas_obj, doc, page_num=doc.page)
        
        canvas_obj.restoreState()
    
    
    def _draw_header(self, canvas_obj, doc):
        """
        Draw header on page.
        
        Contains:
        - Company name
        - Horizontal line
        """
        # Company name (left)
        canvas_obj.setFont('Helvetica-Bold', 10)
        canvas_obj.setFillColor(colors.HexColor('#2E86DE'))
        canvas_obj.drawString(
            0.75*inch, 
            doc.height + doc.topMargin - 0.3*inch,
            self.company_name
        )
        
        # Tagline (right)
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(colors.HexColor('#7F8C8D'))
        canvas_obj.drawRightString(
            doc.width + doc.leftMargin + 0.75*inch,
            doc.height + doc.topMargin - 0.3*inch,
            self.tagline
        )
        
        # Horizontal line
        canvas_obj.setStrokeColor(colors.HexColor('#2E86DE'))
        canvas_obj.setLineWidth(2)
        canvas_obj.line(
            0.75*inch,
            doc.height + doc.topMargin - 0.45*inch,
            doc.width + doc.leftMargin + 0.75*inch,
            doc.height + doc.topMargin - 0.45*inch
        )
    
    
    def _draw_footer(self, canvas_obj, doc, page_num: int = 1):
        """
        Draw footer on page.
        
        Contains:
        - Horizontal line
        - Page number (center)
        - Generation timestamp (right)
        - Disclaimer (left)
        """
        # Horizontal line
        canvas_obj.setStrokeColor(colors.HexColor('#DEE2E6'))
        canvas_obj.setLineWidth(1)
        canvas_obj.line(
            0.75*inch,
            0.6*inch,
            doc.width + doc.leftMargin + 0.75*inch,
            0.6*inch
        )
        
        # Page number (center)
        canvas_obj.setFont('Helvetica', 9)
        canvas_obj.setFillColor(colors.HexColor('#7F8C8D'))
        page_text = f"Page {page_num}"
        canvas_obj.drawCentredString(
            doc.width/2 + doc.leftMargin + 0.75*inch,
            0.4*inch,
            page_text
        )
        
        # Generation timestamp (right)
        canvas_obj.setFont('Helvetica', 7)
        timestamp = datetime.now().strftime('%B %d, %Y')
        canvas_obj.drawRightString(
            doc.width + doc.leftMargin + 0.75*inch,
            0.4*inch,
            timestamp
        )
        
        # Disclaimer (left)
        canvas_obj.setFont('Helvetica', 7)
        canvas_obj.drawString(
            0.75*inch,
            0.4*inch,
            "Not Financial Advice"
        )


# Testing
if __name__ == "__main__":
    print("ReportTemplate helper class created.")
    print("This is used by PDFReportBuilder for headers/footers.")