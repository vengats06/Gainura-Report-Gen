"""
Chart Styling Configuration
===========================

Centralized styling for all charts in StockPulse Analytics.
Ensures consistent, professional appearance across all visualizations.

Usage:
    from visualization.chart_styles import ChartStyles
    
    styles = ChartStyles()
    plt.plot(..., color=styles.COLORS['primary'])
"""

class ChartStyles:
    """
    Professional chart styling configuration.
    """
    
    # Color palette (modern, professional)
    COLORS = {
        # Primary colors
        'primary': '#2E86DE',      # Blue
        'secondary': '#10AC84',    # Green
        'accent': '#EE5A6F',       # Red/Pink
        'warning': '#F79F1F',      # Orange
        
        # Stock-specific colors
        'bullish': '#26A69A',      # Teal (up/gain)
        'bearish': '#EF5350',      # Red (down/loss)
        
        # Technical indicators
        'ma_20': '#2E86DE',        # Blue
        'ma_50': '#F79F1F',        # Orange
        'ma_200': '#EE5A6F',       # Red
        'rsi': '#8E44AD',          # Purple
        'macd': '#2E86DE',         # Blue
        'signal': '#E74C3C',       # Red
        'prediction': '#9B59B6',   # Purple
        
        # Neutral colors
        'neutral': '#95A5A6',      # Gray
        'dark': '#2C3E50',         # Dark blue-gray
        'light': '#ECF0F1',        # Light gray
        'white': '#FFFFFF',
        'black': '#000000'
    }
    
    # Font configuration
    FONTS = {
        'title': {
            'family': 'sans-serif',
            'size': 16,
            'weight': 'bold'
        },
        'subtitle': {
            'family': 'sans-serif',
            'size': 14,
            'weight': 'bold'
        },
        'label': {
            'family': 'sans-serif',
            'size': 12,
            'weight': 'normal'
        },
        'text': {
            'family': 'sans-serif',
            'size': 10,
            'weight': 'normal'
        }
    }
    
    # Figure sizes (width, height in inches)
    FIGURE_SIZES = {
        'small': (10, 6),
        'medium': (14, 8),
        'large': (16, 10),
        'wide': (18, 6)
    }
    
    # DPI for export
    DPI = {
        'screen': 100,
        'print': 300,
        'high': 600
    }
    
    # Grid styling
    GRID_STYLE = {
        'alpha': 0.3,
        'linestyle': '-',
        'linewidth': 0.5
    }
    
    # Legend styling
    LEGEND_STYLE = {
        'loc': 'upper left',
        'framealpha': 0.9,
        'edgecolor': '#95A5A6',
        'fontsize': 10
    }
    
    @staticmethod
    def get_color_scheme(name: str = 'default'):
        """
        Get predefined color schemes.
        
        Args:
            name: Scheme name ('default', 'dark', 'light', 'colorblind')
            
        Returns:
            dict: Color scheme
        """
        schemes = {
            'default': {
                'background': '#FFFFFF',
                'text': '#2C3E50',
                'grid': '#ECF0F1',
                'primary': '#2E86DE',
                'secondary': '#10AC84'
            },
            'dark': {
                'background': '#1E272E',
                'text': '#ECF0F1',
                'grid': '#2C3E50',
                'primary': '#48DBFB',
                'secondary': '#1DD1A1'
            },
            'light': {
                'background': '#F8F9FA',
                'text': '#2C3E50',
                'grid': '#DEE2E6',
                'primary': '#007BFF',
                'secondary': '#28A745'
            },
            'colorblind': {
                'background': '#FFFFFF',
                'text': '#000000',
                'grid': '#CCCCCC',
                'primary': '#0173B2',
                'secondary': '#DE8F05'
            }
        }
        
        return schemes.get(name, schemes['default'])
    
    @staticmethod
    def format_currency(value: float, symbol: str = '₹') -> str:
        """
        Format value as currency.
        
        Args:
            value: Numeric value
            symbol: Currency symbol
            
        Returns:
            str: Formatted string
        """
        if value >= 10000000:  # >= 1 crore
            return f"{symbol}{value/10000000:.2f}Cr"
        elif value >= 100000:  # >= 1 lakh
            return f"{symbol}{value/100000:.2f}L"
        elif value >= 1000:  # >= 1 thousand
            return f"{symbol}{value/1000:.2f}K"
        else:
            return f"{symbol}{value:.2f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """
        Format value as percentage.
        
        Args:
            value: Decimal value (0.05 = 5%)
            
        Returns:
            str: Formatted percentage
        """
        return f"{value*100:+.2f}%"