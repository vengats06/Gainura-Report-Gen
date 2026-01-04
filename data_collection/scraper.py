"""
Web Scraper for Stock Fundamentals
==================================

This module scrapes fundamental data (financial metrics) from screener.in
for Indian stocks (NSE/BSE).

Data Collected:
- Market Cap, P/E Ratio, P/B Ratio
- ROE, ROCE, Debt-to-Equity
- Revenue, Net Profit, Operating Margin
- Dividend Yield, Current Ratio
- Sector, Industry information

Why screener.in?
- Free (no API key needed)
- Comprehensive Indian stock data
- Well-structured HTML (easy to scrape)

Usage:
    from data_collection.scraper import StockScraper
    scraper = StockScraper()
    fundamentals = scraper.get_fundamentals('TCS')
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Optional
import time
from utils.logger import get_logger
from utils.validators import validate_stock_symbol

logger = get_logger(__name__)


class StockScraper:
    """
    Web scraper for Indian stock fundamental data.
    
    Scrapes data from screener.in (no authentication required).
    Respects rate limits (waits between requests).
    """
    
    def __init__(self):
        """
        Initialize scraper with headers to mimic browser.
        
        Why headers?
        - Websites can block Python scripts
        - Headers make request look like it's from a browser
        - Reduces chance of being blocked
        """
        self.base_url = "https://www.screener.in/company/{}/consolidated/"
        
        # Headers to mimic browser (avoid being blocked)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        logger.info("StockScraper initialized")
    
    
    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract number from text (handles crores, lakhs, etc.)
        
        Examples:
            "₹1,234.56 Cr." → 123456000000.0 (1234.56 crores)
            "45.67%" → 45.67
            "2,500" → 2500.0
            "N/A" → None
            
        Args:
            text: Text containing number
            
        Returns:
            float or None: Extracted number
        """
        if not text or text.strip() in ['', 'N/A', '-', 'NA']:
            return None
        
        try:
            # Remove currency symbols and commas
            text = text.replace('₹', '').replace(',', '').strip()
            
            # Handle percentages
            if '%' in text:
                return float(text.replace('%', '').strip())
            
            # Handle crores (Cr.)
            if 'Cr.' in text or 'Cr' in text:
                number = float(re.findall(r'[\d.]+', text)[0])
                return number * 10000000  # 1 crore = 10 million
            
            # Handle lakhs (Lac)
            if 'Lac' in text:
                number = float(re.findall(r'[\d.]+', text)[0])
                return number * 100000  # 1 lakh = 100 thousand
            
            # Regular number
            numbers = re.findall(r'[\d.]+', text)
            if numbers:
                return float(numbers[0])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract number from '{text}': {str(e)}")
            return None
    
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Scrape fundamental data for a stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS', 'RELIANCE')
            
        Returns:
            Dict or None: Fundamental data
            
        Example:
            data = scraper.get_fundamentals('TCS')
            print(f"Market Cap: ₹{data['market_cap']/10000000:.2f} Cr")
            print(f"P/E Ratio: {data['pe_ratio']}")
        """
        # Validate symbol
        is_valid, symbol_or_msg = validate_stock_symbol(symbol)
        if not is_valid:
            logger.error(f"Invalid symbol: {symbol_or_msg}")
            return None
        
        symbol = symbol_or_msg
        
        try:
            # Build URL
            url = self.base_url.format(symbol)
            logger.info(f"Scraping fundamentals for {symbol}")
            logger.info(f"URL: {url}")
            
            # Make request
            response = requests.get(url, headers=self.headers, timeout=10)
            
            # Check if page exists
            if response.status_code == 404:
                logger.error(f"Stock {symbol} not found on screener.in")
                return None
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch page: Status {response.status_code}")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract company name
            company_name = None
            name_element = soup.find('h1', class_='h2')
            if name_element:
                company_name = name_element.text.strip()
            
            # Initialize data dictionary
            fundamentals = {
                'symbol': symbol,
                'company_name': company_name,
                'sector': None,
                'industry': None,
                'market_cap': None,
                'pe_ratio': None,
                'pb_ratio': None,
                'dividend_yield': None,
                'roe': None,
                'roce': None,
                'debt_to_equity': None,
                'current_ratio': None,
                'revenue': None,
                'net_profit': None,
                'operating_margin': None,
                'net_margin': None
            }
            
            # Extract ratios from top ratios section
            ratios_section = soup.find('ul', id='top-ratios')
            if ratios_section:
                ratio_items = ratios_section.find_all('li', class_='flex flex-space-between')
                
                for item in ratio_items:
                    name_elem = item.find('span', class_='name')
                    value_elem = item.find('span', class_='number')
                    
                    if name_elem and value_elem:
                        name = name_elem.text.strip().lower()
                        value = value_elem.text.strip()
                        
                        # Map to our keys
                        if 'market cap' in name:
                            fundamentals['market_cap'] = self._extract_number(value)
                        elif 'current price' in name:
                            fundamentals['current_price'] = self._extract_number(value)
                        elif 'high / low' in name:
                            # Extract 52-week high and low
                            prices = re.findall(r'[\d,]+', value)
                            if len(prices) >= 2:
                                fundamentals['week_52_high'] = self._extract_number(prices[0])
                                fundamentals['week_52_low'] = self._extract_number(prices[1])
                        elif 'stock p/e' in name:
                            fundamentals['pe_ratio'] = self._extract_number(value)
                        elif 'book value' in name:
                            fundamentals['book_value'] = self._extract_number(value)
                        elif 'dividend yield' in name:
                            fundamentals['dividend_yield'] = self._extract_number(value)
                        elif 'roce' in name:
                            fundamentals['roce'] = self._extract_number(value)
                        elif 'roe' in name:
                            fundamentals['roe'] = self._extract_number(value)
                        elif 'face value' in name:
                            fundamentals['face_value'] = self._extract_number(value)
            
            # Extract quarterly results (for revenue and profit)
            quarterly_section = soup.find('section', id='quarters')
            if quarterly_section:
                table = quarterly_section.find('table')
                if table:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if cells and len(cells) >= 2:
                            metric = cells[0].text.strip().lower()
                            # Get latest quarter value (second column)
                            value = cells[1].text.strip()
                            
                            if 'sales' in metric or 'revenue' in metric:
                                fundamentals['revenue'] = self._extract_number(value)
                            elif 'operating profit' in metric or 'ebit' in metric:
                                fundamentals['operating_profit'] = self._extract_number(value)
                            elif 'net profit' in metric:
                                fundamentals['net_profit'] = self._extract_number(value)
            
            # Calculate margins if we have revenue and profit
            if fundamentals['revenue'] and fundamentals['operating_profit']:
                fundamentals['operating_margin'] = (
                    fundamentals['operating_profit'] / fundamentals['revenue'] * 100
                )
            
            if fundamentals['revenue'] and fundamentals['net_profit']:
                fundamentals['net_margin'] = (
                    fundamentals['net_profit'] / fundamentals['revenue'] * 100
                )
            
            # Extract P/B ratio
            pb_section = soup.find(text=re.compile('Price to Book Value'))
            if pb_section:
                # Try to find value near this text
                parent = pb_section.find_parent()
                if parent:
                    value_elem = parent.find_next('span', class_='number')
                    if value_elem:
                        fundamentals['pb_ratio'] = self._extract_number(value_elem.text)
            
            # Extract debt to equity
            debt_section = soup.find(text=re.compile('Debt to Equity'))
            if debt_section:
                parent = debt_section.find_parent()
                if parent:
                    value_elem = parent.find_next('span', class_='number')
                    if value_elem:
                        fundamentals['debt_to_equity'] = self._extract_number(value_elem.text)
            
            # Extract current ratio
            current_section = soup.find(text=re.compile('Current Ratio'))
            if current_section:
                parent = current_section.find_parent()
                if parent:
                    value_elem = parent.find_next('span', class_='number')
                    if value_elem:
                        fundamentals['current_ratio'] = self._extract_number(value_elem.text)
            
            logger.info(f" Successfully scraped fundamentals for {symbol}")
            logger.info(f"   Company: {fundamentals['company_name']}")
            logger.info(f"   Market Cap: ₹{fundamentals['market_cap']/10000000:.2f} Cr" if fundamentals['market_cap'] else "   Market Cap: N/A")
            
            return fundamentals
            
        except requests.exceptions.Timeout:
            logger.error(f" Request timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f" Failed to scrape {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def get_multiple_stocks(self, symbols: list, delay: float = 2.0) -> Dict[str, Dict]:
        """
        Scrape fundamentals for multiple stocks with delay between requests.
        
        Args:
            symbols: List of stock symbols
            delay: Delay in seconds between requests (default: 2s)
            
        Returns:
            Dict: {symbol: fundamentals_data}
            
        Example:
            data = scraper.get_multiple_stocks(['TCS', 'INFY', 'WIPRO'])
            for symbol, fundamentals in data.items():
                print(f"{symbol}: Market Cap = {fundamentals['market_cap']}")
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Scraping {i+1}/{len(symbols)}: {symbol}")
            
            fundamentals = self.get_fundamentals(symbol)
            if fundamentals:
                results[symbol] = fundamentals
            
            # Wait before next request (be respectful to server)
            if i < len(symbols) - 1:
                logger.info(f"Waiting {delay}s before next request...")
                time.sleep(delay)
        
        logger.info(f" Scraped {len(results)}/{len(symbols)} stocks successfully")
        return results


# Example usage and testing
if __name__ == "__main__":
    """
    Test web scraper
    """
    print("\n" + "="*60)
    print("Testing Stock Scraper")
    print("="*60 + "\n")
    
    try:
        scraper = StockScraper()
        
        # Test 1: Scrape single stock
        print("Test 1: Scraping TCS fundamentals...")
        fundamentals = scraper.get_fundamentals('TCS')
        
        if fundamentals:
            print(f"\n Successfully scraped TCS data:")
            print(f"   Company: {fundamentals['company_name']}")
            if fundamentals['market_cap']:
                print(f"   Market Cap: ₹{fundamentals['market_cap']/10000000:.2f} Cr")
            if fundamentals['pe_ratio']:
                print(f"   P/E Ratio: {fundamentals['pe_ratio']:.2f}")
            if fundamentals['roe']:
                print(f"   ROE: {fundamentals['roe']:.2f}%")
            if fundamentals['debt_to_equity']:
                print(f"   Debt/Equity: {fundamentals['debt_to_equity']:.2f}")
            print()
        else:
            print(" Failed to scrape TCS\n")
        
        # Test 2: Scrape multiple stocks
        print("Test 2: Scraping multiple stocks...")
        stocks = ['RELIANCE', 'INFY']
        results = scraper.get_multiple_stocks(stocks, delay=3.0)
        
        print(f"\n Scraped {len(results)} stocks:")
        for symbol, data in results.items():
            print(f"   {symbol}: {data['company_name']}")
        
        print("\n" + "="*60)
        print("Tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()