"""
News Fetcher for Stock Sentiment Analysis
=========================================

This module fetches news articles related to stocks using News API.
News data is used for sentiment analysis to predict stock movements.

Why News Matters:
- "TCS wins $2B contract" → Positive news → Stock price ↑
- "CEO resigns" → Negative news → Stock price ↓
- News sentiment predicts short-term price movements

Data Collected:
- Headlines
- Publication date
- Source (Economic Times, Bloomberg, etc.)
- Article URL
- Description

API: NewsAPI.org (Free tier: 100 requests/day)

Usage:
    from data_collection.news_fetcher import NewsFetcher
    fetcher = NewsFetcher()
    news = fetcher.get_stock_news('TCS', days=30)
"""

from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from backend.config import Config
from utils.logger import get_logger
from utils.validators import validate_stock_symbol

logger = get_logger(__name__)


class NewsFetcher:
    """
    Fetches news articles for stock symbols using News API.
    
    Features:
    - Search by stock symbol and company name
    - Filter by date range
    - Get top headlines
    - Sort by relevance or date
    """
    
    def __init__(self):
        """
        Initialize News API client with API key from config.
        
        Get free API key from: https://newsapi.org/register
        Free tier: 100 requests per day
        """
        self.api_key = Config.NEWS_API_KEY
        
        if not self.api_key or self.api_key == 'not_configured_yet':
            logger.warning(" News API key not configured!")
            logger.warning("Get free key from: https://newsapi.org/register")
            self.client = None
        else:
            try:
                self.client = NewsApiClient(api_key=self.api_key)
                logger.info(" NewsFetcher initialized successfully")
            except Exception as e:
                logger.error(f" Failed to initialize News API: {str(e)}")
                self.client = None
        
        # Company name mapping (for better search results)
        self.company_names = {
            'TCS': 'Tata Consultancy Services TCS',
            'RELIANCE': 'Reliance Industries',
            'INFY': 'Infosys',
            'HDFCBANK': 'HDFC Bank',
            'ICICIBANK': 'ICICI Bank',
            'HINDUNILVR': 'Hindustan Unilever HUL',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India SBI',
            'BHARTIARTL': 'Bharti Airtel',
            'KOTAKBANK': 'Kotak Mahindra Bank',
            'LT': 'Larsen Toubro L&T',
            'AXISBANK': 'Axis Bank',
            'MARUTI': 'Maruti Suzuki',
            'WIPRO': 'Wipro',
            'HCLTECH': 'HCL Technologies'
        }
    
    
    def _build_search_query(self, symbol: str) -> str:
        """
        Build search query for better news results.
        
        Uses exact match with quotes to get specific company news only.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            str: Search query
            
        Example:
            symbol = 'TCS'
            query = '"Tata Consultancy Services" OR "TCS"'
        """
        # Get company name or use symbol
        company_name = self.company_names.get(symbol, symbol)
        
        # Split company name and symbol for better matching
        if symbol in self.company_names:
            # Use exact match with quotes for better precision
            query = f'"{company_name}" OR "{symbol}"'
        else:
            # For unknown symbols, use exact match
            query = f'"{symbol}"'
        
        return query
    
    
    def get_stock_news(self, symbol: str, days: int = 30, 
                       language: str = 'en', 
                       sort_by: str = 'publishedAt') -> Optional[List[Dict]]:
        """
        Fetch news articles for a stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            days: Number of days to look back (default: 30)
            language: Language code (default: 'en' for English)
            sort_by: Sort order - 'publishedAt', 'relevancy', 'popularity'
            
        Returns:
            List[Dict] or None: News articles with metadata
            
        Example:
            news = fetcher.get_stock_news('TCS', days=7)
            for article in news:
                print(f"{article['publishedAt']}: {article['title']}")
        """
        if not self.client:
            logger.error(" News API client not initialized")
            return None
        
        # Validate symbol
        is_valid, symbol_or_msg = validate_stock_symbol(symbol)
        if not is_valid:
            logger.error(f"Invalid symbol: {symbol_or_msg}")
            return None
        
        symbol = symbol_or_msg
        
        try:
            # Build search query
            query = self._build_search_query(symbol)
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            logger.info(f"Fetching news for {symbol}")
            logger.info(f"Query: '{query}'")
            logger.info(f"Date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
            
            # API call
            response = self.client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language=language,
                sort_by=sort_by,
                page_size=100  # Maximum per request
            )
            
            if response['status'] != 'ok':
                logger.error(f" API error: {response.get('message', 'Unknown error')}")
                return None
            
            articles = response.get('articles', [])
            
            if not articles:
                logger.warning(f"No news found for {symbol}")
                return []
            
            # Process articles and filter relevant ones
            processed_articles = []
            for article in articles:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                # Filter: Article must contain company name or symbol
                company_name = self.company_names.get(symbol, symbol).lower()
                symbol_lower = symbol.lower()
                
                # Check if article mentions the company
                is_relevant = (
                    symbol_lower in title or 
                    symbol_lower in description or
                    any(word in title for word in company_name.split()) or
                    any(word in description for word in company_name.split())
                )
                
                if not is_relevant:
                    continue  # Skip irrelevant articles
                
                processed = {
                    'symbol': symbol,
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'author': article.get('author', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'image_url': article.get('urlToImage', '')
                }
                processed_articles.append(processed)
            
            logger.info(f" Fetched {len(processed_articles)} news articles for {symbol}")
            
            return processed_articles
            
        except Exception as e:
            logger.error(f" Failed to fetch news: {str(e)}")
            return None
    
    
    def get_top_headlines(self, symbol: str, country: str = 'in') -> Optional[List[Dict]]:
        """
        Get top headlines related to stock from specific country.
        
        Args:
            symbol: Stock symbol
            country: Country code (default: 'in' for India)
            
        Returns:
            List[Dict] or None: Top headlines
            
        Example:
            headlines = fetcher.get_top_headlines('TCS')
        """
        if not self.client:
            logger.error(" News API client not initialized")
            return None
        
        try:
            query = self._build_search_query(symbol)
            
            logger.info(f"Fetching top headlines for {symbol}")
            
            response = self.client.get_top_headlines(
                q=query,
                country=country,
                page_size=20
            )
            
            if response['status'] != 'ok':
                logger.error(f" API error: {response.get('message')}")
                return None
            
            articles = response.get('articles', [])
            
            logger.info(f" Fetched {len(articles)} top headlines for {symbol}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch headlines: {str(e)}")
            return None
    
    
    def get_recent_headlines_summary(self, symbol: str, count: int = 5) -> List[str]:
        """
        Get just the headlines (text only) for quick summary.
        
        Args:
            symbol: Stock symbol
            count: Number of headlines to return
            
        Returns:
            List[str]: List of headlines
            
        Example:
            headlines = fetcher.get_recent_headlines_summary('TCS', count=5)
            for headline in headlines:
                print(f"• {headline}")
        """
        articles = self.get_stock_news(symbol, days=7)
        
        if not articles:
            return []
        
        # Extract just titles
        headlines = [article['title'] for article in articles[:count]]
        
        return headlines


# Example usage and testing
if __name__ == "__main__":
    """
    Test News Fetcher
    """
    print("\n" + "="*60)
    print("Testing News Fetcher")
    print("="*60 + "\n")
    
    try:
        fetcher = NewsFetcher()
        
        if not fetcher.client:
            print(" News API not configured!")
            print("To test this:")
            print("1. Get free API key from: https://newsapi.org/register")
            print("2. Add to .env file: NEWS_API_KEY=your_key_here")
            print("3. Run test again")
        else:
            # Test 1: Get stock news
            print("Test 1: Fetching TCS news (last 7 days)...")
            news = fetcher.get_stock_news('TCS', days=7)
            
            if news:
                print(f"\nFetched {len(news)} articles")
                print("\nLatest 3 headlines:")
                for i, article in enumerate(news[:3], 1):
                    print(f"\n{i}. {article['title']}")
                    print(f"   Source: {article['source']}")
                    print(f"   Date: {article['published_at']}")
                print()
            else:
                print(" No news found\n")
            
            # Test 2: Get headlines summary
            print("Test 2: Getting headline summary...")
            headlines = fetcher.get_recent_headlines_summary('RELIANCE', count=5)
            
            if headlines:
                print(f"\nTop 5 RELIANCE headlines:")
                for i, headline in enumerate(headlines, 1):
                    print(f"{i}. {headline}")
                print()
            
        print("="*60)
        print("Tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()