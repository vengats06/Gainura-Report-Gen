"""
Angel One SmartAPI Integration
===============================

This module handles all interactions with Angel One (Angel Broking) SmartAPI
to fetch Indian stock market data (NSE/BSE).

Data Available:
- Historical OHLC (Open, High, Low, Close) prices
- Current market price (LTP - Last Traded Price)
- Intraday data (1min, 5min, 15min, 1hr candles)
- Volume data
- Market depth

Official Documentation: https://smartapi.angelbroking.com/docs

Usage:
    from data_collection.angel_one_api import AngelOneAPI
    angel = AngelOneAPI()
    angel.login()
    data = angel.get_historical_data('TCS', '2024-01-01', '2024-12-17')
"""

from SmartApi import SmartConnect
import pyotp
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import pandas as pd
from backend.config import Config
from utils.logger import get_logger
from utils.validators import validate_stock_symbol

logger = get_logger(__name__)


class AngelOneAPI:
    """
    Angel One SmartAPI Wrapper
    
    Handles authentication and data fetching from Angel One.
    Manages session tokens and TOTP generation automatically.
    """
    
    # NSE symbol tokens (most common stocks)
    # You can find more tokens at: https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
    SYMBOL_TOKENS = {
        'TCS': '11536',
        'RELIANCE': '2885',
        'INFY': '1594',
        'HDFCBANK': '1333',
        'ICICIBANK': '4963',
        'HINDUNILVR': '1394',
        'ITC': '1660',
        'SBIN': '3045',
        'BHARTIARTL': '10604',
        'KOTAKBANK': '1922',
        'LT': '11483',
        'AXISBANK': '5900',
        'ASIANPAINT': '236',
        'MARUTI': '10999',
        'TITAN': '3506',
        'WIPRO': '3787',
        'ULTRACEMCO': '11532',
        'NESTLEIND': '17963',
        'BAJFINANCE': '16669',
        'HCLTECH': '7229'
    }
    
    def __init__(self):
        """
        Initialize Angel One API client with credentials from Config.
        
        Note: Authentication happens in login() method, not here.
        """
        self.api_key = Config.ANGEL_API_KEY
        self.client_id = Config.ANGEL_CLIENT_ID
        self.password = Config.ANGEL_PASSWORD
        self.totp_secret = Config.ANGEL_TOTP_SECRET
        
        self.smart_api = None
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        self.session_active = False
        
        logger.info("AngelOneAPI initialized")
    
    
    def generate_totp(self) -> str:
        """
        Generate Time-based One-Time Password (TOTP).
        
        TOTP is like Google Authenticator - generates 6-digit code
        that changes every 30 seconds.
        
        Returns:
            str: 6-digit TOTP code
            
        Example:
            totp = angel.generate_totp()
            print(totp)  # Output: "458293"
        """
        try:
            totp = pyotp.TOTP(self.totp_secret)
            totp_code = totp.now()
            logger.info(f"Generated TOTP: {totp_code}")
            return totp_code
        except Exception as e:
            logger.error(f"Failed to generate TOTP: {str(e)}")
            raise
    
    
    def login(self) -> bool:
        """
        Login to Angel One SmartAPI and get access token.
        
        Authentication Flow:
        1. Generate TOTP code
        2. Send API_KEY + CLIENT_ID + PASSWORD + TOTP to Angel One
        3. Receive AUTH_TOKEN (valid for 1 day)
        4. Use AUTH_TOKEN for all subsequent API calls
        
        Returns:
            bool: True if login successful, False otherwise
            
        Example:
            angel = AngelOneAPI()
            if angel.login():
                # Now you can fetch data
                data = angel.get_historical_data('TCS', ...)
        """
        try:
            logger.info("Attempting to login to Angel One...")
            
            # Initialize SmartConnect
            self.smart_api = SmartConnect(api_key=self.api_key)
            
            # Generate TOTP
            totp_code = self.generate_totp()
            
            # Login
            data = self.smart_api.generateSession(
                clientCode=self.client_id,
                password=self.password,
                totp=totp_code
            )
            
            if data['status']:
                self.auth_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.smart_api.getfeedToken()
                self.session_active = True
                
                logger.info("Angel One login successful!")
                logger.info(f"Session will expire in 24 hours")
                
                return True
            else:
                logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Login exception: {str(e)}")
            return False
    
    
    def get_symbol_token(self, symbol: str) -> Optional[str]:
        """
        Get NSE symbol token for a stock symbol.
        
        Angel One API requires symbol tokens, not just symbol names.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            
        Returns:
            str or None: Symbol token (e.g., '11536')
            
        Example:
            token = angel.get_symbol_token('TCS')
            # Returns: '11536'
        """
        symbol = symbol.upper().strip()
        
        if symbol in self.SYMBOL_TOKENS:
            return self.SYMBOL_TOKENS[symbol]
        else:
            logger.warning(f"Symbol token not found for: {symbol}")
            logger.warning("Add it to SYMBOL_TOKENS dict or fetch from Angel One master list")
            return None
    
    
    def get_historical_data(self, symbol: str, from_date: str, to_date: str, 
                           interval: str = 'ONE_DAY') -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            interval: Candle interval
                     Options: 'ONE_MINUTE', 'THREE_MINUTE', 'FIVE_MINUTE',
                             'TEN_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE',
                             'ONE_HOUR', 'ONE_DAY'
                     
        Returns:
            pandas.DataFrame or None: OHLC data with columns:
                - timestamp: Date/time
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
                
        Example:
            data = angel.get_historical_data('TCS', '2024-01-01', '2024-12-17')
            print(data.head())
            #    timestamp    open    high     low   close    volume
            # 0  2024-01-01  3500.0  3550.0  3490.0  3542.5  2500000
            # 1  2024-01-02  3545.0  3580.0  3540.0  3575.0  2300000
        """
        try:
            # Validate inputs
            is_valid, symbol_or_msg = validate_stock_symbol(symbol)
            if not is_valid:
                logger.error(f"Invalid symbol: {symbol_or_msg}")
                return None
            
            symbol = symbol_or_msg  # Use validated symbol (uppercase)
            
            # Get symbol token
            token = self.get_symbol_token(symbol)
            if not token:
                logger.error(f"Cannot fetch data for {symbol} - token not found")
                return None
            
            # Check if session is active
            if not self.session_active:
                logger.warning("Session not active, attempting login...")
                if not self.login():
                    logger.error("Login failed, cannot fetch data")
                    return None
            
            # Convert dates to Angel One format (YYYY-MM-DD HH:MM)
            from_date_formatted = f"{from_date} 09:15"
            to_date_formatted = f"{to_date} 15:30"
            
            logger.info(f"Fetching historical data for {symbol}")
            logger.info(f"Period: {from_date} to {to_date}")
            logger.info(f"Interval: {interval}")
            
            # API call
            historicParam = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date_formatted,
                "todate": to_date_formatted
            }
            
            response = self.smart_api.getCandleData(historicParam)
            
            if response['status']:
                data = response['data']
                
                if not data:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    data,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Convert prices to float
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
                df['volume'] = df['volume'].astype(int)
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Sort by date
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Fetched {len(df)} records for {symbol}")
                
                return df
                
            else:
                logger.error(f" API error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            return None
    
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current market price (LTP - Last Traded Price) for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            
        Returns:
            Dict or None: Current price data
                {
                    'symbol': 'TCS',
                    'ltp': 3842.50,
                    'open': 3800.00,
                    'high': 3850.00,
                    'low': 3790.00,
                    'close': 3842.50,
                    'timestamp': '2024-12-17 15:30:00'
                }
                
        Example:
            price = angel.get_current_price('TCS')
            print(f"TCS current price: ₹{price['ltp']}")
        """
        try:
            # Validate symbol
            is_valid, symbol_or_msg = validate_stock_symbol(symbol)
            if not is_valid:
                return None
            
            symbol = symbol_or_msg
            
            # Get token
            token = self.get_symbol_token(symbol)
            if not token:
                return None
            
            # Check session
            if not self.session_active:
                if not self.login():
                    return None
            
            logger.info(f"Fetching current price for {symbol}")
            
            # API call
            ltp_data = self.smart_api.ltpData("NSE", symbol, token)
            
            if ltp_data['status']:
                data = ltp_data['data']
                
                result = {
                    'symbol': symbol,
                    'ltp': float(data.get('ltp', 0)),
                    'open': float(data.get('open', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'close': float(data.get('close', 0)),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                logger.info(f"Current price for {symbol}: ₹{result['ltp']}")
                
                return result
            else:
                logger.error(f"Failed to fetch current price: {ltp_data.get('message')}")
                return None
                
        except Exception as e:
            logger.error(f"Exception fetching current price: {str(e)}")
            return None
    
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed market data including market depth.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict or None: Detailed market data
            
        Example:
            data = angel.get_market_data('TCS')
            print(f"Buy quantity: {data['totalbuyqty']}")
            print(f"Sell quantity: {data['totalsellqty']}")
        """
        try:
            token = self.get_symbol_token(symbol)
            if not token:
                return None
            
            if not self.session_active:
                if not self.login():
                    return None
            
            logger.info(f"Fetching market data for {symbol}")
            
            response = self.smart_api.marketData(
                mode="FULL",
                exchangeTokens={"NSE": [token]}
            )
            
            if response['status']:
                logger.info(f"Fetched market data for {symbol}")
                return response['data']['fetched'][0]
            else:
                logger.error(f" Failed to fetch market data")
                return None
                
        except Exception as e:
            logger.error(f"Exception: {str(e)}")
            return None
    
    
    def logout(self):
        """
        Logout from Angel One and invalidate session token.
        
        Good practice: Call this when done to free up resources.
        """
        try:
            if self.smart_api and self.session_active:
                self.smart_api.terminateSession(self.client_id)
                self.session_active = False
                logger.info("Logged out from Angel One")
        except Exception as e:
            logger.warning(f"Logout warning: {str(e)}")
    
    
    def get_profile(self) -> Optional[Dict]:
        """
        Get user profile information.
        
        Returns:
            Dict: User profile data (name, email, exchanges, etc.)
        """
        try:
            if not self.session_active:
                if not self.login():
                    return None
            
            profile = self.smart_api.getProfile(self.refresh_token)
            
            if profile['status']:
                logger.info("Fetched user profile")
                return profile['data']
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch profile: {str(e)}")
            return None


# Utility function for easy access
def fetch_stock_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch stock data without manual login/logout.
    
    Args:
        symbol: Stock symbol (e.g., 'TCS')
        days: Number of days of historical data (default: 365 = 1 year)
        
    Returns:
        pandas.DataFrame or None: Historical OHLC data
        
    Example:
        # Quick way to get 1 year of TCS data
        df = fetch_stock_data('TCS', days=365)
        print(df.head())
    """
    angel = AngelOneAPI()
    
    try:
        # Login
        if not angel.login():
            return None
        
        # Calculate dates
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Fetch data
        df = angel.get_historical_data(symbol, from_date, to_date)
        
        return df
        
    finally:
        # Always logout
        angel.logout()


# Example usage and testing
if __name__ == "__main__":
    """
    Test Angel One API integration
    """
    print("\n" + "="*60)
    print("Testing Angel One API")
    print("="*60 + "\n")
    
    try:
        # Initialize API
        angel = AngelOneAPI()
        
        # Test 1: Login
        print("Test 1: Login to Angel One...")
        if angel.login():
            print("Login successful!\n")
        else:
            print("Login failed!")
            exit()
        
        # Test 2: Get current price
        print("Test 2: Fetching current price for TCS...")
        current = angel.get_current_price('TCS')
        if current:
            print(f"TCS Current Price: ₹{current['ltp']}")
            print(f"   Open: ₹{current['open']}, High: ₹{current['high']}, Low: ₹{current['low']}\n")
        
        # Test 3: Get historical data (last 30 days)
        print("Test 3: Fetching 30 days historical data for TCS...")
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = angel.get_historical_data('TCS', from_date, to_date)
        
        if df is not None:
            print(f"Fetched {len(df)} records")
            print("\nFirst 5 records:")
            print(df.head())
            print("\nLast 5 records:")
            print(df.tail())
            print()
        
        # Test 4: Get profile
        print("Test 4: Fetching user profile...")
        profile = angel.get_profile()
        if profile:
            print(f"User: {profile.get('name', 'Unknown')}")
            print(f"   Client ID: {profile.get('clientcode', 'Unknown')}\n")
        
        # Test 5: Using convenience function
        print("Test 5: Testing convenience function...")
        df2 = fetch_stock_data('RELIANCE', days=7)
        if df2 is not None:
            print(f"Fetched {len(df2)} records for RELIANCE")
            print(f"   Latest close: ₹{df2.iloc[-1]['close']:.2f}\n")
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
        
        # Logout
        angel.logout()
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
