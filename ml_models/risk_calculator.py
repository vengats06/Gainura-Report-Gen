
"""
Stock Risk Calculator
====================

Calculates various risk metrics for stock investments.

Risk Metrics:
1. Volatility - How much price fluctuates
2. Beta - Compared to market (Nifty 50)
3. Sharpe Ratio - Risk-adjusted returns
4. Max Drawdown - Biggest loss from peak
5. Value at Risk (VaR) - Maximum expected loss

Usage:
    from ml_models.risk_calculator import RiskCalculator
    
    calc = RiskCalculator()
    risk = calc.calculate_all_risks(df)
    print(f"Risk Score: {risk['risk_score']}/10")
"""

import pandas as pd
import numpy as np
from typing import Dict,Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class RiskCalculator:
    """
    Calculate risk metrics for stocks.
    
    Higher risk = Higher potential returns (and losses)
    Lower risk = More stable, predictable returns
    """
    
    def __init__(self):
        """Initialize risk calculator."""
        logger.info("RiskCalculator initialized")
        self.risk_free_rate = 0.07  # 7% (India government bonds)
    
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 30) -> float:
        """
        Calculate annualized volatility.
        
        Volatility = Standard deviation of daily returns * sqrt(252)
        252 = trading days in a year
        
        Args:
            df: DataFrame with 'close' prices
            period: Rolling window period
            
        Returns:
            float: Annualized volatility (0.2 = 20%)
        """
        if 'daily_return' in df.columns:
            returns = df['daily_return']
        else:
            returns = df['close'].pct_change()
        
        # Rolling volatility
        volatility = returns.rolling(window=period).std() * np.sqrt(252)
        
        # Latest volatility
        latest_vol = volatility.iloc[-1]
        
        logger.info(f"Volatility ({period}d): {latest_vol:.2%}")
        
        return latest_vol
    
    
    def calculate_beta(self, stock_df: pd.DataFrame, market_df: pd.DataFrame = None) -> float:
        """
        Calculate Beta (volatility compared to market).
        
        Beta interpretation:
        - Beta = 1: Moves with market
        - Beta > 1: More volatile than market
        - Beta < 1: Less volatile than market
        
        Args:
            stock_df: Stock price DataFrame
            market_df: Market index DataFrame (e.g., Nifty 50)
            
        Returns:
            float: Beta value
        """
        # If no market data, assume beta = 1
        if market_df is None:
            logger.warning("No market data provided, assuming Beta = 1.0")
            return 1.0
        
        # Calculate returns
        stock_returns = stock_df['close'].pct_change().dropna()
        market_returns = market_df['close'].pct_change().dropna()
        
        # Align dates
        combined = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        # Calculate covariance and variance
        covariance = np.cov(combined['stock'], combined['market'])[0][1]
        market_variance = np.var(combined['market'])
        
        beta = covariance / market_variance if market_variance != 0 else 1.0
        
        logger.info(f"Beta: {beta:.2f}")
        
        return beta
    
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted returns).
        
        Sharpe Ratio = (Returns - Risk Free Rate) / Volatility
        
        Higher Sharpe = Better risk-adjusted returns
        - > 1.0 = Good
        - > 2.0 = Very good
        - > 3.0 = Excellent
        
        Args:
            df: DataFrame with 'close' prices
            
        Returns:
            float: Sharpe ratio
        """
        if 'daily_return' in df.columns:
            returns = df['daily_return']
        else:
            returns = df['close'].pct_change()
        
        # Annualized return
        avg_return = returns.mean() * 252
        
        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (avg_return - self.risk_free_rate) / volatility if volatility != 0 else 0
        
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        
        return sharpe
    
    
    def calculate_max_drawdown(self, df: pd.DataFrame) -> Tuple[float, str, str]:
        """
        Calculate maximum drawdown (biggest loss from peak).
        
        Max Drawdown = (Trough - Peak) / Peak
        
        Args:
            df: DataFrame with 'close' prices
            
        Returns:
            Tuple: (max_drawdown, peak_date, trough_date)
        """
        prices = df['close']
        
        # Calculate cumulative maximum
        cummax = prices.cummax()
        
        # Calculate drawdown
        drawdown = (prices - cummax) / cummax
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Find dates
        max_dd_idx = drawdown.idxmin()
        peak_idx = prices[:max_dd_idx].idxmax()
        
        peak_date = df.loc[peak_idx, 'date'] if 'date' in df.columns else str(peak_idx)
        trough_date = df.loc[max_dd_idx, 'date'] if 'date' in df.columns else str(max_dd_idx)
        
        logger.info(f"Max Drawdown: {max_dd:.2%}")
        
        return max_dd, peak_date, trough_date
    
    
    def calculate_var(self, df: pd.DataFrame, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR = Maximum expected loss with given confidence
        
        Example: VaR(95%) = -2.5%
        Meaning: 95% confidence that daily loss won't exceed 2.5%
        
        Args:
            df: DataFrame with 'close' prices
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            float: VaR (negative = loss)
        """
        if 'daily_return' in df.columns:
            returns = df['daily_return']
        else:
            returns = df['close'].pct_change()
        
        # Calculate VaR
        var = np.percentile(returns.dropna(), (1 - confidence) * 100)
        
        logger.info(f"VaR (95%): {var:.2%}")
        
        return var
    
    
    def calculate_risk_score(self, volatility: float, beta: float, 
                            sharpe: float, max_dd: float) -> int:
        """
        Calculate overall risk score (1-10 scale).
        
        1-3: Low risk (safe, stable)
        4-6: Medium risk (moderate)
        7-10: High risk (volatile, risky)
        
        Args:
            volatility: Annualized volatility
            beta: Beta value
            sharpe: Sharpe ratio
            max_dd: Maximum drawdown
            
        Returns:
            int: Risk score (1-10)
        """
        score = 0
        
        # Volatility component (0-3 points)
        if volatility < 0.15:  # <15%
            score += 1
        elif volatility < 0.25:  # 15-25%
            score += 2
        else:  # >25%
            score += 3
        
        # Beta component (0-3 points)
        if beta < 0.8:
            score += 1
        elif beta < 1.2:
            score += 2
        else:
            score += 3
        
        # Sharpe component (0-2 points, inverted)
        if sharpe > 2.0:
            score += 0
        elif sharpe > 1.0:
            score += 1
        else:
            score += 2
        
        # Max drawdown component (0-2 points)
        if max_dd > -0.15:  # <15% drawdown
            score += 0
        elif max_dd > -0.30:  # 15-30% drawdown
            score += 1
        else:  # >30% drawdown
            score += 2
        
        return min(10, max(1, score))
    
    
    def calculate_all_risks(self, df: pd.DataFrame, market_df: pd.DataFrame = None) -> Dict:
        """
        Calculate all risk metrics.
        
        Args:
            df: Stock price DataFrame
            market_df: Market index DataFrame (optional)
            
        Returns:
            Dict with all risk metrics
        """
        logger.info("Calculating all risk metrics...")
        
        volatility = self.calculate_volatility(df)
        beta = self.calculate_beta(df, market_df)
        sharpe = self.calculate_sharpe_ratio(df)
        max_dd, peak_date, trough_date = self.calculate_max_drawdown(df)
        var_95 = self.calculate_var(df)
        
        risk_score = self.calculate_risk_score(volatility, beta, sharpe, max_dd)
        
        # Risk category
        if risk_score <= 3:
            risk_category = "Low Risk"
        elif risk_score <= 6:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        logger.info(f"Risk Score: {risk_score}/10 ({risk_category})")
        
        return {
            'volatility': volatility,
            'beta': beta,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_dd_peak_date': peak_date,
            'max_dd_trough_date': trough_date,
            'var_95': var_95,
            'risk_score': risk_score,
            'risk_category': risk_category
        }


# Testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Risk Calculator")
    print("="*60 + "\n")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    price = 3000
    prices = [price]
    for _ in range(len(dates) - 1):
        change = np.random.randn() * 30  # Random walk
        price = max(price + change, 100)
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    print(f"Sample data: {len(df)} days")
    print(f"Price range: Rs.{df['close'].min():.2f} to Rs.{df['close'].max():.2f}\n")
    
    calc = RiskCalculator()
    
    print("Calculating all risk metrics...\n")
    risks = calc.calculate_all_risks(df)
    
    print("\nRisk Metrics Summary:")
    print(f"  Volatility: {risks['volatility']:.2%}")
    print(f"  Beta: {risks['beta']:.2f}")
    print(f"  Sharpe Ratio: {risks['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {risks['max_drawdown']:.2%}")
    print(f"  VaR (95%): {risks['var_95']:.2%}")
    print(f"\n  Risk Score: {risks['risk_score']}/10")
    print(f"  Risk Category: {risks['risk_category']}")
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)