"""
Stock Price Prediction Model
============================

Predicts future stock prices using historical data and technical indicators.

Models Used:
1. Linear Regression - Simple, fast, interpretable
2. Random Forest - Better for non-linear patterns
3. Prophet - Facebook's time series forecasting

Features:
- Historical prices (OHLC)
- Technical indicators (MA, RSI, MACD)
- Volume data
- Trend features

Usage:
    from ml_models.price_predictor import PricePredictor
    
    predictor = PricePredictor()
    predictor.train(df)
    predictions = predictor.predict(days=30)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class PricePredictor:
    """
    Stock price prediction using machine learning.
    
    Uses multiple features:
    - Moving averages
    - RSI, MACD
    - Volume
    - Previous day prices
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize price predictor.
        
        Args:
            model_type: 'linear' or 'random_forest'
        """
        self.model_type = model_type
        
        if model_type == 'linear':
            self.model = LinearRegression()
            logger.info("Initialized Linear Regression model")
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            logger.info("Initialized Random Forest model")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
        self.feature_columns = []
        self.scaler_mean = None
        self.scaler_std = None
    
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction.
        
        Features created:
        - Lagged prices (previous 1, 2, 3 days)
        - Moving averages (20, 50)
        - RSI
        - MACD
        - Volume ratio
        - Day of week
        
        Args:
            df: DataFrame with OHLCV and indicators
            
        Returns:
            DataFrame with features
        """
        df_features = df.copy()
        
        # Lagged features (previous day prices)
        df_features['close_lag_1'] = df_features['close'].shift(1)
        df_features['close_lag_2'] = df_features['close'].shift(2)
        df_features['close_lag_3'] = df_features['close'].shift(3)
        
        # Price changes
        df_features['price_change_1d'] = df_features['close'].pct_change(1)
        df_features['price_change_3d'] = df_features['close'].pct_change(3)
        
        # Volume change
        df_features['volume_change'] = df_features['volume'].pct_change(1)
        
        # Moving average ratios
        if 'ma_20' in df_features.columns and 'ma_50' in df_features.columns:
            df_features['ma_ratio'] = df_features['ma_20'] / df_features['ma_50']
        
        # Target: Next day's close price
        df_features['target'] = df_features['close'].shift(-1)
        
        # Remove rows with NaN values
        df_features = df_features.dropna()
        
        logger.info(f"Prepared {len(df_features)} samples with features")
        
        return df_features
    
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the prediction model.
        
        Process:
        1. Prepare features
        2. Split train/test
        3. Train model
        4. Evaluate performance
        
        Args:
            df: DataFrame with OHLCV and indicators
            test_size: Fraction of data for testing
            
        Returns:
            Dict with training metrics
            
        Example:
            predictor = PricePredictor()
            metrics = predictor.train(df)
            print(f"R2 Score: {metrics['r2_score']:.3f}")
        """
        logger.info(f"Training {self.model_type} model")
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        # Select feature columns
        self.feature_columns = [
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'price_change_1d', 'price_change_3d',
            'volume_change'
        ]
        
        # Add technical indicators if available
        optional_features = ['ma_20', 'ma_50', 'rsi_14', 'macd', 'volume_ratio', 'ma_ratio']
        for feat in optional_features:
            if feat in df_prepared.columns:
                self.feature_columns.append(feat)
        
        # Prepare X and y
        X = df_prepared[self.feature_columns]
        y = df_prepared['target']
        
        # Normalize features (important for linear regression)
        self.scaler_mean = X.mean()
        self.scaler_std = X.std()
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False  # Don't shuffle time series!
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        self.is_trained = True
        
        logger.info("Training completed!")
        logger.info(f"  Test R2 Score: {metrics['test_r2']:.3f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        logger.info(f"  Test MAE: {metrics['test_mae']:.2f}")
        
        return metrics
    
    
    def predict_next_day(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict next day's closing price.
        
        Args:
            df: DataFrame with recent data (at least last 3 days)
            
        Returns:
            Tuple: (predicted_price, confidence_interval)
            
        Example:
            price, confidence = predictor.predict_next_day(df)
            print(f"Tomorrow's predicted price: Rs.{price:.2f} +/- {confidence:.2f}")
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        # Get latest data point
        latest = df_prepared.iloc[-1]
        
        # Extract features
        X = latest[self.feature_columns].values.reshape(1, -1)
        
        # Normalize
        X_scaled = (X - self.scaler_mean.values) / self.scaler_std.values
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Calculate confidence interval (simple: +/- 2% for now)
        confidence = prediction * 0.02
        
        logger.info(f"Predicted next day price: Rs.{prediction:.2f} +/- {confidence:.2f}")
        
        return prediction, confidence
    
    
    def predict_multiple_days(self, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Predict prices for multiple days ahead.
        
        Note: Accuracy decreases for longer predictions.
        
        Args:
            df: DataFrame with historical data
            days: Number of days to predict
            
        Returns:
            DataFrame with predictions
            
        Example:
            predictions = predictor.predict_multiple_days(df, days=30)
            print(predictions[['date', 'predicted_price']])
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        logger.info(f"Predicting {days} days ahead")
        
        df_work = df.copy()
        predictions = []
        
        for day in range(days):
            # Predict next day
            pred_price, confidence = self.predict_next_day(df_work)
            
            # Create prediction date
            last_date = df_work['date'].max() if 'date' in df_work.columns else df_work['timestamp'].max()
            pred_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
            
            predictions.append({
                'date': pred_date,
                'predicted_price': pred_price,
                'confidence_lower': pred_price - confidence,
                'confidence_upper': pred_price + confidence,
                'day_ahead': day + 1
            })
            
            # Append prediction to dataset for next iteration
            new_row = df_work.iloc[-1].copy()
            date_col = 'date' if 'date' in df_work.columns else 'timestamp'
            new_row[date_col] = pred_date
            new_row['close'] = pred_price
            
            # Append (using concat instead of append)
            df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)
        
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"Predictions complete. Range: Rs.{predictions_df['predicted_price'].min():.2f} to Rs.{predictions_df['predicted_price'].max():.2f}")
        
        return predictions_df
    
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (only for Random Forest).
        
        Returns:
            Dict with feature importance scores
        """
        if self.model_type != 'random_forest':
            logger.warning("Feature importance only available for Random Forest")
            return {}
        
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        importances = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("Feature importance:")
        for feat, imp in importances.items():
            logger.info(f"  {feat}: {imp:.3f}")
        
        return importances


# Example usage and testing
if __name__ == "__main__":
    """
    Test price predictor
    """
    print("\n" + "="*60)
    print("Testing Price Predictor")
    print("="*60 + "\n")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate stock prices (random walk with trend)
    price = 3000
    prices = [price]
    for _ in range(len(dates) - 1):
        change = np.random.randn() * 20 + 2  # Slight upward trend
        price = max(price + change, 100)
        prices.append(price)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'ma_20': pd.Series(prices).rolling(20).mean(),
        'ma_50': pd.Series(prices).rolling(50).mean(),
        'rsi_14': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.randn(len(dates)) * 10,
        'volume_ratio': np.random.uniform(0.8, 1.2, len(dates))
    })
    
    print(f"Created sample data: {len(df)} days")
    print(f"Price range: Rs.{df['close'].min():.2f} to Rs.{df['close'].max():.2f}\n")
    
    # Test 1: Train Linear Regression
    print("Test 1: Training Linear Regression model...")
    predictor_lr = PricePredictor(model_type='linear')
    metrics_lr = predictor_lr.train(df, test_size=0.2)
    print(f"Linear Regression - R2 Score: {metrics_lr['test_r2']:.3f}\n")
    
    # Test 2: Train Random Forest
    print("Test 2: Training Random Forest model...")
    predictor_rf = PricePredictor(model_type='random_forest')
    metrics_rf = predictor_rf.train(df, test_size=0.2)
    print(f"Random Forest - R2 Score: {metrics_rf['test_r2']:.3f}\n")
    
    # Test 3: Predict next day
    print("Test 3: Predicting next day...")
    pred_price, confidence = predictor_lr.predict_next_day(df)
    print(f"Predicted price: Rs.{pred_price:.2f} +/- {confidence:.2f}\n")
    
    # Test 4: Predict 30 days
    print("Test 4: Predicting 30 days ahead...")
    predictions = predictor_lr.predict_multiple_days(df, days=30)
    print(f"30-day predictions:")
    print(predictions.head())
    print(f"\nPredicted price range: Rs.{predictions['predicted_price'].min():.2f} to Rs.{predictions['predicted_price'].max():.2f}\n")
    
    # Test 5: Feature importance (Random Forest only)
    print("Test 5: Feature importance (Random Forest)...")
    importance = predictor_rf.get_feature_importance()
    print("\nTop 3 important features:")
    for i, (feat, imp) in enumerate(list(importance.items())[:3], 1):
        print(f"  {i}. {feat}: {imp:.3f}")
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)