
"""
News Sentiment Analysis
======================

Analyzes sentiment of news headlines to predict market sentiment.

Sentiment Score:
- Positive (+1.0 to 0.0) = Good news, stock likely to rise
- Negative (0.0 to -1.0) = Bad news, stock likely to fall
- Neutral (close to 0) = No strong opinion

Uses TextBlob for sentiment analysis (simple and effective).

Usage:
    from ml_models.sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    score = analyzer.analyze_text("TCS wins major contract")
    # Returns: 0.85 (very positive)
"""

from textblob import TextBlob
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis for stock news.
    
    Uses TextBlob for polarity detection.
    Polarity: -1 (most negative) to +1 (most positive)
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        logger.info("SentimentAnalyzer initialized")
        
        # Keywords that indicate strong sentiment
        self.positive_keywords = [
            'profit', 'growth', 'gain', 'rise', 'win', 'surge', 'beat',
            'strong', 'record', 'high', 'success', 'positive', 'up'
        ]
        
        self.negative_keywords = [
            'loss', 'decline', 'fall', 'drop', 'crash', 'miss', 'weak',
            'concern', 'risk', 'low', 'negative', 'down', 'fail'
        ]
    
    
    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: News headline or article text
            
        Returns:
            Tuple: (sentiment_score, sentiment_label)
            
        Sentiment Score:
        - > 0.3: Very Positive
        - 0.1 to 0.3: Positive
        - -0.1 to 0.1: Neutral
        - -0.3 to -0.1: Negative
        - < -0.3: Very Negative
        
        Example:
            score, label = analyzer.analyze_text("TCS wins $2B contract")
            # Returns: (0.7, "Very Positive")
        """
        if not text or len(text.strip()) == 0:
            return 0.0, "Neutral"
        
        # Use TextBlob for sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Adjust score based on keywords
        text_lower = text.lower()
        
        # Check for positive keywords
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        
        # Check for negative keywords
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        # Adjust polarity
        keyword_adjustment = (positive_count - negative_count) * 0.1
        adjusted_polarity = polarity + keyword_adjustment
        
        # Clamp to [-1, 1]
        adjusted_polarity = max(-1.0, min(1.0, adjusted_polarity))
        
        # Classify
        if adjusted_polarity > 0.3:
            label = "Very Positive"
        elif adjusted_polarity > 0.1:
            label = "Positive"
        elif adjusted_polarity > -0.1:
            label = "Neutral"
        elif adjusted_polarity > -0.3:
            label = "Negative"
        else:
            label = "Very Negative"
        
        return adjusted_polarity, label
    
    
    def analyze_multiple(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of news headlines
            
        Returns:
            DataFrame with sentiment scores
            
        Example:
            headlines = [
                "TCS wins major contract",
                "Stock market crashes",
                "Quarterly results announced"
            ]
            df = analyzer.analyze_multiple(headlines)
        """
        results = []
        
        for text in texts:
            score, label = self.analyze_text(text)
            results.append({
                'text': text,
                'sentiment_score': score,
                'sentiment_label': label
            })
        
        df = pd.DataFrame(results)
        
        logger.info(f"Analyzed {len(texts)} texts")
        logger.info(f"Average sentiment: {df['sentiment_score'].mean():.3f}")
        
        return df
    
    
    def get_overall_sentiment(self, texts: List[str]) -> Dict:
        """
        Get overall sentiment from multiple texts.
        
        Args:
            texts: List of news headlines
            
        Returns:
            Dict with overall metrics
            
        Example:
            sentiment = analyzer.get_overall_sentiment(headlines)
            print(f"Overall: {sentiment['overall_label']}")
        """
        df = self.analyze_multiple(texts)
        
        avg_score = df['sentiment_score'].mean()
        
        # Count by label
        label_counts = df['sentiment_label'].value_counts().to_dict()
        
        # Overall label
        if avg_score > 0.1:
            overall_label = "Positive"
        elif avg_score < -0.1:
            overall_label = "Negative"
        else:
            overall_label = "Neutral"
        
        return {
            'average_score': avg_score,
            'overall_label': overall_label,
            'total_articles': len(texts),
            'label_distribution': label_counts,
            'positive_ratio': len(df[df['sentiment_score'] > 0]) / len(df) if len(df) > 0 else 0,
            'negative_ratio': len(df[df['sentiment_score'] < 0]) / len(df) if len(df) > 0 else 0
        }


# Testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Sentiment Analyzer")
    print("="*60 + "\n")
    
    analyzer = SentimentAnalyzer()
    
    # Test cases
    test_headlines = [
        "TCS wins $2 billion contract from UK bank",
        "Stock market crashes amid global concerns",
        "Quarterly results announced",
        "Company reports record profits and strong growth",
        "CEO resigns following poor performance",
        "Shares surge 10% on positive earnings beat",
        "Analyst downgrades stock to sell on weak outlook"
    ]
    
    print("Test 1: Individual sentiment analysis...")
    for headline in test_headlines[:3]:
        score, label = analyzer.analyze_text(headline)
        print(f"\nHeadline: {headline}")
        print(f"Score: {score:.3f} ({label})")
    
    print("\n\nTest 2: Batch analysis...")
    df = analyzer.analyze_multiple(test_headlines)
    print(df[['text', 'sentiment_score', 'sentiment_label']])
    
    print("\n\nTest 3: Overall sentiment...")
    overall = analyzer.get_overall_sentiment(test_headlines)
    print(f"Overall sentiment: {overall['overall_label']}")
    print(f"Average score: {overall['average_score']:.3f}")
    print(f"Positive ratio: {overall['positive_ratio']:.1%}")
    print(f"Negative ratio: {overall['negative_ratio']:.1%}")
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)