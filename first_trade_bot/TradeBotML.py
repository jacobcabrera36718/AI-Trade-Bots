import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSentimentAnalyzer:
    """Simplified sentiment analyzer that doesn't require heavy ML models"""
   
    def __init__(self):
        # Simple word-based sentiment scoring
        self.positive_words = {
            'surge', 'soar', 'rally', 'gain', 'rise', 'boost', 'strong', 'bullish',
            'growth', 'profit', 'beat', 'exceed', 'optimistic', 'positive', 'up',
            'high', 'record', 'breakthrough', 'success', 'outperform', 'upgrade'
        }
       
        self.negative_words = {
            'fall', 'drop', 'plunge', 'decline', 'crash', 'bear', 'bearish', 'loss',
            'weak', 'disappointing', 'miss', 'concern', 'worry', 'fear', 'down',
            'low', 'recession', 'crisis', 'downgrade', 'sell-off', 'volatile'
        }
       
        logger.info("Simple sentiment analyzer initialized")
   
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        if not text:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
       
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
       
        total_sentiment_words = positive_count + negative_count
       
        if total_sentiment_words == 0:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
       
        positive_score = positive_count / len(words)
        negative_score = negative_count / len(words)
        neutral_score = max(0, 1 - positive_score - negative_score)
       
        # Normalize scores
        total = positive_score + negative_score + neutral_score
        if total > 0:
            return {
                "positive": positive_score / total,
                "negative": negative_score / total,
                "neutral": neutral_score / total
            }
       
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
   
    def analyze_multiple(self, texts: List[str]) -> Dict[str, float]:
        """Analyze sentiment across multiple texts"""
        if not texts:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
       
        all_scores = [self.analyze_text(text) for text in texts]
       
        # Average the scores
        avg_scores = {
            "positive": np.mean([s["positive"] for s in all_scores]),
            "negative": np.mean([s["negative"] for s in all_scores]),
            "neutral": np.mean([s["neutral"] for s in all_scores])
        }
       
        return avg_scores


# Global analyzer instance
analyzer = SimpleSentimentAnalyzer()


def estimate_sentiment(news: List[str]) -> Tuple[float, str]:
    """
    Estimate sentiment from news headlines
   
    Args:
        news: List of news headlines
       
    Returns:
        Tuple of (confidence, sentiment)
    """
    if not news:
        return 0.5, "neutral"
   
    # Filter and clean news
    clean_news = [headline.strip() for headline in news if headline.strip() and len(headline.strip()) > 5]
   
    if not clean_news:
        return 0.5, "neutral"
   
    # Get sentiment scores
    scores = analyzer.analyze_multiple(clean_news)
   
    # Determine dominant sentiment
    max_sentiment = max(scores.items(), key=lambda x: x[1])
    sentiment = max_sentiment[0]
    confidence = max_sentiment[1]
   
    logger.info(f"Sentiment analysis: {sentiment} ({confidence:.3f}) from {len(clean_news)} headlines")
   
    return float(confidence), sentiment


def get_technical_indicators(symbol: str, days: int = 30) -> Dict[str, float]:
    """
    Get technical indicators with robust error handling
   
    Args:
        symbol: Stock symbol
        days: Number of days of data to fetch
       
    Returns:
        Dictionary of technical indicators
    """
    try:
        # Calculate date range with buffer
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 20)
       
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval="1d")
       
        if data.empty or len(data) < 10:
            logger.warning(f"Insufficient data for {symbol}")
            return {}
       
        indicators = {}
       
        # Current price
        current_price = float(data['Close'].iloc[-1])
        indicators['current_price'] = current_price
       
        # Simple Moving Averages
        if len(data) >= 10:
            sma_10 = data['Close'].rolling(window=10).mean().iloc[-1]
            indicators['sma_10'] = float(sma_10)
            indicators['price_above_sma10'] = current_price > sma_10
       
        if len(data) >= 20:
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_20'] = float(sma_20)
            indicators['price_above_sma20'] = current_price > sma_20
       
        # RSI calculation
        if len(data) >= 15:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
           
            # Avoid division by zero
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
           
            if not rsi.empty:
                indicators['rsi'] = float(rsi.iloc[-1])
                indicators['rsi_oversold'] = indicators['rsi'] < 30
                indicators['rsi_overbought'] = indicators['rsi'] > 70
       
        # Volatility (simplified)
        if len(data) >= 10:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(252) * 100
                indicators['volatility'] = float(volatility)
                indicators['high_volatility'] = volatility > 25
       
        # Price momentum
        if len(data) >= 5:
            price_5d_ago = float(data['Close'].iloc[-6])
            momentum = (current_price - price_5d_ago) / price_5d_ago * 100
            indicators['momentum_5d'] = momentum
            indicators['positive_momentum'] = momentum > 2
       
        # Volume analysis
        if 'Volume' in data.columns and len(data) >= 10:
            current_volume = float(data['Volume'].iloc[-1])
            avg_volume = data['Volume'].rolling(window=10).mean().iloc[-1]
           
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                indicators['volume_ratio'] = float(volume_ratio)
                indicators['high_volume'] = volume_ratio > 1.5
       
        logger.info(f"Technical indicators calculated for {symbol}: {len(indicators)} metrics")
        return indicators
       
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")
        return {}


def get_market_sentiment_signal(symbol: str = "SPY") -> Dict[str, Any]:
    """
    Get overall market sentiment signal combining price action and technical indicators
   
    Args:
        symbol: Market symbol to analyze
       
    Returns:
        Dictionary with market sentiment signal
    """
    try:
        # Get technical indicators
        tech_data = get_technical_indicators(symbol, 30)
       
        if not tech_data:
            return {"signal": "neutral", "confidence": 0.0, "factors": []}
       
        # Score different factors
        bullish_factors = []
        bearish_factors = []
       
        # Price vs moving averages
        if tech_data.get('price_above_sma10', False):
            bullish_factors.append("Price above 10-day MA")
        else:
            bearish_factors.append("Price below 10-day MA")
       
        if tech_data.get('price_above_sma20', False):
            bullish_factors.append("Price above 20-day MA")
        else:
            bearish_factors.append("Price below 20-day MA")
       
        # RSI analysis
        if 'rsi' in tech_data:
            rsi = tech_data['rsi']
            if rsi > 50:
                bullish_factors.append(f"RSI bullish ({rsi:.1f})")
            elif rsi < 50:
                bearish_factors.append(f"RSI bearish ({rsi:.1f})")
           
            if tech_data.get('rsi_oversold', False):
                bullish_factors.append("RSI oversold (potential bounce)")
            elif tech_data.get('rsi_overbought', False):
                bearish_factors.append("RSI overbought (potential pullback)")
       
        # Momentum
        if tech_data.get('positive_momentum', False):
            bullish_factors.append(f"Positive momentum ({tech_data.get('momentum_5d', 0):.1f}%)")
        elif tech_data.get('momentum_5d', 0) < -2:
            bearish_factors.append(f"Negative momentum ({tech_data.get('momentum_5d', 0):.1f}%)")
       
        # Volume confirmation
        if tech_data.get('high_volume', False):
            bullish_factors.append("High volume confirmation")
       
        # Determine overall signal
        bullish_score = len(bullish_factors)
        bearish_score = len(bearish_factors)
        total_factors = bullish_score + bearish_score
       
        if total_factors == 0:
            signal = "neutral"
            confidence = 0.0
        elif bullish_score > bearish_score:
            signal = "bullish"
            confidence = bullish_score / total_factors
        elif bearish_score > bullish_score:
            signal = "bearish"
            confidence = bearish_score / total_factors
        else:
            signal = "neutral"
            confidence = 0.5
       
        return {
            "signal": signal,
            "confidence": confidence,
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "technical_data": tech_data
        }
       
    except Exception as e:
        logger.error(f"Error getting market sentiment signal: {e}")
        return {"signal": "neutral", "confidence": 0.0, "factors": []}


def simulate_news_sentiment(symbol: str, current_price: float, recent_performance: float) -> Tuple[float, str]:
    """
    Simulate news sentiment based on recent price performance
    This is used when actual news data is not available
   
    Args:
        symbol: Stock symbol
        current_price: Current stock price
        recent_performance: Recent return percentage
       
    Returns:
        Tuple of (confidence, sentiment)
    """
    try:
        # Create sentiment based on recent performance with some randomness
        base_sentiment = 0.5
       
        # Strong positive performance
        if recent_performance > 3:
            base_sentiment = 0.75 + np.random.normal(0, 0.1)
            sentiment_label = "positive"
        # Strong negative performance  
        elif recent_performance < -3:
            base_sentiment = 0.75 + np.random.normal(0, 0.1)
            sentiment_label = "negative"
        # Moderate positive
        elif recent_performance > 1:
            base_sentiment = 0.6 + np.random.normal(0, 0.15)
            sentiment_label = "positive"
        # Moderate negative
        elif recent_performance < -1:
            base_sentiment = 0.6 + np.random.normal(0, 0.15)
            sentiment_label = "negative"
        # Neutral
        else:
            base_sentiment = 0.4 + np.random.normal(0, 0.2)
            sentiment_label = "neutral"
       
        # Clamp confidence between 0.3 and 0.9
        confidence = np.clip(base_sentiment, 0.3, 0.9)
       
        logger.info(f"Simulated sentiment for {symbol}: {sentiment_label} ({confidence:.3f}) "
                   f"based on {recent_performance:.1f}% performance")
       
        return float(confidence), sentiment_label
       
    except Exception as e:
        logger.error(f"Error simulating news sentiment: {e}")
        return 0.5, "neutral"


if __name__ == "__main__":
    print("Testing Enhanced ML Trading Components")
    print("=" * 50)
   
    # Test simple sentiment analysis
    test_headlines = [
        "Stock market rallies on strong earnings reports",
        "Tech sector faces headwinds amid regulatory concerns",
        "Economic data shows resilient growth momentum",
        "Federal Reserve signals cautious approach to policy"
    ]
   
    confidence, sentiment = estimate_sentiment(test_headlines)
    print(f"Sentiment Analysis: {sentiment} (confidence: {confidence:.3f})")
   
    # Test technical indicators
    print(f"\nTechnical Analysis for SPY:")
    tech_indicators = get_technical_indicators("SPY")
    for key, value in tech_indicators.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
   
    # Test market sentiment signal
    print(f"\nMarket Sentiment Signal:")
    market_signal = get_market_sentiment_signal("SPY")
    print(f"Signal: {market_signal['signal']} (confidence: {market_signal['confidence']:.3f})")
    print(f"Bullish factors: {market_signal['bullish_factors']}")
    print(f"Bearish factors: {market_signal['bearish_factors']}")
   
    # Test simulated sentiment
    print(f"\nSimulated News Sentiment:")
    sim_conf, sim_sent = simulate_news_sentiment("SPY", 450.0, 2.5)
    print(f"Simulated: {sim_sent} ({sim_conf:.3f})")

