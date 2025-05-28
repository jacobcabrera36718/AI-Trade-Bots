import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_stock_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get stock price data from Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)
       
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
       
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
       
        return data
       
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_momentum(symbol: str, days: int = 5) -> float:
    """Calculate price momentum over specified days"""
    try:
        data = get_stock_data(symbol, days + 5)
       
        if len(data) < days:
            return 0.0
       
        current_price = data['Close'].iloc[-1]
        past_price = data['Close'].iloc[-days-1]
       
        momentum = ((current_price - past_price) / past_price) * 100
        return float(momentum)
       
    except Exception as e:
        logger.error(f"Error calculating momentum for {symbol}: {e}")
        return 0.0


def calculate_rsi(symbol: str, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    try:
        data = get_stock_data(symbol, period + 10)
       
        if len(data) < period:
            return 50.0  # Neutral RSI
       
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
       
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
       
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
       
    except Exception as e:
        logger.error(f"Error calculating RSI for {symbol}: {e}")
        return 50.0


def calculate_moving_averages(symbol: str) -> Dict[str, float]:
    """Calculate moving averages"""
    try:
        data = get_stock_data(symbol, 50)
       
        if len(data) < 20:
            return {}
       
        current_price = data['Close'].iloc[-1]
        sma_10 = data['Close'].rolling(window=10).mean().iloc[-1] if len(data) >= 10 else current_price
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_price
       
        return {
            'current_price': float(current_price),
            'sma_10': float(sma_10),
            'sma_20': float(sma_20),
            'above_sma_10': current_price > sma_10,
            'above_sma_20': current_price > sma_20
        }
       
    except Exception as e:
        logger.error(f"Error calculating moving averages for {symbol}: {e}")
        return {}


def get_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Get all technical indicators for a symbol"""
    try:
        momentum = calculate_momentum(symbol, 5)
        rsi = calculate_rsi(symbol)
        ma_data = calculate_moving_averages(symbol)
       
        # Calculate volatility
        data = get_stock_data(symbol, 20)
        volatility = 0.0
        if len(data) >= 10:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
       
        indicators = {
            'momentum_5d': momentum,
            'rsi': rsi,
            'volatility': float(volatility),
            'rsi_oversold': rsi < 30,
            'rsi_overbought': rsi > 70,
            'positive_momentum': momentum > 1,
            'negative_momentum': momentum < -1
        }
       
        # Add moving average data
        indicators.update(ma_data)
       
        logger.info(f"Technical indicators for {symbol}: RSI={rsi:.1f}, Momentum={momentum:.1f}%")
        return indicators
       
    except Exception as e:
        logger.error(f"Error getting technical indicators for {symbol}: {e}")
        return {}


def generate_sentiment_signal(symbol: str, momentum: float) -> Tuple[float, str]:
    """Generate sentiment signal based on price action - AGGRESSIVE VERSION"""
    try:
        # MUCH MORE AGGRESSIVE - Generate high confidence more often
        if momentum < -1:  # Any decent drop
            confidence = np.random.uniform(0.7, 0.95)  # High confidence
            sentiment = "positive"  # Contrarian - buy the dip
           
        elif momentum < -0.5:  # Small drop
            confidence = np.random.uniform(0.6, 0.8)
            sentiment = "positive"
           
        elif momentum < 0:  # Any decline at all
            confidence = np.random.uniform(0.5, 0.7)
            sentiment = "positive"
           
        elif momentum > 2:  # Major gain - still buy (momentum)
            confidence = np.random.uniform(0.5, 0.7)
            sentiment = "positive"  # Changed from negative
           
        else:  # Neutral/small moves - still bullish
            confidence = np.random.uniform(0.4, 0.6)
            sentiment = "positive"  # Always favor buying
       
        logger.info(f"AGGRESSIVE Sentiment for {symbol}: {sentiment} ({confidence:.2f}) on {momentum:.1f}% momentum")
        return float(confidence), sentiment
       
    except Exception as e:
        logger.error(f"Error generating sentiment for {symbol}: {e}")
        return 0.7, "positive"  # Default to strong buy


def get_market_sentiment_signal(symbol: str) -> Dict[str, Any]:
    """Get overall market sentiment signal"""
    try:
        indicators = get_technical_indicators(symbol)
        momentum = indicators.get('momentum_5d', 0)
        rsi = indicators.get('rsi', 50)
       
        bullish_factors = []
        bearish_factors = []
       
        # Analyze factors
        if momentum < -1:
            bullish_factors.append(f"Dip buying opportunity ({momentum:.1f}%)")
        elif momentum > 2:
            bearish_factors.append(f"Overbought momentum ({momentum:.1f}%)")
       
        if rsi < 35:
            bullish_factors.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:
            bearish_factors.append(f"RSI overbought ({rsi:.1f})")
       
        if not indicators.get('above_sma_10', True):
            bullish_factors.append("Price below SMA10")
       
        if not indicators.get('above_sma_20', True):
            bullish_factors.append("Price below SMA20")
       
        # Determine signal
        bullish_score = len(bullish_factors)
        bearish_score = len(bearish_factors)
       
        if bullish_score > bearish_score:
            signal = "bullish"
            confidence = bullish_score / (bullish_score + bearish_score + 1)
        elif bearish_score > bullish_score:
            signal = "bearish"
            confidence = bearish_score / (bullish_score + bearish_score + 1)
        else:
            signal = "neutral"
            confidence = 0.5
       
        return {
            "signal": signal,
            "confidence": confidence,
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "technical_data": indicators
        }
       
    except Exception as e:
        logger.error(f"Error getting market sentiment for {symbol}: {e}")
        return {"signal": "neutral", "confidence": 0.5, "bullish_factors": [], "bearish_factors": []}


# Legacy function for compatibility
def estimate_sentiment(news_list) -> Tuple[float, str]:
    """Legacy sentiment function - returns random sentiment for compatibility"""
    confidence = np.random.uniform(0.4, 0.7)
    sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3])
    return confidence, sentiment


if __name__ == "__main__":
    # Test the functions
    symbol = "SPY"
    print(f"Testing ML functions for {symbol}")
    print("=" * 40)
   
    # Test technical indicators
    indicators = get_technical_indicators(symbol)
    print(f"Technical Indicators: {indicators}")
   
    # Test market sentiment
    sentiment = get_market_sentiment_signal(symbol)
    print(f"Market Sentiment: {sentiment}")
   
    # Test momentum
    momentum = calculate_momentum(symbol)
    print(f"5-day Momentum: {momentum:.2f}%")
   
    print(" All tests completed!")
 
