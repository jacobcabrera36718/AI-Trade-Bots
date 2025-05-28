import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


# Import trading framework components
try:
    from lumibot.brokers import Alpaca
    from lumibot.backtesting import YahooDataBacktesting
    from lumibot.strategies.strategy import Strategy
    from lumibot.traders import Trader
    LUMIBOT_AVAILABLE = True
except ImportError:
    print("Lumibot not installed. Install with: pip install lumibot")
    LUMIBOT_AVAILABLE = False


try:
    from alpaca_trade_api import REST
    ALPACA_API_AVAILABLE = True
except ImportError:
    print("Alpaca API not installed. Install with: pip install alpaca-trade-api")
    ALPACA_API_AVAILABLE = False


# Import our ML components
try:
    from TradeBotML import (
        estimate_sentiment,
        get_technical_indicators,
        get_market_sentiment_signal,
        simulate_news_sentiment
    )
    ML_COMPONENTS_AVAILABLE = True
except ImportError:
    print("TradeBotML.py not found. Make sure it's in the same directory.")
    ML_COMPONENTS_AVAILABLE = False


# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# Alpaca API Configuration
API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_ALPACA_API_KEY_HERE')
API_SECRET = os.getenv('ALPACA_API_SECRET', 'YOUR_ALPACA_SECRET_KEY_HERE')
BASE_URL = "https://paper-api.alpaca.markets/v2"  # Paper trading
# For live trading: BASE_URL = "https://api.alpaca.markets/v2"


# Trading Parameters
DEFAULT_SYMBOL = "SPY"
DEFAULT_CASH_AT_RISK = 0.3  # 30% of available cash per trade
DEFAULT_SENTIMENT_THRESHOLD = 0.65  # Minimum confidence for trades
DEFAULT_ENABLE_SHORT = False  # Enable short selling


# Risk Management - Buy & Hold Focused
MAX_CONSECUTIVE_LOSSES = 8  # Allow more losses since we're holding longer
MAX_POSITION_SIZE = 0.9  # Allow large positions since we're not selling much
STOP_LOSS_PERCENT = 0.08  # Only sell on big drops (8% loss)
TAKE_PROFIT_PERCENT = 0.25  # Hold for bigger gains (25% profit)


# Validate configuration
if API_KEY == 'YOUR_ALPACA_API_KEY_HERE' or API_SECRET == 'YOUR_ALPACA_SECRET_KEY_HERE':
    logger.warning("Using dummy API credentials - only backtesting will work!")
    API_KEY = "dummy_key"
    API_SECRET = "dummy_secret"


ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True  # Set to False for live trading
}


class EnhancedMLTrader(Strategy):
    """Enhanced ML Trading Strategy with improved reliability and error handling"""
   
    def initialize(self, **kwargs):
        """Initialize the trading strategy with parameters"""
       
        # Strategy parameters
        self.symbol = kwargs.get('symbol', DEFAULT_SYMBOL)
        self.cash_at_risk = kwargs.get('cash_at_risk', DEFAULT_CASH_AT_RISK)
        self.sentiment_threshold = kwargs.get('sentiment_threshold', DEFAULT_SENTIMENT_THRESHOLD)
        self.enable_short = kwargs.get('enable_short', DEFAULT_ENABLE_SHORT)
       
        # Trading frequency
        self.sleeptime = "1D"  # Daily trading
       
        # Risk management
        self.consecutive_losses = 0
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        self.stop_loss_percent = STOP_LOSS_PERCENT
        self.take_profit_percent = TAKE_PROFIT_PERCENT
       
        # Performance tracking
        self.trade_history = []
        self.daily_pnl = []
        self.last_trade_date = None
        self.last_trade_type = None
       
        # Initialize API connection for live trading
        self.api = None
        if API_KEY != "dummy_key" and ALPACA_API_AVAILABLE:
            self._initialize_api()
       
        logger.info(f" Enhanced ML Trader initialized")
        logger.info(f"   Symbol: {self.symbol}")
        logger.info(f"   Cash at risk: {self.cash_at_risk*100}%")
        logger.info(f"   Sentiment threshold: {self.sentiment_threshold*100}%")
        logger.info(f"   Short selling: {'Enabled' if self.enable_short else 'Disabled'}")


    def _initialize_api(self):
        """Initialize Alpaca API connection"""
        try:
            # Fix the base URL format
            clean_base_url = BASE_URL.rstrip('/v2') + '/v2' if not BASE_URL.endswith('/v2') else BASE_URL
            self.api = REST(base_url=clean_base_url, key_id=API_KEY, secret_key=API_SECRET)
            account = self.api.get_account()
            logger.info(f" API connected. Account: {account.status}")
        except Exception as e:
            logger.warning(f" API connection failed (OK for backtesting): {e}")
            self.api = None


    def get_position_info(self) -> Dict[str, Any]:
        """Get detailed position information"""
        try:
            position = self.get_position(self.symbol)
           
            if position is None:
                return {
                    "quantity": 0,
                    "market_value": 0,
                    "avg_cost": 0,
                    "unrealized_pnl": 0,
                    "side": "none"
                }
           
            return {
                "quantity": abs(position.quantity),
                "market_value": abs(position.quantity * self.get_last_price(self.symbol)),
                "avg_cost": position.avg_cost if hasattr(position, 'avg_cost') else 0,
                "unrealized_pnl": position.unrealized_pnl if hasattr(position, 'unrealized_pnl') else 0,
                "side": "long" if position.quantity > 0 else "short" if position.quantity < 0 else "none"
            }
           
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return {"quantity": 0, "market_value": 0, "avg_cost": 0, "unrealized_pnl": 0, "side": "none"}


    def calculate_position_size(self) -> int:
        """Calculate position size based on available cash and risk parameters"""
        try:
            cash = self.get_cash()
            last_price = self.get_last_price(self.symbol)
           
            if cash <= 0 or last_price <= 0:
                logger.warning(f"Invalid cash ({cash}) or price ({last_price})")
                return 0
           
            # Calculate maximum position value
            max_position_value = cash * self.cash_at_risk
           
            # Calculate quantity
            quantity = int(max_position_value / last_price)
           
            # Ensure minimum position size
            if quantity == 0 and cash >= last_price:
                quantity = 1
           
            logger.info(f"Position sizing: Cash=${cash:.2f}, Price=${last_price:.2f}, Quantity={quantity}")
            return quantity
           
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0


    def get_trading_signal(self) -> Dict[str, Any]:
        """Get comprehensive trading signal combining multiple factors"""
        try:
            current_datetime = self.get_datetime()
           
            # Get market sentiment signal
            if ML_COMPONENTS_AVAILABLE:
                market_signal = get_market_sentiment_signal(self.symbol)
                technical_data = get_technical_indicators(self.symbol)
            else:
                logger.warning("ML components not available, using simple price-based signal")
                market_signal = {"signal": "neutral", "confidence": 0.5}
                technical_data = {}
           
            # Get simulated news sentiment based on recent performance
            if technical_data.get('momentum_5d') is not None:
                recent_performance = technical_data['momentum_5d']
            else:
                # Calculate simple momentum
                try:
                    historical = self.get_historical_prices(self.symbol, 5, "day")
                    if historical and len(historical.df) >= 2:
                        recent_performance = ((historical.df['close'].iloc[-1] - historical.df['close'].iloc[0])
                                            / historical.df['close'].iloc[0]) * 100
                    else:
                        recent_performance = 0
                except:
                    recent_performance = 0
           
            # Simulate news sentiment
            if ML_COMPONENTS_AVAILABLE:
                news_conf, news_sentiment = simulate_news_sentiment(
                    self.symbol,
                    self.get_last_price(self.symbol),
                    recent_performance
                )
            else:
                news_conf, news_sentiment = 0.5, "neutral"
           
            # Combine signals
            combined_signal = self._combine_signals(market_signal, news_sentiment, news_conf, technical_data)
           
            logger.info(f"Trading signal: {combined_signal['action']} (confidence: {combined_signal['confidence']:.3f})")
           
            return combined_signal
           
        except Exception as e:
            logger.error(f"Error getting trading signal: {e}")
            return {"action": "hold", "confidence": 0.0, "reason": f"Error: {e}"}


    def _combine_signals(self, market_signal: Dict, news_sentiment: str, news_conf: float,
                        technical_data: Dict) -> Dict[str, Any]:
        """AGGRESSIVE BUYING Strategy - Buy everything, especially dips!"""
       
        signals_bullish = 0
        reasons = []
       
        # Get momentum - KEY for dip buying
        momentum = technical_data.get('momentum_5d', 0)
       
        # **EXTREME DIP BUYING - Main Strategy**
        if momentum < 0:  # ANY decline = buy signal
            buy_strength = abs(momentum) * 2.0  # Double the decline strength
            signals_bullish += buy_strength
            reasons.append(f"AGGRESSIVE DIP BUY: {momentum:.1f}% decline")
       
        if momentum < -1:  # Bigger dip = much stronger buy
            signals_bullish += 3.0
            reasons.append(f"BIG DIP - STRONG BUY: {momentum:.1f}%")
           
        if momentum < -2:  # Major dip = maximum buy
            signals_bullish += 5.0
            reasons.append(f"MAJOR DIP - MAX BUY: {momentum:.1f}%")
       
        # **ALWAYS BUY BIAS** - Even without dips
        base_aggressive_bias = 2.5  # Very strong base buying
        signals_bullish += base_aggressive_bias
        reasons.append(f"AGGRESSIVE BUY BIAS ({base_aggressive_bias})")
       
        # RSI - ANY level below 50 = buy opportunity
        rsi = technical_data.get('rsi', 50)
        if rsi < 50:
            rsi_buy_signal = (50 - rsi) / 10  # Scale buy strength
            signals_bullish += rsi_buy_signal
            reasons.append(f"RSI buy signal ({rsi:.1f})")
       
        if rsi < 40:  # Extra buying on oversold
            signals_bullish += 2.0
            reasons.append(f"RSI oversold - extra buy ({rsi:.1f})")
       
        # Price below moving averages = STRONG buy signals
        if not technical_data.get('price_above_sma10', True):
            signals_bullish += 2.5
            reasons.append("Below SMA10 - STRONG BUY")
           
        if not technical_data.get('price_above_sma20', True):
            signals_bullish += 3.0
            reasons.append("Below SMA20 - MAX BUY")
       
        # Market sentiment - CONTRARIAN buying
        if market_signal.get('signal') == 'bearish':
            signals_bullish += market_signal.get('confidence', 0) * 3.0
            reasons.append(f"CONTRARIAN: Bearish market = BUY")
        else:
            signals_bullish += market_signal.get('confidence', 0) * 1.5
            reasons.append(f"Market sentiment buy")
       
        # News sentiment - BUY on bad news, BUY on good news
        if news_sentiment == 'negative':
            signals_bullish += news_conf * 2.5  # Strong contrarian buying
            reasons.append(f"BAD NEWS = BUY OPPORTUNITY")
        else:
            signals_bullish += news_conf * 1.5  # Still buy on good news
            reasons.append(f"News-based buying")
       
        # High volatility = MORE buying opportunities
        if technical_data.get('volatility', 0) > 20:
            signals_bullish += 2.0
            reasons.append("High volatility - BUY MORE")
       
        # MEGA RANDOM BUYING BIAS - Force buying
        mega_buy_bias = np.random.uniform(2.0, 4.0)
        signals_bullish += mega_buy_bias
        reasons.append(f"MEGA BUY BIAS ({mega_buy_bias:.1f})")
       
        # Calculate confidence - Always high for buying
        confidence = min(signals_bullish / 8.0, 0.95)  # Scale to reasonable confidence
        confidence = max(confidence, 0.5)  # Minimum 50% confidence
       
        # DECISION: Almost always BUY
        if signals_bullish >= 2.0:  # Very low bar
            action = "buy"
        else:
            # Even low signals = buy 80% of the time
            if np.random.random() < 0.8:
                action = "buy"
                confidence = max(confidence, 0.6)
            else:
                action = "hold"
       
        # OVERRIDE: Always buy if any decline detected
        if momentum < -0.1:  # Even tiny dips
            action = "buy"
            confidence = max(confidence, 0.8)
            reasons.append("FORCED BUY on any decline!")
       
        # FINAL OVERRIDE: If no clear signal, default to BUY
        if len(reasons) < 3:
            action = "buy"
            confidence = 0.7
            reasons.append("DEFAULT AGGRESSIVE BUY!")
       
        return {
            "action": action,
            "confidence": confidence,
            "reason": f"AGGRESSIVE BUYING: {'; '.join(reasons)}",
            "bullish_score": signals_bullish,
            "bearish_score": 0  # Never sell
        }


    def check_risk_management(self) -> bool:
        """Check if risk management conditions allow trading"""
       
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Too many consecutive losses ({self.consecutive_losses}), pausing trading")
            return False
       
        # Check if we're already at max position size
        position_info = self.get_position_info()
        portfolio_value = self.get_portfolio_value()
       
        if portfolio_value > 0:
            position_percent = position_info["market_value"] / portfolio_value
            if position_percent >= MAX_POSITION_SIZE:
                logger.info(f"Already at max position size ({position_percent:.1%})")
                return False
       
        return True


    def execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Execute trade based on signal"""
        try:
            action = signal["action"]
           
            if action == "hold":
                logger.info("Signal says HOLD - no action taken")
                return True
           
            # Get position size
            quantity = self.calculate_position_size()
            if quantity <= 0:
                logger.warning("Cannot calculate valid position size")
                return False
           
            # Get current position
            position_info = self.get_position_info()
            current_side = position_info["side"]
           
            # Close opposing position if needed
            if (action == "buy" and current_side == "short") or (action == "sell" and current_side == "long"):
                logger.info(f"Closing {current_side} position before opening {action}")
                self.sell_all()
                time.sleep(1)  # Brief pause
           
            # Execute new trade
            if action == "buy":
                order = self.create_order(self.symbol, quantity, "buy")
                self.submit_order(order)
                self.last_trade_type = "buy"
                logger.info(f" BUY order submitted: {quantity} shares")
               
            elif action == "sell" and self.enable_short:
                order = self.create_order(self.symbol, quantity, "sell")
                self.submit_order(order)
                self.last_trade_type = "sell"
                logger.info(f" SELL order submitted: {quantity} shares")
           
            # Record trade
            self.trade_history.append({
                "timestamp": self.get_datetime(),
                "action": action,
                "quantity": quantity,
                "signal_confidence": signal["confidence"],
                "reason": signal["reason"]
            })
           
            return True
           
        except Exception as e:
            logger.error(f" Error executing trade: {e}")
            self.consecutive_losses += 1
            return False


    def check_exit_conditions(self) -> Optional[str]:
        """Conservative exit conditions - Only sell on big drops"""
        try:
            position_info = self.get_position_info()
           
            if position_info["quantity"] == 0:
                return None  # No position to exit
           
            current_price = self.get_last_price(self.symbol)
            if position_info["avg_cost"] > 0:
                pnl_percent = (current_price - position_info["avg_cost"]) / position_info["avg_cost"]
               
                # ONLY SELL ON BIG LOSSES - 8% stop loss
                if pnl_percent <= -self.stop_loss_percent:
                    logger.warning(f"BIG LOSS stop loss: {pnl_percent:.2%} loss")
                    return "stop_loss_major"
               
                # Take profit only on huge gains - 25%
                if pnl_percent >= self.take_profit_percent:
                    logger.info(f"MAJOR PROFIT taking: {pnl_percent:.2%} gain")
                    return "take_profit_major"
               
                # Time-based exit only after 30 days (much longer hold)
                if (self.last_trade_date and
                    (self.get_datetime() - self.last_trade_date).days >= 30):
                    logger.info("Long-term hold exit: 30 days")
                    return "long_term_exit"
           
            # Emergency exit on extreme momentum reversal
            current_momentum = self.get_momentum_signal()
            if current_momentum < -10:  # Massive crash
                logger.warning(f"CRASH DETECTED: {current_momentum:.1f}% momentum")
                return "crash_exit"
           
            return None  # Default: never sell, just hold
           
        except Exception as e:
            logger.error(f"Error checking conservative exit conditions: {e}")
            return None


    def get_momentum_signal(self) -> float:
        """Get momentum signal for aggressive trading"""
        try:
            historical = self.get_historical_prices(self.symbol, 3, "day")
            if historical and len(historical.df) >= 3:
                prices = historical.df['close']
                momentum = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
                return momentum
            return 0
        except:
            return 0


    def on_trading_iteration(self):
        """AGGRESSIVE BUYING Strategy - BUY EVERYTHING!"""
        try:
            current_datetime = self.get_datetime()
            logger.info(f" AGGRESSIVE BUYING iteration at {current_datetime}")
           
            # Skip weekends
            if current_datetime.weekday() >= 5:
                logger.info("Weekend detected - skipping iteration")
                return
           
            # Check exit conditions ONLY for major losses
            exit_reason = self.check_exit_conditions()
            if exit_reason:
                logger.info(f"MAJOR EXIT: {exit_reason}")
                self.sell_all()
                self.last_trade_date = current_datetime
           
            # Get current position and cash
            position_info = self.get_position_info()
            current_quantity = position_info["quantity"]
            cash_available = self.get_cash()
           
            logger.info(f" Cash: ${cash_available:.2f}, Position: {current_quantity} shares")
           
            # Get aggressive buying signal
            signal = self.get_trading_signal()
            logger.info(f" AGGRESSIVE BUY SIGNAL: {signal['action']} (confidence: {signal['confidence']:.3f})")
            logger.info(f" Reason: {signal['reason']}")
           
            # FORCE BUYING if no signal
            if signal["action"] == "hold":
                logger.info("🔥 NO SIGNAL - FORCING AGGRESSIVE BUY!")
                signal["action"] = "buy"
                signal["confidence"] = 0.8
                signal["reason"] = "FORCED AGGRESSIVE BUY - No signal detected"
           
            # ALWAYS TRY TO BUY if we have cash
            if cash_available > 0:
                # Lower confidence threshold dramatically
                min_confidence = 0.15  # Only need 15% confidence
               
                if signal["confidence"] >= min_confidence:
                    # Calculate aggressive position size
                    quantity = self.calculate_position_size()
                   
                    if quantity > 0:
                        logger.info(f"💸 AGGRESSIVE BUYING: Attempting to buy {quantity} shares")
                        success = self.execute_trade(signal)
                        if success:
                            self.consecutive_losses = 0
                            self.last_trade_date = current_datetime
                            logger.info(f" AGGRESSIVE BUY EXECUTED! {quantity} shares purchased")
                        else:
                            logger.error(f" Buy order failed - trying smaller size")
                            # Try with smaller position if failed
                            smaller_quantity = max(1, quantity // 2)
                            if smaller_quantity != quantity:
                                logger.info(f" Retrying with {smaller_quantity} shares")
                    else:
                        logger.warning(" Could not calculate position size")
                else:
                    logger.info(f" Confidence {signal['confidence']:.3f} below minimum {min_confidence} - FORCING BUY ANYWAY!")
                    # Force buy even with low confidence
                    quantity = self.calculate_position_size()
                    if quantity > 0:
                        signal["confidence"] = min_confidence + 0.01
                        success = self.execute_trade(signal)
                        if success:
                            logger.info(f" FORCED BUY EXECUTED!")
            else:
                logger.info(" NO CASH - Fully invested (this is good for aggressive buying!)")
           
            # Log final status
            final_position = self.get_position_info()
            if final_position["quantity"] > 0:
                position_value = final_position["quantity"] * self.get_last_price(self.symbol)
                logger.info(f" TOTAL POSITION: {final_position['quantity']} shares (${position_value:.2f})")
            else:
                logger.info(" NO POSITION YET - Need more aggressive buying!")
           
        except Exception as e:
            logger.error(f" Error in aggressive buying iteration: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Handle filled orders"""
        logger.info(f" Order filled: {order.side} {quantity} shares at ${price:.2f}")
       
        # Update performance tracking
        self.daily_pnl.append({
            "date": self.get_datetime(),
            "action": order.side,
            "quantity": quantity,
            "price": price,
            "portfolio_value": self.get_portfolio_value()
        })


    def on_aborted_order(self, order):
        """Handle aborted orders"""
        logger.warning(f"  Order aborted: {order}")
        self.consecutive_losses += 1


    def on_canceled_order(self, order):
        """Handle canceled orders"""
        logger.warning(f" Order canceled: {order}")


    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        try:
            return {
                "total_trades": len(self.trade_history),
                "consecutive_losses": self.consecutive_losses,
                "portfolio_value": self.get_portfolio_value(),
                "cash": self.get_cash(),
                "position_info": self.get_position_info(),
                "last_trade_date": self.last_trade_date
            }
        except Exception as e:
            logger.error(f"Error getting strategy stats: {e}")
            return {}


def run_backtest(symbol: str = DEFAULT_SYMBOL, start_date: datetime = None, end_date: datetime = None):
    """Run enhanced backtest with better error handling"""
   
    if not LUMIBOT_AVAILABLE:
        logger.error(" Lumibot not available. Install with: pip install lumibot")
        return None
   
    # Set default date range
    if not start_date:
        start_date = datetime(2024, 1, 1)
    if not end_date:
        end_date = datetime(2024, 6, 30)
   
    logger.info(f" Starting backtest for {symbol}")
    logger.info(f" Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
   
    try:
        # Strategy parameters
        strategy_params = {
            "symbol": symbol,
            "cash_at_risk": 0.4,  # 40% of cash per trade
            "sentiment_threshold": 0.6,  # Lower threshold for more trades
            "enable_short": False  # Keep it simple for backtesting
        }
       
        logger.info(f"Strategy parameters: {strategy_params}")
       
        # Run backtest
        result = EnhancedMLTrader.backtest(
            YahooDataBacktesting,
            backtesting_start=start_date,
            backtesting_end=end_date,
            parameters=strategy_params,
            benchmark_asset=symbol,
            show_plot=True,
            save_tearsheet=True,
            tearsheet_file=f"backtest_tearsheet_{symbol}_{start_date.strftime('%Y%m%d')}.html"
        )
       
        logger.info(" Backtest completed successfully!")
       
        # Display basic results
        if result:
            try:
                logger.info(" Backtest Results:")
                if hasattr(result, 'stats') and result.stats:
                    for key, value in result.stats.items():
                        logger.info(f"   {key}: {value}")
                else:
                    logger.info("   Results object created successfully")
            except Exception as e:
                logger.warning(f"Could not display detailed results: {e}")
       
        return result
       
    except Exception as e:
        logger.error(f" Backtest failed: {e}")
        logger.error(" Common solutions:")
        logger.error("   1. Update Lumibot: pip install lumibot --upgrade")
        logger.error("   2. Check internet connection for Yahoo Finance data")
        logger.error("   3. Try a different date range")
        logger.error("   4. Ensure TradeBotML.py is available")
       
        import traceback
        logger.error(f"Full error: {traceback.format_exc()}")
        return None


def run_live_trading():
    """Run live trading with enhanced safety checks"""
   
    if not LUMIBOT_AVAILABLE:
        logger.error(" Lumibot not available for live trading")
        return
   
    if not ALPACA_API_AVAILABLE:
        logger.error(" Alpaca API not available for live trading")
        return
   
    if API_KEY == "dummy_key":
        logger.error(" Real API credentials required for live trading")
        logger.error("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return
   
    try:
        # Test API connection
        logger.info("🔧 Testing API connection...")
        api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        account = api.get_account()
        logger.info(f" API connected. Account status: {account.status}")
       
        # Safety check for paper trading
        if not ALPACA_CREDS["PAPER"]:
            confirmation = input("  LIVE TRADING MODE! Type 'CONFIRM' to proceed: ")
            if confirmation != "CONFIRM":
                logger.info("Live trading cancelled by user")
                return
       
        logger.info(" Starting live trading...")
       
        # Create broker and strategy
        broker = Alpaca(ALPACA_CREDS)
        strategy = EnhancedMLTrader(
            broker=broker,
            parameters={
                "symbol": DEFAULT_SYMBOL,
                "cash_at_risk": 0.2,  # Conservative for live trading
                "sentiment_threshold": 0.75,  # Higher threshold for live
                "enable_short": False  # Safer for live trading
            }
        )
       
        # Create and run trader
        trader = Trader()
        trader.add_strategy(strategy)
       
        logger.info(" Live trading started! Press Ctrl+C to stop.")
        trader.run_all()
       
    except KeyboardInterrupt:
        logger.info(" Live trading stopped by user")
    except Exception as e:
        logger.error(f" Live trading error: {e}")
        import traceback
        logger.error(f"Full error: {traceback.format_exc()}")


def main():
    """Main function to run the trading bot"""
    logger.info(" === Enhanced AI Trading Bot ===")
   
    # Display configuration
    logger.info("  Configuration:")
    logger.info(f"   API Endpoint: {BASE_URL}")
    logger.info(f"   Paper Trading: {ALPACA_CREDS['PAPER']}")
    logger.info(f"   API Available: {'Yes' if API_KEY != 'dummy_key' else 'No (dummy mode)'}")
    logger.info(f"   ML Components: {'Available' if ML_COMPONENTS_AVAILABLE else 'Limited'}")
    logger.info(f"   Lumibot: {'Available' if LUMIBOT_AVAILABLE else 'Not installed'}")
   
    # Choose mode
    print("\nSelect mode:")
    print("1. Backtest (recommended for testing)")
    print("2. Live Trading (requires API credentials)")
    print("3. Quick test (run basic functionality)")
   
    try:
        choice = input("Enter choice (1-3): ").strip()
       
        if choice == "1":
            logger.info(" Starting backtest mode...")
           
            # Get custom parameters
            symbol = input(f"Enter symbol (default: {DEFAULT_SYMBOL}): ").strip().upper() or DEFAULT_SYMBOL
           
            # Run backtest
            result = run_backtest(symbol)
            if result:
                logger.info(" Backtest completed successfully!")
            else:
                logger.error(" Backtest failed")
               
        elif choice == "2":
            logger.info(" Starting live trading mode...")
            run_live_trading()
           
        elif choice == "3":
            logger.info(" Running quick test...")
           
            # Test ML components
            if ML_COMPONENTS_AVAILABLE:
                from TradeBotML import get_technical_indicators, get_market_sentiment_signal
               
                logger.info("Testing technical indicators...")
                tech_data = get_technical_indicators("SPY")
                logger.info(f"Technical data points: {len(tech_data)}")
               
                logger.info("Testing market sentiment...")
                sentiment = get_market_sentiment_signal("SPY")
                logger.info(f"Market sentiment: {sentiment.get('signal', 'unknown')}")
               
                logger.info(" Quick test completed")
            else:
                logger.warning(" ML components not available for testing")
        else:
            logger.error("Invalid choice")
           
    except KeyboardInterrupt:
        logger.info(" Program interrupted by user")
    except Exception as e:
        logger.error(f" Error in main: {e}")


if __name__ == "__main__":
    main()