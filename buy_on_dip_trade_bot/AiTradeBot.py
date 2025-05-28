import os
import logging
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Dict, Any


# Import trading framework
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


# Import ML components
try:
    from TradeBotML import (
        get_technical_indicators,
        get_market_sentiment_signal,
        generate_sentiment_signal,
        calculate_momentum
    )
    ML_AVAILABLE = True
except ImportError:
    print("TradeBotML.py not found. Make sure it's in the same directory.")
    ML_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# Configuration
API_KEY = os.getenv('ALPACA_API_KEY', 'dummy_key')
API_SECRET = os.getenv('ALPACA_API_SECRET', 'dummy_secret')
BASE_URL = "https://paper-api.alpaca.markets/v2"


ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}


class SmartDipTrader(Strategy):
    """Smart Dip Buying Strategy with Partial Profit Taking"""
   
    def initialize(self, **kwargs):
        """Initialize the strategy"""
        # Strategy parameters
        self.symbol = kwargs.get('symbol', 'SPY')
        self.cash_at_risk = kwargs.get('cash_at_risk', 0.25)  # More aggressive - 25%
        self.sentiment_threshold = kwargs.get('sentiment_threshold', 0.3)  # Lower threshold
       
        # Trading settings
        self.sleeptime = "1D"  # Daily trading
        self.last_trade_date = None
       
        # Risk management
        self.min_cash_reserve = 10000  # Lower reserve - $10k minimum
        self.consecutive_losses = 0
        self.max_consecutive_losses = 8  # Allow more losses
       
        # Profit taking levels - MORE AGGRESSIVE
        self.profit_level_1 = 0.03  # 3% - sell 25%
        self.profit_level_2 = 0.07  # 7% - sell 50%
        self.profit_level_3 = 0.20  # 20% - sell all
        self.stop_loss_level = -0.06  # 6% stop loss
       
        # Tracking
        self.trade_history = []
        self.partial_sales_made = []  # Track which profit levels hit
       
        logger.info(f" Smart Dip Trader initialized for {self.symbol}")
        logger.info(f"   Profit taking: 5%→25%, 10%→50%, 25%→100%")
        logger.info(f"   Stop loss: 8%")


    def get_current_momentum(self) -> float:
        """Get current price momentum"""
        try:
            if ML_AVAILABLE:
                return calculate_momentum(self.symbol, 5)
            else:
                # Fallback calculation
                historical = self.get_historical_prices(self.symbol, 7, "day")
                if historical and len(historical.df) >= 6:
                    prices = historical.df['close']
                    return ((prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]) * 100
                return 0.0
        except:
            return 0.0


    def calculate_position_size(self, momentum: float) -> int:
        """Calculate position size - CONSERVATIVE cash management to keep reserves"""
        try:
            cash = self.get_cash()
            price = self.get_last_price(self.symbol)
           
            if cash <= self.min_cash_reserve or price <= 0:
                return 0
           
            # SMART CASH MANAGEMENT - Scale down to preserve cash for future dips
            available_cash = cash - self.min_cash_reserve
           
            # Much smaller position sizes to keep cash for dips
            if momentum < -3:  # Major dip - bigger position but still conservative
                target_amount = min(8000, available_cash * 0.4)  # Max $8k or 40% of available
            elif momentum < -1.5:  # Medium dip  
                target_amount = min(6000, available_cash * 0.3)  # Max $6k or 30% of available
            elif momentum < -0.5:  # Small dip
                target_amount = min(4000, available_cash * 0.25) # Max $4k or 25% of available
            else:  # No dip - very small position
                target_amount = min(3000, available_cash * 0.15) # Max $3k or 15% of available
           
            # Ensure minimum viable trade
            if target_amount < 1000 and available_cash > 1000:
                target_amount = 1000  # Minimum $1k trade
           
            quantity = int(target_amount / price)
           
            logger.info(f"CONSERVATIVE sizing: {momentum:.1f}% momentum, ${available_cash:.0f} available → ${target_amount:.0f} → {quantity} shares")
            logger.info(f"CASH MANAGEMENT: Keeping ${cash - target_amount:.0f} for future opportunities")
           
            return max(0, quantity)
           
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0


    def generate_buy_signal(self) -> Dict[str, Any]:
        """Generate buy signal based on technical analysis - AGGRESSIVE VERSION"""
        try:
            momentum = self.get_current_momentum()
           
            # Get technical data
            if ML_AVAILABLE:
                tech_data = get_technical_indicators(self.symbol)
                market_signal = get_market_sentiment_signal(self.symbol)
            else:
                tech_data = {}
                market_signal = {"signal": "neutral", "confidence": 0.5}
           
            # Calculate buy score - MUCH MORE AGGRESSIVE
            buy_score = 0
            reasons = []
           
            # AGGRESSIVE: Always add base buying bias
            buy_score += 2  # Strong base bias
            reasons.append("Aggressive base buying bias")
           
            # Momentum-based scoring (main factor)
            if momentum < -1:  # Any decent dip
                buy_score += 4  # Much higher score
                reasons.append(f"Strong dip buy ({momentum:.1f}%)")
            elif momentum < -0.5:  # Small dip
                buy_score += 3
                reasons.append(f"Medium dip buy ({momentum:.1f}%)")
            elif momentum < 0:  # Any decline at all
                buy_score += 2
                reasons.append(f"Small decline ({momentum:.1f}%)")
            else:  # Even no decline
                buy_score += 1  # Still some buying interest
                reasons.append("Momentum buying")
           
            # RSI - MUCH more lenient
            rsi = tech_data.get('rsi', 50)
            if rsi < 60:  # Buy when RSI below 60 (was 35)
                buy_score += 2
                reasons.append(f"RSI favorable ({rsi:.1f})")
            if rsi < 40:
                buy_score += 1  # Extra boost
                reasons.append(f"RSI oversold bonus")
           
            # Below moving averages - always good
            if not tech_data.get('above_sma_10', True):
                buy_score += 2  # Higher score
                reasons.append("Below SMA10 - discount")
            if not tech_data.get('above_sma_20', True):
                buy_score += 2
                reasons.append("Below SMA20 - strong discount")
           
            # Market sentiment - favor any signal
            if market_signal.get('signal') == 'bullish':
                buy_score += 2
                reasons.append("Bullish market sentiment")
            elif market_signal.get('signal') == 'bearish':
                buy_score += 3  # Contrarian buying
                reasons.append("Contrarian buy on bearish sentiment")
            else:
                buy_score += 1
                reasons.append("Neutral market - still buying")
           
            # Add random factor to force more trades
            random_boost = np.random.uniform(1, 3)
            buy_score += random_boost
            reasons.append(f"Random boost ({random_boost:.1f})")
           
            # Calculate confidence - always high
            confidence = min(buy_score / 8.0, 0.95)  # Scale but ensure high confidence
            confidence = max(confidence, 0.5)  # Minimum 50% confidence
           
            # AGGRESSIVE Decision - almost always buy
            if buy_score >= 4:  # Lower threshold
                action = "buy"
            elif buy_score >= 2:  # Very low threshold
                action = "buy"
                confidence = max(confidence, 0.6)
            else:
                # Even low scores have chance to buy
                if np.random.random() < 0.7:  # 70% chance to buy anyway
                    action = "buy"
                    confidence = 0.6
                else:
                    action = "hold"
           
            # FORCE BUY on any decline
            if momentum < -0.1:  # Even tiny declines
                action = "buy"
                confidence = max(confidence, 0.8)
                reasons.append("FORCED BUY on any decline!")
           
            return {
                "action": action,
                "confidence": confidence,
                "buy_score": buy_score,
                "reasons": reasons,
                "momentum": momentum
            }
           
        except Exception as e:
            logger.error(f"Error generating buy signal: {e}")
            return {"action": "buy", "confidence": 0.7, "reasons": ["Default buy"], "momentum": 0}


    def check_profit_taking(self) -> Optional[str]:
        """Check if we should take profits - MUCH MORE AGGRESSIVE"""
        try:
            position = self.get_position(self.symbol)
            if not position or position.quantity <= 0:
                return None
           
            current_price = self.get_last_price(self.symbol)
           
            # Try to get average cost - handle different position object types
            try:
                avg_cost = position.avg_cost
            except AttributeError:
                try:
                    avg_cost = position.avg_fill_price
                except AttributeError:
                    # Fallback - estimate from recent trades
                    avg_cost = current_price * 0.95  # Assume 5% profit minimum
           
            if avg_cost <= 0:
                avg_cost = current_price * 0.95
           
            profit_pct = (current_price - avg_cost) / avg_cost
           
            logger.info(f"Current P&L: {profit_pct:.2%} (Current: ${current_price:.2f}, Avg Cost: ${avg_cost:.2f})")
           
            # MUCH MORE AGGRESSIVE PROFIT TAKING - Lower thresholds
            if profit_pct >= 0.20:  # 20% profit (was 25%)
                return "sell_all_20"
            elif profit_pct >= 0.07 and "7%" not in self.partial_sales_made:  # 7% profit (was 10%)
                return "sell_half_7"
            elif profit_pct >= 0.03 and "3%" not in self.partial_sales_made:  # 3% profit (was 5%)
                return "sell_quarter_3"
            elif profit_pct <= -0.06:  # 6% stop loss (was 8%)
                return "stop_loss"
           
            return None
           
        except Exception as e:
            logger.error(f"Error checking profit taking: {e}")
            return None


    def execute_partial_sale(self, sale_type: str) -> bool:
        """Execute partial sale - Updated for new profit levels"""
        try:
            position = self.get_position(self.symbol)
            if not position or position.quantity <= 0:
                return False
           
            current_quantity = position.quantity
            current_price = self.get_last_price(self.symbol)
           
            # Determine sell quantity based on new levels
            if sale_type == "sell_quarter_3":  # 3% profit
                sell_quantity = max(1, int(current_quantity * 0.25))
                self.partial_sales_made.append("3%")
                logger.info(f" Taking 25% profits at 3% gain: {sell_quantity} shares")
               
            elif sale_type == "sell_half_7":  # 7% profit
                sell_quantity = max(1, int(current_quantity * 0.50))
                self.partial_sales_made.append("7%")
                logger.info(f" Taking 50% profits at 7% gain: {sell_quantity} shares")
               
            elif sale_type == "sell_all_20":  # 20% profit
                sell_quantity = current_quantity
                self.partial_sales_made = []  # Reset for next position
                logger.info(f" Taking all profits at 20% gain: {sell_quantity} shares")
               
            elif sale_type == "stop_loss":
                sell_quantity = current_quantity
                self.partial_sales_made = []  # Reset for next position
                logger.info(f" Stop loss triggered: {sell_quantity} shares")
               
            else:
                return False
           
            # Execute sell order
            sell_order = self.create_order(self.symbol, sell_quantity, "sell")
            self.submit_order(sell_order)
           
            # Log the sale
            sale_value = sell_quantity * current_price
            remaining = current_quantity - sell_quantity
           
            logger.info(f" SOLD: {sell_quantity} shares for ${sale_value:.2f}")
            logger.info(f" REMAINING: {remaining} shares")
            logger.info(f" CASH FREED UP: ${sale_value:.2f} for new dip buying!")
           
            return True
           
        except Exception as e:
            logger.error(f"Error executing partial sale: {e}")
            return False


    def on_trading_iteration(self):
        """Main trading logic"""
        try:
            current_time = self.get_datetime()
            logger.info(f" Trading iteration: {current_time}")
           
            # Skip weekends
            if current_time.weekday() >= 5:
                return
           
            # Check for profit taking first
            profit_action = self.check_profit_taking()
            if profit_action:
                logger.info(f" Profit taking triggered: {profit_action}")
                self.execute_partial_sale(profit_action)
                return
           
            # Get current status
            cash = self.get_cash()
            position = self.get_position(self.symbol)
            current_shares = position.quantity if position else 0
           
            logger.info(f" Cash: ${cash:.2f}, Shares: {current_shares}")
           
            # Generate buy signal
            signal = self.generate_buy_signal()
            momentum = signal["momentum"]
           
            logger.info(f" Signal: {signal['action']} (confidence: {signal['confidence']:.2f})")
            logger.info(f" Momentum: {momentum:.2f}%")
            logger.info(f" Reasons: {', '.join(signal['reasons'])}")
           
            # Execute buy if conditions are met - MUCH MORE LENIENT
            if signal["action"] == "buy" and signal["confidence"] >= self.sentiment_threshold:
               
                # Calculate position size
                quantity = self.calculate_position_size(momentum)
               
                if quantity > 0:  # Removed strict cash check
                   
                    logger.info(f" EXECUTING BUY: {quantity} shares")
                   
                    # Create and submit buy order
                    buy_order = self.create_order(self.symbol, quantity, "buy")
                    self.submit_order(buy_order)
                   
                    # Reset partial sales tracking for new position
                    if current_shares == 0:
                        self.partial_sales_made = []
                   
                    # Record trade
                    self.trade_history.append({
                        "timestamp": current_time,
                        "action": "buy",
                        "quantity": quantity,
                        "momentum": momentum,
                        "confidence": signal["confidence"]
                    })
                   
                    self.consecutive_losses = 0
                    self.last_trade_date = current_time
                   
                    logger.info(f" BUY ORDER SUBMITTED: {quantity} shares")
                   
                else:
                    logger.info(f" Cannot buy: quantity={quantity}")
           
            # FORCE BUY if no signal but conditions are right
            elif signal["action"] == "hold" and momentum < -0.5 and cash > self.min_cash_reserve + 5000:
                logger.info(" FORCING BUY on dip despite weak signal!")
                quantity = self.calculate_position_size(momentum)
                if quantity > 0:
                    buy_order = self.create_order(self.symbol, quantity, "buy")
                    self.submit_order(buy_order)
                    logger.info(f" FORCED BUY: {quantity} shares")
           
            else:
                logger.info(" No buy signal - waiting for opportunity")
           
            # Status summary
            total_value = cash + (current_shares * self.get_last_price(self.symbol) if current_shares > 0 else 0)
            logger.info(f" Total Portfolio Value: ${total_value:.2f}")
           
        except Exception as e:
            logger.error(f" Error in trading iteration: {e}")
            import traceback
            logger.error(traceback.format_exc())


def run_backtest(symbol="SPY"):
    """Run backtest"""
    if not LUMIBOT_AVAILABLE:
        print(" Lumibot not available")
        return None
   
    logger.info(f" Starting backtest for {symbol}")
   
    try:
        # Backtest parameters
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2025, 1, 1)
       
        strategy_params = {
            "symbol": symbol,
            "cash_at_risk": 0.25,  # More aggressive
            "sentiment_threshold": 0.3  # Lower threshold
        }
       
        logger.info(f" Period: {start_date.date()} to {end_date.date()}")
        logger.info(f" Parameters: {strategy_params}")
       
        # Run backtest
        result = SmartDipTrader.backtest(
            YahooDataBacktesting,
            backtesting_start=start_date,
            backtesting_end=end_date,
            parameters=strategy_params,
            benchmark_asset=symbol,
            show_plot=True,
            save_tearsheet=True
        )
       
        logger.info(" Backtest completed successfully!")
        return result
       
    except Exception as e:
        logger.error(f" Backtest failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_live_trading():
    """Run live trading"""
    if not LUMIBOT_AVAILABLE or not ALPACA_API_AVAILABLE:
        print(" Required libraries not available")
        return
   
    if API_KEY == "dummy_key":
        print(" Please set real API credentials for live trading")
        return
   
    logger.info(" Starting live trading...")
   
    try:
        broker = Alpaca(ALPACA_CREDS)
        strategy = SmartDipTrader(
            broker=broker,
            parameters={
                "symbol": "SPY",
                "cash_at_risk": 0.1,  # Conservative for live trading
                "sentiment_threshold": 0.5
            }
        )
       
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
       
    except Exception as e:
        logger.error(f" Live trading error: {e}")


def main():
    """Main function"""
    print(" === Smart Dip Trading Bot ===")
    print("\nSelect mode:")
    print("1. Backtest")
    print("2. Live Trading")
    print("3. Test ML Components")
   
    try:
        choice = input("Enter choice (1-3): ").strip()
       
        if choice == "1":
            symbol = input("Enter symbol (default SPY): ").strip().upper() or "SPY"
            print(f"\n Starting backtest for {symbol}...")
            result = run_backtest(symbol)
           
        elif choice == "2":
            print("\n Starting live trading...")
            run_live_trading()
           
        elif choice == "3":
            print("\n Testing ML components...")
            if ML_AVAILABLE:
                from TradeBotML import get_technical_indicators, get_market_sentiment_signal
               
                symbol = "SPY"
                tech = get_technical_indicators(symbol)
                sentiment = get_market_sentiment_signal(symbol)
               
                print(f"Technical indicators: {tech}")
                print(f"Market sentiment: {sentiment}")
                print(" ML components working!")
            else:
                print(" ML components not available")
            
        else:
            print("Invalid choice")
           
    except KeyboardInterrupt:
        print("\n Interrupted by user")
    except Exception as e:
        logger.error(f" Error: {e}")


if __name__ == "__main__":
    main()

