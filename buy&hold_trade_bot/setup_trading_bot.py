#!/usr/bin/env python3
"""
Setup script for Enhanced AI Trading Bot
This script helps install dependencies and configure the trading bot
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    logger.info(f" Python {version.major}.{version.minor}.{version.micro} - compatible")
    return True

def install_package(package):
    """Install a Python package using pip"""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f" {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f" Failed to install {package}: {e}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    logger.info(" Installing required dependencies...")
    
    # Core dependencies
    core_packages = [
        "numpy",
        "pandas", 
        "yfinance",
        "lumibot",
        "alpaca-trade-api",
        "python-dotenv"
    ]
    
    # Optional ML dependencies (will try to install but not fail if they don't work)
    optional_packages = [
        "torch",
        "transformers",
        "scikit-learn"
    ]
    
    success_count = 0
    total_core = len(core_packages)
    
    # Install core packages
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    if success_count < total_core:
        logger.warning(f"Only {success_count}/{total_core} core packages installed successfully")
        logger.warning("Some functionality may be limited")
    else:
        logger.info(f" All {total_core} core packages installed successfully")
    
    # Try to install optional packages
    logger.info("📦 Installing optional ML packages (may take time)...")
    optional_success = 0
    for package in optional_packages:
        if install_package(package):
            optional_success += 1
    
    logger.info(f" Optional packages: {optional_success}/{len(optional_packages)} installed")
    
    return success_count == total_core

def create_env_file():
    """Create .env file for API credentials"""
    env_path = Path(".env")
    
    if env_path.exists():
        logger.info(" .env file already exists")
        return
    
    logger.info(" Creating .env file for API credentials...")
    
    env_content = """# Alpaca Trading API Credentials
# Get these from: https://app.alpaca.markets/account/keys

# Paper Trading (Safe for testing)
ALPACA_API_KEY=YOUR_ALPACA_API_KEY_HERE
ALPACA_API_SECRET=YOUR_ALPACA_SECRET_KEY_HERE

# For live trading, set this to False (BE VERY CAREFUL!)
PAPER_TRADING=True

# Trading Parameters
DEFAULT_SYMBOL=SPY
CASH_AT_RISK=0.3
SENTIMENT_THRESHOLD=0.65
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        logger.info(" .env file created successfully")
        logger.info(" Please edit .env file with your actual API credentials")
    except Exception as e:
        logger.error(f" Failed to create .env file: {e}")

def create_requirements_txt():
    """Create requirements.txt file"""
    requirements_content = """# Core trading dependencies
lumibot>=3.0.0
alpaca-trade-api>=3.0.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.21.0
python-dotenv>=0.19.0

# Optional ML dependencies (comment out if installation fails)
torch>=2.0.0
transformers>=4.20.0
scikit-learn>=1.1.0

# Development and testing
pytest>=7.0.0
"""
    
    try:
        with open("requirements.txt", 'w') as f:
            f.write(requirements_content)
        logger.info(" requirements.txt created")
    except Exception as e:
        logger.error(f" Failed to create requirements.txt: {e}")

def verify_installation():
    """Verify that key components can be imported"""
    logger.info(" Verifying installation...")
    
    test_imports = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("yfinance", "Market data"),
        ("lumibot", "Trading framework"),
        ("alpaca_trade_api", "Alpaca API")
    ]
    
    success_count = 0
    
    for module, description in test_imports:
        try:
            __import__(module)
            logger.info(f" {description} ({module}) - OK")
            success_count += 1
        except ImportError as e:
            logger.error(f" {description} ({module}) - FAILED: {e}")
    
    # Test optional imports
    optional_imports = [
        ("torch", "PyTorch ML framework"),
        ("transformers", "Hugging Face transformers")
    ]
    
    for module, description in optional_imports:
        try:
            __import__(module)
            logger.info(f" {description} ({module}) - Available")
        except ImportError:
            logger.warning(f"  {description} ({module}) - Not available (optional)")
    
    if success_count == len(test_imports):
        logger.info(" All core components verified successfully!")
        return True
    else:
        logger.warning(f"  {success_count}/{len(test_imports)} core components available")
        return False

def setup_logging():
    """Create logging directory and configuration"""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.info(" Logging directory created")
    except Exception as e:
        logger.error(f" Failed to create logging directory: {e}")

def display_next_steps():
    """Display instructions for next steps"""
    logger.info("\n" + "="*50)
    logger.info(" SETUP COMPLETE!")
    logger.info("="*50)
    
    print("""
 NEXT STEPS:

1.  SET UP API CREDENTIALS:
   - Go to https://app.alpaca.markets/account/keys
   - Create paper trading API keys (safe for testing)
   - Edit the .env file with your credentials

2.  TEST THE BOT:
   - Run: python AITradeBot.py
   - Choose option 3 for quick test
   - Choose option 1 for backtesting

3.  BACKTESTING (RECOMMENDED FIRST):
   - Run backtests to understand the strategy
   - No real money involved
   - Uses historical data from Yahoo Finance

4.  LIVE TRADING (ADVANCED):
   - Only after successful backtesting
   - Start with paper trading (PAPER_TRADING=True in .env)
   - Never risk more than you can afford to lose

  IMPORTANT WARNINGS:
   - This is educational software
   - Past performance doesn't guarantee future results
   - Always start with paper trading
   - Never invest more than you can afford to lose
   - Trading involves significant risk

 FILES CREATED:
   - .env (API credentials - EDIT THIS!)
   - requirements.txt (dependencies list)
   - logs/ (directory for log files)

 TO START TRADING BOT:
   python AITradeBot.py
""")

def main():
    """Main setup function"""
    logger.info(" Enhanced AI Trading Bot Setup")
    logger.info("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create requirements.txt
    create_requirements_txt()
    
    # Install dependencies
    if not install_dependencies():
        logger.error(" Dependency installation failed")
        logger.info(" Try installing manually with: pip install -r requirements.txt")
    
    # Create configuration files
    create_env_file()
    setup_logging()
    
    # Verify installation
    verify_installation()
    
    # Show next steps
    display_next_steps()
    
    logger.info(" Setup completed!")

if __name__ == "__main__":
    main()
