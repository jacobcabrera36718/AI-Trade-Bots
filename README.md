# AI-Trade-Bots

AI-Trade-Bots is a Python-based collection of algorithmic trading bots implementing various trading strategies using machine learning principles. Each bot is modularly designed and provides a different perspective on automated trading in financial markets.

## Table of Contents

- [Features](#features)
- [Bot Strategies](#bot-strategies)
  - [1. First Trade Bot](#1-first-trade-bot)
  - [2. Buy & Hold Bot](#2-buy--hold-bot)
  - [3. Buy on Dip Bot](#3-buy-on-dip-bot)
  - [4. Extreme Trade Bot](#4-extreme-trade-bot)
- [File Structure](#file-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- Modular codebase with separate folders for each trading strategy
- Basic and advanced strategies: from "buy & hold" to "extreme condition trading"
- Integration-ready for broker APIs (e.g., Alpaca)
- Easily extendable for more strategies or AI models

## Bot Strategies

### 1. First Trade Bot

A basic template bot that introduces the structure used across other bots. It is useful for understanding how the trading environment is set up.

**Files:**

- `AITradeBot.py`: Core logic for decision-making
- `TradeBotML.py`: Placeholder for potential ML integration
- `setup_trading_bot.py`: Entry point to initialize and run the bot

### 2. Buy & Hold Bot

Implements a passive investment strategy that buys and holds an asset regardless of price fluctuations. Ideal for long-term investors.

### 3. Buy on Dip Bot

Executes trades when there’s a noticeable price dip, aiming to buy assets at a discount and sell once they recover. Includes logic for detecting dips and responding dynamically.

### 4. Extreme Trade Bot

Targets highly volatile market conditions. This bot may use thresholds or ML models to detect when volatility is unusually high and make trades accordingly.

## File Structure

```
AI-Trade-Bots-main/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── first_trade_bot/
│   ├── AITradeBot.py
│   ├── TradeBotML.py
│   └── setup_trading_bot.py
│
├── buy&hold_trade_bot/
│   ├── AITradeBot.py
│   ├── TradeBotML.py
│   └── setup_trading_bot.py
│
├── buy_on_dip_trade_bot/
│   ├── AiTradeBot.py
│   ├── TradeBotML.py
│   └── setup_trading_bot.py
│
└── extreme_trade_bot/
    ├── AITradeBot.py
    ├── TradeBotML.py
    └── setup_trading_bot.py
```

## Setup Instructions

1. Clone the repository or unzip the project.
2. Navigate into any strategy folder (e.g., `cd buy_on_dip_trade_bot`).
3. Run the `setup_trading_bot.py` file to install dependencies and initialize the environment:

```bash
python setup_trading_bot.py
```

4. Creates a `.env` file in the root of your strategy folder and add your Alpaca API keys:

```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

> Note: The trading bot is designed to work with Alpaca’s brokerage API. You must create an Alpaca account and generate API keys to connect.

## Usage

Each bot runs independently by executing its `setup_trading_bot.py` file. Trading logic and ML enhancements can be customized in `AITradeBot.py` and `TradeBotML.py`.

## Dependencies

These are required for most bots. Additional libraries may be needed depending on your modifications.

```bash
pip install numpy pandas scikit-learn matplotlib yfinance python-dotenv alpaca-trade-api
```

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

## Disclaimer

This trading bot is provided for educational and informational purposes only. By using this software, you acknowledge and agree that:

- I am not a financial advisor.
- This software does not constitute financial or investment advice.
- You use this software at your own risk.
- I am not responsible or liable for any losses, damages, or consequences resulting from the use of this software.

Always do your own research and consult a licensed financial advisor before making investment decisions.
