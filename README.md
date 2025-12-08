# S&P 500 Portfolio Rotation Trading Bot

A sophisticated backtesting system for implementing portfolio rotation strategies on S&P 500 stocks with bear market protection.

## 📊 Overview

This trading bot implements a momentum-based portfolio rotation strategy that:
- Automatically rotates between top-performing S&P 500 stocks
- Protects capital during bear markets by increasing cash reserves
- Uses technical analysis (RSI, EMA, momentum) to score and rank stocks
- Provides comprehensive visualization and performance analytics

### Key Features

- **Bear Market Protection**: Automatically adjusts cash reserves (60% - optimized) when SPY falls below 200-day moving average
- **Monthly Rebalancing**: Systematically rotates into top 10 momentum stocks
- **Historical Performance**: 7.9% annual return (2005-2024) with optimized parameters
- **Interactive Visualizations**: Bokeh-based charts for portfolio analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Levietduc1104/trading_bot.git
cd trading_bot
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
trading_bot/
├── src/
│   ├── backtest/
│   │   ├── portfolio_bot_demo.py          # Main portfolio rotation bot
│   │   ├── run_best_bear_protection.py    # Run best strategy config
│   │   ├── generate_sp500_data.py         # Generate S&P 500 data
│   │   ├── create_sp500_index.py          # Create SPY index data
│   │   └── optimize_strategy.py           # Strategy optimization
│   ├── visualize/
│   │   ├── visualize_trades.py            # Create interactive charts
│   │   └── visualize_results.py           # Performance visualization
│   └── risk/
│       └── risk_management_backtest.py    # Risk management tools
├── sp500_data/
│   └── daily/                             # S&P 500 stock CSV files
├── results/
│   ├── trades_log.json                    # Detailed trade history
│   └── portfolio_values.csv               # Portfolio value over time
└── README.md
```

## 📈 Generating Data

### Option 1: Generate All S&P 500 Data

Generate historical data for all S&P 500 stocks from 2005 to present:

```bash
python3 src/backtest/generate_sp500_data.py
```

This will:
- Create `sp500_data/daily/` directory
- Generate realistic OHLCV data for 500+ stocks
- Include proper price trends, volatility, and volume patterns

### Option 2: Download Real Data

To use real market data, you can:
1. Use yfinance or similar API to download actual S&P 500 data
2. Save as CSV files in `sp500_data/daily/` with format: `TICKER.csv`
3. Required columns: `Date, Open, High, Low, Close, Volume`

## 🎯 Running Backtests

### Run Best Strategy Configuration

Run the optimized bear protection strategy (60% cash in bear markets - Phase 1 optimized):

```bash
python3 src/backtest/run_best_bear_protection.py
```

**Expected Output:**
```
================================================================================
RUNNING BEST BEAR PROTECTION STRATEGY
================================================================================

Optimized Configuration (Phase 1):
  - Monthly rebalancing
  - 60% cash in bear markets (when SPY < 200-day MA)
  - 10% cash in bull markets

INFO: Loading 472 stocks...
INFO: Loaded 472 stocks
INFO: Starting BEAR PROTECTION backtest...

============================================================
BEAR PROTECTION RESULTS
============================================================
Initial Capital: $100,000
Final Value: $456,610
Total Return: 356.6%
Annual Return: 7.9%
Max Drawdown: -32.8%
Sharpe Ratio: 1.28
============================================================

Yearly Returns:
  2005:    2.0% ✅
  2006:    4.5% ✅
  ...
  2024:    6.4% ✅
```

### Custom Backtest

Run a custom backtest with your own parameters:

```python
from src.backtest.portfolio_bot_demo import PortfolioRotationBot

# Initialize bot
bot = PortfolioRotationBot(data_dir='sp500_data/daily')
bot.prepare_data()
bot.score_all_stocks()

# Run backtest
portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,                    # Number of stocks to hold
    rebalance_freq='M',          # Monthly rebalancing
    bear_cash_reserve=0.60,      # 60% cash in bear markets (optimized)
    bull_cash_reserve=0.10       # 10% cash in bull markets
)
```

## 📊 Visualization

### Generate Interactive Charts

Create comprehensive Bokeh visualizations:

```bash
python3 src/visualize/visualize_trades.py
```

This generates `results/trading_dashboard.html` with:

#### Tab 1: Trading Analysis
- Portfolio composition over time
- Holdings timeline (top 30 stocks)
- Trade frequency analysis

#### Tab 2: Performance Metrics
- Portfolio value vs SPY benchmark
- Cumulative returns comparison
- Drawdown analysis
- Monthly/yearly returns heatmap

#### Tab 3: Stock Selector
- Interactive stock price charts (top 50 most traded)
- Technical indicators (RSI, EMA 20/50/100/200, Volume)
- Buy/sell markers on price charts
- Fundamental data display

### View Results

Open the dashboard in your browser:

```bash
open results/trading_dashboard.html
# On Windows: start results/trading_dashboard.html
# On Linux: xdg-open results/trading_dashboard.html
```

## 🔧 Configuration

### Strategy Parameters

Edit parameters in your backtest script:

```python
# Portfolio settings
top_n = 10                      # Number of stocks to hold
rebalance_freq = 'M'            # M (monthly), Q (quarterly), Y (yearly)

# Bear market protection
bear_cash_reserve = 0.60        # 60% cash when SPY < 200-day MA (optimized)
bull_cash_reserve = 0.10        # 10% cash in normal markets

# Risk management
max_position_size = 0.15        # Max 15% per position
stop_loss = 0.20               # 20% stop loss
```

### Scoring System

Stocks are scored based on:
- **RSI (30%)**: Relative Strength Index
- **Price vs EMA (40%)**: Price momentum above moving averages
- **Volume (10%)**: Trading volume strength
- **Volatility (20%)**: Risk-adjusted returns

## 📉 Results

Results are saved in the `results/` directory:

### trades_log.json
Detailed trade history with:
- Buy/sell transactions
- Holdings at each rebalance
- Bull/bear market indicators

```json
[
  {
    "date": "2024-01-01",
    "action": "BUY",
    "ticker": "AAPL",
    "price": 185.64,
    "shares": 50,
    "value": 9282.00
  },
  ...
]
```

### portfolio_values.csv
Daily portfolio values and returns:
- Date
- Portfolio value
- Cash
- Invested amount
- Daily/cumulative returns

## 🎓 Strategy Details

### Bear Market Protection

The strategy identifies bear markets when:
- SPY (S&P 500 ETF) closes below its 200-day moving average

Actions taken:
- **Bear Market (🐻)**: Move to 60% cash (optimized), hold top 10 stocks
- **Bull Market (🐂)**: Stay 90% invested in top 10 stocks

### Historical Performance Highlights (Optimized Configuration)

| Metric | Value |
|--------|-------|
| Total Return (2005-2024) | 356.6% |
| Annual Return | 7.9% |
| Max Drawdown | -32.8% |
| Sharpe Ratio | 1.28 |
| Improvement vs Baseline | +0.4% annual |

**Notable Years:**
- **2008 Financial Crisis**: -17.9% (vs S&P 500 -37%)
- **2021 Bull Market**: +46.8%
- **2022 Bear Market**: +7.8% (vs S&P 500 -18%)

## 🛠️ Advanced Usage

### Optimize Strategy Parameters

Run grid search to find optimal parameters:

```bash
python3 src/backtest/optimize_strategy.py
```

### Custom Stock Universe

Use your own stock list:

```python
bot = PortfolioRotationBot(data_dir='your_data_dir')
bot.prepare_data(stock_list=['AAPL', 'MSFT', 'GOOGL'])
```

### Export to Excel

Export detailed results:

```python
import pandas as pd

# Load results
df = pd.read_csv('results/portfolio_values.csv')
trades = pd.read_json('results/trades_log.json')

# Export
df.to_excel('results/portfolio_analysis.xlsx')
trades.to_excel('results/trade_history.xlsx')
```

## 📝 Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **src/backtest/**: Core backtesting engine and strategy logic
- **src/visualize/**: Visualization and reporting tools
- **src/risk/**: Risk management and portfolio optimization

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes only. Not financial advice.

## ⚠️ Disclaimer

Past performance does not guarantee future results. This backtesting system is for educational and research purposes only. Always do your own research and consult with financial professionals before making investment decisions.

## 🔗 Resources

- [Repository](https://github.com/Levietduc1104/trading_bot)
- [Bokeh Documentation](https://docs.bokeh.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with Claude Code** 🤖
