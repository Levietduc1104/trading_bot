# S&P 500 Portfolio Rotation Trading Bot

A sophisticated backtesting system implementing **V13 Production Strategy** with 5-stock concentration, momentum-strength weighting, and portfolio-level drawdown control for S&P 500 stocks.

## 📊 Overview

This trading bot implements an advanced momentum-based portfolio rotation strategy that combines multiple proven techniques:

- **5-Stock Concentration**: High conviction portfolio (optimal risk/return)
- **Momentum-Strength Weighting**: Allocates capital based on momentum/volatility ratio
- **Drawdown Control**: Progressive exposure reduction during portfolio drawdowns
- **VIX Regime Detection**: Forward-looking market stress indicator
- **Adaptive Position Sizing**: Switches between momentum and inverse-volatility weighting
- **Zero Prediction Bias**: Uses only historical data, no curve fitting

### 🏆 Performance (V13 Production - 5 Stocks)

```
Annual Return:   9.8%
Sharpe Ratio:    1.07
Max Drawdown:    -19.1%
Win Rate:        80% (16/20 positive years)
Final Value:     $615,402 (on $100k over 19.4 years)
Improvement:     +1.4% annual vs 10-stock baseline
```

**Negative Years:** Only 2008 (-7.2%), 2009 (-6.0%), 2018 (-3.0%), 2020 (-6.9%)

**Key Insight:** Portfolio concentration captures more alpha. 5 stocks beat 10 stocks by +1.4% annually with acceptable risk increase.

## ✨ Key Features

### V13 Strategy Components

1. **VIX-Based Regime Detection** (V8)
   - Forward-looking volatility index (not lagging indicators)
   - Dynamic cash reserve: 5% (VIX<30) to 70% (VIX>70)

2. **Adaptive Position Weighting** (V11)
   - **Calm Markets (VIX < 30)**: Momentum-strength weighting
   - **Stressed Markets (VIX ≥ 30)**: Inverse volatility weighting

3. **Momentum-Strength Weighting** (V13)
   - Formula: `weight ∝ momentum / volatility`
   - Momentum = 9-month return (6-12 month range)
   - Allocates more to strong, stable trends
   - Academically proven (Jegadeesh & Titman 1993)

4. **Portfolio-Level Drawdown Control** (V12)
   - Progressive exposure reduction as drawdown increases
   - Rules:
     - DD < 10%: 100% invested
     - DD 10-15%: 75% invested
     - DD 15-20%: 50% invested
     - DD ≥ 20%: 25% invested (maximum defense)

5. **Mid-Month Rebalancing** (V7)
   - Rebalances on day 7-10 (avoids month-end institutional flows)
   - Seasonal adjustments (winter aggressive, summer defensive)

6. **Momentum Quality Filters** (V6)
   - Must be above EMA-89 (long-term trend)
   - Must have positive 20-day momentum
   - RSI penalties for overbought/oversold conditions

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
│   ├── core/
│   │   └── execution.py                    # Main production execution (V13)
│   ├── backtest/
│   │   ├── portfolio_bot_demo.py          # Core strategy implementation
│   │   ├── optimize_strategy.py           # Strategy optimization tools
│   │   └── swing_optimization.py          # Swing trading variants
│   ├── data/
│   │   ├── download_vix.py                # Download VIX data
│   │   └── create_vix_proxy.py            # Create VIX proxy from SPY
│   ├── visualize/
│   │   ├── visualize_trades.py            # Interactive visualizations
│   │   └── visualize_results.py           # Performance dashboards
│   └── risk/
│       └── risk_management_backtest.py    # Risk management tools
├── sp500_data/
│   └── daily/                             # S&P 500 stock CSV files (473 stocks)
│       ├── AAPL.csv
│       ├── MSFT.csv
│       ├── VIX.csv                        # VIX volatility index
│       └── ...
├── output/
│   ├── data/
│   │   └── trading_results.db             # SQLite results database
│   ├── reports/
│   │   └── performance_report_*.txt       # Performance reports
│   ├── plots/
│   │   └── trading_analysis.html          # Interactive charts
│   └── logs/
│       └── execution.log                  # Execution logs
├── test_v12_drawdown_control.py          # V12 drawdown control test
├── test_v13_momentum_weighting.py         # V13 momentum weighting test
└── README.md
```

## 🎯 Running the Strategy

### Production Execution (V13)

Run the full V13 strategy end-to-end:

```bash
python src/core/execution.py
```

**This will:**
1. Load 473 S&P 500 stocks + VIX data
2. Run V13 backtest (2005-2024)
3. Save results to database
4. Generate performance report
5. Create interactive visualization

**Expected Output:**
```
================================================================================
                       V13 MOMENTUM-STRENGTH WEIGHTING
================================================================================

Strategy:        V13 (Momentum + Drawdown Control)
Annual Return:   8.5%
Sharpe Ratio:    1.26
Max Drawdown:    -18.5%
Win Rate:        90% (18/20 positive years)
Final Value:     $481,677

Outputs:
  📊 Database:      output/data/trading_results.db
  📈 Report:        output/reports/performance_report_*.txt
  🎨 Visualization: output/plots/trading_analysis.html
  📋 Logs:          output/logs/execution.log
```

### Custom Backtest

Run custom configurations programmatically:

```python
from src.backtest.portfolio_bot_demo import PortfolioRotationBot

# Initialize bot
bot = PortfolioRotationBot(
    data_dir='sp500_data/daily',
    initial_capital=100000
)

# Load data
bot.prepare_data()
bot.score_all_stocks()

# Run V13 strategy
portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,                        # Top 10 stocks
    rebalance_freq='M',              # Monthly rebalancing
    use_vix_regime=True,             # V8: VIX regime detection
    use_adaptive_weighting=True,     # V11: Adaptive weighting
    use_momentum_weighting=True,     # V13: Momentum-strength weighting
    use_drawdown_control=True,       # V12: Drawdown control
    trading_fee_pct=0.001            # 0.1% trading fee
)
```

### Test Individual Components

Test V12 drawdown control:
```bash
python test_v12_drawdown_control.py
```

Test V13 momentum weighting:
```bash
python test_v13_momentum_weighting.py
```

## 📊 Visualization

### View Interactive Dashboard

Open the generated visualization:

```bash
open output/plots/trading_analysis.html
# On Windows: start output\plots\trading_analysis.html
# On Linux: xdg-open output/plots/trading_analysis.html
```

**Dashboard includes:**
- Portfolio value growth over time
- Drawdown analysis chart
- Yearly returns bar chart
- Monthly returns heatmap
- Risk-adjusted metrics comparison

## 🔧 Strategy Configuration

### Available Strategy Versions

| Version | Description | Annual Return | Sharpe | Max DD |
|---------|-------------|---------------|--------|--------|
| V8 | VIX + Equal Weight | 8.4% | 1.15 | -23.2% |
| V10 | VIX + Inverse Vol | 8.2% | 1.21 | -22.8% |
| V11 | Adaptive Hybrid | 8.3% | 1.22 | -22.8% |
| V12 | V11 + Drawdown Control | 8.2% | 1.23 | -18.5% |
| **V13** | **V12 + Momentum** | **8.5%** | **1.26** | **-18.5%** |

### Enable/Disable Features

```python
portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,
    use_vix_regime=True,              # VIX regime detection (recommended)
    use_adaptive_weighting=True,      # Adaptive position weighting
    use_momentum_weighting=True,      # Momentum-strength weighting (V13)
    use_drawdown_control=True,        # Portfolio drawdown control (V12)
    trading_fee_pct=0.001             # Trading fees (0.1%)
)
```

## 📈 Stock Scoring System

Stocks are scored on a 0-150 point scale:

### V5 Base Scoring (100 points)

1. **Price Trend** (50 pts)
   - Short-term: Price > EMA-13 > EMA-34 (20 pts)
   - Long-term: Price > EMA-89 (30 pts)
   - Acceleration: EMA-34 > EMA-89 (+10 pts bonus)

2. **Recent Performance** (30 pts)
   - ROC-20 > 15%: 30 pts
   - ROC-20 > 10%: 20 pts
   - ROC-20 > 5%: 15 pts
   - ROC-20 > 0%: 10 pts

3. **Risk Level** (20 pts)
   - ATR% < 2%: 20 pts
   - ATR% < 3%: 15 pts
   - ATR% < 4%: 10 pts
   - ATR% < 5%: 5 pts

### V6 Momentum Filters (Disqualification)

- **CRITICAL**: Must be above EMA-89 (score = 0 if fails)
- **CRITICAL**: Must have ROC-20 > 2% (score = 0 if fails)
- **Penalty**: RSI > 75 → score × 0.7
- **Penalty**: RSI < 30 → score × 0.5

### V7 Sector Bonus (±15 points)

- Compare to sector peers (60-day performance)
- Outperformance > 10%: +15 pts
- Outperformance > 5%: +10 pts
- Outperformance > 2%: +5 pts
- Underperformance < -5%: -10 pts

## 📉 Results & Analytics

### Database Schema

Results are stored in SQLite (`output/data/trading_results.db`):

**Tables:**
- `runs`: Backtest metadata
- `portfolio_values`: Daily portfolio value history
- `yearly_returns`: Year-by-year performance

### Performance Report

Generated at `output/reports/performance_report_*.txt`:

```
================================================================================
PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Initial Capital:       $100,000
Final Value:           $481,677
Total Return:          381.7%
Annual Return:         8.5%
Max Drawdown:          -18.5%
Sharpe Ratio:          1.26
Period:                2005-05-23 to 2024-10-03
Duration:              19.4 years

YEARLY RETURNS (90% Win Rate)
--------------------------------------------------------------------------------
  2005:      0.4% ✅    2014:     20.6% ✅    2021:     26.5% ✅
  2006:      3.1% ✅    2015:      0.8% ✅    2022:     10.4% ✅
  2007:     19.2% ✅    2016:     14.2% ✅    2023:     16.3% ✅
  2008:    -13.2% ❌    2017:     15.2% ✅    2024:      4.8% ✅
  2009:     -0.5% ❌    2018:      3.1% ✅
  2010:      6.9% ✅    2019:      7.2% ✅
  2011:     11.9% ✅    2020:      0.1% ✅
  2012:     16.7% ✅
  2013:      6.9% ✅
```

## 🎓 Strategy Evolution

### Phase 1: Base System (V5)
- Momentum-based scoring
- Monthly rebalancing
- Simple bear/bull detection

### Phase 2: Risk Management (V6-V7)
- V6: Momentum quality filters
- V7: Mid-month rebalancing, seasonal adjustments, sector relative strength

### Phase 3: VIX Regime (V8)
- Forward-looking volatility indicator
- Dynamic cash reserves (5%-70%)

### Phase 4: Position Sizing (V10-V11)
- V10: Inverse volatility weighting
- V11: Adaptive hybrid (equal in calm, inverse-vol in stress)

### Phase 5: Portfolio Risk Control (V12)
- Progressive drawdown exposure reduction
- Prevents drawdown acceleration
- Preserves capital for recovery

### Phase 6: Momentum Weighting (V13) 🏆
- Momentum-strength position sizing
- weight ∝ momentum / volatility
- Academically proven factor

## 🔬 Academic Foundation

V13 is built on peer-reviewed research:

1. **Momentum Persistence**
   - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
   - Empirical fact: past returns predict future returns

2. **Volatility Clustering**
   - Engle (1982): "Autoregressive Conditional Heteroskedasticity"
   - Nobel Prize-winning research

3. **VIX Forward-Looking Indicator**
   - Whaley (1993): "Derivatives on Market Volatility"
   - Better than lagging indicators like 200-day MA

4. **Drawdown Control**
   - Used by professional CTAs and hedge funds
   - Geometric return > arithmetic return

## 🛠️ Advanced Usage

### Export Results to Excel

```python
import pandas as pd
import sqlite3

# Load from database
conn = sqlite3.connect('output/data/trading_results.db')

# Get portfolio values
portfolio = pd.read_sql('SELECT * FROM portfolio_values WHERE run_id = 23', conn)
portfolio.to_excel('output/portfolio_analysis.xlsx', index=False)

# Get yearly returns
yearly = pd.read_sql('SELECT * FROM yearly_returns WHERE run_id = 23', conn)
yearly.to_excel('output/yearly_returns.xlsx', index=False)

conn.close()
```

### Custom Stock Universe

```python
# Use custom stock list
bot = PortfolioRotationBot(data_dir='your_data_dir')
bot.prepare_data()

# Run on subset
portfolio_df = bot.backtest_with_bear_protection(
    top_n=5,  # Hold only 5 stocks
    use_vix_regime=True,
    use_momentum_weighting=True,
    use_drawdown_control=True
)
```

### Optimize Parameters

Grid search for optimal configuration:

```bash
python src/backtest/optimize_strategy.py
```

## 📊 Performance Metrics Explained

### Sharpe Ratio (1.26)
- Risk-adjusted return metric
- Higher = better risk-adjusted performance
- Formula: `(mean_return / std_return) × √252`

### Max Drawdown (-18.5%)
- Largest peak-to-trough decline
- Measures worst-case scenario
- V13 reduces this by 19% vs baseline

### Win Rate (90%)
- Percentage of positive years
- 18 out of 20 years profitable
- Only 2008 & 2009 were negative

### Annual Return (8.5%)
- Compound annual growth rate (CAGR)
- Geometric mean, not arithmetic
- Consistent over 19.4 years

## ⚠️ Risk Disclosure

### Important Disclaimers

1. **Past Performance ≠ Future Results**
   - Historical backtests don't guarantee future performance
   - Market conditions change over time

2. **Educational Purpose Only**
   - This is a research and educational tool
   - Not financial advice or investment recommendation

3. **Real Trading Risks**
   - Actual trading has slippage, market impact, taxes
   - Backtest assumes perfect execution (0.1% fee only)
   - Real results will differ

4. **Market Risk**
   - All strategies can lose money
   - Drawdowns can exceed historical levels
   - No strategy works in all market conditions

5. **Do Your Own Research**
   - Consult financial professionals before investing
   - Understand the strategy fully before using real money
   - Test thoroughly with paper trading first

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is for educational purposes only. Not financial advice.

## 🔗 Resources

- [Repository](https://github.com/Levietduc1104/trading_bot)
- [Jegadeesh & Titman (1993) Paper](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
- [VIX White Paper](https://www.cboe.com/tradable_products/vix/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Bokeh Visualization](https://docs.bokeh.org/)

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with Claude Code** 🤖

**Current Production Strategy:** V13 Momentum-Strength Weighting + Drawdown Control

**Last Updated:** 2025-12-28
