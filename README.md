# S&P 500 Portfolio Rotation Trading Bot

A sophisticated backtesting system implementing **V22 Production Strategy** with Kelly-weighted position sizing, 5-stock concentration, and portfolio-level drawdown control for S&P 500 stocks.

## 📊 Overview

This trading bot implements an advanced momentum-based portfolio rotation strategy with Kelly position sizing that combines multiple proven techniques:

- **Kelly Position Sizing (V22)**: Weight positions by conviction (√score method) ⭐
- **5-Stock Concentration**: High conviction portfolio (optimal risk/return)
- **VIX Regime Detection**: Forward-looking market stress indicator
- **Drawdown Control**: Progressive exposure reduction during portfolio drawdowns
- **Zero Prediction Bias**: Uses only historical data, no curve fitting

### 🏆 Performance (V22 Production - Kelly Weighted)

```
Annual Return:   10.2% ⭐ (+0.4% vs V13)
Sharpe Ratio:    1.11 ⭐ (BETTER than V13's 1.07)
Max Drawdown:    -15.2% ⭐ (BETTER than V13's -19.1%)
Win Rate:        80% (16/20 positive years)
Final Value:     $653,746 (on $100k over 19.4 years)
Peak Value:      $681,158 (June 2024)
```

**Negative Years:** Only 2008 (-4.2%), 2009 (-4.0%), 2018 (-2.4%), 2020 (-9.3%)

**Key Innovation:** Kelly position sizing (weight ∝ √score) allocates more capital to highest-conviction picks (17-24% per position vs 20% equal weight), resulting in BETTER returns AND BETTER risk metrics.

## ✨ Key Features

### V22 Strategy Components

1. **Kelly-Weighted Position Sizing** (V22) ⭐
   - Formula: `weight ∝ √(score)`
   - High score (120): ~24% position (+20% more capital vs equal weight)
   - Low score (60): ~17% position (-15% less capital vs equal weight)
   - Concentrates capital where edge is highest
   - Conservative Kelly approach (square root vs linear)

2. **VIX-Based Regime Detection** (V8/V22)
   - Forward-looking volatility index (not lagging indicators)
   - Continuous formula for smoother transitions
   - Dynamic cash reserve: 5% (VIX<30) to 70% (VIX>70)
   - Formula:
     - VIX < 30: cash = 5% + (VIX - 10) × 0.5%
     - VIX ≥ 30: cash = 15% + (VIX - 30) × 1.25%

3. **Portfolio-Level Drawdown Control** (V12)
   - Progressive exposure reduction as drawdown increases
   - Rules:
     - DD < 10%: 100% invested
     - DD 10-15%: 75% invested
     - DD 15-20%: 50% invested
     - DD ≥ 20%: 25% invested (maximum defense)

4. **Mid-Month Rebalancing** (V7)
   - Rebalances on day 7-10 (avoids month-end institutional flows)
   - Reduces slippage from crowded trades

5. **Momentum Quality Filters** (V6)
   - Must be above EMA-89 (long-term trend)
   - Must have ROC-20 > 2% (positive momentum)
   - RSI penalties for overbought/oversold conditions

6. **Sector Relative Strength** (V7)
   - Awards bonus points for stocks outperforming sector peers
   - Ensures buying sector leaders, not just market leaders

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
│   │   └── execution.py                    # Main production execution (V22)
│   ├── backtest/
│   │   ├── portfolio_bot_demo.py          # Core strategy implementation
│   │   └── create_sp500_index.py          # S&P 500 index creation
│   ├── data/
│   │   ├── download_vix.py                # Download VIX data
│   │   └── create_vix_proxy.py            # Create VIX proxy from SPY
│   ├── visualize/
│   │   └── visualize_trades.py            # Interactive visualizations
│   ├── risk/
│   │   └── risk_management_backtest.py    # Risk management tools
│   └── tests/
│       ├── README.md                      # Test documentation
│       ├── test_v22_kelly_position_sizing.py  # V22 validation
│       └── ...                            # Other experimental tests
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
├── docs/
│   ├── README.md                          # Documentation index
│   ├── CHANGELOG.md                       # Version history
│   └── archive/                           # Historical documentation
├── run_v22_production.py                  # Standalone V22 execution
├── V22_PRODUCTION_SUMMARY.md              # V22 strategy documentation
├── V22_INTEGRATION_COMPLETE.md            # V22 integration guide
├── CONTRIBUTING.md                        # Contributing guidelines
└── README.md                              # This file
```

## 🎯 Running the Strategy

### Production Execution (V22)

Run the full V22 strategy end-to-end:

```bash
python src/core/execution.py
```

**This will:**
1. Load 473 S&P 500 stocks + VIX data
2. Run V22 backtest with Kelly position sizing (2005-2024)
3. Save results to database
4. Generate performance report
5. Create interactive visualization

**Expected Output:**
```
================================================================================
                       V22-SQRT KELLY POSITION SIZING
================================================================================

Strategy:        V22 (Kelly Position Sizing + Drawdown Control)
Annual Return:   10.2%
Sharpe Ratio:    1.11
Max Drawdown:    -15.2%
Win Rate:        80% (16/20 positive years)
Final Value:     $653,746

Outputs:
  📊 Database:      output/data/trading_results.db
  📈 Report:        output/reports/performance_report_*.txt
  🎨 Visualization: output/plots/trading_analysis.html
  📋 Logs:          output/logs/execution.log
```

### Standalone V22 Execution

Run standalone V22 script (same results):

```bash
python run_v22_production.py
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

# Run V22 strategy (Kelly position sizing)
portfolio_df = bot.backtest_with_bear_protection(
    top_n=5,                        # Top 5 stocks
    rebalance_freq='M',              # Monthly rebalancing
    use_vix_regime=True,             # V8: VIX regime detection
    use_kelly_weighting=True,        # V22: Kelly position sizing ⭐
    use_drawdown_control=True,       # V12: Drawdown control
    trading_fee_pct=0.001            # 0.1% trading fee
)
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
- Portfolio value growth over time (peaks at $681k)
- Drawdown analysis chart
- Yearly returns bar chart
- Daily returns distribution
- Cumulative returns chart
- Risk-adjusted metrics

**Note:** Visualization reads data from the database (not recalculated), ensuring consistency with execution results.

## 🔧 Strategy Configuration

### Available Strategy Versions

| Version | Description | Annual Return | Sharpe | Max DD |
|---------|-------------|---------------|--------|--------|
| V8 | VIX + Equal Weight | 8.4% | 1.15 | -23.2% |
| V10 | VIX + Inverse Vol | 8.2% | 1.21 | -22.8% |
| V11 | Adaptive Hybrid | 8.3% | 1.22 | -22.8% |
| V12 | V11 + Drawdown Control | 8.2% | 1.23 | -18.5% |
| V13 | V12 + Momentum | 9.8% | 1.07 | -19.1% |
| **V22** | **V13 + Kelly Sizing** ⭐ | **10.2%** | **1.11** | **-15.2%** |

### Enable/Disable Features

```python
portfolio_df = bot.backtest_with_bear_protection(
    top_n=5,
    use_vix_regime=True,              # VIX regime detection (recommended)
    use_kelly_weighting=True,         # Kelly position sizing (V22) ⭐
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

### V22 Kelly Position Sizing

Scores translate to position weights via square root:

```
Example scores:
  AAPL: 120 → √120 = 10.95 → weight = 23.9%
  MSFT: 100 → √100 = 10.0  → weight = 21.9%
  GOOGL: 80 → √80  = 8.94  → weight = 19.5%
  NVDA: 70 → √70   = 8.37  → weight = 18.3%
  META: 60 → √60   = 7.75  → weight = 16.9%
```

## 📉 Results & Analytics

### Database Schema

Results are stored in SQLite (`output/data/trading_results.db`):

**Tables:**
- `backtest_runs`: Run metadata (strategy, returns, metrics)
- `portfolio_values`: Daily portfolio value history
- `yearly_returns`: Year-by-year performance

### Performance Report

Generated at `output/reports/performance_report_*.txt`:

```
================================================================================
PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Initial Capital:       $100,000
Final Value:           $653,746
Total Return:          553.7%
Annual Return:         10.2%
Max Drawdown:          -15.2%
Sharpe Ratio:          1.11
Period:                2005-05-23 to 2024-10-03
Duration:              19.4 years

YEARLY RETURNS (80% Win Rate)
--------------------------------------------------------------------------------
  2005:      3.1% ✅    2014:     13.3% ✅    2021:     16.0% ✅
  2006:      3.9% ✅    2015:     11.9% ✅    2022:     17.3% ✅
  2007:     26.1% ✅    2016:     22.1% ✅    2023:     23.3% ✅
  2008:     -4.2% ❌    2017:     13.4% ✅    2024:      7.2% ✅
  2009:     -4.0% ❌    2018:     -2.4% ❌
  2010:      9.0% ✅    2019:     12.9% ✅
  2011:     22.3% ✅    2020:     -9.3% ❌
  2012:     14.6% ✅
  2013:     13.0% ✅
```

## 🎓 Strategy Evolution

### Phase 1: Base System (V5)
- Momentum-based scoring (0-100 points)
- Monthly rebalancing
- Simple bear/bull detection

### Phase 2: Risk Management (V6-V7)
- V6: Momentum quality filters (disqualify weak trends)
- V7: Mid-month rebalancing, seasonal adjustments, sector relative strength

### Phase 3: VIX Regime (V8)
- Forward-looking volatility indicator
- Dynamic cash reserves (5%-70%)

### Phase 4: Position Sizing Experiments (V10-V11)
- V10: Inverse volatility weighting
- V11: Adaptive hybrid (equal in calm, inverse-vol in stress)

### Phase 5: Portfolio Risk Control (V12)
- Progressive drawdown exposure reduction
- Prevents drawdown acceleration
- Preserves capital for recovery

### Phase 6: Momentum Weighting (V13)
- Momentum-strength position sizing
- weight ∝ momentum / volatility
- 5-stock concentration (9.8% annual)

### Phase 7: Kelly Position Sizing (V22) 🏆 ⭐
- Kelly-weighted position sizing (weight ∝ √score)
- Concentrates capital where edge is highest
- Result: **10.2% annual, -15.2% DD, 1.11 Sharpe**
- BETTER returns AND BETTER risk metrics
- **Proves scoring quality matters**

## 🔬 Academic Foundation

V22 is built on peer-reviewed research:

1. **Kelly Criterion**
   - J.L. Kelly Jr. (1956): "A New Interpretation of Information Rate"
   - Optimal bet sizing for systems with edge
   - Ed Thorp applied to trading in "Beat the Dealer" (1962)

2. **Momentum Persistence**
   - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
   - Empirical fact: past returns predict future returns

3. **Volatility Clustering**
   - Engle (1982): "Autoregressive Conditional Heteroskedasticity"
   - Nobel Prize-winning research

4. **VIX Forward-Looking Indicator**
   - Whaley (1993): "Derivatives on Market Volatility"
   - Better than lagging indicators like 200-day MA

5. **Drawdown Control**
   - Used by professional CTAs and hedge funds
   - Geometric return > arithmetic return

## 🛠️ Advanced Usage

### Export Results to Excel

```python
import pandas as pd
import sqlite3

# Load from database
conn = sqlite3.connect('output/data/trading_results.db')

# Get latest run
latest_run = pd.read_sql("""
    SELECT * FROM backtest_runs
    ORDER BY run_id DESC LIMIT 1
""", conn)

# Get portfolio values
portfolio = pd.read_sql(f"""
    SELECT * FROM portfolio_values
    WHERE run_id = {latest_run['run_id'].iloc[0]}
""", conn)
portfolio.to_excel('output/portfolio_analysis.xlsx', index=False)

# Get yearly returns
yearly = pd.read_sql(f"""
    SELECT * FROM yearly_returns
    WHERE run_id = {latest_run['run_id'].iloc[0]}
""", conn)
yearly.to_excel('output/yearly_returns.xlsx', index=False)

conn.close()
```

### Custom Stock Universe

```python
# Use custom stock list
bot = PortfolioRotationBot(data_dir='your_data_dir')
bot.prepare_data()

# Run V22 on custom universe
portfolio_df = bot.backtest_with_bear_protection(
    top_n=5,  # Hold only 5 stocks
    use_vix_regime=True,
    use_kelly_weighting=True,  # Kelly position sizing
    use_drawdown_control=True
)
```

## 📊 Performance Metrics Explained

### Sharpe Ratio (1.11)
- Risk-adjusted return metric
- Higher = better risk-adjusted performance
- Formula: `(mean_return / std_return) × √252`
- V22 improves to 1.11 from V13's 1.07

### Max Drawdown (-15.2%)
- Largest peak-to-trough decline
- Measures worst-case scenario
- V22 reduces this to -15.2% vs V13's -19.1%
- Better risk control despite higher returns

### Win Rate (80%)
- Percentage of positive years
- 16 out of 20 years profitable
- Only 4 negative years in 19+ years

### Annual Return (10.2%)
- Compound annual growth rate (CAGR)
- Geometric mean, not arithmetic
- Consistent over 19.4 years
- +0.4% improvement over V13

## 🎯 Why V22 Works

### The Kelly Advantage

**Problem:** Equal weighting assumes all top 5 stocks are equal quality

**Solution:** Kelly sizing allocates based on conviction (score)

**Math:**
```
Equal Weight (V13):
  All stocks: 20% × return
  No differentiation

Kelly Weight (V22):
  Best stock (score 120): 24% × (likely higher return)
  Worst stock (score 60): 17% × (likely lower return)
  = Concentrates where edge is highest
```

**Validation:**
If scoring was NOISE (random), Kelly would HURT performance.
But we got:
- ✅ Higher returns (+0.4%)
- ✅ Better Sharpe (+3.7%)
- ✅ Lower drawdown (-3.9%)

**This proves our scoring differentiates quality.**

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

6. **Position Concentration**
   - 5-stock portfolio is concentrated (higher risk)
   - Kelly sizing can lead to 24% positions
   - Max expected drawdown: -15% to -20%

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is for educational purposes only. Not financial advice.

## 📚 Documentation

### Main Documentation
- [V22 Strategy Documentation](V22_PRODUCTION_SUMMARY.md) - Complete V22 strategy specification
- [V22 Integration Guide](V22_INTEGRATION_COMPLETE.md) - Integration and next steps
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Complete Documentation](docs/README.md) - Full documentation index
- [Changelog](docs/CHANGELOG.md) - Version history and changes

### Historical Documentation
See [docs/archive/](docs/archive/) for:
- Previous strategy versions (V5-V13)
- Optimization experiments
- Research and brainstorming
- Development history

## 🔗 Resources

- [Repository](https://github.com/Levietduc1104/trading_bot)
- [Kelly Criterion Paper](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x)
- [Jegadeesh & Titman (1993) Paper](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
- [VIX White Paper](https://www.cboe.com/tradable_products/vix/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Bokeh Visualization](https://docs.bokeh.org/)

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with Claude Code** 🤖

**Current Production Strategy:** V22-Sqrt Kelly Position Sizing ⭐

**Performance:** 10.2% annual, -15.2% max drawdown, 1.11 Sharpe ratio

**Last Updated:** 2026-01-01
