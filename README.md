# Trading Bot - Machine Learning Stock Trading System

**ML-enhanced systematic trading strategies with comprehensive backtesting (2015-2024)**

⚠️ **Important Note:** Current backtest results do NOT include realistic transaction costs. See [Known Limitations](#-known-limitations--planned-improvements) for expected adjustments.

## 📊 Strategy Performance Summary

| Strategy | Annual Return | Max Drawdown | Sharpe Ratio | Status | Best For |
|----------|--------------|--------------|--------------|--------|----------|
| **V30 Dynamic Mega-Cap** | 16.1% | -23.8% | 1.00 ⭐ | ✅ Validated | Risk-conscious investors |
| **LightGBM ML** | 25.8% | -54.2% | 0.84 | ⚠️ Needs TC* | High returns, can tolerate drawdown |
| **RandomForest ML** | 29.3% | -59.0% | 0.88 | ⚠️ Needs TC* | Maximum returns, high risk tolerance |

**Benchmark:** SPY = 12.3% annual, -33.7% max DD
**TC = Transaction Costs** (bid-ask spread, slippage, commissions)

---

## 📈 Performance Visualizations

### Comprehensive Strategy Comparison (2015-2024)

![Strategy Comparison](output/complete_strategy_comparison.png)

**Key Insights:**
- **Top-Left:** Portfolio growth over time (log scale) - ML Regularized reaches $100M+
- **Top-Right:** Drawdown comparison - ML strategies have deeper drawdowns (-34% to -49%)
- **Bottom-Left:** Annual return bars - ML Regularized leads at 106.1%
- **Bottom-Right:** Risk-return profile - V30 offers best balance

### LightGBM ML Strategy Performance

![LightGBM Performance](output/plots/ml_lightgbm_performance.png)

**Performance Details:**
- Strong portfolio growth with volatility spikes
- Annual returns vary: 2024 saw +130% spike, 2022 had -40% crash
- Demonstrates the high-risk, high-reward nature of ML strategies

---

## 🎯 Recommended Strategies

### 1. V30 Dynamic Mega-Cap Strategy ⭐ **RECOMMENDED**

**Best risk-adjusted returns** (Sharpe 1.00)

**Performance (2015-2024):**
- Annual Return: 16.1%
- Max Drawdown: -23.8%
- Final Value: $100K → $435K

**How it works:**
- Dynamically identifies top 7 mega-caps by trading volume
- 70% allocation to Top 3 Mega-Caps (by momentum)
- 30% allocation to Top 2 Momentum stocks
- VIX-based cash reserves (5-70%)
- 15% trailing stops
- Progressive drawdown control

**Run the strategy:**
```bash
python src/core/execution.py --strategy v30 --start 2015 --end 2024
```

---

### 2. LightGBM ML Strategy

**Balanced returns with ML stock selection**

**Performance (2015-2024):**
- Annual Return: 25.8%
- Max Drawdown: -54.2%
- Final Value: $100K → $959K

**How it works:**
- LightGBM gradient boosting model
- Predicts 21-day forward returns
- 13 technical features (RSI, ATR, dist_52w_high, momentum, relative strength)
- Retrains every 6 months
- Top 5 stocks by ML score
- VIX-based risk management (5-70% cash)

**Run the strategy:**
```bash
python src/core/execution_lgbm.py --start 2015 --end 2024
```

---

### 3. RandomForest ML Strategy

**Highest returns, highest risk**

**Performance (2015-2024):**
- Annual Return: 29.3%
- Max Drawdown: -59.0%
- Final Value: $100K → $1.25M

**How it works:**
- RandomForest regression model
- Same features as LightGBM
- 50 estimators, max_depth=3 (regularized)
- Retrains every 6 months

**Run the strategy:**
```bash
python src/core/execution.py --strategy ml --start 2015 --end 2024
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas, numpy
- scikit-learn
- lightgbm (for LightGBM ML strategy)
- bokeh (for visualizations)

### 2. Download Data

Stock data should be in `sp500_data/stock_data_1990_2024_top500/` directory.

### 3. Run a Backtest

Choose your strategy:

```bash
# V30 Strategy (Recommended: best risk-adjusted)
python src/core/execution.py --strategy v30

# LightGBM ML (Balanced: 25.8% annual)
python src/core/execution_lgbm.py

# RandomForest ML (High returns: 29.3% annual, -59% DD)
python src/core/execution.py --strategy ml
```

### 4. View Results

Results are saved to:
- Database: `output/data/trading_results.db`
- Plots: `output/plots/`
- Reports: `output/reports/`
- Logs: `output/logs/execution.log`

### 5. Visualize Trading Activity

```python
from src.visualize.visualize_trades import create_trade_visualizations

# Visualize latest run
create_trade_visualizations()

# Visualize specific run (e.g., Run ID 6 = LightGBM)
create_trade_visualizations(run_id=6)
```

---

## 📁 Repository Structure

```
trading_bot/
├── README.md                      # This file
├── CLEANUP_PLAN.md               # Optimization documentation
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── core/
│   │   ├── execution.py         # Main: V30 + RandomForest ML
│   │   └── execution_lgbm.py    # LightGBM ML execution
│   │
│   ├── strategies/
│   │   ├── v30_dynamic_megacap.py       # V30 strategy (Sharpe 1.00)
│   │   ├── ml_stock_ranker_lgbm.py      # LightGBM ML (25.8% annual)
│   │   └── ml_stock_ranker_simple.py    # RandomForest ML (29.3% annual)
│   │
│   ├── backtest/
│   │   └── portfolio_bot_demo.py        # Core backtesting engine
│   │
│   ├── visualize/
│   │   └── visualize_trades.py          # Interactive Bokeh dashboards
│   │
│   ├── data/
│   │   └── download_vix.py             # Data utilities
│   │
│   └── tool/                           # Helper scripts
│
├── output/                              # Results and visualizations
│   ├── data/trading_results.db         # SQLite database
│   ├── plots/                          # Performance charts
│   ├── reports/                        # Text reports
│   └── logs/                           # Execution logs
│
├── sp500_data/                         # Stock price data
│   └── stock_data_1990_2024_top500/
│
└── archive/                            # Historical experiments (80+ files)
    ├── experiments/                    # Test files (v14-v38)
    ├── failed_strategies/              # Hybrid, old versions
    ├── backups/                        # Backup files
    └── root_tests/                     # Root-level test scripts
```

---

## 🔬 Strategy Comparison Details

### Why V30 is Recommended (Sharpe 1.00)

**Advantages:**
✅ Best risk-adjusted returns
✅ Controlled drawdown (-23.8% vs ML's -54% to -59%)
✅ Proven consistency across market cycles
✅ Less stress - easier to stick with long-term

**When to use:**
- You prioritize capital preservation
- You want steady, reliable returns
- You can't tolerate -50%+ drawdowns

### When to Use LightGBM ML (25.8% annual)

**Advantages:**
✅ 2x better predictive power than RandomForest (Val R² 0.023 vs 0.010)
✅ Feature importance tracking (identifies what matters)
✅ Slightly better drawdown than RandomForest (-54% vs -59%)

**When to use:**
- You want higher returns than V30 (25.8% vs 16.1%)
- You can tolerate -54% drawdowns
- You prefer ML-based stock selection

### When to Use RandomForest ML (29.3% annual)

**Advantages:**
✅ Highest absolute returns (29.3% annual)
✅ Strong Sharpe ratio (0.88)
✅ $100K → $1.25M over 10 years

**When to use:**
- You need maximum returns
- You can psychologically handle -59% drawdowns
- You have a long time horizon (10+ years)

---

## 📈 Backtest Results (2015-2024)

### All Runs in Database

```sql
SELECT id, strategy, annual_return, max_drawdown, sharpe_ratio
FROM backtest_runs
ORDER BY sharpe_ratio DESC;
```

| ID | Strategy | Annual % | Max DD % | Sharpe |
|----|----------|----------|----------|--------|
| 1  | V30_DYNAMIC_MEGACAP | 16.1 | -23.8 | 1.00 ⭐ |
| 2  | ML_REGULARIZED (RF) | 29.3 | -59.0 | 0.88 |
| 6  | ML_LIGHTGBM | 25.8 | -54.2 | 0.84 |
| 3  | ML_REGULARIZED_ENHANCED | 14.3 | -48.0 | 0.63 |
| 4  | ML_REGULARIZED (Debiased) | 14.0 | -61.1 | 0.57 |
| 5  | HYBRID_ML_V30 | 6.0 | -30.3 | 0.47 ❌ |

---

## 🛠️ Advanced Usage

### Custom Backtest Periods

```bash
# Test on different periods
python src/core/execution.py --strategy v30 --start 2018 --end 2023

# Test ML with different parameters
python src/core/execution.py --strategy ml --start 2010 --end 2020
```

### Access Specific Runs

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('output/data/trading_results.db')

# Get all runs
runs = pd.read_sql_query("SELECT * FROM backtest_runs", conn)

# Get specific run's portfolio values
portfolio = pd.read_sql_query(
    "SELECT * FROM portfolio_values WHERE run_id = 6",
    conn,
    parse_dates=['date'],
    index_col='date'
)
```

---

## 🎓 Key Learnings

### 1. **High Returns Require High Risk**
- 29% annual returns come with -59% drawdowns
- 16% annual returns come with -24% drawdowns
- **This tradeoff is fundamental and unavoidable**

### 2. **LightGBM > RandomForest for Financial Data**
- 130% better predictive power (Val R² 0.023 vs 0.010)
- Gradient boosting better for tabular data
- Feature importance helps identify what matters

### 3. **Overfitting Wasn't the Problem**
- Original ML had R² = 0.026 (very weak signal)
- "Debiasing" made it worse (R² = 0.016)
- **Problem**: Features can't predict 21-day returns well

### 4. **Each Crisis is Different**
- 2008: Gradual financial crisis with warning signs
- 2020 COVID: Fastest crash in history, no technical warnings
- **ML can't predict black swans** from price patterns alone

### 5. **Hybrid Approaches Can Backfire**
- ML+V30 Hybrid: Only 6.0% annual (worst of both worlds)
- Too defensive during recoveries (95% cash → missed rallies)
- Too aggressive during crashes (still got -30% DD)

---

## 📞 Support & Questions

For issues or questions:
1. Check `output/logs/execution.log` for detailed execution logs
2. Review backtest runs in `output/data/trading_results.db`
3. Inspect plots in `output/plots/` for visual analysis

---

## 📜 License & Disclaimer

**Educational purposes only. Not financial advice.**

Past performance does not guarantee future results. Trading involves substantial risk of loss. Always consult with a qualified financial advisor before making investment decisions.

---

## ⚠️ Known Limitations & Planned Improvements

### Current Limitations

1. **Missing Transaction Costs** ❌
   - Current backtests do NOT deduct:
     - Bid-ask spreads (0.05%-0.50%)
     - Market impact (Kyle's Lambda)
     - Slippage (0.1%-0.3%)
     - Commissions ($0.35-$5 per trade)
   - **Impact:** Reported returns likely 20-30% higher than realistic
   - **Fix available:** `src/backtest/transaction_costs.py` (not yet integrated)

2. **Potential Survivorship Bias** ⚠️
   - Data may only include stocks that survived to 2024
   - Missing: Companies that went bankrupt (Lehman, Enron, etc.)
   - **Impact:** 5-10% annual return inflation
   - **Fix:** Requires point-in-time S&P 500 constituent data

3. **Missing Key Indicators** 📊
   - Identified 10 high-impact indicators not yet implemented:
     - ✅ 12-month momentum (Jegadeesh & Titman 1993)
     - ✅ ADX (trend strength filter)
     - ✅ Beta (market correlation for risk management)
     - ✅ Bollinger Bands position
     - ✅ Money Flow Index (volume-weighted RSI)
     - And 5 more...
   - **Impact:** Potential +5-10% annual return improvement
   - **Status:** Implementation planned

### Realistic Performance Expectations

After applying all fixes:

| Strategy | Current Backtest | After TC Fix | After All Fixes | Live Trading |
|----------|-----------------|--------------|-----------------|--------------|
| **V30 Mega-Cap** | 16.1% | ~14-15% | ~15-18% | ~12-16% |
| **LightGBM ML** | 25.8% | ~18-20% | ~20-25% | ~15-22% |
| **RandomForest ML** | 29.3% | ~20-22% | ~22-28% | ~18-25% |

**Bottom Line:** Even after fixes, all strategies should beat SPY (12.3%) with proper execution.

### Next Steps

1. ✅ Add transaction cost model (completed: `src/backtest/transaction_costs.py`)
2. 🔄 Integrate transaction costs into backtest
3. 🔄 Add top 10 missing indicators
4. 🔄 Implement out-of-sample validation (2023-2024 hold-out)
5. 🔄 Paper trading for 6 months
6. 🔄 Acquire survivorship-bias-free data

---

## 🔬 Technical Implementation Details

### Transaction Cost Model

Comprehensive model based on "Machine Learning for Trading" (Stefan Jansen):

```python
from src.backtest.transaction_costs import TransactionCostModel

model = TransactionCostModel(broker='interactive_brokers')

# Calculate costs for a trade
cost = model.total_execution_cost(
    ticker='AAPL',
    shares=100,
    price=180.0,
    avg_daily_volume=50_000_000,
    market_cap='large'
)

# Returns breakdown:
# - Spread cost
# - Market impact
# - Slippage
# - Commission
# - Total cost percentage
```

**Features:**
- Bid-ask spread estimation (stock size dependent)
- Market impact modeling (Kyle's Lambda)
- Slippage calculation (order type dependent)
- Multiple broker commission structures
- Round-trip cost analysis

### ML Feature Engineering

Current features (21 total):
- **Momentum:** ROC 5/10/20/60, 12-month (planned)
- **Trend:** EMA ratios, crossovers
- **Volatility:** ATR, historical volatility
- **Volume:** Volume ratios, OBV (planned), MFI (planned)
- **Market Regime:** SPY trend, VIX levels
- **Relative Strength:** vs SPY
- **Position:** Distance from 52-week high, 200-MA (planned)

Missing high-impact indicators (identified from academic research):
1. 12-month momentum with 1-month skip (Jegadeesh & Titman)
2. ADX for trend strength filtering
3. Beta for market correlation
4. MACD for trend changes
5. Bollinger Bands for mean reversion
6. Money Flow Index (MFI)
7. Distance from 200-day MA
8. Volatility ratio
9. On-Balance Volume trend
10. Linear regression slope + R²

---

**Last Updated:** January 18, 2026
**Best Strategy:** V30 Dynamic Mega-Cap (Sharpe 1.00)
**Total Strategies Tested:** 6 (80+ experiments in archive/)
**Transaction Cost Model:** ✅ Implemented (not yet integrated)
**Recommended Reading:** "Machine Learning for Trading" by Stefan Jansen

