# V10 Production Strategy: Inverse Volatility Position Weighting

## Current Production Strategy (as of Dec 27, 2024)

**V10: VIX Regime + Inverse Volatility Weighting**

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Annual Return** | 8.2% |
| **Max Drawdown** | -22.8% |
| **Sharpe Ratio** | 1.21 |
| **Final Value (19.4 years)** | $458,354 (from $100k) |
| **Negative Years** | 2/20 (10%) |
| **Positive Years** | 18/20 (90%) |

## How to Run V10

```python
from src.backtest.portfolio_bot_demo import PortfolioRotationBot

# Initialize bot
bot = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
bot.prepare_data()

# Run V10 strategy
portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,
    use_vix_regime=True,              # VIX-based regime detection
    use_inverse_vol_weighting=True,   # Inverse volatility weighting
    trading_fee_pct=0.001              # 0.1% per trade
)
```

## What V10 Does

### Components

1. **Stock Selection (V7)**
   - Momentum scoring (150 points max)
   - Quality filters (disqualify weak stocks)
   - Sector relative strength bonus

2. **Regime Detection (V8)**
   - VIX-based market regime
   - Dynamic cash reserves (5-70%)
   - Forward-looking volatility indicator

3. **Position Sizing (V10)** ✨ NEW
   - **Inverse volatility weighting**
   - Formula: `weight_i = (score_i / vol_i) / Σ(score_j / vol_j)`
   - Higher momentum + lower volatility = larger position

### Position Weighting Formula

```python
# For each stock in top 10:
volatility = std_dev(returns_40_days) * sqrt(252)  # Annualized
risk_adjusted_score = momentum_score / volatility
weight = risk_adjusted_score / sum(all_risk_adjusted_scores)
position_size = total_capital * weight
```

**Effect:**
- **High-quality, low-volatility stocks** → Larger positions (15-20%)
- **High-quality, high-volatility stocks** → Smaller positions (5-10%)
- **Equal-weighted baseline** → Each stock gets 10%

## Why V10 Wins

### Advantages vs V8 (Equal Weighting)

| Aspect | V8 Equal | V10 Inverse Vol | Winner |
|--------|----------|-----------------|--------|
| Annual Return | 8.4% | 8.2% | V8 (-0.3%) |
| Max Drawdown | -23.2% | -22.8% | V10 (+0.4%) |
| Sharpe Ratio | 1.15 | **1.21** | **V10 (+0.06)** |
| Negative Years | 5/20 (25%) | **2/20 (10%)** | **V10 (-60%)** |
| Consistency | Volatile | **Smooth** | **V10** |

### Key Improvements

1. **60% Fewer Losing Years**
   - 2005: -0.8% → +0.4% ✅
   - 2015: -1.6% → +0.8% ✅
   - 2020: -2.4% → +0.1% ✅

2. **Better Risk-Adjusted Returns**
   - Sharpe 1.21 vs 1.15 (+5% improvement)
   - More consistent monthly returns

3. **Better Crisis Protection**
   - 2008: -14.5% vs -15.1% (V8)
   - 2020: +0.1% vs -2.4% (V8)

## Trade-offs

**What You Give Up:**
- **-0.3% annual return** (8.2% vs 8.4%)
- **-$21k over 19 years** ($458k vs $480k)
- **Lower peak gains in bull runs**
  - 2017: 12.1% vs 16.7% (V8)
  - 2016: 14.2% vs 17.1% (V8)

**What You Get:**
- **Smoother equity curve**
- **Fewer negative years**
- **Better sleep at night** (less volatility)
- **Better Sharpe ratio** (risk-adjusted)

## No Hindsight Bias

V10 is **bias-free**:

✅ Uses only **historical volatility** (40-day lookback)
✅ Calculated at each **rebalancing date** (no future data)
✅ Logical formula (not curve-fitted)
✅ Works across **all market periods** (2005-2024)

**Volatility calculation:**
```python
# At rebalancing date:
recent_prices = stock_prices[-41:]  # Last 40 days + 1
returns = recent_prices.pct_change()
volatility = returns.std() * sqrt(252)  # Annualize
```

## Yearly Returns (V10)

| Year | Return | Status | Notes |
|------|--------|--------|-------|
| 2005 | +0.4% | ✅ | Fixed (was -0.8% in V8) |
| 2006 | +4.3% | ✅ | |
| 2007 | +19.7% | ✅ | |
| 2008 | -14.5% | ❌ | Financial crisis (better than V8's -15.1%) |
| 2009 | -0.3% | ❌ | Recovery year |
| 2010 | +9.4% | ✅ | |
| 2011 | +11.7% | ✅ | |
| 2012 | +16.7% | ✅ | |
| 2013 | +6.9% | ✅ | |
| 2014 | +15.2% | ✅ | |
| 2015 | +0.8% | ✅ | Fixed (was -1.6% in V8) |
| 2016 | +14.2% | ✅ | |
| 2017 | +12.1% | ✅ | |
| 2018 | +3.1% | ✅ | |
| 2019 | +7.2% | ✅ | |
| 2020 | +0.1% | ✅ | COVID crisis - Fixed (was -2.4% in V8) |
| 2021 | +26.5% | ✅ | |
| 2022 | +10.4% | ✅ | |
| 2023 | +16.3% | ✅ | |
| 2024 | +4.8% | ✅ | |

**Win Rate: 90% (18/20 years positive)**

## Implementation Details

### Methods Added

1. **`calculate_stock_volatility(ticker, date, lookback_days=40)`**
   - Location: `src/backtest/portfolio_bot_demo.py:748-790`
   - Calculates annualized volatility from historical returns
   - Default 40-day lookback (2 months)
   - Returns 20% default if insufficient data
   - Bounds: 5% to 100% (prevents extremes)

2. **Inverse volatility allocation logic**
   - Location: `src/backtest/portfolio_bot_demo.py:999-1024`
   - Calculates `score / volatility` for each stock
   - Normalizes to allocate total capital
   - Fallback to equal weight if calculation fails

### Parameters

```python
use_inverse_vol_weighting=True   # Enable V10
lookback_days=40                  # Volatility lookback period
```

## Testing

**Test Script:** `test_v10_inverse_volatility.py`

```bash
python test_v10_inverse_volatility.py
```

**Results:** `test_v10_results.log`

## Version History

| Version | Date | Strategy | Annual Return | Sharpe | Status |
|---------|------|----------|---------------|--------|--------|
| V1-V5 | Nov 2024 | Stock scoring refinements | 7.7% | 1.02 | Deprecated |
| V6 | Dec 2024 | Momentum filters | - | - | Integrated |
| V7 | Dec 2024 | Seasonal + sector strength | - | - | Integrated |
| V8 | Dec 26, 2024 | VIX regime detection | 8.4% | 1.15 | Previous best |
| **V10** | **Dec 27, 2024** | **Inverse vol weighting** | **8.2%** | **1.21** | **✅ PRODUCTION** |

(V9 = Market trend strength - tested but not adopted)

## Next Potential Improvements

1. **Momentum Persistence** - Hold winners longer, reduce turnover
2. **Sector Diversification** - Cap at 40% per sector
3. **Trailing Stops** - Exit deteriorating positions mid-month
4. **Dynamic Portfolio Size** - 5-15 stocks based on market conditions

## Usage Notes

- **Rebalancing:** Monthly (day 7-10 of each month)
- **Cash Reserves:** 5-70% based on VIX regime
- **Portfolio Size:** Top 10 stocks (variable weighting)
- **Transaction Costs:** 0.1% per trade
- **Minimum Data:** 200 days for regime, 40 days for volatility

## Monitoring

**Key Metrics to Track:**
- Monthly return consistency
- Sharpe ratio (target: >1.20)
- Max drawdown (target: <25%)
- Number of negative months
- Turnover rate

## Files

- **Strategy:** `src/backtest/portfolio_bot_demo.py`
- **Test:** `test_v10_inverse_volatility.py`
- **Results:** `test_v10_results.log`
- **This doc:** `V10_PRODUCTION_STRATEGY.md`

---

**Last Updated:** December 27, 2024
**Status:** ✅ Production
**Maintainer:** Trading Bot Team
