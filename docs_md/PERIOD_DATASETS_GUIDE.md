# Period-Specific Stock Datasets - Summary

## Overview

Created 3 separate datasets for different historical testing periods. Each dataset contains only stocks with actual price data during that specific period.

## Datasets Created

### 1. stock_data_1963_1983 (20 years)
- **Period:** 1963-01-01 to 1983-12-31
- **Stocks:** 184 valid stocks (39.5% of S&P 500)
- **Location:** `sp500_data/stock_data_1963_1983/`
- **Use case:** Test strategy on 1960s-1980s market conditions
- **Notable:** Includes high-inflation 1970s, early tech stocks

**Sample stocks:** AAPL (from 1980), BA, MO, XOM, IBM, GE, MRK, MMM

### 2. stock_data_1983_2003 (20 years)
- **Period:** 1983-01-01 to 2003-12-31
- **Stocks:** 353 valid stocks (75.8% of S&P 500)
- **Location:** `sp500_data/stock_data_1983_2003/`
- **Use case:** Test strategy on dot-com boom/bust era
- **Notable:** Includes 1987 crash, 1990s bull market, 2000-2003 bear

**Sample stocks:** CSCO (from 1990), MSFT, AMZN, GOOGL, AAPL, NVDA

### 3. stock_data_1990_2024 (34 years)
- **Period:** 1990-01-01 to 2024-12-31
- **Stocks:** 466 valid stocks (100% of filtered S&P 500)
- **Location:** `sp500_data/stock_data_1990_2024/`
- **Use case:** Current production dataset (same as sp500_filtered)
- **Notable:** Includes 2000 crash, 2008 crisis, 2020 COVID, full modern era

**All stocks from sp500_filtered are included**

## Key Features

### Data Quality
- ✅ Only stocks with **at least 100 trading days** in the period
- ✅ Only stocks with **valid price data** (no all-zeros, no all-NaN)
- ✅ Adjusted prices + dividends + stock splits
- ✅ VIX data filtered to matching period

### Coverage by Period

| Period | Years | Stocks | Coverage | Major Events |
|--------|-------|--------|----------|--------------|
| 1963-1983 | 20 | 184 | 39.5% | Oil crisis, stagflation, early computing |
| 1983-2003 | 20 | 353 | 75.8% | Dot-com boom/bust, 1987 crash |
| 1990-2024 | 34 | 466 | 100% | 2000 crash, 2008 crisis, COVID |

### Why Different Coverage?

**1963-1983 has fewer stocks because:**
- Many current S&P 500 companies didn't exist yet
- Technology sector was much smaller
- Some companies were private or not yet founded
- Data availability for older periods is limited

**1983-2003 has more stocks because:**
- Tech boom created many new public companies (CSCO, MSFT, AMZN, etc.)
- More mature markets with better data coverage
- S&P 500 became more diversified

**1990-2024 has all stocks because:**
- This is the reference period used to create sp500_filtered
- Nearly all current S&P 500 companies were public by 1990
- Complete modern data availability

## Files in Each Dataset

Each dataset folder contains:
```
stock_data_YYYY_YYYY/
├── AAPL.csv              (individual stock data)
├── MSFT.csv
├── ... (other stocks)
├── VIX.csv               (volatility index for the period)
└── stock_list.txt        (list of all valid stocks)
```

## How to Use

### Run Backtest on Specific Period

Edit `src/core/execution.py` line 227 to change the dataset:

```python
# Test 1960s-1980s
def run_backtest(data_dir='sp500_data/stock_data_1963_1983', initial_capital=100000):

# Test dot-com era
def run_backtest(data_dir='sp500_data/stock_data_1983_2003', initial_capital=100000):

# Test modern era (current)
def run_backtest(data_dir='sp500_data/stock_data_1990_2024', initial_capital=100000):
```

Then run:
```bash
venv/bin/python3 -m src.core.execution
```

### Compare Strategy Across Eras

You can now test if your V28 Momentum Leaders strategy works well across different market regimes:

1. **1963-1983:** High inflation, oil shocks, stagflation
   - Expected: Lower returns, value stocks may dominate
   - Momentum may struggle in sideways markets

2. **1983-2003:** Tech boom, globalization, dot-com crash
   - Expected: High volatility, momentum should excel in 1990s
   - Risk management crucial for 2000-2003 bear market

3. **1990-2024:** Modern diversified markets (current results)
   - Actual: 10.6% annual return, -27.5% max drawdown
   - Benchmark for comparison

## Data Verification

### Sample: Boeing (BA) in 1963-1983
- Start: 1963-01-02, price: $0.145
- End: 1983-12-30, price: $2.94
- Trading days: 5,273 days (~21 years)

### Sample: Cisco (CSCO) in 1983-2003
- Start: 1990-02-16 (IPO), price: $0.057
- End: 2003-12-31, price: $17.87
- Trading days: 3,499 days (~14 years after IPO)

## Expected Performance Differences

Based on historical market characteristics:

| Period | Expected Annual Return | Expected Max DD | Notes |
|--------|----------------------|----------------|-------|
| 1963-1983 | 5-8% | -30% to -40% | High inflation, sideways markets |
| 1983-2003 | 12-15% | -40% to -50% | Tech boom, dot-com crash |
| 1990-2024 | **10.6%** (actual) | **-27.5%** (actual) | Current results |

## Testing Recommendations

1. **Start with 1990-2024** (current dataset) as baseline
2. **Test 1983-2003** to see how strategy handles dot-com crash
3. **Test 1963-1983** to validate in different economic regime

## Created By

- Script: `create_period_datasets.py`
- Date: 2026-01-10
- Source: `sp500_data/sp500_filtered/` (466 clean S&P 500 stocks)
- Method: Date-range filtering with data validation

## Notes

- All datasets use the **same VIX calculation** (30-day rolling volatility)
- Survivorship bias is present (only includes stocks that made it to S&P 500)
- Earlier periods have fewer stocks due to historical availability
- Data is adjusted for splits and dividends consistently across all periods
