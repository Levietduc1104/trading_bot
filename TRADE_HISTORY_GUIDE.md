# Trade History CSV Export Guide

## Overview

Every time you run the Tier 2 strategy backtest, all trades (buys and sells) are automatically saved to a CSV file for your review.

## File Location

```
output/tier2_trades.csv
```

## How to Generate

Simply run any Tier 2 backtest:

```bash
# Run backtest - trades are automatically saved
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024

# Trades saved to: output/tier2_trades.csv
```

## CSV Format

The CSV contains the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| date | Trade execution date | 2024-01-02 |
| ticker | Stock symbol | AAPL |
| action | BUY or SELL | BUY |
| reason | Why the trade happened | rebalance_megacap |
| shares | Number of shares traded | 127.5 |
| price | Price per share | $182.50 |
| value | Total trade value (shares × price) | $23,268.75 |
| cost | Transaction cost (spread + commission) | $15.25 |

## Trade Reasons

| Reason | Description |
|--------|-------------|
| `rebalance_megacap` | Bought as part of quarterly rebalancing (mega-cap position) |
| `rebalance_momentum` | Bought as part of quarterly rebalancing (momentum position) |
| `rebalance` | Sold during quarterly rebalancing (liquidating old position) |
| `trailing_stop` | Sold due to 15% trailing stop being triggered |

## Example Trades

### Sample CSV Content:

```csv
date,ticker,action,reason,shares,price,value,cost
2024-01-02,AAPL,BUY,rebalance_megacap,127.5,182.50,23268.75,15.25
2024-01-02,MSFT,BUY,rebalance_megacap,56.2,415.20,23333.44,16.30
2024-01-02,GOOGL,BUY,rebalance_megacap,166.8,139.80,23322.24,15.80
2024-01-02,NVDA,BUY,rebalance_momentum,4.9,878.50,4304.65,3.15
2024-01-02,META,BUY,rebalance_momentum,8.6,495.30,4259.58,3.12
2024-04-01,JPM,SELL,rebalance,85.3,175.20,14944.56,10.25
2024-04-01,V,SELL,trailing_stop,45.2,282.30,12759.96,9.80
```

## Using the Trade History

### 1. Verify Backtest Accuracy

Check that trades match your expectations:
- Are trades happening quarterly (Jan/Apr/Jul/Oct)?
- Are the correct stocks being bought/sold?
- Are position sizes reasonable?

### 2. Calculate Performance by Stock

Use Excel/Python to analyze:
```python
import pandas as pd

# Load trades
trades = pd.read_csv('output/tier2_trades.csv')

# Group by ticker
by_ticker = trades.groupby('ticker').agg({
    'value': 'sum',
    'cost': 'sum',
    'action': 'count'
})

print(by_ticker)
```

### 3. Analyze Transaction Costs

Calculate total costs:
```python
trades = pd.read_csv('output/tier2_trades.csv')
total_costs = trades['cost'].sum()
total_volume = trades['value'].sum()
cost_percentage = (total_costs / total_volume) * 100

print(f"Total transaction costs: ${total_costs:,.2f}")
print(f"Cost percentage: {cost_percentage:.2f}%")
```

### 4. Review Stop Losses

See which stocks were stopped out:
```python
trades = pd.read_csv('output/tier2_trades.csv')
stopped_out = trades[trades['reason'] == 'trailing_stop']

print(f"Stopped out {len(stopped_out)} times:")
print(stopped_out[['date', 'ticker', 'price', 'value']])
```

### 5. Track Holding Periods

Match buys to sells to see how long stocks were held:
```python
trades = pd.read_csv('output/tier2_trades.csv')
trades['date'] = pd.to_datetime(trades['date'])

for ticker in trades['ticker'].unique():
    ticker_trades = trades[trades['ticker'] == ticker].sort_values('date')
    buys = ticker_trades[ticker_trades['action'] == 'BUY']
    sells = ticker_trades[ticker_trades['action'] == 'SELL']

    if len(buys) > 0 and len(sells) > 0:
        first_buy = buys['date'].min()
        first_sell = sells['date'].min()
        holding_days = (first_sell - first_buy).days
        print(f"{ticker}: held for {holding_days} days")
```

## Quarterly Rebalancing Example

Typical quarterly rebalancing will show:

**Day before rebalance (e.g., March 31):**
- No trades

**First day of quarter (e.g., April 1):**
1. **SELL all old positions** (reason: `rebalance`)
   - Usually 8-10 stocks sold
2. **BUY new mega-caps** (reason: `rebalance_megacap`)
   - Top 3 mega-caps by market cap × momentum score
   - 70% of portfolio (~$23,333 each for $100k portfolio)
3. **BUY new momentum stocks** (reason: `rebalance_momentum`)
   - Top 7 momentum stocks (excluding mega-caps)
   - 30% of portfolio (~$4,286 each for $100k portfolio)

## Tips

### Expected Trade Counts

**For a full year backtest (e.g., 2024):**
- 4 quarterly rebalances
- Each rebalance: ~10 sells + ~10 buys = 20 trades
- Occasional trailing stops: 5-10 trades
- **Total: 85-95 trades per year**

### Red Flags

- ❌ More than 200 trades per year → too frequent
- ❌ Large transaction costs (>3% of portfolio) → something wrong
- ❌ No trades for multiple quarters → rebalancing not working
- ❌ All stocks stopped out → strategy not working

### Green Flags

- ✅ ~80-100 trades per year
- ✅ Transaction costs <2% of portfolio annually
- ✅ Rebalances happening in Jan/Apr/Jul/Oct
- ✅ Mix of mega-cap and momentum positions

## Integration with Other Tools

### Compare with Recommendations

```bash
# 1. Run backtest (generates tier2_trades.csv)
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024

# 2. Check current recommendations (generates tier2_recommendations.csv)
python3 export_tier2_recommendations.py

# 3. Compare to see if live positions match backtest
```

### Visualize Trades

The visualize_trades.py script reads from this CSV:
```bash
cd src/visualize
python3 visualize_trades.py
# Creates trading_analysis.html with interactive trade timeline
```

## Troubleshooting

### "File not found: output/tier2_trades.csv"

**Cause:** Backtest hasn't been run yet or failed

**Fix:** Run a backtest first:
```bash
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024
```

### "CSV is empty"

**Cause:** No trades happened during backtest period

**Fix:** Check:
- Is the date range too short? (Use at least 1 year)
- Is the strategy configured correctly?
- Are there any error messages in the logs?

### "Trade dates don't match quarterly schedule"

**Cause:** Strategy might be using monthly rebalancing

**Fix:** Verify quarterly rebalancing is enabled:
```bash
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024 --quarterly
```

## Summary

✅ **Automatic:** Trades saved every backtest run
✅ **Complete:** All buys, sells, and stops recorded
✅ **Detailed:** Price, shares, costs, reasons included
✅ **Useful:** Verify strategy, analyze costs, track performance

**Location:** `output/tier2_trades.csv`

**Updated:** After every backtest run

---

**Related Documentation:**
- `HOW_TO_CHECK_POSITIONS.md` - Check current buy/sell recommendations
- `TIER2_PRODUCTION_DEPLOYMENT_GUIDE.md` - Full deployment guide
- `LIVE_TRADING_MONITORING_CHECKLIST.md` - Monitoring framework
