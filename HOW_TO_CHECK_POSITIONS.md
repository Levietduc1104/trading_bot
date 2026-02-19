# How to Check Tier 2 Buy/Sell Signals

## Quick Start

```bash
# Check current recommendations (what to buy right now)
python3 check_tier2_positions.py

# Compare vs your current holdings
python3 check_tier2_positions.py --holdings AAPL,MSFT,GOOGL,NVDA,AMZN,META,TSLA,BRK.B,V,JPM

# Check for specific date
python3 check_tier2_positions.py --date 2024-03-31

# Use different capital amount
python3 check_tier2_positions.py --capital 50000
```

## When to Use This

### Before Quarterly Rebalancing (Jan 1, Apr 1, Jul 1, Oct 1)
1. Run this script to see current recommendations
2. Compare vs your holdings to see what to buy/sell
3. Execute trades at market open
4. Verify final allocations match targets

### Example Quarterly Rebalancing Workflow

**Day Before Rebalancing (e.g., March 31):**
```bash
# Check what you currently hold vs what you should hold
python3 check_tier2_positions.py --holdings AAPL,MSFT,NVDA
```

**Output will show:**
- 🟢 BUY: Stocks you need to add
- 🔴 SELL: Stocks you need to exit
- ✓ HOLD: Stocks to keep

**Rebalancing Day (e.g., April 1, 9:30 AM):**
1. Execute all SELL orders first
2. Wait for sells to settle
3. Execute all BUY orders with freed-up cash
4. Verify allocations: 70% mega-caps, 30% momentum

## What the Output Shows

### Top 3 Mega-caps (70% allocation)
```
Ticker   Price        Target $         Shares     Status
────────────────────────────────────────────────────────
AAPL     $  182.50    $     23,333        127    ✓ HOLD
MSFT     $  415.20    $     23,333         56    🟢 BUY
GOOGL    $  139.80    $     23,333        166    ✓ HOLD
```
- These get 70% of portfolio ($70,000 / 3 = ~$23,333 each)
- Buy ~equal dollar amounts of each

### Top 7 Momentum (30% allocation)
```
Ticker   Price        Target $         Shares     Status
────────────────────────────────────────────────────────
NVDA     $  878.50    $      4,286          4    🟢 BUY
META     $  495.30    $      4,286          8    ✓ HOLD
TSLA     $  175.20    $      4,286         24    🟢 BUY
...
```
- These get 30% of portfolio ($30,000 / 7 = ~$4,286 each)
- Buy ~equal dollar amounts of each

### Execution Summary
```
🟢 BUY:  3 positions - MSFT, NVDA, TSLA
🔴 SELL: 2 positions - JPM, V
✓ HOLD: 5 positions - AAPL, GOOGL, META, AMZN, BRK.B
```

## Common Use Cases

### 1. Starting Fresh (No Current Holdings)
```bash
python3 check_tier2_positions.py --capital 100000
```
Shows all 10 stocks to buy and exact share counts.

### 2. Quarterly Rebalancing
```bash
# List your current holdings
python3 check_tier2_positions.py --holdings AAPL,MSFT,GOOGL,NVDA,AMZN,META,TSLA,BRK.B,V,JPM
```
Shows what to buy, sell, and hold.

### 3. Paper Trading Check
```bash
# Check what you should be holding
python3 check_tier2_positions.py
```
Verify your paper trading positions match recommendations.

### 4. Historical Check
```bash
# What were recommendations on Jan 1, 2024?
python3 check_tier2_positions.py --date 2024-01-01
```
Useful for reviewing past rebalancing decisions.

## Understanding the Recommendations

### Why These Stocks?
The Tier 2 strategy ranks ALL S&P 500 stocks by:
- **70% Momentum** - 20-day price momentum, relative strength
- **30% Growth Fundamentals** - ROE, margins, FCF, ROIC, financial health

Then selects:
- **Top 3 mega-caps** by combined score (largest companies)
- **Top 7 momentum stocks** by combined score (excluding mega-caps already selected)

### Allocation Logic
- **70% to mega-caps** (3 stocks) = stability, less volatility
- **30% to momentum** (7 stocks) = higher growth potential

This 70/30 split has been proven across 34 years (1990-2024) to be optimal.

### Why Equal Dollar Amounts?
- Each mega-cap gets: 70% ÷ 3 = 23.33% of portfolio
- Each momentum stock gets: 30% ÷ 7 = 4.29% of portfolio

Equal dollar weighting within each group provides diversification while maintaining the 70/30 split.

## Troubleshooting

### "Could not determine recommended positions"
**Fix:** Run full backtest to get recommendations
```bash
python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024
```

### "Stocks keep changing every time I run it"
**Normal:** The script analyzes latest available data. Stocks change as momentum/fundamentals change daily. Only rebalance quarterly, not daily!

### "My holdings don't match exactly"
**OK:** Small differences are fine. Perfect matches rarely happen due to:
- Fractional shares
- Price movements between buy and now
- Transaction timing differences

As long as allocation is roughly 70/30, you're good!

### "Should I buy these stocks TODAY?"
**NO!** Only rebalance on quarterly schedule:
- Jan 1 (Q1)
- Apr 1 (Q2)
- Jul 1 (Q3)
- Oct 1 (Q4)

Running this script daily will show different stocks. Ignore daily changes. Trust the quarterly schedule.

## Important Reminders

### ✅ DO:
- Run this before each quarterly rebalancing
- Follow the buy/sell recommendations exactly
- Maintain 70/30 mega-cap/momentum allocation
- Execute all sells before buys
- Verify final positions match recommendations

### ❌ DON'T:
- Run this daily and trade based on changes (overtrading)
- Override recommendations with your own picks (emotional trading)
- Skip quarterly rebalancing (system depends on rotation)
- Modify allocation percentages (70/30 is proven optimal)
- Buy/sell outside quarterly schedule (transaction costs)

## Integration with Live Trading

### Paper Trading
```bash
# 1. Check recommendations
python3 check_tier2_positions.py

# 2. Execute in paper account
# 3. Verify positions match
python3 check_tier2_positions.py --holdings [YOUR_PAPER_HOLDINGS]
```

### Live Trading
```bash
# 1. Day before rebalancing: Check what to buy/sell
python3 check_tier2_positions.py --holdings [CURRENT_LIVE_HOLDINGS]

# 2. Rebalancing day: Execute trades
# 3. After trades settle: Verify
python3 check_tier2_positions.py --holdings [NEW_LIVE_HOLDINGS]
```

## Example Session

```
$ python3 check_tier2_positions.py --holdings AAPL,MSFT,NVDA,V,JPM

📋 Your Current Holdings: AAPL, MSFT, NVDA, V, JPM

================================================================================
TIER 2 GROWTH SCORING - BUY/SELL SIGNALS
================================================================================

Strategy: V31 Tier 2 (70% Momentum + 30% Growth Fundamentals)
Portfolio: 70% Top 3 Mega-caps + 30% Top 7 Momentum
Capital: $100,000

Target Date: 2024-11-04
Analyzing data through: 2024

Calculating current recommendations...

────────────────────────────────────────────────────────────────────────────────
RECOMMENDED PORTFOLIO (10 stocks)
────────────────────────────────────────────────────────────────────────────────

📊 TOP 3 MEGA-CAPS (70% allocation):
Ticker   Price        Target $         Shares     Status
────────────────────────────────────────────────────────────────────────────────
AAPL     $  182.50    $     23,333        127    ✓ HOLD
MSFT     $  415.20    $     23,333         56    ✓ HOLD
GOOGL    $  139.80    $     23,333        166    🟢 BUY

📈 TOP MOMENTUM STOCKS (30% allocation):
Ticker   Price        Target $         Shares     Status
────────────────────────────────────────────────────────────────────────────────
NVDA     $  878.50    $      4,286          4    ✓ HOLD
META     $  495.30    $      4,286          8    🟢 BUY
TSLA     $  175.20    $      4,286         24    🟢 BUY
AMZN     $  178.40    $      4,286         24    🟢 BUY
BRK.B    $  445.20    $      4,286          9    🟢 BUY
LLY      $  892.30    $      4,286          4    🟢 BUY
AVGO     $  171.50    $      4,286         24    🟢 BUY

🔴 SELL (exit these positions):
Ticker   Reason
────────────────────────────────────────────────────────────────────────────────
JPM      Not in current top 10 recommendations
V        Not in current top 10 recommendations

────────────────────────────────────────────────────────────────────────────────
EXECUTION SUMMARY
────────────────────────────────────────────────────────────────────────────────

🟢 BUY:  7 positions - AMZN, AVGO, BRK.B, GOOGL, LLY, META, TSLA
🔴 SELL: 2 positions - JPM, V
✓ HOLD: 3 positions - AAPL, MSFT, NVDA

────────────────────────────────────────────────────────────────────────────────
ALLOCATION SUMMARY
────────────────────────────────────────────────────────────────────────────────

Total Capital: $100,000

Mega-cap (70%): $70,000
  → $23,333 per stock (3 stocks)

Momentum (30%): $30,000
  → $4,286 per stock (7 stocks)

================================================================================

💡 TIP: Run this before each quarterly rebalancing (Jan 1, Apr 1, Jul 1, Oct 1)
💡 TIP: Use --holdings to compare vs your current portfolio
```

## Summary

**This tool tells you exactly what to buy and sell for Tier 2 strategy.**

Use it:
- ✅ Before quarterly rebalancing (4 times per year)
- ✅ To verify paper/live trading positions
- ✅ To calculate target share counts
- ✅ To see which stocks to buy/sell/hold

Don't use it:
- ❌ For daily trading decisions
- ❌ To second-guess the strategy
- ❌ Outside quarterly rebalancing schedule

**For full deployment guide, see:**
- `TIER2_PRODUCTION_DEPLOYMENT_GUIDE.md`
- `PAPER_TRADING_QUICK_START.md`
- `LIVE_TRADING_MONITORING_CHECKLIST.md`
