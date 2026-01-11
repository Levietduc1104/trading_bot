# Step 2: Building a Basic Monte Carlo Simulator

## Goal
Create a simple simulator that can:
1. Run your V28 backtest with different parameters
2. Collect the results
3. Compare them

## The Plan

### What We'll Build (Simple Version)

```
MonteCarloSimulator
├── Load your trading bot
├── Run backtest with custom parameters
├── Calculate metrics (return, drawdown, Sharpe)
└── Return results as a dictionary
```

### Why This Approach?

We'll **reuse** your existing V28 backtest code (src/backtest/portfolio_bot_demo.py)
We just need to:
- Call it with different parameters
- Collect the results
- That's it!

---

## Step 2.1: Understanding What We Need

Your current V28 backtest has these parameters we can vary:

### Parameters We Can Test:
1. **portfolio_size**: How many stocks to hold (3, 5, 7, 10)
2. **kelly_exponent**: Position sizing aggressiveness (0.4, 0.5, 0.6)
3. **vix_multiplier**: Cash reserve sensitivity (0.8, 1.0, 1.2)
4. **fee_pct**: Trading costs (0.0005, 0.001, 0.002)
5. **start_offset_days**: When to start (0, 30, 60, 90...)

### Metrics We'll Track:
- Annual Return (%)
- Max Drawdown (%)
- Sharpe Ratio
- Sortino Ratio
- Final Portfolio Value

---

## Step 2.2: The Code Structure

We'll create these files:

```
src/monte_carlo/
├── __init__.py              (empty, just makes it a package)
├── simple_simulator.py      (our main simulator - STEP 2)
└── statistics.py            (calculate stats - STEP 3)
```

---

## Next: Let's write the code!

Starting with the simplest possible simulator...
