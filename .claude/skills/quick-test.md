# Quick Test Skill

Rapidly test a code change or strategy modification without running full backtest.

## What this skill does:
1. Detect recent changes in tracked files (git diff)
2. Run a quick validation test (last 2 years of data instead of full 19 years)
3. Compare results to baseline
4. Provide fast feedback (< 30 seconds vs 2-3 minutes for full backtest)

## Use cases:
- Testing a new indicator before full backtest
- Validating a bug fix
- Exploring parameter tweaks
- Quick sanity check after code changes

## Steps to execute:
1. Check git status for modified files
2. Show user what changed (git diff summary)
3. Ask user: "Run quick test on recent changes?"
4. If yes:
   - Modify date range in backtest to last 2-3 years
   - Run abbreviated backtest
   - Compare to known baseline for same period
5. Display quick results
6. Ask if user wants to run full backtest

## Expected output format:
```
⚡ QUICK TEST RESULTS
=====================
Modified Files:
  - src/core/execution.py (Kelly weighting adjustment)
  - src/backtest/portfolio_bot_demo.py (scoring tweak)

Test Period: 2022-01-01 to 2024-12-31 (3 years)

Quick Test:
  Annual Return:    12.1%
  Max Drawdown:    -18.3%
  Sharpe Ratio:     1.15

Baseline (V22):
  Annual Return:    11.8%
  Max Drawdown:    -16.2%
  Sharpe Ratio:     1.13

Delta:
  Annual Return:    +0.3%  ✅ IMPROVEMENT
  Max Drawdown:    -2.1%  ⚠️  WORSE
  Sharpe Ratio:     +0.02  ✅ BETTER

Assessment: Mixed results - better returns but higher drawdown
Recommendation: Run full backtest to confirm over complete 19-year period

Run full backtest? [yes/no]
```

## Implementation:
```python
# Quick test modification (in test script):
# Instead of:
dates = bot.stocks_data[first_ticker].index

# Use:
import pandas as pd
cutoff_date = pd.Timestamp('2022-01-01')
dates = bot.stocks_data[first_ticker].index
dates = dates[dates >= cutoff_date]
```

## Success criteria:
- Test completes in < 30 seconds
- Results are comparable to baseline
- Clear recommendation provided
- Option to run full backtest if promising
