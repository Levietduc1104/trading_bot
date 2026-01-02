# Optimize Skill

Run optimization experiments to test parameter variations and find optimal settings.

## What this skill does:
1. Identify a specific parameter to optimize (e.g., portfolio size, cash reserve, rebalancing frequency)
2. Define a parameter range to test
3. Run multiple backtests with different parameter values
4. Compare results and identify optimal parameter
5. Visualize parameter sensitivity (optional)

## Steps to execute:
1. Ask user which parameter to optimize:
   - Portfolio size (3, 5, 7, 10 stocks)
   - Cash reserve (5%, 10%, 15%, 20%)
   - Kelly exponent (0.3, 0.4, 0.5, 0.6, 0.7)
   - Drawdown thresholds (5/10/15, 10/15/20, 15/20/25)
   - VIX regime thresholds

2. Create test variations:
   - Copy src/core/execution.py or relevant file
   - Modify parameter programmatically
   - Run backtest for each value

3. Collect results for each parameter value:
   - Annual return
   - Max drawdown
   - Sharpe ratio
   - Win rate

4. Identify optimal value:
   - Maximize Sharpe ratio (risk-adjusted)
   - Or maximize return subject to drawdown constraint

5. Generate optimization report with recommendation

## Expected output format:
```
🔬 OPTIMIZATION RESULTS
=======================
Parameter: Portfolio Size (number of stocks)
Range Tested: 3, 5, 7, 10, 15 stocks
Optimization Goal: Maximize Sharpe Ratio

Results:
Portfolio Size   Annual    Drawdown   Sharpe    Win Rate   Status
-------------    -------   --------   -------   --------   ------
3 stocks         11.2%     -22.4%     1.05      15/20      Too risky
5 stocks         10.2%     -15.2%     1.11      16/20      ⭐ OPTIMAL
7 stocks          9.7%     -14.1%     1.08      17/20      Over-diversified
10 stocks         8.9%     -13.8%     0.95      17/20      Too diluted
15 stocks         7.8%     -12.9%     0.88      18/20      Excessive diversification

RECOMMENDATION:
✅ Optimal: 5 stocks
   - Best Sharpe ratio (1.11)
   - Balanced return/risk (10.2% / -15.2%)
   - Current production setting is correct!

Sensitivity Analysis:
- Each additional stock reduces return by ~0.5-0.7%
- Drawdown improves marginally with more stocks
- Sharpe ratio peaks at 5 stocks then declines
```

## Parameter Optimization Examples:

### Example 1: Kelly Exponent
```
Test values: 0.3, 0.4, 0.5, 0.6, 0.7
Find: Best risk-adjusted returns
Modify: src/core/execution.py line 77
```

### Example 2: VIX Cash Reserve Formula
```
Test formulas:
- Current: 0.05 + (VIX-10)*0.005
- Conservative: 0.10 + (VIX-10)*0.008
- Aggressive: 0.03 + (VIX-10)*0.003
Modify: src/core/execution.py lines 161-165
```

### Example 3: Drawdown Thresholds
```
Test threshold sets:
- Current: 10/15/20 (1.0/0.75/0.50/0.25)
- Tight: 5/10/15 (earlier defense)
- Loose: 15/20/25 (let it run)
Modify: src/backtest/portfolio_bot_demo.py lines 1081-1092
```

## Success criteria:
- Parameter sweep completes successfully
- Clear optimal value identified
- Sensitivity analysis provided
- Recommendation includes confidence level

## User interaction:
- Ask which parameter to optimize
- Confirm parameter range
- Show progress during optimization
- Provide clear recommendation with reasoning
