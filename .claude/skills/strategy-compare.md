# Strategy Compare Skill

Compare multiple strategy versions side-by-side to identify which performs best.

## What this skill does:
1. Identify all available strategy test files (test_v*.py)
2. Run 2-4 strategies specified by the user
3. Collect performance metrics for each
4. Generate a comparison table
5. Highlight the winner and key differences

## Steps to execute:
1. Ask user which strategies to compare (e.g., "V13, V20, V22")
2. Locate the corresponding test files in src/tests/
3. Run each test script sequentially
4. Parse output for metrics:
   - Annual return
   - Max drawdown
   - Sharpe ratio
   - Win rate
   - Final value
5. Create comparison table
6. Calculate deltas vs baseline (V13)
7. Identify best strategy by Sharpe ratio (risk-adjusted returns)

## Expected output format:
```
📊 STRATEGY COMPARISON
======================

Strategy          Annual    Drawdown   Sharpe    Win Rate   Final Value   vs V13
----------------  --------  ---------  --------  ---------  ------------  ------
V13 (Baseline)     9.8%      -19.1%     1.07     16/20      $547,632      --
V20 (Leverage)     9.1%      -22.3%     0.98     15/20      $512,441      -0.7%
V22 (Kelly)       10.2%      -15.2%     1.11     16/20      $653,746      +0.4% ⭐

Winner: V22-Sqrt Kelly Position Sizing
Reason: Best risk-adjusted returns (highest Sharpe) with lowest drawdown
```

## User interaction:
- Prompt user to select 2-4 strategies to compare
- Provide recommendations based on results
- Suggest next optimization direction

## Success criteria:
- All selected strategies run successfully
- Metrics are accurately extracted
- Clear winner is identified
- Actionable insights provided
