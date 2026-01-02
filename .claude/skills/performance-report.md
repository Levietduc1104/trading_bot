# Performance Report Skill

Generate a comprehensive performance analysis report from database results.

## What this skill does:
1. Connect to output/data/trading_results.db
2. Query the most recent backtest run
3. Extract all performance metrics
4. Generate detailed analysis report
5. Compare to historical runs (if available)
6. Identify strengths and weaknesses

## Steps to execute:
1. Check if database exists
2. Query backtest_runs table for most recent run
3. Extract metrics:
   - Returns (annual, total, yearly)
   - Risk (drawdown, volatility, Sharpe)
   - Trade statistics
   - Win rate
4. Query portfolio_values for equity curve analysis
5. Calculate additional metrics:
   - Calmar ratio (return/drawdown)
   - Sortino ratio (downside risk)
   - Recovery time from drawdowns
   - Consecutive wins/losses
6. Generate formatted report with insights

## Expected output format:
```
📊 PERFORMANCE ANALYSIS REPORT
==============================
Generated: 2025-01-02 10:30:00
Strategy: V22-Sqrt Kelly Position Sizing
Run ID: 42
Period: 2005-01-01 to 2024-12-31 (19.4 years)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 RETURN METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial Capital:        $100,000
Final Value:            $653,746
Total Return:           553.7%
Annual Return:          10.2%
CAGR:                   10.2%

Benchmark Comparison (SPY):
  Strategy:             10.2%
  SPY:                   9.1%
  Outperformance:       +1.1%  ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  RISK METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Max Drawdown:          -15.2%
Avg Drawdown:           -5.3%
Volatility (annual):    9.2%
Downside Deviation:     6.1%

Drawdown Analysis:
  Worst drawdown:       -15.2% (Mar 2020 COVID crash)
  Recovery time:        4.2 months
  Drawdowns > 10%:      3 events
  Avg recovery:         2.8 months

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 RISK-ADJUSTED RETURNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sharpe Ratio:           1.11  ⭐ (Excellent)
Sortino Ratio:          1.67  ⭐⭐ (Outstanding)
Calmar Ratio:           0.67  ✅ (Good)

Risk-Return Efficiency:
  Strategy positions in TOP QUARTILE of risk-adjusted returns
  Sharpe > 1.0 = Institutional quality
  Sortino > 1.5 = Excellent downside protection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 YEARLY PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Positive Years: 16/20 (80% win rate)
Negative Years:  4/20 (20%)

Best year:  2007 (+26.1%)
Worst year: 2020 (-9.3%)

Consistency Score: 8.5/10
  - High win rate (80%)
  - Negative years are manageable (<10% loss)
  - Positive years cluster around 10-15% (predictable)

Decade Analysis:
  2005-2009:  +8.1% avg/year (Financial Crisis recovery)
  2010-2019: +11.4% avg/year (Bull market)
  2020-2024:  +9.8% avg/year (COVID + inflation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 KEY STRENGTHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Strong risk-adjusted returns (Sharpe 1.11)
✅ Excellent downside protection (Sortino 1.67)
✅ High win rate (80% positive years)
✅ Manageable drawdowns (<20%)
✅ Consistent across market cycles
✅ Outperforms benchmark (SPY +1.1%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  AREAS FOR IMPROVEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔸 Absolute returns moderate (10.2% vs target >12%)
🔸 Drawdown recovery could be faster
🔸 2020 COVID crash had -9.3% loss (defensive positioning could improve)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Consider adaptive Kelly weighting to push returns toward 12%
2. Improve VIX regime detection for better crash protection
3. Test conditional rebalancing to let winners run longer
4. Explore multi-timeframe momentum for better entry timing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Database:       output/data/trading_results.db
Report:         output/reports/performance_report_20250102_103000.txt
Visualization:  output/plots/trading_analysis.html
Logs:           output/logs/execution.log

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Analysis Depth:
- Basic: Returns, drawdown, Sharpe
- Intermediate: Add Sortino, Calmar, yearly analysis
- Advanced: Add regime analysis, correlation studies, factor attribution

## Success criteria:
- All key metrics calculated
- Comparison to benchmark provided
- Clear strengths/weaknesses identified
- Actionable recommendations given
