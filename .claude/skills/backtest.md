# Backtest Skill

Run a backtest with the current V22 production strategy and generate a comprehensive report.

## What this skill does:
1. Runs the V22-Sqrt Kelly Position Sizing backtest
2. Generates performance metrics (annual return, drawdown, Sharpe ratio)
3. Creates visualizations (trading analysis HTML dashboard)
4. Saves results to database
5. Generates text performance report

## Steps to execute:
1. Read the current src/core/execution.py to understand the configuration
2. Run the backtest using: `python -m src.core.execution`
3. Parse the output for key metrics
4. Display results in a formatted summary table
5. Check if visualizations were generated successfully
6. Provide paths to all output files (database, reports, plots)

## Expected output format:
```
📊 BACKTEST RESULTS
===================
Strategy: V22-Sqrt Kelly Position Sizing
Period: [start_date] to [end_date]
Duration: X.X years

Performance Metrics:
  Annual Return:    XX.X%
  Max Drawdown:    -XX.X%
  Sharpe Ratio:     X.XX
  Win Rate:         XX/XX years (XX%)
  Final Value:      $XXX,XXX

Output Files:
  📊 Database:      output/data/trading_results.db
  📈 Report:        output/reports/performance_report_YYYYMMDD_HHMMSS.txt
  🎨 Visualization: output/plots/trading_analysis.html
  📋 Logs:          output/logs/execution.log
```

## Success criteria:
- Backtest runs without errors
- Annual return is calculated and displayed
- All output files are created
- Visualizations are accessible
