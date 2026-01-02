# Debug Skill

Debug issues with the trading bot by systematically checking common failure points.

## What this skill does:
1. Runs comprehensive diagnostics on the trading bot
2. Checks data integrity, code health, and configuration
3. Identifies common issues and suggests fixes
4. Validates that all components are working correctly

## Steps to execute:

### 1. Environment Check
- Verify Python version (>=3.8 required)
- Check installed dependencies (pandas, numpy, etc.)
- Validate project structure

### 2. Data Validation
- Check if sp500_data/daily/ exists and has data
- Verify number of stock CSV files (should be ~473)
- Validate VIX data exists (VIX.csv)
- Check date ranges (should cover 2005-2024)
- Validate data format (OHLCV columns)

### 3. Code Health Check
- Run Python syntax validation on key files:
  - src/core/execution.py
  - src/backtest/portfolio_bot_demo.py
  - src/visualize/visualize_trades.py
- Check for import errors
- Validate database schema

### 4. Quick Sanity Test
- Run minimal backtest (1 year of data)
- Verify basic functionality works
- Check if output directories are writable

### 5. Database Check
- Verify database file exists or can be created
- Check database schema
- List recent backtest runs
- Validate data integrity

## Expected output format:
```
🔍 TRADING BOT DIAGNOSTICS
==========================

✅ Environment Check
  ✓ Python 3.11.5
  ✓ pandas 2.1.0
  ✓ numpy 1.24.3
  ✓ Project structure valid

✅ Data Validation
  ✓ Data directory: sp500_data/daily/
  ✓ Stock files: 473 CSV files found
  ✓ VIX data: VIX.csv present
  ✓ Date range: 2005-01-03 to 2024-12-31 (19.4 years)
  ✓ Data format: OHLCV columns validated

✅ Code Health
  ✓ src/core/execution.py - No syntax errors
  ✓ src/backtest/portfolio_bot_demo.py - No syntax errors
  ✓ src/visualize/visualize_trades.py - No syntax errors
  ✓ All imports successful

✅ Database
  ✓ output/data/trading_results.db exists
  ✓ Schema valid (3 tables: backtest_runs, portfolio_values, yearly_returns)
  ✓ Recent runs: 5 backtest runs found
  ✓ Latest run: V22_SQRT_KELLY_20250102_103000 (10.2% annual)

✅ Quick Sanity Test
  ✓ Minimal backtest executed successfully
  ✓ Basic functionality working
  ✓ Output directories writable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 ALL SYSTEMS OPERATIONAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trading bot is healthy and ready to use!
```

## Error Detection Format:
```
🔍 TRADING BOT DIAGNOSTICS
==========================

✅ Environment Check
  ✓ Python 3.11.5
  ✓ pandas 2.1.0

⚠️  Data Validation
  ✓ Data directory: sp500_data/daily/
  ✗ Stock files: Only 420 CSV files found (expected ~473)
  ✗ VIX data: VIX.csv NOT FOUND
  ✓ Date range: 2005-01-03 to 2024-12-31

❌ Code Health
  ✓ src/core/execution.py - No syntax errors
  ✗ src/backtest/portfolio_bot_demo.py - SyntaxError line 245
     IndentationError: unexpected indent

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ISSUES DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Found 3 issues:

1. MISSING VIX DATA
   Problem: VIX.csv not found in sp500_data/daily/
   Impact: VIX regime detection will fail
   Fix: Run: python src/data/download_vix.py

2. INCOMPLETE STOCK DATA
   Problem: Only 420/473 stocks found (53 missing)
   Impact: Reduced stock universe
   Fix: Download missing stocks or verify data source

3. SYNTAX ERROR
   Problem: IndentationError in portfolio_bot_demo.py line 245
   Impact: Backtest will fail to run
   Fix: Check indentation at line 245

Run fixes? [yes/no]
```

## Diagnostic Checks:

### Critical Issues (❌):
- Missing data files
- Syntax errors in code
- Import failures
- Database corruption
- Permission errors

### Warnings (⚠️):
- Incomplete data (some stocks missing)
- Old data (last update >30 days ago)
- Deprecated dependencies
- Large log files (>100MB)

### Info (ℹ️):
- Data statistics
- Recent backtest runs
- Disk space usage
- Performance trends

## Auto-Fix Capability:

If user confirms, skill can auto-fix:
1. Download missing VIX data
2. Create missing directories
3. Initialize database schema
4. Clear old log files
5. Update stale data

## Common Issues & Solutions:

### Issue 1: ModuleNotFoundError
```
Problem: pandas not installed
Fix: pip install -r requirements.txt
```

### Issue 2: FileNotFoundError
```
Problem: sp500_data/daily/ not found
Fix: Verify data directory or update path in execution.py
```

### Issue 3: Database OperationalError
```
Problem: Database schema outdated
Fix: Drop and recreate tables or run migration
```

### Issue 4: VIX Data Missing
```
Problem: VIX.csv not found
Fix: python src/data/download_vix.py
```

### Issue 5: Backtest Returns NaN
```
Problem: Insufficient data for indicators (need 100+ days)
Fix: Check date ranges or reduce indicator lookback periods
```

## Success criteria:
- All critical checks pass (✅)
- No errors detected (❌)
- Warnings documented with fixes (⚠️)
- Sanity test runs successfully
- Clear action items if issues found

## Usage examples:

### Basic diagnostic:
```
Claude, run debug skill
```

### After errors:
```
Claude, debug - backtest is failing
```

### Pre-deployment:
```
Claude, debug and verify everything is ready for production
```

### With auto-fix:
```
Claude, debug and fix any issues automatically
```
