# Test Files and Experiments

This directory contains test scripts, experiments, and historical strategy tests.

## 📂 Organization

### Strategy Tests (V14-V22)
Tests for experimental strategies that were explored during development:

- **test_v14_rebalance_bands.py** - Rebalance bands experiment
- **test_v15_multi_timeframe.py** - Multi-timeframe momentum test
- **test_v16_sector_rotation.py** - Sector rotation experiment
- **test_v17_correlation_diversification.py** - Correlation-based diversification
- **test_v17_debug.py** - Debug helper for V17
- **test_v18_progressive_cash_redeployment.py** - Progressive cash redeployment
- **test_v19_asymmetric_redeployment.py** - Asymmetric redeployment test
- **test_v20_volatility_targeted_leverage.py** - Volatility-targeted leverage
- **test_v21_tactical_portfolio_size.py** - Tactical portfolio sizing
- **test_v22_kelly_position_sizing.py** - Kelly position sizing (Winner! 10.2%) ⭐
- **test_v22_leverage.py** - Kelly + leverage experiment

### Analysis Tests

- **test_comprehensive_comparison.py** - Compare multiple strategies
- **test_portfolio_concentration.py** - Portfolio concentration analysis
- **test_kelly_viz.py** - Kelly weighting visualization test

### Old Production Scripts

- **run_v13_production_5stocks.py** - V13 production (superseded by V22)
- **optimize_v20_leverage.py** - V20 leverage optimization

### Test Outputs

- **concentration_test.log** - Portfolio concentration test output
- **v20_leverage_optimization_results.csv** - V20 optimization results
- **v21_calibrated_output.log** - V21 calibration output
- **v21_output.log** - V21 test output
- **v22_kelly_output.log** - V22 Kelly test output
- **v22_leverage_output.log** - V22 leverage test output

## 🏆 Key Findings

### V22-Sqrt Kelly Position Sizing (Winner!)
- **File**: test_v22_kelly_position_sizing.py
- **Result**: 10.2% annual, -15.2% DD, 1.11 Sharpe
- **Status**: Integrated into production (src/core/execution.py)

### Failed Experiments
Most V14-V21 experiments did NOT beat V13 baseline (9.8%):
- V14-V19: Various approaches, all underperformed
- V20: Leverage (too risky, higher DD)
- V21: Tactical sizing (no improvement)

### Why They Failed
- **Over-optimization**: Curve fitting to historical data
- **Complexity**: More parameters = more ways to fail
- **Leverage risk**: Higher returns but unacceptable drawdowns
- **No edge**: Ideas that sound good but don't work

### Why V22 Won
- **Simple**: Only changed position sizing (weight ∝ √score)
- **Validated edge**: Kelly sizing PROVES our scoring has quality
- **Better risk**: Improved returns AND reduced drawdown
- **Fundamental**: Based on Kelly Criterion (academically sound)

## 🔬 Running Tests

### Run a specific test:
```bash
cd /path/to/trading_bot
python src/tests/test_v22_kelly_position_sizing.py
```

### Compare strategies:
```bash
python src/tests/test_comprehensive_comparison.py
```

## ⚠️ Note

These are experimental tests and should NOT be used in production. The only production-ready strategy is **V22-Sqrt Kelly Position Sizing** located in:
- `src/core/execution.py` (main production script)
- `run_v22_production.py` (standalone script)

## 📊 Test Results Summary

| Test | Annual Return | Max DD | Sharpe | Status |
|------|---------------|--------|--------|--------|
| V13 Baseline | 9.8% | -19.1% | 1.07 | ✅ Good |
| V14-V19 | 7-9% | -18 to -25% | 0.9-1.1 | ❌ Failed |
| V20 Leverage | 11.5% | -28.3% | 0.92 | ❌ Too risky |
| V21 Tactical | 9.6% | -19.5% | 1.05 | ❌ No improvement |
| **V22 Kelly** | **10.2%** | **-15.2%** | **1.11** | **✅ Winner!** ⭐ |

---

**Current Production:** V22-Sqrt Kelly Position Sizing

**Last Updated:** 2026-01-01
