# Documentation Index

This folder contains all markdown documentation for the trading bot project.

## 📚 Documentation Files

### Strategy & Results

1. **README.md** - Main project overview and getting started guide
2. **V13_PRODUCTION_SUMMARY.md** - V13 strategy summary (baseline momentum strategy)
3. **V22_PRODUCTION_SUMMARY.md** - V22 strategy with Kelly position sizing
4. **V22_INTEGRATION_COMPLETE.md** - V22 integration details and testing
5. **PRODUCTION_SUMMARY.txt** - Production deployment summary (see root folder)

### Analysis & Optimization

6. **KELLY_RESULTS.md** - Kelly criterion position sizing analysis
7. **MONTE_CARLO_EXPLAINED.md** - Monte Carlo parameter optimization guide
8. **STEP2_PLAN.md** - Monte Carlo testing plan
9. **STEP2_RESULTS.md** - Monte Carlo testing results
10. **VIX_RESULTS.md** - VIX multiplier optimization analysis

### Data & Visualization

11. **PERIOD_DATASETS_GUIDE.md** - Guide for period-specific datasets (1963-1983, 1983-2003, 1990-2024)
12. **VISUALIZATION_GUIDE.md** - Interactive Bokeh visualization guide

### Contributing

13. **CONTRIBUTING.md** - Contribution guidelines

---

## 📊 Quick Links by Topic

### Getting Started
- [README.md](README.md) - Start here

### Strategy Evolution
- V13 → V22 → V27 → **V28 (Current)**
- [V13_PRODUCTION_SUMMARY.md](V13_PRODUCTION_SUMMARY.md)
- [V22_PRODUCTION_SUMMARY.md](V22_PRODUCTION_SUMMARY.md)

### Optimization Studies
- [KELLY_RESULTS.md](KELLY_RESULTS.md) - Position sizing
- [VIX_RESULTS.md](VIX_RESULTS.md) - Regime detection
- [MONTE_CARLO_EXPLAINED.md](MONTE_CARLO_EXPLAINED.md) - Parameter sweep

### Data Management
- [PERIOD_DATASETS_GUIDE.md](PERIOD_DATASETS_GUIDE.md) - Historical period datasets

### Visualization
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Interactive charts

---

## 🎯 Current Production Strategy: V28

**Strategy:** Momentum Leaders + Regime-Based Portfolio Sizing
**Performance:** 10.6% annual, -27.5% max drawdown, 0.75 Sharpe
**Period:** 1990-2024 (34.3 years)
**Dataset:** 466 S&P 500 stocks

See root folder for:
- `PRODUCTION_SUMMARY.txt` - Latest production summary
- `execution_sp500_filtered.log` - Latest backtest log

---

## 📁 Folder Organization

```
docs_md/                    # All markdown documentation (this folder)
docs/                       # Additional documentation
output/                     # Backtest results
  ├── data/                # Database files
  ├── reports/             # Text reports
  ├── plots/               # Visualizations
  └── logs/                # Execution logs
sp500_data/                # Stock datasets
  ├── stock_data_1963_1983/  # 1960s-1980s period
  ├── stock_data_1983_2003/  # Dot-com era
  ├── stock_data_1990_2024/  # Modern era (production)
  └── sp500_filtered/        # Clean S&P 500 stocks
src/                       # Source code
tests/                     # Test files
```

---

## 📝 Last Updated

- **Date:** 2026-01-11
- **Latest Strategy:** V28 Momentum Leaders
- **Latest Dataset:** stock_data_1990_2024 (466 stocks)
- **Latest Run ID:** 40 (10.6% annual return)
