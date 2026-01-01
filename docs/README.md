# Trading Bot Documentation

This directory contains additional documentation and historical references for the S&P 500 Portfolio Rotation Trading Bot.

## 📚 Documentation Structure

### Root Documentation (Main Files)

Located in the project root:

- **[README.md](../README.md)** - Main project documentation and quick start guide
- **[V22_PRODUCTION_SUMMARY.md](../V22_PRODUCTION_SUMMARY.md)** - Current production strategy (V22-Sqrt Kelly Position Sizing)
- **[V22_INTEGRATION_COMPLETE.md](../V22_INTEGRATION_COMPLETE.md)** - V22 integration guide and next steps
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Guidelines for contributing to the project

### Historical Documentation

Located in `docs/archive/`:

- **5_STOCK_INTEGRATION_SUMMARY.md** - V13 5-stock concentration summary
- **ADAPTIVE_REGIME_RESULTS.md** - Early adaptive regime testing
- **BRANCH_PROTECTION_SETUP.md** - GitHub branch protection guide
- **END_TO_END_GUIDE.md** - Original end-to-end guide
- **IMPROVEMENT_SUMMARY.md** - Improvement brainstorming and analysis
- **MARKET_TREND_STRENGTH.md** - Market trend strength research
- **ML_STRATEGY_PROPOSAL.md** - Machine learning strategy proposals
- **OPTIMIZATION_BRAINSTORM.md** - Optimization ideas and experiments
- **PROJECT_RECAP.md** - Early project recap
- **QUICK_START.md** - Original quick start guide (superseded by README)
- **V10_PRODUCTION_STRATEGY.md** - V10 production strategy (superseded by V22)

## 🏆 Current Production Strategy

**V22-Sqrt Kelly Position Sizing**

The current production strategy is **V22-Sqrt**, which uses Kelly Criterion position sizing with square root transformation.

**Key Features:**
- Kelly-weighted position sizing (weight ∝ √score)
- 5-stock high conviction portfolio
- VIX-based regime detection
- Portfolio-level drawdown control

**Performance:**
- Annual Return: 10.2%
- Sharpe Ratio: 1.11
- Max Drawdown: -15.2%
- Win Rate: 80% (16/20 positive years)

See [V22_PRODUCTION_SUMMARY.md](../V22_PRODUCTION_SUMMARY.md) for complete details.

## 📊 Strategy Evolution

| Version | Year | Key Innovation | Annual Return |
|---------|------|----------------|---------------|
| V5 | 2024 | Base momentum scoring | 7.8% |
| V6 | 2024 | Momentum quality filters | 8.1% |
| V7 | 2024 | Mid-month rebalancing | 8.3% |
| V8 | 2024 | VIX regime detection | 8.4% |
| V10 | 2024 | Inverse volatility weighting | 8.2% |
| V11 | 2024 | Adaptive hybrid weighting | 8.3% |
| V12 | 2024 | Drawdown control | 8.2% |
| V13 | 2024 | Momentum-strength weighting + 5 stocks | 9.8% |
| **V22** | **2026** | **Kelly position sizing** | **10.2%** ⭐ |

## 🔍 Finding Information

### For New Users
Start with the main [README.md](../README.md) for:
- Quick start guide
- Installation instructions
- Basic usage examples
- Strategy overview

### For Implementation Details
See [V22_PRODUCTION_SUMMARY.md](../V22_PRODUCTION_SUMMARY.md) for:
- Complete V22 strategy specification
- Kelly weighting formula
- Performance analysis
- Risk metrics

### For Integration & Next Steps
See [V22_INTEGRATION_COMPLETE.md](../V22_INTEGRATION_COMPLETE.md) for:
- How to use V22 in production
- Paper trading setup
- Live trading considerations
- API integration guide

### For Historical Context
Browse `docs/archive/` for:
- Previous strategy versions
- Optimization experiments
- Research and brainstorming
- Development history

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and changes.

## 🤝 Contributing

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) before submitting pull requests.

---

**Current Version:** V22-Sqrt Kelly Position Sizing

**Last Updated:** 2026-01-01
