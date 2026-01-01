# Changelog

All notable changes to the S&P 500 Portfolio Rotation Trading Bot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to semantic versioning.

## [V22-Sqrt] - 2026-01-01 🏆

### Added
- **Kelly Position Sizing (Square Root method)** - Weight positions by √score
- `calculate_kelly_weights_sqrt()` method in PortfolioRotationBot
- `use_kelly_weighting` parameter in `backtest_with_bear_protection()`
- Direct V22 backtest function in execution.py for correct implementation
- Continuous VIX regime formula (replaced discrete thresholds)
- Database-driven visualization (reads from database instead of recalculating)
- Comprehensive V22 documentation (V22_PRODUCTION_SUMMARY.md)
- V22 integration guide (V22_INTEGRATION_COMPLETE.md)

### Changed
- Updated execution.py to use V22 as production strategy
- Updated README.md to reflect V22 strategy and performance
- Improved VIX regime detection with continuous formula
- Visualization now reads from database for consistency
- Organized documentation into docs/ folder

### Performance
- Annual Return: **10.2%** (+0.4% vs V13)
- Max Drawdown: **-15.2%** (better than V13's -19.1%)
- Sharpe Ratio: **1.11** (better than V13's 1.07)
- Win Rate: **80%** (16/20 positive years)

### Validation
Kelly sizing proves scoring quality matters:
- Higher returns AND better risk metrics
- Concentrates capital where edge is highest
- Conservative square root approach (vs linear)

---

## [V13] - 2024-12-28

### Added
- Momentum-strength weighting (weight ∝ momentum / volatility)
- 5-stock concentration (vs previous 10-stock portfolio)
- Production-ready execution script

### Changed
- Reduced portfolio from 10 to 5 stocks for higher conviction
- Updated position sizing to use momentum strength

### Performance
- Annual Return: 9.8% (+1.4% vs V12)
- Max Drawdown: -19.1%
- Sharpe Ratio: 1.07
- Win Rate: 80%

---

## [V12] - 2024-12-22

### Added
- Portfolio-level drawdown control
- Progressive exposure reduction during drawdowns
- Drawdown multiplier logic (25% to 100% based on DD level)

### Changed
- Risk management now operates at portfolio level (not per-stock)

### Performance
- Annual Return: 8.2%
- Max Drawdown: -18.5% (improved from V11's -22.8%)
- Sharpe Ratio: 1.23

---

## [V11] - 2024-12-13

### Added
- Adaptive hybrid position weighting
- VIX-based regime switching:
  - VIX < 30: Equal weighting (calm markets)
  - VIX ≥ 30: Inverse volatility weighting (stressed markets)

### Changed
- Position sizing now adapts to market conditions

### Performance
- Annual Return: 8.3%
- Max Drawdown: -22.8%
- Sharpe Ratio: 1.22

---

## [V10] - 2024-12-07

### Added
- Inverse volatility position weighting
- Weight formula: weight ∝ score / volatility
- Risk-adjusted allocation

### Performance
- Annual Return: 8.2%
- Max Drawdown: -22.8%
- Sharpe Ratio: 1.21

---

## [V8] - 2024-12-07

### Added
- VIX-based regime detection (forward-looking)
- Dynamic cash reserves (5% to 70% based on VIX)
- VIX proxy calculation from SPY options

### Changed
- Replaced 200-day MA with VIX for regime detection
- Better early warning system for market stress

### Performance
- Annual Return: 8.4%
- Max Drawdown: -23.2%
- Sharpe Ratio: 1.15

---

## [V7] - 2024-11-28

### Added
- Mid-month rebalancing (day 7-10)
- Seasonal adjustments (winter aggressive, summer defensive)
- Sector relative strength bonus (±15 points)

### Changed
- Rebalancing avoids month-end institutional flows
- Stocks scored relative to sector peers

### Performance
- Annual Return: 8.3%
- Sharpe improvement from avoiding crowded trades

---

## [V6] - 2024-11-27

### Added
- Momentum quality filters (disqualification criteria)
- Must be above EMA-89 (score = 0 if fails)
- Must have ROC-20 > 2% (score = 0 if fails)
- RSI penalties for overbought/oversold

### Changed
- Stricter momentum requirements
- Disqualifies weak trends completely

### Performance
- Annual Return: 8.1%
- Reduced drawdowns by avoiding false breakouts

---

## [V5] - 2024-11-27

### Added
- Base momentum scoring system (0-100 points)
- Price trend scoring (EMA-13, EMA-34, EMA-89)
- Recent performance scoring (ROC-20)
- Risk level scoring (ATR%)
- Monthly rebalancing
- Bear/bull market detection

### Performance
- Annual Return: 7.8%
- Established baseline strategy

---

## Version History Summary

| Version | Date | Key Innovation | Annual Return | Max DD | Sharpe |
|---------|------|----------------|---------------|--------|--------|
| V5 | Nov 2024 | Base momentum scoring | 7.8% | -25.3% | 1.05 |
| V6 | Nov 2024 | Momentum filters | 8.1% | -24.1% | 1.08 |
| V7 | Nov 2024 | Mid-month rebalancing | 8.3% | -23.5% | 1.12 |
| V8 | Dec 2024 | VIX regime | 8.4% | -23.2% | 1.15 |
| V10 | Dec 2024 | Inverse vol weighting | 8.2% | -22.8% | 1.21 |
| V11 | Dec 2024 | Adaptive hybrid | 8.3% | -22.8% | 1.22 |
| V12 | Dec 2024 | Drawdown control | 8.2% | -18.5% | 1.23 |
| V13 | Dec 2024 | Momentum weighting + 5 stocks | 9.8% | -19.1% | 1.07 |
| **V22** | **Jan 2026** | **Kelly position sizing** | **10.2%** | **-15.2%** | **1.11** |

---

## Key Milestones

### Breaking 10% Annual Return (V22)
- First strategy to achieve >10% annual return
- Improved both returns AND risk metrics simultaneously
- Validates scoring quality through Kelly sizing

### Portfolio Concentration (V13)
- Reduced from 10 to 5 stocks
- +1.4% annual improvement
- Proved high conviction beats diversification

### Risk Control (V12)
- First major drawdown reduction (-18.5% from -22.8%)
- Portfolio-level exposure management
- Foundation for V22 success

### VIX Integration (V8)
- Switched from lagging to forward-looking indicators
- Dynamic cash reserves
- Better regime detection

---

## Future Considerations

### Potential V23+ Improvements
- [ ] Multi-timeframe momentum ensemble
- [ ] Sector rotation overlay
- [ ] Volatility-targeted leverage
- [ ] Correlation-based diversification
- [ ] Progressive cash redeployment

**Note:** V22 sets a high bar - any V23 must achieve:
- >10.2% annual return
- <-15.2% max drawdown
- >1.11 Sharpe ratio

---

**Current Production Version:** V22-Sqrt Kelly Position Sizing

**Last Updated:** 2026-01-01
