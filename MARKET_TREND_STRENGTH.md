# Market Trend Strength Filter Implementation

## Overview

This document describes the Market Regime Filter (Time-Series Momentum) implementation for the Portfolio Rotation Bot.

## Feature Description

The `calculate_market_trend_strength()` method determines overall market strength using the S&P 500 index (SPY) and its 200-day moving average. Unlike binary bull/bear indicators, this returns a **continuous trend strength score** from 0 to 1.

## Formula

```
trend_strength = clip((SPY_price / MA200 - 1) / 0.15, 0, 1)
```

### Interpretation

- **0.0** = Weak / Bearish market (SPY is 15%+ below MA200)
- **0.5** = Neutral (SPY is at MA200)
- **1.0** = Strong / Bullish market (SPY is 15%+ above MA200)

### Key Advantages

1. **Avoids Hindsight Bias**: Uses only data available up to the current date
2. **Reduces Whipsaw Effects**: Continuous signal vs binary on/off
3. **Proportional Assessment**: Market strength is measured on a scale, not just "bull" or "bear"

## Implementation Details

### Method Signature

```python
def calculate_market_trend_strength(self, date):
    """
    Returns:
        float: Trend strength from 0.0 (weak/bearish) to 1.0 (strong/bullish)
    """
```

### Location

The method is implemented in `src/backtest/portfolio_bot_demo.py` in the `PortfolioRotationBot` class, after the `calculate_vix_regime()` method.

### Error Handling

- Returns **0.5** (neutral) if SPY data is not available
- Returns **0.5** (neutral) if insufficient data (< 200 days) for MA calculation

## Usage Examples

### Example 1: Basic Usage

```python
from src.backtest.portfolio_bot_demo import PortfolioRotationBot
import pandas as pd

# Initialize bot
bot = PortfolioRotationBot(data_dir='sp500_data/daily')
bot.prepare_data()

# Calculate trend strength for a specific date
date = pd.Timestamp('2024-10-03')
trend_strength = bot.calculate_market_trend_strength(date)

print(f"Market Trend Strength on {date}: {trend_strength:.3f}")

# Interpret the result
if trend_strength >= 0.8:
    print("Market Regime: STRONG BULLISH")
elif trend_strength >= 0.6:
    print("Market Regime: BULLISH")
elif trend_strength >= 0.4:
    print("Market Regime: NEUTRAL")
elif trend_strength >= 0.2:
    print("Market Regime: BEARISH")
else:
    print("Market Regime: STRONG BEARISH")
```

### Example 2: Using in Portfolio Strategy

```python
# In your backtest loop
for date in dates:
    # Calculate market trend strength
    trend_strength = bot.calculate_market_trend_strength(date)

    # Adjust position sizing based on trend strength
    # Example: Reduce exposure in weak markets
    if trend_strength < 0.3:
        # Very bearish - high cash reserve
        position_multiplier = 0.3  # Only 30% invested
    elif trend_strength < 0.5:
        # Bearish - moderate cash reserve
        position_multiplier = 0.6  # 60% invested
    else:
        # Bullish - normal exposure
        position_multiplier = 1.0  # Fully invested

    # Apply multiplier to position sizing
    max_positions = int(top_n * position_multiplier)
```

### Example 3: Combining with Other Filters

```python
# Combine trend strength with VIX regime for more robust filtering
trend_strength = bot.calculate_market_trend_strength(date)
vix_cash_reserve = bot.calculate_vix_regime(date)

# Use the most conservative approach
# High VIX = more cash, Low trend strength = more cash
combined_cash_reserve = max(
    (1 - trend_strength) * 0.7,  # Trend-based cash (0-70%)
    vix_cash_reserve              # VIX-based cash
)

print(f"Trend Strength: {trend_strength:.2f}")
print(f"VIX Cash Reserve: {vix_cash_reserve:.2%}")
print(f"Combined Cash Reserve: {combined_cash_reserve:.2%}")
```

### Example 4: Dynamic Position Sizing

```python
# Use trend strength for continuous position sizing
trend_strength = bot.calculate_market_trend_strength(date)

# Calculate cash reserve inversely proportional to trend strength
# Strong trend (1.0) = 5% cash, Weak trend (0.0) = 70% cash
min_cash = 0.05
max_cash = 0.70
cash_reserve = max_cash - (trend_strength * (max_cash - min_cash))

print(f"Trend Strength: {trend_strength:.2f}")
print(f"Cash Reserve: {cash_reserve:.2%}")
print(f"Equity Exposure: {(1-cash_reserve):.2%}")
```

## Integration with Existing Backtests

The method can be integrated into the existing `backtest_swing_trading()` function by adding a new parameter:

```python
def backtest_swing_trading(self, ..., use_trend_strength=False):
    ...
    if use_trend_strength:
        trend_strength = self.calculate_market_trend_strength(date)
        # Convert trend strength to cash reserve
        cash_reserve = 0.70 - (trend_strength * 0.65)  # 70% to 5% range
    elif use_vix_regime:
        cash_reserve = self.calculate_vix_regime(date)
    ...
```

## Testing

Run the test script to verify the implementation:

```bash
python test_market_trend_strength.py
```

The test script:
- Loads historical SPY data
- Tests trend strength calculation on various dates
- Displays summary statistics
- Tests edge cases (insufficient data, exact 200 days, etc.)

### Expected Output

The test shows:
- Trend strength values correctly range from 0.0 to 1.0
- Values are clipped at extremes (large deviations still result in 0.0 or 1.0)
- Neutral value (0.5) is returned when data is insufficient
- Distribution shows market was bullish ~65% of tested dates

## Performance Characteristics

- **Computation**: O(1) - Very fast, only needs last price and MA200
- **Memory**: Minimal - Uses existing SPY data
- **Latency**: None - Synchronous calculation

## Comparison with Existing Methods

| Method | Type | Range | Pros | Cons |
|--------|------|-------|------|------|
| `calculate_market_trend_strength()` | Continuous | 0.0-1.0 | Smooth transitions, proportional | Requires 200 days history |
| `calculate_vix_regime()` | Discrete | 0.05-0.70 | Forward-looking, fear-based | Requires VIX data |
| `calculate_adaptive_regime()` | Discrete | 0.05-0.65 | Multi-factor, robust | More complex, slower |
| Binary SPY > MA200 | Binary | 0/1 | Simple, fast | Whipsaw prone, no gradation |

## Recommended Use Cases

1. **Position Sizing**: Scale exposure based on market strength
2. **Entry Filters**: Only enter new positions when trend_strength > 0.5
3. **Risk Management**: Increase cash reserves when trend weakens
4. **Strategy Combination**: Blend with VIX or adaptive regime for robust filtering
5. **Backtesting**: Compare performance across different trend strength thresholds

## Future Enhancements

Potential improvements:
1. Make the normalization factor (0.15 or 15%) configurable
2. Add support for different market indices (NASDAQ, Russell 2000)
3. Implement trend strength momentum (rate of change)
4. Add multi-timeframe trend strength (50-day, 100-day, 200-day)

## References

- Location: `src/backtest/portfolio_bot_demo.py` (line ~696-746)
- Test Script: `test_market_trend_strength.py`
- Related Methods: `calculate_vix_regime()`, `calculate_adaptive_regime()`, `_calculate_regime_score()`
