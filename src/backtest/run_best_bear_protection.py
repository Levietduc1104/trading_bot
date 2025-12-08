"""
RUN BEST BEAR PROTECTION WITH ADAPTIVE REGIME
Based on unbiased timing strategy comparison:
- Adaptive Multi-Factor Regime Detection (winner: 7.92% annual return)
- Uses 4 standard factors: trend, momentum, volatility, breadth
- Dynamic cash reserves: 5% (very bullish) to 65% (very bearish)
- No parameter tuning - all standard industry parameters
"""
import sys
import os

# Get the project root directory (2 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'src', 'backtest'))

from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("RUNNING ADAPTIVE MULTI-FACTOR REGIME STRATEGY")
print("="*80)

# Initialize bot
data_dir = os.path.join(project_root, 'sp500_data', 'daily')
bot = PortfolioRotationBot(data_dir=data_dir)
bot.prepare_data()
bot.score_all_stocks()

# Run adaptive regime strategy
print("\nAdaptive Multi-Factor Configuration:")
print("  - Monthly rebalancing")
print("  - Dynamic cash reserve based on 4 market factors:")
print("    1. Trend (200-day MA)")
print("    2. Momentum (50-day ROC)")
print("    3. Volatility (30-day vs 1-year)")
print("    4. Market Breadth (% stocks > 200 MA)")
print("  - Cash range: 5% (very bullish) to 65% (very bearish)")
print("")

portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,
    rebalance_freq='M',
    use_adaptive_regime=True  # Enable adaptive regime detection
)

print("\n" + "="*80)
print("ADAPTIVE REGIME BACKTEST COMPLETE")
print("="*80)

