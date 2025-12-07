"""
RUN BEST BEAR PROTECTION AND VISUALIZE
Based on testing, the best config is:
- Monthly rebalancing
- 70% cash in bear markets
- 10% cash in bull markets
- Result: 18.3% annual, 2021: +2.4%, 2022: -10.1%
"""
import sys
import os

# Get the project root directory (2 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'src', 'backtest'))

from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("RUNNING BEST BEAR PROTECTION STRATEGY")
print("="*80)

# Initialize bot
data_dir = os.path.join(project_root, 'sp500_data', 'daily')
bot = PortfolioRotationBot(data_dir=data_dir)
bot.prepare_data()
bot.score_all_stocks()

# Run best configuration
print("\nBest Configuration:")
print("  - Monthly rebalancing")
print("  - 70% cash in bear markets (when SPY < 200-day MA)")
print("  - 10% cash in bull markets")
print("")

portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,
    rebalance_freq='M',
    bear_cash_reserve=0.70,
    bull_cash_reserve=0.10
)

print("\n" + "="*80)
print("SAVED! Now updating visualization...")
print("="*80)
