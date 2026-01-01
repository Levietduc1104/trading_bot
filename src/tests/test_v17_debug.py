"""
Debug test to verify correlation diversification is working
"""

import sys
import pandas as pd
sys.path.insert(0, 'src')

from backtest.portfolio_bot_demo import PortfolioRotationBot

# Initialize bot
bot = PortfolioRotationBot(
    data_dir='sp500_data/daily',
    initial_capital=100000
)

# Load data
bot.prepare_data()

# Test correlation calculation on a specific date
test_date = pd.Timestamp('2010-06-15')

# Create mock holdings
mock_holdings = {
    'AAPL': 100,
    'MSFT': 100,
    'GOOGL': 100
}

# Test correlation for a tech stock (should be high correlation)
print("Testing correlation for NVDA (tech stock):")
corr_nvda = bot.calculate_stock_correlation_to_portfolio('NVDA', mock_holdings, test_date, lookback_days=60)
penalty_nvda = bot.calculate_correlation_penalty(corr_nvda)
print(f"  Correlation: {corr_nvda:.3f}")
print(f"  Penalty multiplier: {penalty_nvda:.3f}")

# Test correlation for a non-tech stock (should be lower correlation)
print("\nTesting correlation for WMT (consumer staples):")
corr_wmt = bot.calculate_stock_correlation_to_portfolio('WMT', mock_holdings, test_date, lookback_days=60)
penalty_wmt = bot.calculate_correlation_penalty(corr_wmt)
print(f"  Correlation: {corr_wmt:.3f}")
print(f"  Penalty multiplier: {penalty_wmt:.3f}")

# Test with empty holdings (should return neutral)
print("\nTesting correlation with empty holdings:")
corr_empty = bot.calculate_stock_correlation_to_portfolio('AAPL', {}, test_date, lookback_days=60)
penalty_empty = bot.calculate_correlation_penalty(corr_empty)
print(f"  Correlation: {corr_empty:.3f}")
print(f"  Penalty multiplier: {penalty_empty:.3f}")
