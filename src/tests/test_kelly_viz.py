"""Quick test to verify Kelly weighting works"""
import sys
sys.path.append('src')
from backtest.portfolio_bot_demo import PortfolioRotationBot

# Initialize
bot = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
bot.prepare_data()
bot.score_all_stocks()

# Test Kelly weighting method
test_scores = [
    ('AAPL', 120),
    ('MSFT', 100),
    ('GOOGL', 80),
    ('NVDA', 70),
    ('META', 60)
]

print("Testing Kelly weighting function:")
weights = bot.calculate_kelly_weights_sqrt(test_scores)
for ticker, weight in weights.items():
    print(f"  {ticker}: {weight:.1%}")

print("\nRunning backtest with Kelly weighting...")
portfolio_df = bot.backtest_with_bear_protection(
    top_n=5,
    use_vix_regime=True,
    use_kelly_weighting=True,
    use_drawdown_control=True,
    trading_fee_pct=0.001
)

final_value = portfolio_df['value'].iloc[-1]
years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
annual_return = ((final_value / 100000) ** (1/years) - 1) * 100

print(f"\nResults:")
print(f"  Final Value: ${final_value:,.0f}")
print(f"  Annual Return: {annual_return:.1f}%")
print(f"  Expected: $653,746 / 10.2%")
