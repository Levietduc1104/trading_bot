import sys, os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from src.strategies.v30_dynamic_megacap import V30Strategy, calculate_metrics

print("Testing V30 with ETF exclusion fix (2015-2024)...")
bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024_top500', initial_capital=100000)
bot.prepare_data()

strategy = V30Strategy(bot)
portfolio_df = strategy.run_backtest(start_year=2015, end_year=2024)
metrics = calculate_metrics(portfolio_df, 100000)

# Show identified mega-caps
last_date = portfolio_df.index[-1]
megacaps = strategy.identify_megacaps(last_date, 7)

print(f"\n✅ V30 Fixed - Identified Mega-Caps (No ETFs):")
print(f"   {', '.join(megacaps)}")
print(f"\nPerformance (2015-2024):")
print(f"   Annual Return: {metrics['annual_return']:.1f}%")
print(f"   Max Drawdown:  {metrics['max_drawdown']:.1f}%")
print(f"   Sharpe Ratio:  {metrics['sharpe']:.2f}")
print(f"   Final Value:   ${metrics['final_value']:,.0f}")
