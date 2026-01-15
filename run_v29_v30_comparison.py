#\!/usr/bin/env python3
"""V29 vs V30 Strategy Comparison"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from src.strategies.v29_mega_cap_split import V29Strategy
from src.strategies.v30_dynamic_megacap import V30Strategy, calculate_metrics

def run_comparison(data_dir, period_name, start_year, end_year):
    print(f"\n{'='*80}")
    print(f"PERIOD: {period_name} ({start_year}-{end_year})")
    print(f"{'='*80}")
    
    full_path = os.path.join(project_root, data_dir)
    if not os.path.exists(full_path):
        print(f"  Dataset not found")
        return None
    
    bot = PortfolioRotationBot(data_dir=full_path, initial_capital=100000)
    bot.prepare_data()
    print(f"Loaded {len(bot.stocks_data)} stocks")
    
    config = {
        'mag7_allocation': 0.70, 'megacap_allocation': 0.70,
        'num_mag7': 3, 'num_megacap': 3, 'num_momentum': 2,
        'trailing_stop': 0.15, 'max_portfolio_dd': 0.25, 'vix_crisis': 35,
        'num_top_megacaps': 7, 'lookback_trading_value': 20,
    }
    
    print("  Running V29 (Fixed Mag7)...")
    v29_strategy = V29Strategy(bot, config=config)
    v29_portfolio = v29_strategy.run_backtest(start_year=start_year, end_year=end_year)
    v29_metrics = calculate_metrics(v29_portfolio, 100000)
    
    print("  Running V30 (Dynamic Mega-Cap)...")
    v30_strategy = V30Strategy(bot, config=config)
    v30_portfolio = v30_strategy.run_backtest(start_year=start_year, end_year=end_year)
    v30_metrics = calculate_metrics(v30_portfolio, 100000)
    
    last_date = v30_portfolio.index[-1]
    v30_megacaps = v30_strategy.identify_megacaps(last_date, 7)
    
    spy_annual = 10.0
    if 'SPY' in bot.stocks_data:
        spy_df = bot.stocks_data['SPY']
        spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
        if len(spy_period) > 1:
            spy_start, spy_end = spy_period['close'].iloc[0], spy_period['close'].iloc[-1]
            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
            spy_annual = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100
    
    v29_alpha = v29_metrics['annual_return'] - spy_annual
    v30_alpha = v30_metrics['annual_return'] - spy_annual
    
    print(f"\n  V30 Identified Mega-Caps: {', '.join(v30_megacaps[:5])}")
    print(f"  V29 Fixed Mag7: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA")
    print(f"\n  {'Metric':<20} {'V29 (Fixed)':>15} {'V30 (Dynamic)':>15} {'Difference':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Annual Return':<20} {v29_metrics['annual_return']:>14.1f}% {v30_metrics['annual_return']:>14.1f}% {v30_metrics['annual_return']-v29_metrics['annual_return']:>+14.1f}%")
    print(f"  {'Max Drawdown':<20} {v29_metrics['max_drawdown']:>14.1f}% {v30_metrics['max_drawdown']:>14.1f}% {v30_metrics['max_drawdown']-v29_metrics['max_drawdown']:>+14.1f}%")
    print(f"  {'Sharpe Ratio':<20} {v29_metrics['sharpe']:>14.2f} {v30_metrics['sharpe']:>14.2f} {v30_metrics['sharpe']-v29_metrics['sharpe']:>+14.2f}")
    print(f"  {'Alpha vs SPY':<20} {v29_alpha:>+14.1f}% {v30_alpha:>+14.1f}% {v30_alpha-v29_alpha:>+14.1f}%")
    
    return {'period': period_name, 'v29_annual': v29_metrics['annual_return'],
            'v30_annual': v30_metrics['annual_return'], 'v29_alpha': v29_alpha, 'v30_alpha': v30_alpha}

def main():
    print("="*80)
    print("V29 (FIXED MAG7) vs V30 (DYNAMIC MEGA-CAP) COMPARISON")
    print("="*80)
    
    tests = [
        ('sp500_data/stock_data_1963_1983_top500', '1963-1983', 1965, 1983),
        ('sp500_data/stock_data_1983_2003_top500', '1983-2003', 1985, 2003),
        ('sp500_data/stock_data_1990_2024_top500', '1990-2024', 1992, 2024),
        ('sp500_data/stock_data_1990_2024_top500', '2015-2024', 2015, 2024),
    ]
    
    results = []
    for data_dir, period_name, start_year, end_year in tests:
        result = run_comparison(data_dir, period_name, start_year, end_year)
        if result:
            results.append(result)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Period':<15} {'Winner':<10} {'V29 Annual':>12} {'V30 Annual':>12} {'Difference':>12}")
    print("-"*70)
    
    v29_wins = v30_wins = 0
    for r in results:
        winner = "V30" if r['v30_annual'] > r['v29_annual'] else "V29"
        v30_wins += 1 if winner == "V30" else 0; v29_wins += 1 if winner == "V29" else 0
        diff = r['v30_annual'] - r['v29_annual']
        print(f"{r['period']:<15} {winner:<10} {r['v29_annual']:>11.1f}% {r['v30_annual']:>11.1f}% {diff:>+11.1f}%")
    
    print("-"*70)
    print(f"Win Rate: V29={v29_wins}/{len(results)}  V30={v30_wins}/{len(results)}")

if __name__ == '__main__':
    main()
