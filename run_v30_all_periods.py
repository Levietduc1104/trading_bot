#\!/usr/bin/env python3
"""V30 Dynamic Mega-Cap Strategy - Test Across All Time Periods"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

def run_period(data_dir, period_name, start_year, end_year):
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v30_dynamic_megacap import V30Strategy, calculate_metrics

    print(f"\n{'='*70}")
    print(f"TESTING: {period_name} ({start_year}-{end_year})")
    print(f"{'='*70}")

    full_path = os.path.join(project_root, data_dir)
    if not os.path.exists(full_path):
        print(f"  Dataset not found: {data_dir}")
        return None

    bot = PortfolioRotationBot(data_dir=full_path, initial_capital=100000)
    bot.prepare_data()
    print(f"  Loaded {len(bot.stocks_data)} stocks")

    config = {
        'megacap_allocation': 0.70,
        'num_megacap': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
        'num_top_megacaps': 7,
        'lookback_trading_value': 20,
    }

    strategy = V30Strategy(bot, config=config)
    portfolio_df = strategy.run_backtest(start_year=start_year, end_year=end_year)
    metrics = calculate_metrics(portfolio_df, 100000)

    print(f"\n  Dynamic Mega-Caps Identified:")
    first_date = portfolio_df.index[100] if len(portfolio_df) > 100 else portfolio_df.index[0]
    first_megacaps = strategy.identify_megacaps(first_date, 7)
    print(f"    Start ({first_date.year}): {', '.join(first_megacaps[:5])}...")

    last_date = portfolio_df.index[-1]
    last_megacaps = strategy.identify_megacaps(last_date, 7)
    print(f"    End ({last_date.year}):   {', '.join(last_megacaps[:5])}...")

    spy_annual = 10.0
    spy_dd = -30.0
    if 'SPY' in bot.stocks_data:
        spy_df = bot.stocks_data['SPY']
        spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
        if len(spy_period) > 1:
            spy_start = spy_period['close'].iloc[0]
            spy_end = spy_period['close'].iloc[-1]
            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
            spy_annual = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100
            spy_cummax = spy_period['close'].cummax()
            spy_dd = ((spy_period['close'] - spy_cummax) / spy_cummax * 100).min()

    alpha = metrics['annual_return'] - spy_annual

    print(f"\n  Results:")
    print(f"    V30 Annual Return:  {metrics['annual_return']:.1f}%")
    print(f"    V30 Max Drawdown:   {metrics['max_drawdown']:.1f}%")
    print(f"    V30 Sharpe Ratio:   {metrics['sharpe']:.2f}")
    print(f"    V30 Final Value:    ${metrics['final_value']:,.0f}")
    print(f"    SPY Annual Return:  {spy_annual:.1f}%")
    print(f"    Alpha vs SPY:       {alpha:+.1f}%")

    return {
        'period': period_name,
        'annual_return': metrics['annual_return'],
        'max_drawdown': metrics['max_drawdown'],
        'sharpe': metrics['sharpe'],
        'alpha': alpha,
        'last_megacaps': last_megacaps[:3],
    }

def main():
    print("="*70)
    print("V30 DYNAMIC MEGA-CAP STRATEGY - ALL PERIODS TEST")
    print("="*70)
    print("\nStrategy: 70% Top 3 Dynamic Mega-Caps + 30% Top 2 Momentum")
    print("Innovation: Automatically identifies mega-caps using trading value")

    tests = [
        ('sp500_data/stock_data_1963_1983_top500', '1963-1983 Era', 1965, 1983),
        ('sp500_data/stock_data_1983_2003_top500', '1983-2003 Era', 1985, 2003),
        ('sp500_data/stock_data_1990_2024_top500', '1990-2024 Era', 1992, 2024),
        ('sp500_data/stock_data_1990_2024_top500', '2015-2024', 2015, 2024),
    ]

    results = []
    for data_dir, period_name, start_year, end_year in tests:
        result = run_period(data_dir, period_name, start_year, end_year)
        if result:
            results.append(result)

    print("\n" + "="*70)
    print("SUMMARY: V30 DYNAMIC MEGA-CAP ACROSS ALL ERAS")
    print("="*70)
    print(f"\n{'Period':<18} {'Annual':>10} {'MaxDD':>10} {'Sharpe':>8} {'Alpha':>10} {'Mega-Caps':<20}")
    print("-"*80)

    total_alpha = 0
    positive = 0
    for r in results:
        mc = ', '.join(r['last_megacaps'])
        print(f"{r['period']:<18} {r['annual_return']:>9.1f}% {r['max_drawdown']:>9.1f}% {r['sharpe']:>8.2f} {r['alpha']:>+9.1f}% {mc:<20}")
        total_alpha += r['alpha']
        if r['alpha'] > 0:
            positive += 1

    print("-"*80)
    print(f"Average Alpha: {total_alpha/len(results):+.1f}%  |  Win Rate: {positive}/{len(results)}")

if __name__ == '__main__':
    main()
