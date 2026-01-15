#\!/usr/bin/env python3
"""
V29 Strategy Test: Compare With and Without Drawdown Protection

Tests the impact of drawdown protection mechanisms on:
1. Maximum drawdown reduction
2. Return impact
3. Risk-adjusted returns (Sharpe)
"""

import sys
import os

# Setup paths first
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))


def main():
    """Main test function - must be inside if __name__ == '__main__' for multiprocessing"""
    import pandas as pd
    import numpy as np
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v29_mega_cap_split import V29Strategy, calculate_metrics

    print("="*80)
    print("V29 STRATEGY TEST: DRAWDOWN PROTECTION IMPACT")
    print("="*80)
    print()

    # Load data
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    print(f"Loading data from: {data_dir}")

    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    print("Data loaded.\n")

    # Calculate SPY benchmark
    spy_df = bot.stocks_data.get('SPY')
    spy_return = 12.4  # Default
    spy_dd = -33.7  # Default

    if spy_df is not None:
        spy_2015_2024 = spy_df[(spy_df.index >= '2015-01-01') & (spy_df.index <= '2024-12-31')]
        if len(spy_2015_2024) > 1:
            spy_start = spy_2015_2024['close'].iloc[0]
            spy_end = spy_2015_2024['close'].iloc[-1]
            spy_years = (spy_2015_2024.index[-1] - spy_2015_2024.index[0]).days / 365.25
            spy_return = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100

            spy_cummax = spy_2015_2024['close'].cummax()
            spy_dd = ((spy_2015_2024['close'] - spy_cummax) / spy_cummax * 100).min()

    print(f"SPY Benchmark (2015-2024):")
    print(f"  Annual Return: {spy_return:.1f}%")
    print(f"  Max Drawdown: {spy_dd:.1f}%")
    print()

    # Test configurations
    configs = [
        {
            'name': 'V29 WITHOUT Protection',
            'config': {
                'mag7_allocation': 0.70,
                'num_mag7': 3,
                'num_momentum': 2,
                'trailing_stop': 1.0,
                'max_portfolio_dd': 1.0,
                'vix_crisis': 100,
            }
        },
        {
            'name': 'V29 WITH Protection (Conservative)',
            'config': {
                'mag7_allocation': 0.70,
                'num_mag7': 3,
                'num_momentum': 2,
                'trailing_stop': 0.15,
                'max_portfolio_dd': 0.25,
                'vix_crisis': 35,
            }
        },
        {
            'name': 'V29 WITH Protection (Aggressive)',
            'config': {
                'mag7_allocation': 0.70,
                'num_mag7': 3,
                'num_momentum': 2,
                'trailing_stop': 0.12,
                'max_portfolio_dd': 0.20,
                'vix_crisis': 30,
            }
        },
    ]

    results = []

    print("Testing configurations...")
    print("-" * 80)
    print()

    for cfg in configs:
        print(f"Testing: {cfg['name']}")

        strategy = V29Strategy(bot, config=cfg['config'])
        portfolio_df = strategy.run_backtest(start_year=2015, end_year=2024)

        metrics = calculate_metrics(portfolio_df, 100000)

        if metrics:
            alpha = metrics['annual_return'] - spy_return
            results.append({
                'name': cfg['name'],
                **metrics,
                'alpha': alpha
            })

            print(f"  Return: {metrics['annual_return']:.1f}% | MaxDD: {metrics['max_drawdown']:.1f}% | Sharpe: {metrics['sharpe']:.2f}")
        print()

    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print()

    print(f"{'Configuration':<35} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'Sharpe':>8}")
    print("-" * 80)

    for r in results:
        status = "OK" if r['alpha'] > 0 else "MISS"
        print(f"{r['name']:<35} {r['annual_return']:>9.1f}% {r['alpha']:>+9.1f}% {r['max_drawdown']:>9.1f}% {r['sharpe']:>7.2f} {status}")

    print(f"{'SPY (Benchmark)':<35} {spy_return:>9.1f}% {0:>9.1f}%  {spy_dd:>9.1f}%")

    # Calculate drawdown reduction
    if len(results) >= 2:
        print()
        print("="*80)
        print("DRAWDOWN REDUCTION ANALYSIS")
        print("="*80)
        print()

        base_dd = results[0]['max_drawdown']
        for r in results[1:]:
            reduction = base_dd - r['max_drawdown']
            return_cost = results[0]['annual_return'] - r['annual_return']
            print(f"{r['name']}:")
            print(f"  Drawdown reduction: {reduction:+.1f}% ({base_dd:.1f}% -> {r['max_drawdown']:.1f}%)")
            print(f"  Return cost: {return_cost:.1f}%")
            if return_cost > 0:
                print(f"  Trade-off: {abs(reduction/return_cost):.1f}% DD reduction per 1% return cost")
            else:
                print("  No return cost\!")
            print()

    # Best risk-adjusted
    if results:
        print("="*80)
        print("RECOMMENDATION")
        print("="*80)
        print()

        best_sharpe = max(results, key=lambda x: x['sharpe'])
        print(f"Best Risk-Adjusted: {best_sharpe['name']}")
        print(f"  Sharpe Ratio: {best_sharpe['sharpe']:.2f}")
        print(f"  Return: {best_sharpe['annual_return']:.1f}%")
        print(f"  Max Drawdown: {best_sharpe['max_drawdown']:.1f}%")
        print()

        lowest_dd = min(results, key=lambda x: x['max_drawdown'])
        print(f"Lowest Drawdown: {lowest_dd['name']}")
        print(f"  Max Drawdown: {lowest_dd['max_drawdown']:.1f}%")
        print(f"  Return: {lowest_dd['annual_return']:.1f}%")
        print()

    print("="*80)
    return results


if __name__ == '__main__':
    main()
