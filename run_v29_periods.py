#\!/usr/bin/env python3
"""
V29 Strategy - Test Multiple Time Periods
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))


def main():
    import pandas as pd
    import numpy as np
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v29_mega_cap_split import V29Strategy, calculate_metrics
    
    print("="*80)
    print("V29 STRATEGY - MULTI-PERIOD ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    print("Loading data...")
    
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    print("Data loaded.\n")
    
    # Configuration
    config = {
        'mag7_allocation': 0.70,
        'num_mag7': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
    }
    
    # Different periods to test
    periods = [
        (2000, 2024, "Full History (2000-2024)"),
        (2000, 2010, "Dot-Com & Financial Crisis (2000-2010)"),
        (2010, 2020, "Bull Market (2010-2020)"),
        (2015, 2024, "Recent Decade (2015-2024)"),
        (2018, 2020, "COVID Period (2018-2020)"),
        (2020, 2024, "Post-COVID (2020-2024)"),
        (2022, 2024, "Rate Hike Era (2022-2024)"),
        (2023, 2024, "Most Recent (2023-2024)"),
    ]
    
    spy_df = bot.stocks_data.get('SPY')
    
    results = []
    
    print(f"{'Period':<35} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'Sharpe':>8}")
    print("="*80)
    
    for start_year, end_year, label in periods:
        try:
            strategy = V29Strategy(bot, config=config)
            portfolio_df = strategy.run_backtest(start_year=start_year, end_year=end_year)
            
            metrics = calculate_metrics(portfolio_df, 100000)
            
            if metrics and len(portfolio_df) > 50:
                # Calculate SPY benchmark
                spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
                if len(spy_period) > 1:
                    spy_start = spy_period['close'].iloc[0]
                    spy_end = spy_period['close'].iloc[-1]
                    spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
                    spy_return = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100 if spy_years > 0 else 0
                else:
                    spy_return = 0
                
                alpha = metrics['annual_return'] - spy_return
                status = "OK" if alpha > 0 else "MISS"
                
                results.append({
                    'period': label,
                    'start': start_year,
                    'end': end_year,
                    'return': metrics['annual_return'],
                    'spy_return': spy_return,
                    'alpha': alpha,
                    'max_dd': metrics['max_drawdown'],
                    'sharpe': metrics['sharpe'],
                })
                
                print(f"{label:<35} {metrics['annual_return']:>9.1f}% {alpha:>+9.1f}% {metrics['max_drawdown']:>9.1f}% {metrics['sharpe']:>7.2f} {status}")
        except Exception as e:
            print(f"{label:<35} ERROR: {str(e)[:30]}")
    
    print("="*80)
    print()
    
    # Summary statistics
    if results:
        avg_return = sum(r['return'] for r in results) / len(results)
        avg_alpha = sum(r['alpha'] for r in results) / len(results)
        avg_dd = sum(r['max_dd'] for r in results) / len(results)
        avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
        win_rate = sum(1 for r in results if r['alpha'] > 0) / len(results) * 100
        
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"  Average Annual Return: {avg_return:.1f}%")
        print(f"  Average Alpha vs SPY:  {avg_alpha:+.1f}%")
        print(f"  Average Max Drawdown:  {avg_dd:.1f}%")
        print(f"  Average Sharpe Ratio:  {avg_sharpe:.2f}")
        print(f"  Win Rate vs SPY:       {win_rate:.0f}%")
        print()
        
        # Best and worst periods
        best = max(results, key=lambda x: x['alpha'])
        worst = min(results, key=lambda x: x['alpha'])
        
        print(f"Best Period:  {best['period']}")
        print(f"  Return: {best['return']:.1f}%, Alpha: {best['alpha']:+.1f}%")
        print()
        print(f"Worst Period: {worst['period']}")
        print(f"  Return: {worst['return']:.1f}%, Alpha: {worst['alpha']:+.1f}%")
        print()
        
        # Lowest drawdown
        lowest_dd = min(results, key=lambda x: x['max_dd'])
        highest_dd = max(results, key=lambda x: x['max_dd'])
        
        print(f"Lowest Drawdown:  {lowest_dd['period']} ({lowest_dd['max_dd']:.1f}%)")
        print(f"Highest Drawdown: {highest_dd['period']} ({highest_dd['max_dd']:.1f}%)")
    
    print()
    print("="*80)


if __name__ == '__main__':
    main()
