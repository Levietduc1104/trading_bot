#\!/usr/bin/env python3
"""
V29 Strategy - Test on ALL Available Datasets (1963-2024)
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
    print("V29 STRATEGY - ALL DATASETS (1963-2024)")
    print("="*80)
    print()
    
    # Configuration
    config = {
        'mag7_allocation': 0.70,
        'num_mag7': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
    }
    
    # All available datasets with their periods
    datasets = [
        ('stock_data_1963_1983_top500', 1965, 1983, "1963-1983 Era"),
        ('stock_data_1983_2003_top500', 1985, 2003, "1983-2003 Era"),
        ('stock_data_1990_2024_top500', 1992, 2024, "1990-2024 Era"),
    ]
    
    all_results = []
    
    for data_folder, start_year, end_year, label in datasets:
        data_dir = os.path.join(project_root, 'sp500_data', data_folder)
        
        if not os.path.exists(data_dir):
            print(f"Skipping {label}: folder not found")
            continue
            
        print(f"\n{'='*80}")
        print(f"DATASET: {label}")
        print(f"Folder: {data_folder}")
        print(f"{'='*80}")
        
        try:
            bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
            bot.prepare_data()
            
            # Check available stocks
            num_stocks = len(bot.stocks_data)
            print(f"Loaded {num_stocks} stocks")
            
            # Get date range
            first_ticker = list(bot.stocks_data.keys())[0]
            dates = bot.stocks_data[first_ticker].index
            print(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
            
            # Check if SPY exists for benchmark
            has_spy = 'SPY' in bot.stocks_data
            print(f"SPY available: {has_spy}")
            print()
            
            # Run backtest
            strategy = V29Strategy(bot, config=config)
            portfolio_df = strategy.run_backtest(start_year=start_year, end_year=end_year)
            
            metrics = calculate_metrics(portfolio_df, 100000)
            
            if metrics:
                # Calculate SPY benchmark if available
                if has_spy:
                    spy_df = bot.stocks_data['SPY']
                    spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
                    if len(spy_period) > 1:
                        spy_start = spy_period['close'].iloc[0]
                        spy_end = spy_period['close'].iloc[-1]
                        spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
                        spy_return = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100 if spy_years > 0 else 0
                        spy_cummax = spy_period['close'].cummax()
                        spy_dd = ((spy_period['close'] - spy_cummax) / spy_cummax * 100).min()
                    else:
                        spy_return = 0
                        spy_dd = 0
                else:
                    # Use approximate historical S&P 500 returns
                    if start_year < 1985:
                        spy_return = 8.0  # Historical average
                        spy_dd = -40.0
                    else:
                        spy_return = 10.0
                        spy_dd = -35.0
                
                alpha = metrics['annual_return'] - spy_return
                
                print(f"V29 Strategy Results:")
                print(f"  Annual Return:  {metrics['annual_return']:>7.1f}%")
                print(f"  Total Return:   {metrics['total_return']:>7.1f}%")
                print(f"  Max Drawdown:   {metrics['max_drawdown']:>7.1f}%")
                print(f"  Sharpe Ratio:   {metrics['sharpe']:>7.2f}")
                print(f"  Final Value:    ${metrics['final_value']:>,.0f}")
                print()
                print(f"Benchmark (SPY/S&P500):")
                print(f"  Annual Return:  {spy_return:>7.1f}%")
                print(f"  Max Drawdown:   {spy_dd:>7.1f}%")
                print()
                print(f"Alpha vs Benchmark: {alpha:>+7.1f}%")
                
                if alpha > 0:
                    print(f"  >>> BEATS BENCHMARK\!")
                
                all_results.append({
                    'dataset': label,
                    'start': start_year,
                    'end': end_year,
                    'return': metrics['annual_return'],
                    'total_return': metrics['total_return'],
                    'spy_return': spy_return,
                    'alpha': alpha,
                    'max_dd': metrics['max_drawdown'],
                    'sharpe': metrics['sharpe'],
                    'final_value': metrics['final_value'],
                })
                
        except Exception as e:
            print(f"Error processing {label}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary across all datasets
    print()
    print("="*80)
    print("CROSS-ERA SUMMARY")
    print("="*80)
    print()
    
    print(f"{'Dataset':<25} {'Period':<15} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'Sharpe':>8}")
    print("-"*80)
    
    for r in all_results:
        period = f"{r['start']}-{r['end']}"
        status = "OK" if r['alpha'] > 0 else "MISS"
        print(f"{r['dataset']:<25} {period:<15} {r['return']:>9.1f}% {r['alpha']:>+9.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>7.2f} {status}")
    
    print("-"*80)
    
    if all_results:
        avg_return = sum(r['return'] for r in all_results) / len(all_results)
        avg_alpha = sum(r['alpha'] for r in all_results) / len(all_results)
        avg_dd = sum(r['max_dd'] for r in all_results) / len(all_results)
        avg_sharpe = sum(r['sharpe'] for r in all_results) / len(all_results)
        win_rate = sum(1 for r in all_results if r['alpha'] > 0) / len(all_results) * 100
        
        print(f"{'AVERAGE':<25} {'ALL':<15} {avg_return:>9.1f}% {avg_alpha:>+9.1f}% {avg_dd:>9.1f}% {avg_sharpe:>7.2f}")
        print()
        print(f"Win Rate vs Benchmark: {win_rate:.0f}%")
        print()
        
        # Total growth calculation
        total_growth = 1.0
        for r in all_results:
            years = r['end'] - r['start']
            total_growth *= (1 + r['return']/100) ** years
        
        total_years = sum(r['end'] - r['start'] for r in all_results)
        compound_annual = (total_growth ** (1/total_years) - 1) * 100 if total_years > 0 else 0
        
        print(f"Compound Annual Growth (all eras): {compound_annual:.1f}%")
        print(f"$100K invested in 1965 would be: ${100000 * total_growth:,.0f}")
    
    print()
    print("="*80)


if __name__ == '__main__':
    main()
