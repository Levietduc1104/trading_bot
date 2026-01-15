#!/usr/bin/env python3
"""
V29 Strategy with Era-Appropriate Mega-Cap Leaders

Each era had its own "Magnificent 7" equivalent:
- 1963-1983: IBM, GE, XOM, T, GM, F, MMM (Industrial giants)
- 1983-2003: GE, IBM, WMT, MSFT, INTC, CSCO, PFE (Tech rise)
- 1990-2024: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA (Tech dominance)
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))


# Era-specific mega-cap leaders
ERA_MEGACAPS = {
    '1963-1983': ['IBM', 'GE', 'XOM', 'T', 'GM', 'F', 'MMM', 'JNJ', 'PG', 'KO', 'MRK', 'DD', 'BA', 'CAT'],
    '1983-2003': ['GE', 'IBM', 'WMT', 'MSFT', 'INTC', 'CSCO', 'PFE', 'XOM', 'JNJ', 'PG', 'MRK', 'C', 'AIG', 'T'],
    '1990-2024': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA'],
}


def run_era_backtest(bot, megacaps, config, start_year, end_year):
    """Run V29-style backtest with era-specific mega-caps"""
    import pandas as pd
    import numpy as np
    
    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index
    
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    test_dates = dates[(dates >= start_date) & (dates <= end_date)]
    
    if len(test_dates) < 100:
        return None
    
    # Filter megacaps to only those available
    available_megacaps = [t for t in megacaps if t in bot.stocks_data]
    print(f"  Available mega-caps: {available_megacaps}")
    
    if len(available_megacaps) < 2:
        print("  Not enough mega-caps available")
        return None
    
    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    position_peaks = {}
    
    megacap_alloc = config.get('megacap_allocation', 0.70)
    num_megacap = min(config.get('num_megacap', 3), len(available_megacaps))
    num_momentum = config.get('num_momentum', 2)
    trailing_stop = config.get('trailing_stop', 0.15)
    
    for date in test_dates[100:]:
        # Check trailing stops
        for ticker in list(holdings.keys()):
            df_at_date = bot.get_data_up_to_date(ticker, date)
            if len(df_at_date) > 0:
                current_price = df_at_date['close'].iloc[-1]
                if ticker not in position_peaks:
                    position_peaks[ticker] = current_price
                position_peaks[ticker] = max(position_peaks[ticker], current_price)
                
                drop_pct = (position_peaks[ticker] - current_price) / position_peaks[ticker]
                if drop_pct >= trailing_stop:
                    cash += holdings[ticker] * current_price
                    del holdings[ticker]
                    if ticker in position_peaks:
                        del position_peaks[ticker]
        
        # Monthly rebalancing
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 15
            )
        )
        
        if is_rebalance_day:
            # Liquidate
            for ticker in list(holdings.keys()):
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) > 0:
                    cash += holdings[ticker] * df_at_date['close'].iloc[-1]
            holdings = {}
            position_peaks = {}
            last_rebalance_date = date
            
            # VIX
            if bot.vix_data is not None:
                vix_at_date = bot.vix_data[bot.vix_data.index <= date]
                vix = vix_at_date.iloc[-1]['close'] if len(vix_at_date) > 0 else 20
            else:
                vix = 20
            
            # Score all stocks
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass
            
            # Get mega-cap scores and sort by momentum
            megacap_with_momentum = []
            for ticker in available_megacaps:
                if ticker in current_scores:
                    df_at_date = bot.get_data_up_to_date(ticker, date)
                    if len(df_at_date) >= 20:
                        momentum = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1) * 100
                        megacap_with_momentum.append((ticker, momentum, current_scores[ticker]))
            
            megacap_sorted = sorted(megacap_with_momentum, key=lambda x: (x[1], x[2]), reverse=True)
            top_megacap = [(t, m) for t, m, s in megacap_sorted[:num_megacap]]
            
            # Get other momentum stocks
            other_scores = {t: s for t, s in current_scores.items() if t not in available_megacaps and s > 0}
            other_sorted = sorted(other_scores.items(), key=lambda x: x[1], reverse=True)
            top_momentum = other_sorted[:num_momentum]
            
            if not top_megacap and not top_momentum:
                portfolio_values.append({'date': date, 'value': cash})
                continue
            
            # VIX cash reserve
            if vix < 15:
                cash_reserve = 0.05
            elif vix < 20:
                cash_reserve = 0.10
            elif vix < 25:
                cash_reserve = 0.20
            elif vix < 30:
                cash_reserve = 0.35
            elif vix < 40:
                cash_reserve = 0.50
            else:
                cash_reserve = 0.70
            
            invest_amount = cash * (1 - cash_reserve)
            
            if top_megacap:
                megacap_invest = invest_amount * megacap_alloc
                megacap_per = megacap_invest / len(top_megacap)
            else:
                megacap_per = 0
                
            if top_momentum:
                momentum_invest = invest_amount * (1 - megacap_alloc) if top_megacap else invest_amount
                momentum_per = momentum_invest / len(top_momentum)
            else:
                momentum_per = 0
                if top_megacap:
                    megacap_per = invest_amount / len(top_megacap)
            
            # Buy mega-caps
            for ticker, _ in top_megacap:
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) > 0 and megacap_per > 0:
                    price = df_at_date['close'].iloc[-1]
                    shares = megacap_per / price
                    holdings[ticker] = shares
                    position_peaks[ticker] = price
                    cash -= (megacap_per + megacap_per * 0.001)
            
            # Buy momentum
            for ticker, _ in top_momentum:
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) > 0 and momentum_per > 0:
                    price = df_at_date['close'].iloc[-1]
                    shares = momentum_per / price
                    holdings[ticker] = shares
                    position_peaks[ticker] = price
                    cash -= (momentum_per + momentum_per * 0.001)
        
        # Daily value
        stocks_value = sum(
            holdings[t] * bot.get_data_up_to_date(t, date)['close'].iloc[-1]
            for t in holdings if len(bot.get_data_up_to_date(t, date)) > 0
        )
        portfolio_values.append({'date': date, 'value': cash + stocks_value})
    
    return pd.DataFrame(portfolio_values).set_index('date')


def calculate_metrics(portfolio_df, initial_capital):
    import numpy as np
    if portfolio_df is None or len(portfolio_df) < 2:
        return None
    
    start_value = portfolio_df['value'].iloc[0]
    end_value = portfolio_df['value'].iloc[-1]
    
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    total_return = (end_value / start_value - 1) * 100
    annual_return = ((end_value / start_value) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    returns = portfolio_df['value'].pct_change()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'final_value': end_value
    }


def main():
    import pandas as pd
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    
    print("="*80)
    print("V29 STRATEGY WITH ERA-APPROPRIATE MEGA-CAPS")
    print("="*80)
    print()
    print("Each era uses its own 'Magnificent 7' equivalent:")
    print("  1963-1983: IBM, GE, XOM, T, GM, F, MMM (Industrial giants)")
    print("  1983-2003: GE, IBM, WMT, MSFT, INTC, CSCO, PFE (Tech rise)")
    print("  1990-2024: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA")
    print()
    
    config = {
        'megacap_allocation': 0.70,
        'num_megacap': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
    }
    
    datasets = [
        ('stock_data_1963_1983_top500', 1965, 1983, '1963-1983'),
        ('stock_data_1983_2003_top500', 1985, 2003, '1983-2003'),
        ('stock_data_1990_2024_top500', 1992, 2024, '1990-2024'),
    ]
    
    all_results = []
    
    for data_folder, start_year, end_year, era in datasets:
        data_dir = os.path.join(project_root, 'sp500_data', data_folder)
        
        if not os.path.exists(data_dir):
            continue
        
        print(f"\n{'='*80}")
        print(f"ERA: {era}")
        print(f"{'='*80}")
        
        megacaps = ERA_MEGACAPS.get(era, [])
        print(f"Era mega-caps: {megacaps[:7]}")
        
        try:
            bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
            bot.prepare_data()
            print(f"Loaded {len(bot.stocks_data)} stocks")
            
            portfolio_df = run_era_backtest(bot, megacaps, config, start_year, end_year)
            
            if portfolio_df is not None:
                metrics = calculate_metrics(portfolio_df, 100000)
                
                if metrics:
                    # Benchmark
                    if 'SPY' in bot.stocks_data:
                        spy_df = bot.stocks_data['SPY']
                        spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
                        if len(spy_period) > 1:
                            spy_start = spy_period['close'].iloc[0]
                            spy_end = spy_period['close'].iloc[-1]
                            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
                            spy_return = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100 if spy_years > 0 else 0
                        else:
                            spy_return = 10.0
                    else:
                        # Historical S&P 500 average returns
                        spy_return = 7.0 if era == '1963-1983' else 10.0
                    
                    alpha = metrics['annual_return'] - spy_return
                    
                    print(f"\nResults:")
                    print(f"  Annual Return:  {metrics['annual_return']:>7.1f}%")
                    print(f"  Total Return:   {metrics['total_return']:>7.1f}%")
                    print(f"  Max Drawdown:   {metrics['max_drawdown']:>7.1f}%")
                    print(f"  Sharpe Ratio:   {metrics['sharpe']:>7.2f}")
                    print(f"  Benchmark:      {spy_return:>7.1f}%")
                    print(f"  Alpha:          {alpha:>+7.1f}%")
                    
                    status = "BEATS BENCHMARK!" if alpha > 0 else ""
                    print(f"  {status}")
                    
                    all_results.append({
                        'era': era,
                        'return': metrics['annual_return'],
                        'alpha': alpha,
                        'max_dd': metrics['max_drawdown'],
                        'sharpe': metrics['sharpe'],
                    })
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print()
    print("="*80)
    print("CROSS-ERA SUMMARY")
    print("="*80)
    print()
    
    print(f"{'Era':<15} {'Return':>10} {'Alpha':>10} {'MaxDD':>10} {'Sharpe':>8}")
    print("-"*60)
    
    for r in all_results:
        status = "OK" if r['alpha'] > 0 else "MISS"
        print(f"{r['era']:<15} {r['return']:>9.1f}% {r['alpha']:>+9.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>7.2f} {status}")
    
    if all_results:
        avg_return = sum(r['return'] for r in all_results) / len(all_results)
        avg_alpha = sum(r['alpha'] for r in all_results) / len(all_results)
        win_rate = sum(1 for r in all_results if r['alpha'] > 0) / len(all_results) * 100
        
        print("-"*60)
        print(f"{'AVERAGE':<15} {avg_return:>9.1f}% {avg_alpha:>+9.1f}%")
        print(f"\nWin Rate: {win_rate:.0f}%")
    
    print("="*80)


if __name__ == '__main__':
    main()
