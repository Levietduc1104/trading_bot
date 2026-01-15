#\!/usr/bin/env python3
"""V29 Full Execution with Visualization"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

def main():
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v29_mega_cap_split import V29Strategy, calculate_metrics
    
    print("="*80)
    print("V29 MEGA-CAP SPLIT - FULL EXECUTION")
    print("="*80)
    
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    print("Loading data...")
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    print("Data loaded.")
    
    config = {
        'mag7_allocation': 0.70,
        'num_mag7': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
    }
    
    print("Running V29 backtest (1992-2024)...")
    strategy = V29Strategy(bot, config=config)
    portfolio_df = strategy.run_backtest(start_year=1992, end_year=2024)
    
    spy_df = bot.stocks_data['SPY']
    spy_period = spy_df[(spy_df.index >= '1992-01-01') & (spy_df.index <= '2024-12-31')]
    spy_norm = spy_period['close'] / spy_period['close'].iloc[0] * 100000
    
    metrics = calculate_metrics(portfolio_df, 100000)
    spy_years = (spy_norm.index[-1] - spy_norm.index[0]).days / 365.25
    spy_annual = ((spy_norm.iloc[-1] / 100000) ** (1/spy_years) - 1) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('V29 Mega-Cap Split Strategy (1992-2024)', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Growth
    ax1 = axes[0, 0]
    ax1.semilogy(portfolio_df.index, portfolio_df['value'], 'b-', linewidth=2, label='V29 Strategy')
    ax1.semilogy(spy_norm.index, spy_norm.values, 'gray', linewidth=2, alpha=0.7, label='SPY')
    ax1.set_title('Portfolio Growth (Log Scale)')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    v29_dd = (portfolio_df['value'] - portfolio_df['value'].cummax()) / portfolio_df['value'].cummax() * 100
    spy_dd = (spy_norm - spy_norm.cummax()) / spy_norm.cummax() * 100
    ax2.fill_between(v29_dd.index, v29_dd.values, 0, alpha=0.5, color='blue', label=f'V29 (Max: {v29_dd.min():.1f}%)')
    ax2.fill_between(spy_dd.index, spy_dd.values, 0, alpha=0.3, color='gray', label=f'SPY (Max: {spy_dd.min():.1f}%)')
    ax2.set_title('Drawdown Comparison')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-60, 5)
    
    # 3. Annual Returns
    ax3 = axes[1, 0]
    v29_annual_ret = portfolio_df['value'].resample('YE').last().pct_change() * 100
    spy_annual_ret = spy_norm.resample('YE').last().pct_change() * 100
    v29_vals = v29_annual_ret.dropna()
    spy_vals = spy_annual_ret.dropna()
    common_idx = v29_vals.index.intersection(spy_vals.index)
    v29_vals = v29_vals.loc[common_idx]
    spy_vals = spy_vals.loc[common_idx]
    years = common_idx.year
    x = np.arange(len(years))
    width = 0.35
    ax3.bar(x - width/2, v29_vals.values, width, label='V29', color='blue', alpha=0.7)
    ax3.bar(x + width/2, spy_vals.values, width, label='SPY', color='gray', alpha=0.7)
    ax3.set_title('Annual Returns by Year')
    ax3.set_ylabel('Return (%)')
    ax3.set_xticks(x[::3])
    ax3.set_xticklabels(years[::3], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    txt = f"""
    PERFORMANCE SUMMARY (1992-2024)
    ================================
    
    V29 Strategy:
      Final Value:    ${metrics['final_value']:,.0f}
      Annual Return:  {metrics['annual_return']:.1f}%
      Total Return:   {metrics['total_return']:.1f}%
      Max Drawdown:   {metrics['max_drawdown']:.1f}%
      Sharpe Ratio:   {metrics['sharpe']:.2f}
    
    SPY Benchmark:
      Final Value:    ${spy_norm.iloc[-1]:,.0f}
      Annual Return:  {spy_annual:.1f}%
      Max Drawdown:   {spy_dd.min():.1f}%
    
    ALPHA vs SPY:     {metrics['annual_return'] - spy_annual:+.1f}%
    
    Configuration:
      70% Top 3 Magnificent 7
      30% Top 2 Momentum Stocks
      15% Trailing Stop
    """
    ax4.text(0.1, 0.95, txt, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/v29_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: output/plots/v29_performance.png")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"V29 Strategy:  {metrics['annual_return']:.1f}% annual, {metrics['max_drawdown']:.1f}% max DD, Sharpe {metrics['sharpe']:.2f}")
    print(f"SPY Benchmark: {spy_annual:.1f}% annual, {spy_dd.min():.1f}% max DD")
    print(f"Alpha:         {metrics['annual_return'] - spy_annual:+.1f}%")
    print(f"Final Value:   ${metrics['final_value']:,.0f} (from $100,000)")
    print("="*80)

if __name__ == '__main__':
    main()
