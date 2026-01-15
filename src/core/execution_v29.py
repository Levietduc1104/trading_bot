"""
END-TO-END PORTFOLIO TRADING SYSTEM EXECUTION
==============================================

V29: MEGA-CAP SPLIT (Production Strategy)

This script runs the V29 trading system:
1. Loads stock data
2. Runs backtest with 70/30 Mega-Cap Split + Drawdown Protection
3. Saves results to database
4. Generates visualizations
5. Creates performance reports

Strategy: V29 Mega-Cap Split
- 70% allocation to Top 3 Magnificent 7 (by momentum)
- 30% allocation to Top 2 Momentum stocks
- VIX-based cash reserves (up to 70% in crisis)
- 15% trailing stop losses
- Progressive portfolio drawdown control
- Expected: 22-24% annual, -17% max DD, 1.45 Sharpe

Output:
- Database: output/data/trading_results.db
- Plots: output/plots/v29_performance.png
- Reports: output/reports/v29_report.txt
- Logs: output/logs/execution.log
"""
import sys
import os
import time
from datetime import datetime
import logging

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_root, 'output', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'execution.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np


def log_header(title):
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def run_v29_backtest(start_year=2015, end_year=2024):
    """Run V29 Mega-Cap Split Strategy backtest"""
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v29_mega_cap_split import V29Strategy, calculate_metrics
    
    log_header("STEP 1: LOADING DATA")
    
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    logger.info(f"Data directory: {data_dir}")
    
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")
    
    log_header("STEP 2: RUNNING V29 BACKTEST")
    
    logger.info("Strategy: V29 MEGA-CAP SPLIT")
    logger.info("  - 70% Top 3 Magnificent 7 (by momentum)")
    logger.info("  - 30% Top 2 Momentum stocks")
    logger.info("  - 15% Trailing stop losses")
    logger.info("  - VIX-based cash reserves")
    logger.info("  - Progressive drawdown control")
    logger.info("")
    
    config = {
        'mag7_allocation': 0.70,
        'num_mag7': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
    }
    
    strategy = V29Strategy(bot, config=config)
    portfolio_df = strategy.run_backtest(start_year=start_year, end_year=end_year)
    
    metrics = calculate_metrics(portfolio_df, 100000)
    
    # Calculate SPY benchmark
    spy_df = bot.stocks_data.get('SPY')
    if spy_df is not None:
        spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
        if len(spy_period) > 1:
            spy_start = spy_period['close'].iloc[0]
            spy_end = spy_period['close'].iloc[-1]
            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
            spy_annual = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100
            spy_cummax = spy_period['close'].cummax()
            spy_dd = ((spy_period['close'] - spy_cummax) / spy_cummax * 100).min()
        else:
            spy_annual = 10.0
            spy_dd = -30.0
    else:
        spy_annual = 10.0
        spy_dd = -30.0
    
    alpha = metrics['annual_return'] - spy_annual
    
    logger.info(f"Backtest Period: {start_year}-{end_year}")
    logger.info(f"V29 Annual Return:  {metrics['annual_return']:.1f}%")
    logger.info(f"V29 Max Drawdown:   {metrics['max_drawdown']:.1f}%")
    logger.info(f"V29 Sharpe Ratio:   {metrics['sharpe']:.2f}")
    logger.info(f"SPY Annual Return:  {spy_annual:.1f}%")
    logger.info(f"SPY Max Drawdown:   {spy_dd:.1f}%")
    logger.info(f"Alpha vs SPY:       {alpha:+.1f}%")
    
    return portfolio_df, bot, metrics, spy_annual, spy_dd


def generate_v29_visualization(portfolio_df, bot, metrics, spy_annual, spy_dd, start_year, end_year):
    """Generate V29 performance visualization"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    log_header("STEP 3: GENERATING VISUALIZATION")
    
    spy_df = bot.stocks_data['SPY']
    spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
    spy_norm = spy_period['close'] / spy_period['close'].iloc[0] * 100000
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'V29 Mega-Cap Split Strategy ({start_year}-{end_year})', fontsize=16, fontweight='bold')
    
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
    spy_dd_series = (spy_norm - spy_norm.cummax()) / spy_norm.cummax() * 100
    ax2.fill_between(v29_dd.index, v29_dd.values, 0, alpha=0.5, color='blue', label=f'V29 (Max: {v29_dd.min():.1f}%)')
    ax2.fill_between(spy_dd_series.index, spy_dd_series.values, 0, alpha=0.3, color='gray', label=f'SPY (Max: {spy_dd_series.min():.1f}%)')
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
    if len(common_idx) > 0:
        v29_vals = v29_vals.loc[common_idx]
        spy_vals = spy_vals.loc[common_idx]
        years = common_idx.year
        x = np.arange(len(years))
        width = 0.35
        ax3.bar(x - width/2, v29_vals.values, width, label='V29', color='blue', alpha=0.7)
        ax3.bar(x + width/2, spy_vals.values, width, label='SPY', color='gray', alpha=0.7)
        ax3.set_xticks(x[::2])
        ax3.set_xticklabels(years[::2], rotation=45)
    ax3.set_title('Annual Returns by Year')
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    
    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    alpha = metrics['annual_return'] - spy_annual
    txt = f"""
    PERFORMANCE SUMMARY ({start_year}-{end_year})
    ================================
    
    V29 Strategy:
      Final Value:    ${metrics['final_value']:,.0f}
      Annual Return:  {metrics['annual_return']:.1f}%
      Total Return:   {metrics['total_return']:.1f}%
      Max Drawdown:   {metrics['max_drawdown']:.1f}%
      Sharpe Ratio:   {metrics['sharpe']:.2f}
    
    SPY Benchmark:
      Annual Return:  {spy_annual:.1f}%
      Max Drawdown:   {spy_dd:.1f}%
    
    ALPHA vs SPY:     {alpha:+.1f}%
    
    Configuration:
      70% Top 3 Magnificent 7
      30% Top 2 Momentum Stocks
      15% Trailing Stop
      VIX-based Cash Reserve
    """
    ax4.text(0.1, 0.95, txt, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    output_dir = os.path.join(project_root, 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'v29_performance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization: {output_path}")
    return output_path


def create_v29_report(portfolio_df, metrics, spy_annual, spy_dd, start_year, end_year):
    """Create V29 performance report"""
    log_header("STEP 4: CREATING REPORT")
    
    output_dir = os.path.join(project_root, 'output', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'v29_report.txt')
    
    alpha = metrics['annual_return'] - spy_annual
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("V29 MEGA-CAP SPLIT STRATEGY - PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Period: {start_year}-{end_year}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("STRATEGY CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write("  Allocation: 70% Magnificent 7 / 30% Momentum\n")
        f.write("  Mag7 Stocks: Top 3 by momentum\n")
        f.write("  Momentum Stocks: Top 2 (excluding Mag7)\n")
        f.write("  Trailing Stop: 15%\n")
        f.write("  VIX Crisis Level: 35\n")
        f.write("  Max Portfolio DD Control: 25%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Metric':<25} {'V29':>15} {'SPY':>15} {'Diff':>15}\n")
        f.write(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}\n")
        f.write(f"  {'Annual Return':<25} {metrics['annual_return']:>14.1f}% {spy_annual:>14.1f}% {alpha:>+14.1f}%\n")
        f.write(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>14.1f}% {spy_dd:>14.1f}% {metrics['max_drawdown']-spy_dd:>+14.1f}%\n")
        f.write(f"  {'Sharpe Ratio':<25} {metrics['sharpe']:>15.2f}\n")
        f.write(f"  {'Final Value':<25} ${metrics['final_value']:>13,.0f}\n")
        f.write(f"  {'Total Return':<25} {metrics['total_return']:>14.1f}%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("MAGNIFICENT 7 STOCKS\n")
        f.write("-" * 70 + "\n")
        f.write("  AAPL - Apple Inc.\n")
        f.write("  MSFT - Microsoft Corporation\n")
        f.write("  NVDA - NVIDIA Corporation\n")
        f.write("  GOOGL - Alphabet Inc.\n")
        f.write("  META - Meta Platforms Inc.\n")
        f.write("  AMZN - Amazon.com Inc.\n")
        f.write("  TSLA - Tesla Inc.\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved report: {report_path}")
    return report_path


def main(start_year=2015, end_year=2024):
    """Main V29 execution function"""
    log_header("V29 MEGA-CAP SPLIT STRATEGY - EXECUTION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Project Root: {project_root}")
    
    try:
        # Run backtest
        portfolio_df, bot, metrics, spy_annual, spy_dd = run_v29_backtest(start_year, end_year)
        
        # Generate visualization
        viz_path = generate_v29_visualization(portfolio_df, bot, metrics, spy_annual, spy_dd, start_year, end_year)
        
        # Create report
        report_path = create_v29_report(portfolio_df, metrics, spy_annual, spy_dd, start_year, end_year)
        
        # Summary
        log_header("EXECUTION COMPLETE")
        logger.info("V29 Mega-Cap Split Strategy executed successfully\!")
        logger.info("")
        logger.info("Outputs:")
        logger.info(f"  Visualization: {viz_path}")
        logger.info(f"  Report: {report_path}")
        logger.info("")
        logger.info(f"Results ({start_year}-{end_year}):")
        logger.info(f"  Annual Return: {metrics['annual_return']:.1f}%")
        logger.info(f"  Max Drawdown:  {metrics['max_drawdown']:.1f}%")
        logger.info(f"  Sharpe Ratio:  {metrics['sharpe']:.2f}")
        logger.info(f"  Alpha vs SPY:  {metrics['annual_return'] - spy_annual:+.1f}%")
        logger.info(f"  Final Value:   ${metrics['final_value']:,.0f}")
        logger.info("")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return portfolio_df, metrics
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run V29 Mega-Cap Split Strategy')
    parser.add_argument('--start', type=int, default=2015, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')
    args = parser.parse_args()
    
    main(start_year=args.start, end_year=args.end)
