"""
Portfolio Performance Visualization

Creates charts to visualize the portfolio rotation bot results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_results():
    """Run the bot and get results"""
    from portfolio_bot_demo import PortfolioRotationBot
    
    logger.info("Running portfolio bot to get results...")
    bot = PortfolioRotationBot()
    bot.prepare_data()
    bot.score_all_stocks()
    results = bot.backtest(top_n=10)
    
    return results, bot


def create_visualizations(portfolio_df, bot):
    """Create comprehensive performance visualizations"""
    
    logger.info("Creating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('S&P 500 Portfolio Rotation Bot - Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Calculate metrics
    initial_capital = bot.initial_capital
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
    
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # ========================
    # 1. PORTFOLIO VALUE (Large chart - top)
    # ========================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(portfolio_df.index, portfolio_df['value'], linewidth=2.5, 
             color='#2E86AB', label='Portfolio Value')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', 
                alpha=0.5, linewidth=1.5, label='Initial Capital')
    ax1.fill_between(portfolio_df.index, initial_capital, portfolio_df['value'], 
                     where=(portfolio_df['value'] >= initial_capital), 
                     alpha=0.3, color='green')
    ax1.fill_between(portfolio_df.index, initial_capital, portfolio_df['value'], 
                     where=(portfolio_df['value'] < initial_capital), 
                     alpha=0.3, color='red')
    
    # Add key metrics as text
    textstr = f'Final Value: ${final_value:,.0f}\nAnnual Return: {annual_return:.1f}%\nSharpe Ratio: {sharpe:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    ax1.set_title('Portfolio Value Over Time (2018-2024)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # ========================
    # 2. DRAWDOWN
    # ========================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(portfolio_df.index, 0, drawdown, color='#E63946', alpha=0.6)
    ax2.plot(portfolio_df.index, drawdown, linewidth=1.5, color='#A4161A')
    ax2.axhline(y=-20, color='orange', linestyle='--', alpha=0.7, 
                linewidth=1.5, label='20% Level')
    ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Add max drawdown text
    ax2.text(0.98, 0.05, f'Max Drawdown: {max_drawdown:.1f}%', 
             transform=ax2.transAxes, fontsize=10,
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # ========================
    # 3. MONTHLY RETURNS
    # ========================
    ax3 = fig.add_subplot(gs[1, 1])
    monthly_returns = portfolio_df['value'].resample('M').last().pct_change() * 100
    monthly_returns = monthly_returns.dropna()
    colors = ['#06D6A0' if x > 0 else '#EF476F' for x in monthly_returns]
    ax3.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Monthly Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Month Number', fontsize=10)
    ax3.set_ylabel('Monthly Return (%)', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add stats
    win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100
    avg_win = monthly_returns[monthly_returns > 0].mean()
    avg_loss = monthly_returns[monthly_returns < 0].mean()
    
    stats_text = f'Win Rate: {win_rate:.0f}%\nAvg Win: {avg_win:.1f}%\nAvg Loss: {avg_loss:.1f}%'
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ========================
    # 4. RETURNS HISTOGRAM
    # ========================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(daily_returns * 100, bins=50, color='#118AB2', alpha=0.7, edgecolor='black')
    ax4.axvline(x=daily_returns.mean() * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {daily_returns.mean()*100:.2f}%')
    ax4.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Daily Return (%)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9)
    
    # ========================
    # 5. TOP 10 STOCKS RANKING
    # ========================
    ax5 = fig.add_subplot(gs[2, :2])
    top_10 = bot.rankings[:10]
    tickers = [t for t, s in top_10]
    scores = [s for t, s in top_10]
    
    # Color gradient based on score
    colors_bars = []
    for s in scores:
        if s >= 80:
            colors_bars.append('#06D6A0')  # Excellent - green
        elif s >= 70:
            colors_bars.append('#118AB2')  # Strong - blue
        elif s >= 60:
            colors_bars.append('#FFD166')  # Good - yellow
        else:
            colors_bars.append('#EF476F')  # Neutral/Avoid - red
    
    y_pos = np.arange(len(tickers))
    bars = ax5.barh(y_pos, scores, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(tickers, fontsize=10)
    ax5.set_xlabel('Score (0-100)', fontsize=10)
    ax5.set_xlim(0, 100)
    ax5.set_title('Top 10 Stock Rankings (Current)', fontsize=12, fontweight='bold')
    ax5.axvline(x=80, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax5.axvline(x=70, color='blue', linestyle='--', alpha=0.4, linewidth=1)
    ax5.axvline(x=60, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax5.text(score + 1, i, f'{score:.0f}', va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#06D6A0', label='Excellent (80+)'),
        Patch(facecolor='#118AB2', label='Strong (70-79)'),
        Patch(facecolor='#FFD166', label='Good (60-69)'),
        Patch(facecolor='#EF476F', label='Neutral (<60)')
    ]
    ax5.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # ========================
    # 6. KEY METRICS TABLE
    # ========================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Initial Capital', f'${initial_capital:,.0f}'],
        ['Final Value', f'${final_value:,.0f}'],
        ['Total Return', f'{total_return:.1f}%'],
        ['Annual Return', f'{annual_return:.1f}%'],
        ['Max Drawdown', f'{max_drawdown:.1f}%'],
        ['Sharpe Ratio', f'{sharpe:.2f}'],
        ['Trading Days', f'{len(portfolio_df)}'],
        ['Years', f'{years:.1f}'],
        ['Best Day', f'{(daily_returns.max()*100):.2f}%'],
        ['Worst Day', f'{(daily_returns.min()*100):.2f}%'],
    ]
    
    table = ax6.table(cellText=metrics_data, cellLoc='left',
                     colWidths=[0.6, 0.4],
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#118AB2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
    
    ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # Save figure
    output_file = 'portfolio_performance.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    logger.info(f"✓ Saved visualization to {output_file}")
    
    # Show plot
    plt.show()
    
    return fig


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("PORTFOLIO PERFORMANCE VISUALIZATION")
    logger.info("="*60)
    
    # Get results
    portfolio_df, bot = load_results()
    
    # Create visualizations
    fig = create_visualizations(portfolio_df, bot)
    
    logger.info("\n✓ Visualization complete\!")
