"""
Principal Investment Performance Visualization
Shows PnL, cumulative returns, drawdown, and key metrics
"""

import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from portfolio_bot_demo import PortfolioRotationBot
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, Span, Label, BoxAnnotation
from bokeh.palettes import Category10
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_performance_dashboard():
    """Create comprehensive performance dashboard"""
    
    logger.info("="*80)
    logger.info("CREATING PRINCIPAL INVESTMENT PERFORMANCE DASHBOARD")
    logger.info("="*80)
    
    # Run bot and get results
    logger.info("\nRunning portfolio bot...")
    bot = PortfolioRotationBot(data_dir="../sp500_data/daily")
    bot.prepare_data()
    bot.score_all_stocks()
    portfolio_df = bot.backtest(top_n=10)
    
    # Calculate metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)
    
    # Create output file
    output_file("investment_performance.html")
    
    plots = []
    
    # 1. Portfolio Value Over Time (Main Chart)
    logger.info("Creating portfolio value chart...")
    p1 = create_portfolio_value_chart(portfolio_df, bot.initial_capital, metrics)
    plots.append(p1)
    
    # 2. Returns Analysis (2 charts side by side)
    logger.info("Creating returns analysis...")
    p2a = create_cumulative_returns_chart(portfolio_df, bot.initial_capital)
    p2b = create_daily_returns_histogram(portfolio_df)
    plots.append(row(p2a, p2b))
    
    # 3. Drawdown Analysis
    logger.info("Creating drawdown analysis...")
    p3 = create_drawdown_chart(portfolio_df)
    plots.append(p3)
    
    # 4. Monthly Performance
    logger.info("Creating monthly performance chart...")
    p4 = create_monthly_returns_chart(portfolio_df)
    plots.append(p4)
    
    # 5. Performance Metrics Table
    logger.info("Creating metrics summary...")
    p5 = create_metrics_summary(metrics, portfolio_df, bot.initial_capital)
    plots.append(p5)
    
    # Save all plots
    layout = column(*plots)
    save(layout)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Performance dashboard saved to: visualize/investment_performance.html")
    logger.info("="*80)
    
    # Print summary
    print_performance_summary(metrics, bot.initial_capital, portfolio_df)


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate all performance metrics"""
    
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
    
    # Drawdown
    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
    max_drawdown = drawdown.min()
    
    # Daily returns
    daily_returns = portfolio_df['value'].pct_change().dropna()
    
    # Sharpe ratio
    rf_rate = 0.02  # 2% risk-free rate
    excess_returns = daily_returns - (rf_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    # Volatility
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Monthly returns
    portfolio_df['month'] = portfolio_df.index.to_period('M')
    monthly_returns = portfolio_df.groupby('month')['value'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 0 else 0
    )
    
    winning_months = (monthly_returns > 0).sum()
    losing_months = (monthly_returns < 0).sum()
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'best_day': daily_returns.max() * 100,
        'worst_day': daily_returns.min() * 100,
        'best_month': monthly_returns.max(),
        'worst_month': monthly_returns.min(),
        'winning_months': winning_months,
        'losing_months': losing_months,
        'total_days': len(portfolio_df)
    }


def create_portfolio_value_chart(portfolio_df, initial_capital, metrics):
    """Main chart showing portfolio value over time"""
    
    p = figure(x_axis_type="datetime", width=1400, height=500,
               title="Portfolio Value Over Time - Principal Investment Performance")
    
    # Plot portfolio value
    p.line(portfolio_df.index, portfolio_df['value'], 
           line_width=3, color='navy', alpha=0.8, legend_label='Portfolio Value')
    
    # Add initial capital reference line
    p.line(portfolio_df.index, [initial_capital] * len(portfolio_df),
           line_width=2, color='green', alpha=0.5, line_dash='dashed',
           legend_label=f'Initial Capital (${initial_capital:,.0f})')
    
    # Highlight key events
    # COVID crash (around day 530 = ~Mar 2020)
    covid_date = portfolio_df.index[min(530, len(portfolio_df)-1)]
    covid_box = BoxAnnotation(left=covid_date, right=portfolio_df.index[min(550, len(portfolio_df)-1)],
                               fill_alpha=0.1, fill_color='red')
    p.add_layout(covid_box)
    
    # 2022 bear market (around day 1010)
    if len(portfolio_df) > 1010:
        bear_date = portfolio_df.index[1010]
        bear_box = BoxAnnotation(left=bear_date, right=portfolio_df.index[min(1130, len(portfolio_df)-1)],
                                 fill_alpha=0.1, fill_color='orange')
        p.add_layout(bear_box)
    
    # Format
    p.yaxis.axis_label = "Portfolio Value ($)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "top_left"
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Date", "@x{%F}"),
        ("Value", "$@y{0,0}"),
    ], formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    # Add performance annotation
    final_value = metrics['final_value']
    annual_return = metrics['annual_return']
    
    label = Label(x=10, y=450, x_units='screen', y_units='screen',
                  text=f'Final: ${final_value:,.0f} | Return: {annual_return:.1f}% annually',
                  text_font_size='14pt', text_color='navy')
    p.add_layout(label)
    
    return p


def create_cumulative_returns_chart(portfolio_df, initial_capital):
    """Chart showing cumulative returns percentage"""
    
    cumulative_returns = (portfolio_df['value'] / initial_capital - 1) * 100
    
    p = figure(x_axis_type="datetime", width=700, height=400,
               title="Cumulative Returns (%)")
    
    p.line(portfolio_df.index, cumulative_returns,
           line_width=2, color='darkgreen', alpha=0.8)
    
    # Zero line
    p.line(portfolio_df.index, [0] * len(portfolio_df),
           line_width=1, color='gray', alpha=0.5, line_dash='dashed')
    
    p.yaxis.axis_label = "Cumulative Return (%)"
    p.xaxis.axis_label = "Date"
    
    # Add hover
    hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Return", "@y{0.1f}%")],
                      formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    return p


def create_daily_returns_histogram(portfolio_df):
    """Histogram of daily returns"""
    
    daily_returns = portfolio_df['value'].pct_change().dropna() * 100
    
    hist, edges = np.histogram(daily_returns, bins=50)
    
    p = figure(width=700, height=400, title="Daily Returns Distribution")
    
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color='steelblue', line_color='white', alpha=0.7)
    
    # Add mean line
    mean_return = daily_returns.mean()
    p.line([mean_return, mean_return], [0, hist.max()],
           line_width=2, color='red', line_dash='dashed',
           legend_label=f'Mean: {mean_return:.2f}%')
    
    p.xaxis.axis_label = "Daily Return (%)"
    p.yaxis.axis_label = "Frequency"
    p.legend.location = "top_right"
    
    return p


def create_drawdown_chart(portfolio_df):
    """Chart showing drawdown over time"""
    
    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
    
    p = figure(x_axis_type="datetime", width=1400, height=400,
               title="Drawdown Analysis - How Much Below Peak")
    
    p.line(portfolio_df.index, drawdown,
           line_width=2, color='red', alpha=0.8)
    
    # Fill below zero
    p.varea(x=portfolio_df.index, y1=0, y2=drawdown,
            fill_color='red', fill_alpha=0.2)
    
    # Zero line
    p.line(portfolio_df.index, [0] * len(portfolio_df),
           line_width=1, color='gray', alpha=0.5, line_dash='dashed')
    
    # Mark max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    
    p.circle([max_dd_idx], [max_dd_val], size=10, color='darkred',
             legend_label=f'Max Drawdown: {max_dd_val:.1f}%')
    
    p.yaxis.axis_label = "Drawdown (%)"
    p.xaxis.axis_label = "Date"
    p.legend.location = "bottom_left"
    
    # Add hover
    hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Drawdown", "@y{0.1f}%")],
                      formatters={'@x': 'datetime'})
    p.add_tools(hover)
    
    return p


def create_monthly_returns_chart(portfolio_df):
    """Bar chart of monthly returns"""
    
    portfolio_df['month'] = portfolio_df.index.to_period('M')
    monthly_returns = portfolio_df.groupby('month')['value'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 0 else 0
    )
    
    months = [str(m) for m in monthly_returns.index]
    returns = monthly_returns.values
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    p = figure(x_range=months, width=1400, height=400,
               title="Monthly Returns")
    
    p.vbar(x=months, top=returns, width=0.8, color=colors, alpha=0.7)
    
    # Zero line
    p.line(months, [0] * len(months), line_width=1, color='gray', line_dash='dashed')
    
    p.xaxis.axis_label = "Month"
    p.yaxis.axis_label = "Return (%)"
    p.xaxis.major_label_orientation = 0.785  # 45 degrees
    
    # Only show every 6th month label
    p.xaxis.major_label_overrides = {months[i]: months[i] if i % 6 == 0 else "" 
                                     for i in range(len(months))}
    
    return p


def create_metrics_summary(metrics, portfolio_df, initial_capital):
    """Create text summary of key metrics"""
    
    from bokeh.models import Div
    
    html = f"""
    <div style="font-family: Arial; padding: 20px; background-color: #f5f5f5; border-radius: 10px;">
        <h2 style="color: #2c3e50;">📊 Performance Metrics Summary</h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px;">
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #3498db; margin-top: 0;">Capital</h3>
                <p><strong>Initial:</strong> ${metrics['initial_capital']:,.0f}</p>
                <p><strong>Final:</strong> ${metrics['final_value']:,.0f}</p>
                <p><strong>Change:</strong> ${metrics['final_value'] - metrics['initial_capital']:,.0f}</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #2ecc71; margin-top: 0;">Returns</h3>
                <p><strong>Total:</strong> {metrics['total_return']:.1f}%</p>
                <p><strong>Annual:</strong> {metrics['annual_return']:.1f}%</p>
                <p><strong>Best Month:</strong> {metrics['best_month']:.1f}%</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #e74c3c; margin-top: 0;">Risk</h3>
                <p><strong>Max Drawdown:</strong> {metrics['max_drawdown']:.1f}%</p>
                <p><strong>Volatility:</strong> {metrics['volatility']:.1f}%</p>
                <p><strong>Sharpe Ratio:</strong> {metrics['sharpe_ratio']:.2f}</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #9b59b6; margin-top: 0;">Best/Worst Days</h3>
                <p><strong>Best Day:</strong> {metrics['best_day']:.2f}%</p>
                <p><strong>Worst Day:</strong> {metrics['worst_day']:.2f}%</p>
                <p><strong>Avg Daily:</strong> {(metrics['total_return']/metrics['total_days']):.3f}%</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #f39c12; margin-top: 0;">Monthly Stats</h3>
                <p><strong>Winning Months:</strong> {metrics['winning_months']}</p>
                <p><strong>Losing Months:</strong> {metrics['losing_months']}</p>
                <p><strong>Win Rate:</strong> {metrics['winning_months']/(metrics['winning_months']+metrics['losing_months'])*100:.1f}%</p>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #34495e; margin-top: 0;">Timeline</h3>
                <p><strong>Total Days:</strong> {metrics['total_days']}</p>
                <p><strong>Start:</strong> {portfolio_df.index[0].strftime('%Y-%m-%d')}</p>
                <p><strong>End:</strong> {portfolio_df.index[-1].strftime('%Y-%m-%d')}</p>
            </div>
            
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: {'#d4edda' if metrics['annual_return'] >= 20 else '#f8d7da'}; 
                    border-radius: 5px; border-left: 4px solid {'#28a745' if metrics['annual_return'] >= 20 else '#dc3545'};">
            <h3 style="margin-top: 0;">{'🎯 GOAL ACHIEVED\!' if metrics['annual_return'] >= 20 else '⚠️ BELOW TARGET'}</h3>
            <p>Target: 20% annual return | Actual: {metrics['annual_return']:.1f}% annual return</p>
        </div>
    </div>
    """
    
    div = Div(text=html, width=1400, height=500)
    return div


def print_performance_summary(metrics, initial_capital, portfolio_df):
    """Print summary to console"""
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nInitial Capital:  ${initial_capital:,.0f}")
    print(f"Final Value:      ${metrics['final_value']:,.0f}")
    print(f"Total Return:     {metrics['total_return']:.1f}%")
    print(f"Annual Return:    {metrics['annual_return']:.1f}%")
    print(f"Max Drawdown:     {metrics['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"Volatility:       {metrics['volatility']:.1f}%")
    print(f"\nWinning Months:   {metrics['winning_months']}")
    print(f"Losing Months:    {metrics['losing_months']}")
    print(f"Win Rate:         {metrics['winning_months']/(metrics['winning_months']+metrics['losing_months'])*100:.1f}%")
    print("\n" + "="*80)


if __name__ == "__main__":
    create_performance_dashboard()
