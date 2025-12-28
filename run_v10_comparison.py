"""
V10 COMPARISON SCRIPT
=====================

Compares V8 (Equal Weighting) vs V10 (Inverse Volatility Weighting)

Runs both strategies and creates interactive HTML dashboard showing:
- Side-by-side performance metrics
- Equity curve comparison
- Drawdown comparison
- Yearly returns comparison
- Rolling metrics

Output: output/plots/v10_comparison.html
"""

import sys
import os
from datetime import datetime
import logging

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Setup logging
log_dir = os.path.join(project_root, 'output', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'v10_comparison.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports
from src.backtest.portfolio_bot_demo import PortfolioRotationBot
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, TabPanel, Tabs, Div
from bokeh.palettes import Category20


def log_header(title):
    """Log formatted section header"""
    logger.info("=" * 80)
    logger.info(title.center(80))
    logger.info("=" * 80)


def run_strategy(bot, strategy_name, use_inverse_vol=False):
    """
    Run a single strategy backtest

    Args:
        bot: PortfolioRotationBot instance
        strategy_name: Name for logging
        use_inverse_vol: Whether to use inverse volatility weighting

    Returns:
        portfolio_df: Portfolio value over time
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {strategy_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration:")
    logger.info(f"  - VIX Regime Detection: ENABLED")
    logger.info(f"  - Position Weighting: {'Inverse Volatility' if use_inverse_vol else 'Equal Weight'}")
    logger.info(f"  - Top N stocks: 10")
    logger.info(f"  - Rebalancing: Monthly (day 7-10)")
    logger.info(f"  - Trading fee: 0.1%")

    portfolio_df = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_inverse_vol_weighting=use_inverse_vol,
        trading_fee_pct=0.001
    )

    # Rename 'value' column to 'portfolio_value' for consistency
    if 'value' in portfolio_df.columns:
        portfolio_df = portfolio_df.rename(columns={'value': 'portfolio_value'})

    return portfolio_df


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics"""
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100

    # Drawdown
    cummax = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    returns = portfolio_df['portfolio_value'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = {}
    negative_years = 0

    for year in sorted(portfolio_df_copy['year'].unique()):
        year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['portfolio_value'].iloc[-1] /
                          year_data['portfolio_value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return
            if year_return < 0:
                negative_years += 1

    return {
        'final_value': final_value,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'yearly_returns': yearly_returns,
        'negative_years': negative_years,
        'total_years': len(yearly_returns)
    }


def create_metrics_table_html(v8_metrics, v10_metrics):
    """Create HTML metrics comparison table"""

    html = f"""
    <div style='margin: 30px 0; font-family: Arial, sans-serif;'>
        <h2 style='color: #2E86AB; text-align: center; margin-bottom: 20px; font-size: 28px;'>
            📊 V8 vs V10 Performance Comparison
        </h2>
        <p style='text-align: center; color: #666; font-size: 16px; margin-bottom: 30px;'>
            V8: VIX Regime + Equal Weighting &nbsp;&nbsp;|&nbsp;&nbsp; V10: VIX Regime + Inverse Volatility Weighting
        </p>

        <table style='width: 100%; border-collapse: collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <thead>
                <tr style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                    <th style='padding: 15px; text-align: left; font-size: 16px; border: none;'>Metric</th>
                    <th style='padding: 15px; text-align: center; font-size: 16px; border: none;'>V8 Equal Weight</th>
                    <th style='padding: 15px; text-align: center; font-size: 16px; border: none;'>V10 Inverse Vol</th>
                    <th style='padding: 15px; text-align: center; font-size: 16px; border: none;'>Difference</th>
                </tr>
            </thead>
            <tbody>
                <tr style='background-color: #f8f9fa;'>
                    <td style='padding: 12px; border: 1px solid #dee2e6; font-weight: 600;'>Annual Return</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v8_metrics['annual_return']:.1f}%</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v10_metrics['annual_return']:.1f}%</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px; font-weight: bold; color: {"#28a745" if v10_metrics["annual_return"] > v8_metrics["annual_return"] else "#dc3545"};'>
                        {v10_metrics['annual_return'] - v8_metrics['annual_return']:+.1f}%
                    </td>
                </tr>
                <tr style='background-color: white;'>
                    <td style='padding: 12px; border: 1px solid #dee2e6; font-weight: 600;'>Max Drawdown</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v8_metrics['max_drawdown']:.1f}%</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v10_metrics['max_drawdown']:.1f}%</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px; font-weight: bold; color: {"#28a745" if v10_metrics["max_drawdown"] > v8_metrics["max_drawdown"] else "#dc3545"};'>
                        {v10_metrics['max_drawdown'] - v8_metrics['max_drawdown']:+.1f}%
                    </td>
                </tr>
                <tr style='background-color: #f8f9fa;'>
                    <td style='padding: 12px; border: 1px solid #dee2e6; font-weight: 600;'>Sharpe Ratio</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v8_metrics['sharpe']:.2f}</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v10_metrics['sharpe']:.2f}</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px; font-weight: bold; color: {"#28a745" if v10_metrics["sharpe"] > v8_metrics["sharpe"] else "#dc3545"};'>
                        {v10_metrics['sharpe'] - v8_metrics['sharpe']:+.2f}
                    </td>
                </tr>
                <tr style='background-color: white;'>
                    <td style='padding: 12px; border: 1px solid #dee2e6; font-weight: 600;'>Final Value</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>${v8_metrics['final_value']:,.0f}</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>${v10_metrics['final_value']:,.0f}</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px; font-weight: bold; color: {"#28a745" if v10_metrics["final_value"] > v8_metrics["final_value"] else "#dc3545"};'>
                        ${v10_metrics['final_value'] - v8_metrics['final_value']:+,.0f}
                    </td>
                </tr>
                <tr style='background-color: #f8f9fa;'>
                    <td style='padding: 12px; border: 1px solid #dee2e6; font-weight: 600;'>Negative Years</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v8_metrics['negative_years']}/{v8_metrics['total_years']}</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{v10_metrics['negative_years']}/{v10_metrics['total_years']}</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px; font-weight: bold; color: {"#28a745" if v10_metrics["negative_years"] < v8_metrics["negative_years"] else "#dc3545"};'>
                        {v10_metrics['negative_years'] - v8_metrics['negative_years']:+d}
                    </td>
                </tr>
                <tr style='background-color: white;'>
                    <td style='padding: 12px; border: 1px solid #dee2e6; font-weight: 600;'>Win Rate</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{(v8_metrics['total_years']-v8_metrics['negative_years'])/v8_metrics['total_years']*100:.0f}%</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>{(v10_metrics['total_years']-v10_metrics['negative_years'])/v10_metrics['total_years']*100:.0f}%</td>
                    <td style='padding: 12px; border: 1px solid #dee2e6; text-align: center; font-size: 18px;'>-</td>
                </tr>
            </tbody>
        </table>

        <div style='margin-top: 30px; padding: 20px; background-color: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px;'>
            <h3 style='color: #1976d2; margin-top: 0;'>💡 Key Insights:</h3>
            <ul style='color: #424242; line-height: 1.8;'>
                <li><strong>V10 Inverse Volatility Weighting</strong> reduces negative years from {v8_metrics['negative_years']} to {v10_metrics['negative_years']} ({(1 - v10_metrics['negative_years']/v8_metrics['negative_years'])*100:.0f}% improvement)</li>
                <li><strong>Risk-Adjusted Returns:</strong> Sharpe ratio {'improved' if v10_metrics['sharpe'] > v8_metrics['sharpe'] else 'declined'} by {abs(v10_metrics['sharpe'] - v8_metrics['sharpe']):.2f}</li>
                <li><strong>Total Returns:</strong> {'V10 outperformed' if v10_metrics['final_value'] > v8_metrics['final_value'] else 'V8 outperformed'} by ${abs(v10_metrics['final_value'] - v8_metrics['final_value']):,.0f}</li>
            </ul>
        </div>
    </div>
    """

    return Div(text=html, width=1400)


def create_equity_curve_chart(v8_df, v10_df):
    """Create equity curve comparison chart"""
    p = figure(
        title="Portfolio Value Over Time - V8 vs V10",
        x_axis_label='Date',
        y_axis_label='Portfolio Value ($)',
        x_axis_type='datetime',
        width=1400,
        height=450,
        toolbar_location='above',
        background_fill_color="#fafafa"
    )

    # Plot lines
    p.line(v8_df.index, v8_df['portfolio_value'],
           legend_label='V8 Equal Weight', line_width=3, color='#FF6B6B', alpha=0.8)
    p.line(v10_df.index, v10_df['portfolio_value'],
           legend_label='V10 Inverse Vol', line_width=3, color='#4ECDC4', alpha=0.8)

    # Hover tool
    hover = HoverTool(tooltips=[
        ('Date', '@x{%F}'),
        ('Value', '$@y{0,0}')
    ], formatters={'@x': 'datetime'})
    p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "12pt"

    return p


def create_drawdown_chart(v8_df, v10_df):
    """Create drawdown comparison chart"""
    p = figure(
        title="Drawdown Comparison - V8 vs V10",
        x_axis_label='Date',
        y_axis_label='Drawdown (%)',
        x_axis_type='datetime',
        width=1400,
        height=450,
        toolbar_location='above',
        background_fill_color="#fafafa"
    )

    # Calculate drawdowns
    v8_cummax = v8_df['portfolio_value'].cummax()
    v8_dd = (v8_df['portfolio_value'] - v8_cummax) / v8_cummax * 100

    v10_cummax = v10_df['portfolio_value'].cummax()
    v10_dd = (v10_df['portfolio_value'] - v10_cummax) / v10_cummax * 100

    # Plot
    p.line(v8_df.index, v8_dd, legend_label='V8 Equal Weight',
           line_width=3, color='#FF6B6B', alpha=0.8)
    p.line(v10_df.index, v10_dd, legend_label='V10 Inverse Vol',
           line_width=3, color='#4ECDC4', alpha=0.8)

    # Add zero line
    p.line(v8_df.index, [0]*len(v8_df), line_width=1, color='black',
           line_dash='dashed', alpha=0.3)

    # Hover
    hover = HoverTool(tooltips=[
        ('Date', '@x{%F}'),
        ('Drawdown', '@y{0.1f}%')
    ], formatters={'@x': 'datetime'})
    p.add_tools(hover)

    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "12pt"

    return p


def create_yearly_returns_chart(v8_metrics, v10_metrics):
    """Create yearly returns comparison chart"""
    years = sorted(v8_metrics['yearly_returns'].keys())

    p = figure(
        title="Yearly Returns Comparison - V8 vs V10",
        x_range=[str(y) for y in years],
        y_axis_label='Annual Return (%)',
        width=1400,
        height=450,
        toolbar_location='above',
        background_fill_color="#fafafa"
    )

    v8_returns = [v8_metrics['yearly_returns'][y] for y in years]
    v10_returns = [v10_metrics['yearly_returns'][y] for y in years]

    # Create x positions for grouped bars
    x = list(range(len(years)))
    width = 0.35

    # V8 bars
    v8_colors = ['#90EE90' if r >= 0 else '#FFB6C6' for r in v8_returns]
    p.vbar(x=[i - width/2 for i in x], top=v8_returns, width=width,
           color=v8_colors, legend_label='V8 Equal', alpha=0.8)

    # V10 bars
    v10_colors = ['#228B22' if r >= 0 else '#DC143C' for r in v10_returns]
    p.vbar(x=[i + width/2 for i in x], top=v10_returns, width=width,
           color=v10_colors, legend_label='V10 InvVol', alpha=0.8)

    # Zero line
    p.line(x, [0]*len(x), line_width=2, color='black', line_dash='dashed', alpha=0.3)

    # Customize
    p.xaxis.ticker = x
    p.xaxis.major_label_overrides = {i: str(year) for i, year in enumerate(years)}
    p.xaxis.major_label_orientation = 0.8
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "12pt"

    return p


def create_visualization(v8_df, v10_df, v8_metrics, v10_metrics):
    """Create complete visualization dashboard"""
    log_header("CREATING INTERACTIVE VISUALIZATION")

    # Output file
    output_dir = os.path.join(project_root, 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'v10_comparison.html')
    output_file(output_path)

    logger.info("Building visualization components...")

    # Create components
    logger.info("  - Metrics table")
    metrics_table = create_metrics_table_html(v8_metrics, v10_metrics)

    logger.info("  - Equity curve chart")
    equity_chart = create_equity_curve_chart(v8_df, v10_df)

    logger.info("  - Drawdown chart")
    drawdown_chart = create_drawdown_chart(v8_df, v10_df)

    logger.info("  - Yearly returns chart")
    yearly_chart = create_yearly_returns_chart(v8_metrics, v10_metrics)

    # Combine into layout
    logger.info("Combining components into dashboard...")
    layout = column(
        metrics_table,
        equity_chart,
        drawdown_chart,
        yearly_chart
    )

    # Save
    logger.info(f"Saving dashboard to: {output_path}")
    save(layout)

    logger.info(f"✅ Visualization saved successfully!")
    return output_path


def main():
    """Main execution function"""
    log_header("V10 COMPARISON - V8 vs V10 PERFORMANCE ANALYSIS")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Project Root: {project_root}")
    logger.info("")

    try:
        # Initialize bot
        log_header("STEP 1: LOADING DATA")
        data_dir = os.path.join(project_root, 'sp500_data', 'daily')
        initial_capital = 100000

        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Initial capital: ${initial_capital:,.0f}")

        bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=initial_capital)
        bot.prepare_data()
        logger.info(f"✅ Loaded {len(bot.stocks_data)} stocks")

        # Run V8 (Equal Weight)
        log_header("STEP 2: RUNNING V8 (EQUAL WEIGHTING)")
        v8_df = run_strategy(bot, "V8: VIX Regime + Equal Weighting", use_inverse_vol=False)
        v8_metrics = calculate_metrics(v8_df, initial_capital)
        logger.info(f"✅ V8 Complete - Annual Return: {v8_metrics['annual_return']:.1f}%, Sharpe: {v8_metrics['sharpe']:.2f}")

        # Run V10 (Inverse Vol)
        log_header("STEP 3: RUNNING V10 (INVERSE VOLATILITY)")
        v10_df = run_strategy(bot, "V10: VIX Regime + Inverse Vol Weighting", use_inverse_vol=True)
        v10_metrics = calculate_metrics(v10_df, initial_capital)
        logger.info(f"✅ V10 Complete - Annual Return: {v10_metrics['annual_return']:.1f}%, Sharpe: {v10_metrics['sharpe']:.2f}")

        # Create visualization
        log_header("STEP 4: GENERATING VISUALIZATION")
        viz_path = create_visualization(v8_df, v10_df, v8_metrics, v10_metrics)

        # Summary
        log_header("COMPARISON COMPLETE")
        logger.info("")
        logger.info("📊 RESULTS SUMMARY:")
        logger.info(f"  V8 Equal Weight:")
        logger.info(f"    Annual Return: {v8_metrics['annual_return']:.1f}%")
        logger.info(f"    Max Drawdown:  {v8_metrics['max_drawdown']:.1f}%")
        logger.info(f"    Sharpe Ratio:  {v8_metrics['sharpe']:.2f}")
        logger.info(f"    Negative Years: {v8_metrics['negative_years']}/{v8_metrics['total_years']}")
        logger.info("")
        logger.info(f"  V10 Inverse Vol:")
        logger.info(f"    Annual Return: {v10_metrics['annual_return']:.1f}%")
        logger.info(f"    Max Drawdown:  {v10_metrics['max_drawdown']:.1f}%")
        logger.info(f"    Sharpe Ratio:  {v10_metrics['sharpe']:.2f}")
        logger.info(f"    Negative Years: {v10_metrics['negative_years']}/{v10_metrics['total_years']}")
        logger.info("")
        logger.info("📈 IMPROVEMENT:")
        logger.info(f"    Annual Return: {v10_metrics['annual_return'] - v8_metrics['annual_return']:+.1f}%")
        logger.info(f"    Sharpe Ratio:  {v10_metrics['sharpe'] - v8_metrics['sharpe']:+.2f}")
        logger.info(f"    Negative Years: {v10_metrics['negative_years'] - v8_metrics['negative_years']:+d}")
        logger.info("")
        logger.info(f"🎨 Dashboard: {viz_path}")
        logger.info(f"📋 Log file:  output/logs/v10_comparison.log")
        logger.info("")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

        # Verdict
        if v10_metrics['sharpe'] > v8_metrics['sharpe'] and v10_metrics['negative_years'] < v8_metrics['negative_years']:
            logger.info("✅ RECOMMENDATION: V10 shows better risk-adjusted returns and consistency")
            logger.info("   Consider integrating V10 into production (execution.py)")
        elif v10_metrics['annual_return'] > v8_metrics['annual_return']:
            logger.info("⚠️  V10 has higher returns but review risk metrics carefully")
        else:
            logger.info("ℹ️  V8 remains competitive - review dashboard for detailed comparison")

    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
