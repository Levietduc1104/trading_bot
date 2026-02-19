"""
DUAL DYNAMIC Strategy Visualization
Creates comprehensive interactive Bokeh charts for Money+Risk dual model results
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, Div, TabPanel, Tabs, Legend, ColumnDataSource
from bokeh.palettes import Category20_20

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def visualize_dual_dynamic(start_year=2015, end_year=2024, output_dir='output'):
    """
    Create comprehensive interactive Bokeh visualization for DUAL DYNAMIC strategy

    Args:
        start_year: Backtest start year
        end_year: Backtest end year
        output_dir: Directory to save HTML file
    """

    print("\n" + "="*80)
    print("DUAL DYNAMIC STRATEGY VISUALIZATION (Interactive Bokeh)")
    print("="*80)
    print(f"Period: {start_year}-{end_year}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have database results
    db_path = os.path.join(project_root, 'output', 'data', 'trading_results.db')

    if os.path.exists(db_path):
        import sqlite3
        conn = sqlite3.connect(db_path)

        # Get latest DUAL DYNAMIC run
        query = """
            SELECT * FROM backtest_runs
            WHERE strategy LIKE '%DUAL%DYNAMIC%'
            ORDER BY id DESC LIMIT 1
        """

        run_info = pd.read_sql_query(query, conn)

        if len(run_info) > 0:
            print(f"\n✅ Found DUAL DYNAMIC run in database")
            print(f"   Run ID: {run_info['id'].iloc[0]}")
            print(f"   Date: {run_info['timestamp'].iloc[0]}")
            print(f"   Annual Return: {run_info['annual_return'].iloc[0]:.1f}%")

            # Load portfolio values
            run_id = run_info['id'].iloc[0]
            portfolio_df = pd.read_sql_query(f"""
                SELECT date, value
                FROM portfolio_values
                WHERE run_id = {run_id}
                ORDER BY date
            """, conn, parse_dates=['date'], index_col='date')

            conn.close()

            # Extract metrics
            metrics = {
                'annual_return': run_info['annual_return'].iloc[0],
                'total_return': run_info['total_return'].iloc[0],
                'max_drawdown': run_info['max_drawdown'].iloc[0],
                'sharpe_ratio': run_info['sharpe_ratio'].iloc[0],
                'final_value': run_info['final_value'].iloc[0],
                'spy_annual': run_info['spy_annual_return'].iloc[0],
                'spy_max_dd': run_info['spy_max_drawdown'].iloc[0],
                'alpha': run_info['alpha'].iloc[0]
            }

        else:
            print("\n❌ No DUAL DYNAMIC run found in database")
            print("   Please run: python src/core/execution.py --strategy dual_dynamic")
            return
    else:
        print("\n❌ Database not found")
        print("   Please run: python src/core/execution.py --strategy dual_dynamic")
        return

    # Load SPY for comparison
    try:
        spy_path = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500', 'SPY.csv')
        spy_df = pd.read_csv(spy_path)
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        spy_df = spy_df.set_index('Date')
        spy_df = spy_df[(spy_df.index >= str(start_year)) & (spy_df.index <= str(end_year))]

        # Normalize SPY to match starting value
        initial_value = portfolio_df['value'].iloc[0]
        spy_df['normalized'] = (spy_df['close'] / spy_df['close'].iloc[0]) * initial_value
    except:
        print("⚠️  Could not load SPY data for comparison")
        spy_df = None

    # Calculate drawdown and returns for charting
    running_max = portfolio_df['value'].expanding().max()
    drawdown = ((portfolio_df['value'] - running_max) / running_max * 100)
    daily_returns = portfolio_df['value'].pct_change().dropna() * 100
    cumulative_return = (portfolio_df['value'] / portfolio_df['value'].iloc[0] - 1) * 100

    # Prepare SPY data
    if spy_df is not None:
        spy_max = spy_df['normalized'].expanding().max()
        spy_dd = ((spy_df['normalized'] - spy_max) / spy_max * 100)
        spy_cum_ret = (spy_df['normalized'] / spy_df['normalized'].iloc[0] - 1) * 100

    # Create comprehensive visualization
    print("\n📊 Creating interactive Bokeh visualizations...")

    # Setup output file
    output_path = os.path.join(output_dir, f'dual_dynamic_{start_year}_{end_year}.html')
    output_file(output_path)

    # ========================================
    # CHART 1: Portfolio Value Over Time
    # ========================================
    p1 = figure(
        width=1400, height=400,
        title=f'DUAL DYNAMIC Strategy - Portfolio Growth ({start_year}-{end_year})',
        x_axis_type='datetime',
        tools='pan,wheel_zoom,box_zoom,reset,save'
    )

    # DUAL DYNAMIC line
    p1.line(portfolio_df.index, portfolio_df['value'],
           legend_label='DUAL DYNAMIC', line_width=3, color='#2ecc71', alpha=0.8)

    # SPY line
    if spy_df is not None:
        p1.line(spy_df.index, spy_df['normalized'],
               legend_label='SPY', line_width=2, color='#95a5a6', line_dash='dashed', alpha=0.7)

    p1.add_tools(HoverTool(
        tooltips=[
            ('Date', '@x{%F}'),
            ('Value', '$@y{0,0}')
        ],
        formatters={'@x': 'datetime'},
        mode='vline'
    ))

    p1.legend.location = "top_left"
    p1.legend.click_policy = "hide"
    p1.xaxis.axis_label = "Date"
    p1.yaxis.axis_label = "Portfolio Value ($)"

    # ========================================
    # CHART 2: Drawdown Chart
    # ========================================
    p2 = figure(
        width=1400, height=350,
        title='Drawdown Over Time',
        x_axis_type='datetime',
        tools='pan,wheel_zoom,box_zoom,reset,save'
    )

    # DUAL DYNAMIC drawdown
    p2.varea(x=drawdown.index, y1=0, y2=drawdown,
            color='#e74c3c', alpha=0.5, legend_label='DUAL DYNAMIC DD')

    # SPY drawdown
    if spy_df is not None:
        p2.varea(x=spy_dd.index, y1=0, y2=spy_dd,
                color='#95a5a6', alpha=0.3, legend_label='SPY DD')

    p2.add_tools(HoverTool(
        tooltips=[
            ('Date', '@x{%F}'),
            ('Drawdown', '@y{0.1f}%')
        ],
        formatters={'@x': 'datetime'},
        mode='vline'
    ))

    p2.legend.location = "bottom_left"
    p2.legend.click_policy = "hide"
    p2.xaxis.axis_label = "Date"
    p2.yaxis.axis_label = "Drawdown (%)"

    # ========================================
    # CHART 3: Cumulative Returns
    # ========================================
    p3 = figure(
        width=700, height=350,
        title='Cumulative Returns Comparison',
        x_axis_type='datetime',
        tools='pan,wheel_zoom,box_zoom,reset,save'
    )

    p3.line(cumulative_return.index, cumulative_return,
           legend_label='DUAL DYNAMIC', line_width=3, color='#2ecc71', alpha=0.8)

    if spy_df is not None:
        p3.line(spy_cum_ret.index, spy_cum_ret,
               legend_label='SPY', line_width=2, color='#95a5a6', line_dash='dashed', alpha=0.7)

    p3.add_tools(HoverTool(
        tooltips=[
            ('Date', '@x{%F}'),
            ('Return', '@y{0.1f}%')
        ],
        formatters={'@x': 'datetime'},
        mode='vline'
    ))

    p3.legend.location = "top_left"
    p3.legend.click_policy = "hide"
    p3.xaxis.axis_label = "Date"
    p3.yaxis.axis_label = "Cumulative Return (%)"

    # ========================================
    # CHART 4: Risk-Return Scatter
    # ========================================
    p4 = figure(
        width=700, height=350,
        title='Risk-Return Profile',
        tools='pan,wheel_zoom,box_zoom,reset,save'
    )

    # DUAL DYNAMIC point
    p4.circle([abs(metrics['max_drawdown'])], [metrics['annual_return']],
             size=20, color='#2ecc71', alpha=0.7, legend_label='DUAL DYNAMIC')

    # SPY point
    if spy_df is not None:
        p4.circle([abs(metrics['spy_max_dd'])], [metrics['spy_annual']],
                 size=20, color='#95a5a6', alpha=0.7, legend_label='SPY')

    p4.add_tools(HoverTool(
        tooltips=[
            ('Strategy', '@legend_label'),
            ('Max DD', '@x{0.1f}%'),
            ('Annual Return', '@y{0.1f}%')
        ]
    ))

    p4.legend.location = "top_right"
    p4.xaxis.axis_label = "Max Drawdown (%) - Lower is Better"
    p4.yaxis.axis_label = "Annual Return (%)"

    # ========================================
    # Performance Summary Table
    # ========================================
    final_value = portfolio_df['value'].iloc[-1]
    initial_value = portfolio_df['value'].iloc[0]
    total_return = ((final_value / initial_value) - 1) * 100

    summary_html = f"""
    <div style='width: 1400px; margin: 20px auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h2 style='color: white; text-align: center; margin-bottom: 20px;'>DUAL DYNAMIC PERFORMANCE SUMMARY</h2>

        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
            <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #667eea; margin-top: 0;'>📈 Returns</h3>
                <table style='width: 100%; font-size: 14px;'>
                    <tr><td><strong>Annual Return:</strong></td><td style='text-align: right; color: #2ecc71;'><strong>{metrics['annual_return']:.1f}%</strong></td></tr>
                    <tr><td><strong>Total Return:</strong></td><td style='text-align: right;'>{total_return:.1f}%</td></tr>
                    <tr><td>SPY Annual:</td><td style='text-align: right;'>{metrics['spy_annual']:.1f}%</td></tr>
                    <tr><td><strong>Alpha vs SPY:</strong></td><td style='text-align: right; color: {"#2ecc71" if metrics["alpha"] > 0 else "#e74c3c"};'><strong>{metrics['alpha']:+.1f}%</strong></td></tr>
                </table>
            </div>

            <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #667eea; margin-top: 0;'>🛡️ Risk Metrics</h3>
                <table style='width: 100%; font-size: 14px;'>
                    <tr><td><strong>Max Drawdown:</strong></td><td style='text-align: right; color: #e74c3c;'><strong>{metrics['max_drawdown']:.1f}%</strong></td></tr>
                    <tr><td>SPY Max DD:</td><td style='text-align: right;'>{metrics['spy_max_dd']:.1f}%</td></tr>
                    <tr><td><strong>Sharpe Ratio:</strong></td><td style='text-align: right; color: #2ecc71;'><strong>{metrics['sharpe_ratio']:.2f}</strong></td></tr>
                    <tr><td>Daily Volatility:</td><td style='text-align: right;'>{daily_returns.std():.2f}%</td></tr>
                </table>
            </div>

            <div style='background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px;'>
                <h3 style='color: #667eea; margin-top: 0;'>💰 Portfolio Value</h3>
                <table style='width: 100%; font-size: 14px;'>
                    <tr><td>Initial Capital:</td><td style='text-align: right;'>$100,000</td></tr>
                    <tr><td><strong>Final Value:</strong></td><td style='text-align: right; color: #2ecc71;'><strong>${final_value:,.0f}</strong></td></tr>
                    <tr><td><strong>Total Gain:</strong></td><td style='text-align: right; color: #2ecc71;'><strong>${final_value - initial_value:,.0f}</strong></td></tr>
                    <tr><td>Period:</td><td style='text-align: right;'>{start_year}-{end_year}</td></tr>
                </table>
            </div>
        </div>

        <div style='margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; color: white; text-align: center;'>
            <strong>Strategy:</strong> DUAL DYNAMIC (Money Model + Risk Model with VIX Adaptation)
        </div>
    </div>
    """

    summary_div = Div(text=summary_html, width=1400)

    # ========================================
    # Layout and Save
    # ========================================
    layout = column(
        summary_div,
        p1,
        p2,
        row(p3, p4)
    )

    save(layout)

    # Print summary
    print(f"\n✅ Interactive visualization saved to: {output_path}")
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nCreated interactive HTML file:")
    print(f"  {output_path}")
    print(f"\nTo view: Open in any web browser")
    print("  - Hover over charts for details")
    print("  - Use tools to pan, zoom, and explore")
    print("  - Click legend items to hide/show series")
    print("="*80)

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize DUAL DYNAMIC strategy results with Bokeh')
    parser.add_argument('--start', type=int, default=2015, help='Start year (default: 2015)')
    parser.add_argument('--end', type=int, default=2024, help='End year (default: 2024)')
    parser.add_argument('--output', type=str, default='output', help='Output directory (default: output)')

    args = parser.parse_args()

    visualize_dual_dynamic(start_year=args.start, end_year=args.end, output_dir=args.output)
