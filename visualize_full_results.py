"""
Full Results Visualization: 2006-2024
======================================
Combines:
  - 2006-2014: Phase 1A baseline (pre-ML period, used for training)
  - 2015-2024: Phase 1A vs Phase ML (honest, model never saw this data)

Produces an interactive Bokeh HTML dashboard showing:
  1. Equity curves (full 2006-2024)
  2. Annual returns bar chart (side by side)
  3. Drawdown chart
  4. Summary stats table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000

# ── Step 1: Run Phase 1A for 2006-2014 ────────────────────────────────────────
def run_phase1a_2006_2014():
    cache = 'output/phase1a_2006_2014.csv'
    if os.path.exists(cache):
        logger.info("Loading cached Phase 1A 2006-2014...")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    logger.info("Running Phase 1A backtest 2006-2014...")
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v31_tier2_growth_scoring import V31Tier2GrowthScoringStrategy

    bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024',
                               initial_capital=INITIAL_CAPITAL)
    bot.load_all_stocks()

    strategy = V31Tier2GrowthScoringStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=True,
        momentum_weight=0.50,
        growth_weight=0.50
    )
    results = strategy.run_backtest(start_year=2006, end_year=2014)
    results.to_csv(cache)
    logger.info(f"  Done: {len(results)} rows saved to {cache}")
    return results


# ── Step 2: Chain equity curves ───────────────────────────────────────────────
def chain(early_df, late_df):
    """Scale late_df so it starts where early_df ends."""
    end_val = early_df['value'].iloc[-1]
    start_val = late_df['value'].iloc[0]
    scale = end_val / start_val
    late_scaled = late_df.copy()
    late_scaled['value'] = late_df['value'] * scale
    if 'cash' in late_df.columns:
        late_scaled['cash'] = late_df['cash'] * scale
    if 'stocks_value' in late_df.columns:
        late_scaled['stocks_value'] = late_df['stocks_value'] * scale
    return pd.concat([early_df, late_scaled[late_scaled.index > early_df.index[-1]]])


# ── Step 3: Compute metrics ────────────────────────────────────────────────────
def metrics(df, label):
    v = df['value']
    years = (v.index[-1] - v.index[0]).days / 365.25
    annual = ((v.iloc[-1] / v.iloc[0]) ** (1/years) - 1) * 100
    dd = ((v - v.cummax()) / v.cummax() * 100).min()
    rets = v.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    return dict(label=label, annual=annual, sharpe=sharpe, max_dd=dd,
                final=v.iloc[-1], years=years)


# ── Step 4: Annual returns ─────────────────────────────────────────────────────
def annual_returns(df):
    v = df['value']
    years = {}
    for yr in range(v.index[0].year, v.index[-1].year + 1):
        yr_v = v[v.index.year == yr]
        if len(yr_v) < 2:
            continue
        years[yr] = (yr_v.iloc[-1] / yr_v.iloc[0] - 1) * 100
    return years


# ── Step 5: Build Bokeh dashboard ─────────────────────────────────────────────
def build_dashboard(eq_1a, eq_ml, m1a, mml, ann_1a, ann_ml):
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row
    from bokeh.models import (ColumnDataSource, HoverTool, Span, Label,
                               DataTable, TableColumn, NumberFormatter,
                               Div, BoxAnnotation)
    from bokeh.palettes import Category10
    from bokeh.transform import dodge

    output_file('output/full_results_2006_2024.html')

    BLUE  = '#2196F3'
    GREEN = '#4CAF50'
    RED   = '#F44336'
    GOLD  = '#FF9800'

    # ── Equity curve ─────────────────────────────────────────────────────────
    src_1a = ColumnDataSource(dict(date=eq_1a.index.to_list(),
                                   value=eq_1a['value'].to_list()))
    src_ml = ColumnDataSource(dict(date=eq_ml.index.to_list(),
                                   value=eq_ml['value'].to_list()))

    p1 = figure(width=1100, height=380, x_axis_type='datetime',
                title='Full Equity Curve 2006–2024  |  Train: 2006–2014  |  Honest Test: 2015–2024',
                toolbar_location='above')
    p1.line('date', 'value', source=src_1a, color=BLUE,  line_width=2, legend_label='Phase 1A (Baseline)')
    p1.line('date', 'value', source=src_ml, color=GREEN, line_width=2.5, legend_label='Phase ML')
    p1.add_tools(HoverTool(tooltips=[('Date','@date{%F}'),('Value','$@value{0,0}')],
                            formatters={'@date':'datetime'}, mode='vline'))

    # Shade train vs test region
    cutoff = pd.Timestamp('2015-01-01')
    train_box = BoxAnnotation(right=cutoff, fill_alpha=0.05, fill_color='gray')
    test_box  = BoxAnnotation(left=cutoff,  fill_alpha=0.05, fill_color='green')
    p1.add_layout(train_box)
    p1.add_layout(test_box)
    split_line = Span(location=cutoff.timestamp()*1000, dimension='height',
                      line_color='orange', line_dash='dashed', line_width=2)
    p1.add_layout(split_line)
    p1.add_layout(Label(x=cutoff.timestamp()*1000, y=10, y_units='screen',
                        text=' Train | Test', text_color='orange', text_font_size='11px'))

    p1.yaxis.formatter.use_scientific = False
    p1.yaxis.axis_label = 'Portfolio Value ($)'
    p1.legend.location = 'top_left'
    p1.legend.click_policy = 'hide'

    # ── Drawdown ──────────────────────────────────────────────────────────────
    dd_1a = ((eq_1a['value'] - eq_1a['value'].cummax()) / eq_1a['value'].cummax() * 100)
    dd_ml = ((eq_ml['value'] - eq_ml['value'].cummax()) / eq_ml['value'].cummax() * 100)

    src_dd1a = ColumnDataSource(dict(date=eq_1a.index.to_list(), dd=dd_1a.to_list()))
    src_ddml = ColumnDataSource(dict(date=eq_ml.index.to_list(), dd=dd_ml.to_list()))

    p2 = figure(width=1100, height=220, x_axis_type='datetime',
                title='Drawdown', x_range=p1.x_range, toolbar_location=None)
    p2.line('date', 'dd', source=src_dd1a, color=BLUE,  line_width=1.5, legend_label='Phase 1A')
    p2.line('date', 'dd', source=src_ddml, color=GREEN, line_width=1.5, legend_label='Phase ML')
    p2.add_layout(Span(location=cutoff.timestamp()*1000, dimension='height',
                       line_color='orange', line_dash='dashed', line_width=2))
    p2.yaxis.axis_label = 'Drawdown (%)'
    p2.legend.location = 'bottom_left'

    # ── Annual returns bar chart ──────────────────────────────────────────────
    years = sorted(set(ann_1a.keys()) | set(ann_ml.keys()))
    yr_str = [str(y) for y in years]
    r1a = [ann_1a.get(y, 0) for y in years]
    rml = [ann_ml.get(y, 0) for y in years]

    src_ann = ColumnDataSource(dict(years=yr_str, phase1a=r1a, phaseml=rml))
    p3 = figure(x_range=yr_str, width=1100, height=280,
                title='Annual Returns by Year  (orange dashed = train/test split)',
                toolbar_location=None)

    p3.vbar(x=dodge('years', -0.22, range=p3.x_range), top='phase1a', source=src_ann,
            width=0.4, color=BLUE,  alpha=0.8, legend_label='Phase 1A')
    p3.vbar(x=dodge('years',  0.22, range=p3.x_range), top='phaseml', source=src_ann,
            width=0.4, color=GREEN, alpha=0.8, legend_label='Phase ML')
    p3.add_tools(HoverTool(tooltips=[('Year','@years'),
                                      ('Phase 1A','@phase1a{0.1f}%'),
                                      ('Phase ML','@phaseml{0.1f}%')]))

    # Vertical line at 2015
    split_idx = yr_str.index('2015') if '2015' in yr_str else None
    if split_idx is not None:
        p3.add_layout(Span(location=split_idx - 0.5, dimension='height',
                           line_color='orange', line_dash='dashed', line_width=2))

    p3.add_layout(Span(location=0, dimension='width', line_color='black', line_width=1))
    p3.yaxis.axis_label = 'Annual Return (%)'
    p3.legend.location = 'top_left'
    p3.xgrid.grid_line_color = None

    # ── Summary stats ─────────────────────────────────────────────────────────
    improvement = mml['annual'] - m1a['annual']
    header = Div(text=f"""
    <div style="font-family:sans-serif; padding:15px; background:#f5f5f5; border-radius:8px; margin:10px 0">
      <h2 style="margin:0 0 10px 0">Full Period Results (2006–2024)</h2>
      <table style="border-collapse:collapse; width:100%; font-size:14px">
        <tr style="background:#333; color:white">
          <th style="padding:8px 15px; text-align:left">Metric</th>
          <th style="padding:8px 15px; text-align:right">Phase 1A</th>
          <th style="padding:8px 15px; text-align:right">Phase ML</th>
          <th style="padding:8px 15px; text-align:right">Improvement</th>
        </tr>
        <tr style="background:white">
          <td style="padding:8px 15px">Annual Return</td>
          <td style="padding:8px 15px; text-align:right">{m1a['annual']:.2f}%</td>
          <td style="padding:8px 15px; text-align:right; color:green; font-weight:bold">{mml['annual']:.2f}%</td>
          <td style="padding:8px 15px; text-align:right; color:{'green' if improvement>0 else 'red'}; font-weight:bold">{improvement:+.2f}%</td>
        </tr>
        <tr style="background:#f9f9f9">
          <td style="padding:8px 15px">Sharpe Ratio</td>
          <td style="padding:8px 15px; text-align:right">{m1a['sharpe']:.3f}</td>
          <td style="padding:8px 15px; text-align:right; color:green; font-weight:bold">{mml['sharpe']:.3f}</td>
          <td style="padding:8px 15px; text-align:right; color:{'green' if mml['sharpe']>m1a['sharpe'] else 'red'}">{mml['sharpe']-m1a['sharpe']:+.3f}</td>
        </tr>
        <tr style="background:white">
          <td style="padding:8px 15px">Max Drawdown</td>
          <td style="padding:8px 15px; text-align:right">{m1a['max_dd']:.2f}%</td>
          <td style="padding:8px 15px; text-align:right">{mml['max_dd']:.2f}%</td>
          <td style="padding:8px 15px; text-align:right">{mml['max_dd']-m1a['max_dd']:+.2f}%</td>
        </tr>
        <tr style="background:#f9f9f9">
          <td style="padding:8px 15px">Final Value ($100k start)</td>
          <td style="padding:8px 15px; text-align:right">${m1a['final']:,.0f}</td>
          <td style="padding:8px 15px; text-align:right; color:green; font-weight:bold">${mml['final']:,.0f}</td>
          <td style="padding:8px 15px; text-align:right; color:green">${mml['final']-m1a['final']:+,.0f}</td>
        </tr>
        <tr style="background:white">
          <td style="padding:8px 15px">Period</td>
          <td colspan="3" style="padding:8px 15px; color:#666">
            Train (2006–2014): Phase 1A only &nbsp;|&nbsp;
            <span style="color:orange">▏</span>
            Honest Test (2015–2024): Phase 1A vs Phase ML (model never saw this data)
          </td>
        </tr>
      </table>
    </div>
    """)

    layout = column(header, p1, p2, p3)
    save(layout)
    logger.info("Dashboard saved to output/full_results_2006_2024.html")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("FULL RESULTS: 2006-2024")
    print("="*70)

    # Load honest test results (2015-2024)
    eq_1a_test = pd.read_csv('output/honest_1a_2015_2024.csv', index_col=0, parse_dates=True)
    eq_ml_test  = pd.read_csv('output/honest_ml_2015_2024.csv',  index_col=0, parse_dates=True)

    # Run / load Phase 1A 2006-2014
    eq_1a_train = run_phase1a_2006_2014()

    # Chain: 2006-2014 Phase 1A → 2015-2024 Phase 1A (should be seamless)
    # For ML: use Phase 1A 2006-2014 as the "pre-ML" period, then switch to ML
    eq_1a_full = chain(eq_1a_train, eq_1a_test)
    eq_ml_full  = chain(eq_1a_train, eq_ml_test)   # same start, ML takes over at 2015

    # Metrics over full period
    m1a = metrics(eq_1a_full, 'Phase 1A')
    mml = metrics(eq_ml_full, 'Phase ML')

    print(f"\nFull period 2006-2024:")
    print(f"  Phase 1A : {m1a['annual']:.2f}% annual | Sharpe {m1a['sharpe']:.3f} | DD {m1a['max_dd']:.2f}% | Final ${m1a['final']:,.0f}")
    print(f"  Phase ML : {mml['annual']:.2f}% annual | Sharpe {mml['sharpe']:.3f} | DD {mml['max_dd']:.2f}% | Final ${mml['final']:,.0f}")
    print(f"  ML gain  : {mml['annual']-m1a['annual']:+.2f}%/yr | Sharpe {mml['sharpe']-m1a['sharpe']:+.3f}")

    # Annual returns
    ann_1a = annual_returns(eq_1a_full)
    ann_ml  = annual_returns(eq_ml_full)

    print(f"\nYear-by-year (2006-2024):")
    print(f"  {'Year':<6} {'1A':>8} {'ML':>8} {'Diff':>8}  {'Winner'}")
    print(f"  {'-'*45}")
    ml_wins = 0
    for yr in sorted(set(ann_1a) | set(ann_ml)):
        r1a = ann_1a.get(yr, 0)
        rml = ann_ml.get(yr, 0)
        diff = rml - r1a
        tag = '← ML' if diff > 0.5 else ('← 1A' if diff < -0.5 else '≈tie')
        if diff > 0.5: ml_wins += 1
        period = 'TRAIN' if yr < 2015 else 'test'
        print(f"  {yr}  {r1a:>7.1f}% {rml:>7.1f}% {diff:>+7.1f}%  {tag}  [{period}]")
    print(f"\n  ML wins {ml_wins}/{len(ann_1a)} years in full period")

    # Build visualization
    print("\nBuilding dashboard...")
    build_dashboard(eq_1a_full, eq_ml_full, m1a, mml, ann_1a, ann_ml)

    # Open in browser
    import subprocess
    subprocess.run(['open', 'output/full_results_2006_2024.html'], check=False)
    print("\nDone. Dashboard opened in browser.")


if __name__ == '__main__':
    main()
