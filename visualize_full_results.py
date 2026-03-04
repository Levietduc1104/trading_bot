"""
Full Results Visualization: 2015-2024
=======================================
Runs V31 ML, V32 Balanced, SPY benchmark with the current live model
and produces an interactive Bokeh HTML dashboard showing:
  1. Equity curves (2015-2024)
  2. Drawdown chart
  3. Annual returns bar chart
  4. Trade annotations (buy/sell markers)
  5. Summary stats table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000
START = '2015-01-01'
END   = '2024-12-31'
OUTPUT_HTML = 'output/full_results_2015_2024.html'

# ── Run / cache strategies ─────────────────────────────────────────────────────

def run_strategies():
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.stock_ranker import MLStockRanker
    from src.strategies.v31_ml_strategy import V31MLStrategy
    from src.strategies.v32_sector_diversified import V32SectorDiversifiedStrategy

    print("Loading data...")
    bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024',
                               initial_capital=INITIAL_CAPITAL)
    bot.load_all_stocks()

    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    model_path = 'output/models/live_model_latest.txt'
    ranker = MLStockRanker()
    ranker.load_model(model_path)
    print(f"  Model loaded: {model_path}  ({ranker.n_features_to_select} features)")

    results = {}

    # V31 ML
    cache = 'output/vis_v31_2015_2024.csv'
    trades_cache = 'output/vis_v31_trades.csv'
    if os.path.exists(cache):
        print("  V31: loading cache...")
        eq = pd.read_csv(cache, index_col=0, parse_dates=True)
        trades = pd.read_csv(trades_cache) if os.path.exists(trades_cache) else pd.DataFrame()
    else:
        print("  V31: running backtest...")
        strat = V31MLStrategy(bot=bot, fa_loader=fa_loader, ml_model=ranker,
                              use_transaction_costs=True, broker='interactive_brokers',
                              enable_covered_calls=True)
        eq = strat.run_backtest(start_year=2015, end_year=2024)
        trades = pd.DataFrame(getattr(strat, 'trade_log', []))
        eq.to_csv(cache)
        if not trades.empty:
            trades.to_csv(trades_cache, index=False)
    results['V31 ML'] = (eq, trades)

    # V32 Balanced
    cache = 'output/vis_v32_2015_2024.csv'
    trades_cache = 'output/vis_v32_trades.csv'
    if os.path.exists(cache):
        print("  V32: loading cache...")
        eq = pd.read_csv(cache, index_col=0, parse_dates=True)
        trades = pd.read_csv(trades_cache) if os.path.exists(trades_cache) else pd.DataFrame()
    else:
        print("  V32: running backtest...")
        strat = V32SectorDiversifiedStrategy(bot=bot, fa_loader=fa_loader, ml_model=ranker,
                                              use_transaction_costs=True, broker='interactive_brokers',
                                              mode='balanced')
        eq = strat.run_backtest(start_year=2015, end_year=2024)
        trades = pd.DataFrame(getattr(strat, 'trade_log', []))
        eq.to_csv(cache)
        if not trades.empty:
            trades.to_csv(trades_cache, index=False)
    results['V32 Balanced'] = (eq, trades)

    return bot, results


def spy_curve(bot):
    """Build SPY buy-and-hold equity curve."""
    spy = bot.stocks_data.get('SPY')
    if spy is None:
        return None
    spy = spy[(spy.index >= START) & (spy.index <= END)]['close']
    scale = INITIAL_CAPITAL / spy.iloc[0]
    df = pd.DataFrame({'value': spy * scale}, index=spy.index)
    return df


# ── Metrics helpers ────────────────────────────────────────────────────────────

def metrics(df):
    v = df['value']
    years = (v.index[-1] - v.index[0]).days / 365.25
    annual = ((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1) * 100
    dd = ((v - v.cummax()) / v.cummax() * 100).min()
    rets = v.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    calmar = abs(annual / dd) if dd != 0 else 0
    return dict(annual=annual, sharpe=sharpe, max_dd=dd,
                calmar=calmar, final=v.iloc[-1])


def annual_returns(df):
    v = df['value']
    out = {}
    for yr in range(v.index[0].year, v.index[-1].year + 1):
        yr_v = v[v.index.year == yr]
        if len(yr_v) < 2:
            continue
        out[yr] = (yr_v.iloc[-1] / yr_v.iloc[0] - 1) * 100
    return out


# ── Build dashboard ────────────────────────────────────────────────────────────

def build_dashboard(strategy_data, spy_df):
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column
    from bokeh.models import (ColumnDataSource, HoverTool, Span, Div,
                               BoxAnnotation, Legend, LegendItem)
    from bokeh.palettes import Category10

    output_file(OUTPUT_HTML)

    COLORS = {
        'V31 ML':     '#2196F3',   # blue
        'V32 Balanced': '#4CAF50', # green
        'SPY':        '#FF9800',   # orange
    }
    STRESS = [
        ('2018-09-20', '2018-12-24', '2018 Sell-off'),
        ('2020-02-19', '2020-03-23', 'COVID Crash'),
        ('2022-01-03', '2022-10-12', '2022 Rate Shock'),
    ]

    all_years = sorted({yr for _, (eq, _) in strategy_data.items()
                        for yr in annual_returns(eq)})

    # ── 1. Equity curve ───────────────────────────────────────────────────────
    p1 = figure(width=1200, height=420, x_axis_type='datetime',
                title='Equity Curves 2015–2024  (new model: 2010-2022 train, 80 features)',
                toolbar_location='above')

    for name, (eq, _) in strategy_data.items():
        src = ColumnDataSource(dict(
            date=eq.index.to_list(),
            value=eq['value'].to_list(),
            label=[name] * len(eq)
        ))
        p1.line('date', 'value', source=src,
                color=COLORS[name], line_width=2.5, legend_label=name)

    if spy_df is not None:
        src_spy = ColumnDataSource(dict(
            date=spy_df.index.to_list(),
            value=spy_df['value'].to_list(),
            label=['SPY'] * len(spy_df)
        ))
        p1.line('date', 'value', source=src_spy,
                color=COLORS['SPY'], line_width=1.5,
                line_dash='dashed', legend_label='SPY (buy & hold)')

    # Shade stress periods
    for s, e, label in STRESS:
        p1.add_layout(BoxAnnotation(
            left=pd.Timestamp(s).timestamp() * 1000,
            right=pd.Timestamp(e).timestamp() * 1000,
            fill_alpha=0.08, fill_color='red'
        ))

    p1.add_tools(HoverTool(
        tooltips=[('Date', '@date{%F}'), ('Strategy', '@label'),
                  ('Value', '$@value{0,0}')],
        formatters={'@date': 'datetime'}, mode='vline'))
    p1.yaxis.axis_label = 'Portfolio Value ($)'
    p1.yaxis.formatter.use_scientific = False
    p1.legend.location = 'top_left'
    p1.legend.click_policy = 'hide'

    # ── 2. Drawdown ───────────────────────────────────────────────────────────
    p2 = figure(width=1200, height=220, x_axis_type='datetime',
                title='Drawdown (%)', x_range=p1.x_range, toolbar_location=None)

    for name, (eq, _) in strategy_data.items():
        v = eq['value']
        dd = ((v - v.cummax()) / v.cummax() * 100)
        src = ColumnDataSource(dict(date=eq.index.to_list(), dd=dd.to_list()))
        p2.line('date', 'dd', source=src,
                color=COLORS[name], line_width=1.5, legend_label=name)

    if spy_df is not None:
        v = spy_df['value']
        dd = ((v - v.cummax()) / v.cummax() * 100)
        src = ColumnDataSource(dict(date=spy_df.index.to_list(), dd=dd.to_list()))
        p2.line('date', 'dd', source=src,
                color=COLORS['SPY'], line_width=1, line_dash='dashed', legend_label='SPY')

    for s, e, _ in STRESS:
        p2.add_layout(BoxAnnotation(
            left=pd.Timestamp(s).timestamp() * 1000,
            right=pd.Timestamp(e).timestamp() * 1000,
            fill_alpha=0.08, fill_color='red'
        ))

    p2.add_layout(Span(location=0, dimension='width', line_color='black', line_width=1))
    p2.yaxis.axis_label = 'Drawdown (%)'
    p2.legend.location = 'bottom_left'
    p2.legend.click_policy = 'hide'

    # ── 3. Annual returns bar chart ───────────────────────────────────────────
    from bokeh.transform import dodge

    yr_str = [str(y) for y in all_years]
    bar_data = {'years': yr_str}
    strategy_names = list(strategy_data.keys())
    for name, (eq, _) in strategy_data.items():
        ann = annual_returns(eq)
        bar_data[name] = [ann.get(y, 0) for y in all_years]
    if spy_df is not None:
        ann_spy = annual_returns(spy_df)
        bar_data['SPY'] = [ann_spy.get(y, 0) for y in all_years]
        strategy_names.append('SPY')

    src_ann = ColumnDataSource(bar_data)
    n = len(strategy_names)
    width = 0.8 / n
    offsets = [(-0.4 + width / 2) + i * width for i in range(n)]

    p3 = figure(x_range=yr_str, width=1200, height=300,
                title='Annual Returns by Year (%)', toolbar_location=None)
    for name, offset in zip(strategy_names, offsets):
        color = COLORS.get(name, '#999')
        p3.vbar(x=dodge('years', offset, range=p3.x_range),
                top=name, source=src_ann,
                width=width * 0.9, color=color, alpha=0.85, legend_label=name)

    p3.add_tools(HoverTool(tooltips=[('Year', '@years')] +
                            [(n, f'@{{{n}}}{{0.1f}}%') for n in strategy_names]))
    p3.add_layout(Span(location=0, dimension='width', line_color='black', line_width=1))
    p3.yaxis.axis_label = 'Return (%)'
    p3.xgrid.grid_line_color = None
    p3.legend.location = 'top_left'
    p3.legend.click_policy = 'hide'

    # ── 4. Summary table ──────────────────────────────────────────────────────
    rows = ''
    bg = ['white', '#f9f9f9']
    all_strats = list(strategy_data.items())
    if spy_df is not None:
        all_strats.append(('SPY (B&H)', (spy_df, pd.DataFrame())))

    for i, (name, (eq, _)) in enumerate(all_strats):
        m = metrics(eq)
        rows += f"""
        <tr style="background:{bg[i%2]}">
          <td style="padding:8px 14px; font-weight:bold; color:{COLORS.get(name.split()[0]+' '+name.split()[1] if len(name.split())>1 else name, '#333')}">{name}</td>
          <td style="padding:8px 14px; text-align:right">{m['annual']:.2f}%</td>
          <td style="padding:8px 14px; text-align:right">{m['sharpe']:.3f}</td>
          <td style="padding:8px 14px; text-align:right">{m['max_dd']:.2f}%</td>
          <td style="padding:8px 14px; text-align:right">{m['calmar']:.3f}</td>
          <td style="padding:8px 14px; text-align:right">${m['final']:,.0f}</td>
        </tr>"""

    header = Div(text=f"""
    <div style="font-family:sans-serif; padding:15px; background:#f0f4f8;
                border-radius:8px; margin:10px 0; border-left:4px solid #2196F3">
      <h2 style="margin:0 0 12px 0; color:#1a237e">
        Strategy Performance: 2015–2024 &nbsp;
        <span style="font-size:13px; font-weight:normal; color:#666">
          Model: 2010–2022 train | 80 features | 7 new stock-picker features
        </span>
      </h2>
      <table style="border-collapse:collapse; width:100%; font-size:14px">
        <tr style="background:#1a237e; color:white">
          <th style="padding:8px 14px; text-align:left">Strategy</th>
          <th style="padding:8px 14px; text-align:right">Annual Return</th>
          <th style="padding:8px 14px; text-align:right">Sharpe</th>
          <th style="padding:8px 14px; text-align:right">Max DD</th>
          <th style="padding:8px 14px; text-align:right">Calmar</th>
          <th style="padding:8px 14px; text-align:right">Final Value</th>
        </tr>
        {rows}
      </table>
      <p style="font-size:12px; color:#888; margin:8px 0 0 0">
        Red shading = stress periods: 2018 sell-off, COVID crash, 2022 rate shock
      </p>
    </div>
    """)

    layout = column(header, p1, p2, p3)
    save(layout)
    print(f"\nDashboard saved: {OUTPUT_HTML}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("FULL RESULTS VISUALIZATION: 2015-2024")
    print("=" * 70)

    bot, strategy_data = run_strategies()
    spy_df = spy_curve(bot)

    # Print summary table to console
    print(f"\n{'Strategy':<16} {'Annual':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Final $':>12}")
    print("-" * 60)
    for name, (eq, _) in strategy_data.items():
        m = metrics(eq)
        print(f"{name:<16} {m['annual']:>7.2f}% {m['sharpe']:>7.3f} "
              f"{m['max_dd']:>7.2f}% {m['calmar']:>7.3f} ${m['final']:>11,.0f}")
    if spy_df is not None:
        m = metrics(spy_df)
        print(f"{'SPY (B&H)':<16} {m['annual']:>7.2f}% {m['sharpe']:>7.3f} "
              f"{m['max_dd']:>7.2f}% {m['calmar']:>7.3f} ${m['final']:>11,.0f}")

    # Annual breakdown
    print(f"\n{'Year':<6}", end='')
    for name in strategy_data:
        print(f"  {name[:10]:>10}", end='')
    if spy_df is not None:
        print(f"  {'SPY':>6}", end='')
    print()
    print("-" * 60)

    all_years = sorted({yr for _, (eq, _) in strategy_data.items()
                        for yr in annual_returns(eq)})
    spy_ann = annual_returns(spy_df) if spy_df is not None else {}
    for yr in all_years:
        print(f"{yr:<6}", end='')
        for name, (eq, _) in strategy_data.items():
            ann = annual_returns(eq)
            print(f"  {ann.get(yr, 0):>9.1f}%", end='')
        if spy_df is not None:
            print(f"  {spy_ann.get(yr, 0):>5.1f}%", end='')
        print()

    build_dashboard(strategy_data, spy_df)

    import subprocess
    subprocess.run(['open', OUTPUT_HTML], check=False)
    print("Dashboard opened in browser.")


if __name__ == '__main__':
    main()
