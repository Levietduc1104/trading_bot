"""
Trailing Stop Research: Find the optimal trailing stop for V31 ML
==================================================================
Sweeps trailing stop values from 5% to 30% on the honest 2015-2024 test period.

For each stop value, runs the full V31 ML backtest and records:
  - Annual return
  - Max drawdown
  - Sharpe ratio
  - Number of stops triggered
  - Return in 2022 (rate-shock stress year)
  - Return in 2020 (COVID crash + recovery)

The sweep reuses the saved model (no retraining) and shares the loaded
price/FA data across all runs to avoid redundant I/O.

Usage:
  python research_trailing_stop.py                    # full sweep 5%-30%
  python research_trailing_stop.py --stops 10 15 20  # test specific values only
  python research_trailing_stop.py --strategy v32     # test on V32 instead of V31
"""

import sys
import os
import argparse
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000
MODEL_PATH      = 'output/models/honest_2006_2014_model.txt'
METADATA_DIR    = 'sp500_data/metadata'

DEFAULT_STOPS   = [5, 8, 10, 12, 15, 18, 20, 25, 30]


# ─────────────────────────────────────────────────────────────────────────────

def _header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def calc_metrics(results: pd.DataFrame, trail_pct: float, n_stops: int) -> dict:
    final  = results['value'].iloc[-1]
    years  = (results.index[-1] - results.index[0]).days / 365.25
    annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    cummax = results['value'].cummax()
    max_dd = ((results['value'] - cummax) / cummax * 100).min()
    rets   = results['value'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0
    calmar = annual / abs(max_dd) if max_dd != 0 else 0.0

    # Year-by-year returns
    rc = results.copy()
    rc['year'] = rc.index.year
    yearly = {}
    for yr, grp in rc.groupby('year'):
        yearly[yr] = (grp['value'].iloc[-1] / grp['value'].iloc[0] - 1) * 100

    return {
        'trail_pct': trail_pct,
        'annual':    annual,
        'sharpe':    sharpe,
        'max_dd':    max_dd,
        'calmar':    calmar,
        'final':     final,
        'n_stops':   n_stops,
        'yearly':    yearly,
    }


def _run_v31_with_stop(bot, fa_loader, ml_ranker, trail_pct: float) -> tuple:
    """Run V31 ML backtest with a specific trailing stop. Returns (results_df, n_stops)."""
    from src.strategies.v31_ml_strategy import V31MLStrategy
    s = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='alpaca',
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
    )
    # Override the trailing stop value
    s.config['trailing_stop'] = trail_pct / 100.0

    results = s.run_backtest(start_year=2015, end_year=2024)

    # Count trailing stop triggers from trade history
    n_stops = sum(1 for t in s.trade_history if t.get('reason') == 'trailing_stop')
    return results, n_stops


def _run_v32_with_stop(bot, fa_loader, ml_ranker, trail_pct: float) -> tuple:
    """Run V32 balanced backtest with a specific trailing stop."""
    from src.strategies.v32_sector_diversified import V32SectorDiversifiedStrategy
    s = V32SectorDiversifiedStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='alpaca',
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
        mode='balanced',
        metadata_dir=METADATA_DIR,
        top_per_sector=1,
        min_sectors=5,
        use_cycle_rotation=False,
    )
    s.config['trailing_stop'] = trail_pct / 100.0

    results = s.run_backtest(start_year=2015, end_year=2024)
    n_stops = sum(1 for t in s.trade_history if t.get('reason') == 'trailing_stop')
    return results, n_stops


# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows: list):
    stress_years = [y for y in [2018, 2020, 2022, 2023, 2024]
                    if any(y in m['yearly'] for m in rows)]

    # Column widths
    col_w = 8

    header = (f"  {'Stop':>5}  {'Annual':>7}  {'MaxDD':>7}  "
              f"{'Sharpe':>6}  {'Calmar':>6}  {'#Stops':>6}  {'Final $':>10}"
              + "".join(f"  {yr:>6}" for yr in stress_years))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for m in rows:
        yr_cols = "".join(f"  {m['yearly'].get(yr, float('nan')):>+6.1f}%" for yr in stress_years)
        marker = " ←" if m.get('_best_calmar') else (
                 " ←" if m.get('_best_sharpe') else "")
        print(
            f"  {m['trail_pct']:>4.0f}%  "
            f"{m['annual']:>+6.2f}%  "
            f"{m['max_dd']:>+6.2f}%  "
            f"{m['sharpe']:>6.3f}  "
            f"{m['calmar']:>6.3f}  "
            f"{m['n_stops']:>6}  "
            f"${m['final']:>9,.0f}"
            f"{yr_cols}"
            f"{marker}"
        )


def print_verdict(rows: list, strategy: str):
    best_calmar = max(rows, key=lambda m: m['calmar'])
    best_sharpe = max(rows, key=lambda m: m['sharpe'])
    best_annual = max(rows, key=lambda m: m['annual'])
    best_dd     = max(rows, key=lambda m: -m['max_dd'])   # least negative

    _header("VERDICT")
    print(f"  Strategy tested     : {strategy.upper()}")
    print()
    print(f"  Best Calmar ratio   : {best_calmar['trail_pct']:.0f}%  "
          f"(return={best_calmar['annual']:+.2f}%  dd={best_calmar['max_dd']:.2f}%  "
          f"calmar={best_calmar['calmar']:.3f})")
    print(f"  Best Sharpe ratio   : {best_sharpe['trail_pct']:.0f}%  "
          f"(return={best_sharpe['annual']:+.2f}%  sharpe={best_sharpe['sharpe']:.3f})")
    print(f"  Best annual return  : {best_annual['trail_pct']:.0f}%  "
          f"(return={best_annual['annual']:+.2f}%)")
    print(f"  Smallest drawdown   : {best_dd['trail_pct']:.0f}%  "
          f"(dd={best_dd['max_dd']:.2f}%)")
    print()

    # Specific comparison: 15% vs 20%
    r15 = next((m for m in rows if m['trail_pct'] == 15), None)
    r20 = next((m for m in rows if m['trail_pct'] == 20), None)
    if r15 and r20:
        print(f"  15% vs 20% direct comparison:")
        print(f"    15%  →  annual={r15['annual']:+.2f}%  dd={r15['max_dd']:.2f}%  "
              f"sharpe={r15['sharpe']:.3f}  stops={r15['n_stops']}")
        print(f"    20%  →  annual={r20['annual']:+.2f}%  dd={r20['max_dd']:.2f}%  "
              f"sharpe={r20['sharpe']:.3f}  stops={r20['n_stops']}")
        diff_annual = r20['annual'] - r15['annual']
        diff_dd     = r20['max_dd']  - r15['max_dd']
        print(f"    20% vs 15%: return {diff_annual:+.2f}%,  drawdown {diff_dd:+.2f}pp")

    print()
    print(f"  Recommended: {best_calmar['trail_pct']:.0f}%  (best risk-adjusted: Calmar ratio)")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def run(stop_values: list = None, strategy: str = 'v31'):
    if stop_values is None:
        stop_values = DEFAULT_STOPS

    _header(f"TRAILING STOP RESEARCH — {strategy.upper()} ML  |  2015-2024")
    print(f"  Testing stops: {stop_values}")
    print(f"  Model: {MODEL_PATH}")

    # ── Load shared data once ────────────────────────────────────────────────
    _header("LOADING DATA (shared across all runs)")
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.stock_ranker import MLStockRanker

    bot = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=INITIAL_CAPITAL,
    )
    bot.load_all_stocks()
    print(f"  {len(bot.stocks_data)} stocks loaded")

    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()
    print("  FA data loaded")

    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: model not found at {MODEL_PATH}")
        print("  Run backtest_honest_2015_2024.py first.")
        return

    ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
    ml_ranker.load_model(MODEL_PATH)
    vc = ml_ranker.train_metrics.get('val_corr', 'N/A')
    print(f"  Model loaded  (val_corr={vc:.4f})" if isinstance(vc, float) else
          f"  Model loaded  (val_corr={vc})")

    # ── Sweep ────────────────────────────────────────────────────────────────
    _header(f"SWEEP: {len(stop_values)} trailing stop values")

    run_fn = _run_v31_with_stop if strategy == 'v31' else _run_v32_with_stop

    rows = []
    for stop in stop_values:
        print(f"  Running {stop:>3}% trailing stop...", end=' ', flush=True)
        results, n_stops = run_fn(bot, fa_loader, ml_ranker, stop)
        m = calc_metrics(results, stop, n_stops)
        rows.append(m)
        print(f"  annual={m['annual']:+.2f}%  dd={m['max_dd']:.2f}%  "
              f"sharpe={m['sharpe']:.3f}  stops_triggered={n_stops}")

        # Save individual CSV
        os.makedirs('output', exist_ok=True)
        results.to_csv(f'output/trail_{strategy}_{stop:02d}pct_2015_2024.csv')

    # ── Results table ────────────────────────────────────────────────────────
    _header("FULL RESULTS TABLE")
    print_table(rows)

    # ── Verdict ──────────────────────────────────────────────────────────────
    print_verdict(rows, strategy)

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_rows = []
    for m in rows:
        r = {'strategy': strategy, 'trailing_stop_pct': m['trail_pct'],
             'annual_return': m['annual'], 'max_drawdown': m['max_dd'],
             'sharpe': m['sharpe'], 'calmar': m['calmar'],
             'final_value': m['final'], 'stops_triggered': m['n_stops']}
        for yr, ret in m['yearly'].items():
            r[f'return_{yr}'] = ret
        summary_rows.append(r)

    out_path = f'output/trailing_stop_research_{strategy}.csv'
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"  Full results saved: {out_path}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trailing Stop Research')
    parser.add_argument('--stops', nargs='+', type=int, default=None,
                        help='Stop values to test e.g. --stops 10 15 20 25')
    parser.add_argument('--strategy', choices=['v31', 'v32'], default='v31',
                        help='Which strategy to test (default: v31)')
    args = parser.parse_args()

    run(stop_values=args.stops, strategy=args.strategy)
