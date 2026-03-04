"""
V33 Hybrid Backtest
====================
Tests V33 (50% V31 core + 50% V32 sector) against V31 and V32 on 2015-2024.

Also sweeps the core_pct split: 30/70, 40/60, 50/50, 60/40, 70/30
to find the optimal balance between alpha and diversification.

Usage:
  python backtest_v33_hybrid.py              # full comparison + sweep
  python backtest_v33_hybrid.py --no-sweep   # just 50/50 vs V31 vs V32
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INITIAL_CAPITAL = 100_000
MODEL_PATH      = 'output/models/live_model_latest.txt'
FALLBACK_MODEL  = 'output/models/honest_2006_2014_model.txt'
METADATA_DIR    = 'sp500_data/metadata'


def _header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def calc_metrics(results: pd.DataFrame, label: str) -> dict:
    final  = results['value'].iloc[-1]
    years  = (results.index[-1] - results.index[0]).days / 365.25
    annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    cummax = results['value'].cummax()
    max_dd = ((results['value'] - cummax) / cummax * 100).min()
    rets   = results['value'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0
    calmar = annual / abs(max_dd) if max_dd != 0 else 0.0

    rc = results.copy()
    rc['year'] = rc.index.year
    yearly = {}
    for yr, grp in rc.groupby('year'):
        yearly[yr] = (grp['value'].iloc[-1] / grp['value'].iloc[0] - 1) * 100

    return {
        'label': label, 'annual': annual, 'sharpe': sharpe,
        'max_dd': max_dd, 'calmar': calmar, 'final': final,
        'yearly': yearly,
    }


def load_shared_data():
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.stock_ranker import MLStockRanker

    _header("LOADING SHARED DATA")
    bot = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=INITIAL_CAPITAL,
    )
    bot.load_all_stocks()
    print(f"  Stocks loaded : {len(bot.stocks_data)}")

    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()
    print(f"  FA tickers    : {len(fa_loader.ratios)}")

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
    ml_ranker.load_model(model_path)
    vc = ml_ranker.train_metrics.get('val_corr', 'N/A')
    print(f"  Model         : {model_path}")
    print(f"  Val corr      : {vc:.4f}" if isinstance(vc, float) else f"  Val corr      : {vc}")

    return bot, fa_loader, ml_ranker


def run_v31(bot, fa_loader, ml_ranker) -> dict:
    from src.strategies.v31_ml_strategy import V31MLStrategy
    s = V31MLStrategy(
        bot=bot, use_transaction_costs=True, broker='alpaca',
        enable_covered_calls=True, ml_model=ml_ranker,
        n_features_to_select=50, fa_loader=fa_loader,
    )
    r = s.run_backtest(start_year=2015, end_year=2024)
    return calc_metrics(r, 'V31 ML'), r


def run_v32(bot, fa_loader, ml_ranker, mode='balanced') -> dict:
    from src.strategies.v32_sector_diversified import V32SectorDiversifiedStrategy
    s = V32SectorDiversifiedStrategy(
        bot=bot, use_transaction_costs=True, broker='alpaca',
        enable_covered_calls=True, ml_model=ml_ranker,
        n_features_to_select=50, fa_loader=fa_loader,
        mode=mode, metadata_dir=METADATA_DIR,
        top_per_sector=1, min_sectors=5,
        use_cycle_rotation=False,
    )
    r = s.run_backtest(start_year=2015, end_year=2024)
    return calc_metrics(r, f'V32 {mode.title()}'), r


def run_v33(bot, fa_loader, ml_ranker, core_pct=0.50, mode='balanced') -> dict:
    from src.strategies.v33_hybrid import V33HybridStrategy
    label = f'V33 Hybrid {int(core_pct*100)}/{int((1-core_pct)*100)}'
    s = V33HybridStrategy(
        bot=bot, use_transaction_costs=True, broker='alpaca',
        enable_covered_calls=True, ml_model=ml_ranker,
        n_features_to_select=50, fa_loader=fa_loader,
        core_pct=core_pct, mode=mode, metadata_dir=METADATA_DIR,
        top_per_sector=1, min_sectors=4,
        use_cycle_rotation=False,
    )
    r = s.run_backtest(start_year=2015, end_year=2024)
    return calc_metrics(r, label), r


def print_comparison(rows: list, stress_years=None):
    if stress_years is None:
        stress_years = [2018, 2020, 2022, 2023, 2024]

    avail = [y for y in stress_years if any(y in m['yearly'] for m in rows)]

    print(f"\n  {'Strategy':<26} {'Annual':>8} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'Calmar':>7} {'Final $':>10}"
          + "".join(f"  {y:>5}" for y in avail))
    print(f"  {'-'*80}")

    for m in rows:
        yr_cols = "".join(f"  {m['yearly'].get(y, float('nan')):>+5.1f}%" for y in avail)
        print(f"  {m['label']:<26} {m['annual']:>+7.2f}% {m['max_dd']:>+6.2f}% "
              f"{m['sharpe']:>7.3f} {m['calmar']:>7.3f} ${m['final']:>9,.0f}{yr_cols}")


def print_verdict(rows: list):
    _header("VERDICT")

    v31  = next((m for m in rows if m['label'] == 'V31 ML'), None)
    v32  = next((m for m in rows if 'V32' in m['label']), None)
    best = max(rows, key=lambda m: m['calmar'])

    if v31:
        print(f"\n  V31 ML       : {v31['annual']:+.2f}% annual  DD={v31['max_dd']:.2f}%  "
              f"Sharpe={v31['sharpe']:.3f}")
    if v32:
        print(f"  V32 Sector   : {v32['annual']:+.2f}% annual  DD={v32['max_dd']:.2f}%  "
              f"Sharpe={v32['sharpe']:.3f}")

    hybrids = [m for m in rows if 'V33' in m['label']]
    if hybrids:
        print(f"\n  V33 Hybrid variants:")
        for m in hybrids:
            tag = ' ← best Calmar' if m['label'] == best['label'] else ''
            print(f"    {m['label']:<28} {m['annual']:+.2f}% annual  "
                  f"DD={m['max_dd']:.2f}%  Calmar={m['calmar']:.3f}{tag}")

    print(f"\n  Best overall (Calmar): {best['label']}")
    print(f"    Annual={best['annual']:+.2f}%  DD={best['max_dd']:.2f}%  "
          f"Sharpe={best['sharpe']:.3f}  Final=${best['final']:,.0f}")

    if v31 and best['label'] != 'V31 ML':
        dd_improvement = v31['max_dd'] - best['max_dd']
        ann_cost       = v31['annual'] - best['annual']
        print(f"\n  vs V31:  drawdown improved {dd_improvement:+.2f}pp  |  "
              f"annual return change {-ann_cost:+.2f}%")


def run(no_sweep: bool = False):
    bot, fa_loader, ml_ranker = load_shared_data()

    _header("STEP 1: BASELINE — V31 and V32")
    print("  Running V31 ML (2015-2024)...")
    m_v31, r_v31 = run_v31(bot, fa_loader, ml_ranker)
    print(f"    annual={m_v31['annual']:+.2f}%  DD={m_v31['max_dd']:.2f}%  "
          f"Sharpe={m_v31['sharpe']:.3f}")

    print("  Running V32 Balanced (2015-2024)...")
    m_v32, r_v32 = run_v32(bot, fa_loader, ml_ranker, mode='balanced')
    print(f"    annual={m_v32['annual']:+.2f}%  DD={m_v32['max_dd']:.2f}%  "
          f"Sharpe={m_v32['sharpe']:.3f}")

    _header("STEP 2: V33 HYBRID 50/50")
    print("  Running V33 50% core / 50% sector (2015-2024)...")
    m_v33_50, r_v33_50 = run_v33(bot, fa_loader, ml_ranker, core_pct=0.50)
    print(f"    annual={m_v33_50['annual']:+.2f}%  DD={m_v33_50['max_dd']:.2f}%  "
          f"Sharpe={m_v33_50['sharpe']:.3f}")

    all_rows = [m_v31, m_v32, m_v33_50]
    sweep_rows = []

    if not no_sweep:
        _header("STEP 3: CORE_PCT SWEEP (30% → 70%)")
        for core_pct in [0.30, 0.40, 0.60, 0.70]:
            label = f"V33 {int(core_pct*100)}/{int((1-core_pct)*100)}"
            print(f"  Running {label}...", end=' ', flush=True)
            m, _ = run_v33(bot, fa_loader, ml_ranker, core_pct=core_pct)
            sweep_rows.append(m)
            print(f"annual={m['annual']:+.2f}%  DD={m['max_dd']:.2f}%  "
                  f"Sharpe={m['sharpe']:.3f}")

        all_rows = [m_v31, m_v32] + sorted(
            [m_v33_50] + sweep_rows,
            key=lambda m: float(m['label'].split()[2].split('/')[0])
        )

    _header("FULL COMPARISON TABLE (2015-2024)")
    print_comparison(all_rows)
    print_verdict(all_rows)

    # Save
    os.makedirs('output', exist_ok=True)
    summary = []
    for m in all_rows:
        row = {k: v for k, v in m.items() if k != 'yearly'}
        for yr, ret in m['yearly'].items():
            row[f'return_{yr}'] = ret
        summary.append(row)
    out = 'output/v33_hybrid_comparison.csv'
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\n  Results saved: {out}")

    # Save best V33 equity curve
    r_v33_50.to_csv('output/v33_hybrid_50_50_2015_2024.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V33 Hybrid Backtest')
    parser.add_argument('--no-sweep', action='store_true',
                        help='Skip core_pct sweep, only run 50/50')
    args = parser.parse_args()
    run(no_sweep=args.no_sweep)
