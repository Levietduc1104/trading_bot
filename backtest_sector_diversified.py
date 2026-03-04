"""
Sector-Diversified Backtest: V32 vs V31 ML Baseline
=====================================================
Train  : 2006-2014  (reuses saved model if present)
Test   : 2015-2024  (fully unseen)

Tests four strategies:
  Phase 1A  — V31Tier2GrowthScoringStrategy (no-ML baseline)
  V31 ML    — original mega-cap/momentum split with ML scoring
  V32 Loose     — sector-diversified, 1 stock/sector, 30% sector cap
  V32 Balanced  — sector-diversified, 1 stock/sector, 25% sector cap
  V32 Equal     — sector-diversified, 1 stock/sector, equal $ per sector

Outputs:
  output/v32_sector_loose_2015_2024.csv
  output/v32_sector_balanced_2015_2024.csv
  output/v32_sector_equal_2015_2024.csv
  output/v32_sector_comparison.csv   ← summary table printed + saved

Usage:
  python backtest_sector_diversified.py
  python backtest_sector_diversified.py --skip-training   # reuse saved model
  python backtest_sector_diversified.py --mode balanced   # run one mode only
"""

import sys
import os
import argparse
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000
TRAIN_START     = '2006-01-01'
TRAIN_END       = '2014-12-31'
VAL_START       = '2013-01-01'
VAL_END         = '2014-12-31'
TEST_START      = '2015-01-01'
TEST_END        = '2024-12-31'
MODEL_PATH      = 'output/models/honest_2006_2014_model.txt'
METADATA_DIR    = 'sp500_data/metadata'


# ── helpers ──────────────────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def calc_metrics(results: pd.DataFrame, name: str, initial: float = INITIAL_CAPITAL) -> dict:
    """Compute annual return, Sharpe, max drawdown, and year-by-year returns."""
    final  = results['value'].iloc[-1]
    years  = (results.index[-1] - results.index[0]).days / 365.25
    annual = ((final / initial) ** (1 / years) - 1) * 100
    cummax = results['value'].cummax()
    max_dd = ((results['value'] - cummax) / cummax * 100).min()
    rets   = results['value'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0

    # Per-year returns for stress analysis (especially 2022)
    results_copy = results.copy()
    results_copy['year'] = results_copy.index.year
    yearly = {}
    for yr, grp in results_copy.groupby('year'):
        y_start = grp['value'].iloc[0]
        y_end   = grp['value'].iloc[-1]
        yearly[yr] = (y_end / y_start - 1) * 100

    return {
        'name':   name,
        'annual': annual,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final':  final,
        'yearly': yearly,
    }


def print_comparison(metrics_list: list):
    names  = [m['name'] for m in metrics_list]
    width  = max(len(n) for n in names) + 2

    _header("RESULTS: 2015-2024 (10 YEARS, FULLY UNSEEN)")
    header = f"  {'Metric':<22}" + "".join(f"{n:>{width}}" for n in names)
    print(header)
    print("  " + "-" * (22 + width * len(names)))

    def row(label, fn):
        vals = "".join(fn(m) for m in metrics_list)
        print(f"  {label:<22}{vals}")

    row("Annual Return",    lambda m: f"{m['annual']:>{width}.2f}%")
    row("Sharpe Ratio",     lambda m: f"{m['sharpe']:>{width}.3f} ")
    row("Max Drawdown",     lambda m: f"{m['max_dd']:>{width}.2f}%")
    row("Final Value ($)",  lambda m: f"{m['final']:>{width},.0f} ")

    # Year-by-year stress rows
    all_years = sorted({yr for m in metrics_list for yr in m['yearly']})
    stress_years = [y for y in all_years if y in (2018, 2020, 2022, 2023, 2024)]
    if stress_years:
        print(f"\n  {'Year':^22}" + "".join(f"{'Return':>{width}}" for _ in names))
        print("  " + "-" * (22 + width * len(names)))
        for yr in stress_years:
            row(str(yr), lambda m, y=yr: f"{m['yearly'].get(y, float('nan')):>{width}.2f}%")

    print()


# ── data loading (shared across all strategies) ───────────────────────────────

def load_data():
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor

    _header("LOADING DATA")

    bot = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=INITIAL_CAPITAL,
    )
    bot.load_all_stocks()
    logger.info("  %d stocks loaded", len(bot.stocks_data))

    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()
    logger.info("  FA data loaded")

    feature_extractor = MLFeatureExtractor()
    logger.info("  Feature count: %s", feature_extractor.get_feature_count())

    return bot, fa_loader, feature_extractor


# ── model training / loading ──────────────────────────────────────────────────

def get_model(bot, fa_loader, feature_extractor, skip_training: bool):
    from src.ml.stock_ranker import MLStockRanker

    if skip_training and os.path.exists(MODEL_PATH):
        _header("LOADING SAVED MODEL")
        ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
        ranker.load_model(MODEL_PATH)
        vc = ranker.train_metrics.get('val_corr', 'N/A')
        print(f"  Model: {MODEL_PATH}")
        print(f"  Val corr: {vc:.4f}" if isinstance(vc, float) else f"  Val corr: {vc}")
        return ranker

    _header("TRAINING MODEL (2006-2014)")

    from backtest_honest_2015_2024 import build_dataset, flatten

    ANALYST_FEATURES = ['analyst_eps_revision', 'analyst_rev_revision']

    train_f, train_r = build_dataset(bot, fa_loader, feature_extractor,
                                     TRAIN_START, TRAIN_END, label='TRAIN')
    val_f,   val_r   = build_dataset(bot, fa_loader, feature_extractor,
                                     VAL_START,   VAL_END,   label='VAL')

    X_train, y_train, feature_names = flatten(train_f, train_r, drop_features=ANALYST_FEATURES)
    X_val,   y_val,   _             = flatten(val_f,   val_r,   feature_names)

    print(f"  Train samples : {len(X_train):,}")
    print(f"  Val samples   : {len(X_val):,}")
    print(f"  Features      : {len(feature_names)}")

    ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
    metrics = ranker.train(X_train, y_train, X_val, y_val, feature_names)

    print(f"\n  Val RMSE : {metrics['val_rmse']:.4f}")
    print(f"  Val corr : {metrics['val_corr']:.4f}")
    print(f"  Overfit  : {metrics['overfitting_ratio']:.3f}x")

    os.makedirs('output/models', exist_ok=True)
    ranker.save_model(MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")

    return ranker


# ── run one strategy ──────────────────────────────────────────────────────────

def run_v31_baseline(bot, fa_loader):
    """Phase 1A no-ML baseline."""
    from src.strategies.v31_tier2_growth_scoring import V31Tier2GrowthScoringStrategy
    s = V31Tier2GrowthScoringStrategy(
        bot=bot, use_transaction_costs=True, broker='alpaca',
        enable_covered_calls=True, momentum_weight=0.50, growth_weight=0.50,
    )
    return s.run_backtest(start_year=2015, end_year=2024)


def run_v31_ml(bot, fa_loader, ml_ranker):
    """Original V31 ML: mega-cap 70% + momentum 30%."""
    from src.strategies.v31_ml_strategy import V31MLStrategy
    s = V31MLStrategy(
        bot=bot, use_transaction_costs=True, broker='alpaca',
        enable_covered_calls=True, ml_model=ml_ranker,
        n_features_to_select=50, fa_loader=fa_loader,
    )
    return s.run_backtest(start_year=2015, end_year=2024)


def run_v32(bot, fa_loader, ml_ranker, mode: str, use_cycle_rotation: bool = False):
    """V32 sector-diversified variant. Set use_cycle_rotation=True for cycle-aware weighting."""
    from src.strategies.v32_sector_diversified import V32SectorDiversifiedStrategy
    s = V32SectorDiversifiedStrategy(
        bot=bot, use_transaction_costs=True, broker='alpaca',
        enable_covered_calls=True, ml_model=ml_ranker,
        n_features_to_select=50, fa_loader=fa_loader,
        mode=mode, metadata_dir=METADATA_DIR,
        top_per_sector=1, min_sectors=5,
        use_cycle_rotation=use_cycle_rotation,
    )
    return s.run_backtest(start_year=2015, end_year=2024)


# ── main ─────────────────────────────────────────────────────────────────────

def run(skip_training: bool = False, modes: list = None):
    if modes is None:
        modes = ['loose', 'balanced', 'equal']

    bot, fa_loader, feature_extractor = load_data()
    ml_ranker = get_model(bot, fa_loader, feature_extractor, skip_training)

    all_metrics = []

    # Phase 1A baseline
    _header("RUNNING PHASE 1A BASELINE")
    res_1a = run_v31_baseline(bot, fa_loader)
    m_1a   = calc_metrics(res_1a, 'Phase 1A')
    all_metrics.append(m_1a)
    res_1a.to_csv('output/honest_1a_2015_2024.csv')
    print("  Phase 1A done")

    # V31 ML
    _header("RUNNING V31 ML (MEGA-CAP/MOMENTUM)")
    res_v31 = run_v31_ml(bot, fa_loader, ml_ranker)
    m_v31   = calc_metrics(res_v31, 'V31 ML')
    all_metrics.append(m_v31)
    res_v31.to_csv('output/honest_ml_2015_2024.csv')
    print("  V31 ML done")

    # V32 variants
    for mode in modes:
        _header(f"RUNNING V32 SECTOR-DIVERSIFIED ({mode.upper()})")
        res_v32 = run_v32(bot, fa_loader, ml_ranker, mode, use_cycle_rotation=False)
        m_v32   = calc_metrics(res_v32, f'V32 {mode.capitalize()}')
        all_metrics.append(m_v32)
        res_v32.to_csv(f'output/v32_sector_{mode}_2015_2024.csv')
        print(f"  V32 {mode} done")

    # V32 Balanced + cycle rotation  (the key new variant)
    _header("RUNNING V32 BALANCED + CYCLE ROTATION")
    res_cycle = run_v32(bot, fa_loader, ml_ranker, 'balanced', use_cycle_rotation=True)
    m_cycle   = calc_metrics(res_cycle, 'V32 Cycle')
    all_metrics.append(m_cycle)
    res_cycle.to_csv('output/v32_sector_cycle_2015_2024.csv')
    print("  V32 Cycle done")

    # Print comparison table
    print_comparison(all_metrics)

    # Save summary CSV
    os.makedirs('output', exist_ok=True)
    rows = []
    for m in all_metrics:
        row = {'strategy': m['name'], 'annual_return': m['annual'],
               'sharpe': m['sharpe'], 'max_drawdown': m['max_dd'],
               'final_value': m['final']}
        for yr, ret in m['yearly'].items():
            row[f'return_{yr}'] = ret
        rows.append(row)
    pd.DataFrame(rows).to_csv('output/v32_sector_comparison.csv', index=False)
    print("Summary saved: output/v32_sector_comparison.csv")

    # Verdict
    _header("VERDICT")
    v32_variants = [m for m in all_metrics if m['name'].startswith('V32')]
    best = max(v32_variants, key=lambda m: m['annual'])
    diff_best   = best['annual']  - m_v31['annual']
    diff_cycle  = m_cycle['annual'] - m_v31['annual']
    dd_cycle    = m_cycle['max_dd'] - m_v31['max_dd']
    y22_v31     = m_v31['yearly'].get(2022, 0)
    y22_cycle   = m_cycle['yearly'].get(2022, 0)

    print(f"  Best overall       : {best['name']}  ({diff_best:+.2f}% vs V31 ML)")
    print(f"  V32 Cycle vs V31   : {diff_cycle:+.2f}% annual return")
    print(f"  Drawdown change    : {dd_cycle:+.2f}pp  (negative = less drawdown)")
    print(f"  2022 stress test   : V31={y22_v31:.1f}%  V32 Cycle={y22_cycle:.1f}%")
    print()
    if y22_cycle > y22_v31:
        print("  ✓ Cycle rotation reduced drawdown in 2022 rate-shock year")
    else:
        print("  - Cycle rotation did not improve 2022 specifically")
    if diff_cycle > 0:
        print("  ✓ Cycle rotation improved annual return over baseline V32")
    else:
        print(f"  - Cycle rotation trades {abs(diff_cycle):.1f}% annual return for smoother ride")
    print()

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V32 Sector-Diversified Backtest')
    parser.add_argument('--skip-training', action='store_true',
                        help='Reuse saved model instead of retraining')
    parser.add_argument('--mode', choices=['loose', 'balanced', 'equal'],
                        default=None,
                        help='Run only one V32 mode (default: all three + cycle)')
    args = parser.parse_args()

    modes = [args.mode] if args.mode else ['loose', 'balanced', 'equal']
    run(skip_training=args.skip_training, modes=modes)
