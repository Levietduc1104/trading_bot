"""
Feature Drift Detection
========================
Monitors whether current live feature distributions have drifted significantly
from the training distribution. Silent drift causes model performance to degrade
without any obvious error.

What it checks:
  - For each of the 153 ML features: mean, std, p5, p25, p75, p95
  - Computes Z-score of current mean vs training baseline
  - Flags features where |Z| > DRIFT_THRESHOLD (default 2.0 sigma)
  - Saves drift report to output/feature_drift.json
  - Prints a ranked summary of most-drifted features

Baseline:
  - Built from the training data distributions stored in the model metadata
  - Falls back to computing from historical CSV data if metadata lacks stats

Usage:
  python feature_drift_detector.py                  # check drift today
  python feature_drift_detector.py --rebuild-baseline  # recompute baseline from history
  python feature_drift_detector.py --days 30        # use last 30 days of live data

Schedule (weekly, Monday morning):
  0 8 * * 1  cd /path/to/trading_bot && python feature_drift_detector.py >> logs/drift.log 2>&1
"""

import sys
import os
import json
import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

DRIFT_THRESHOLD  = 2.0     # sigma: |Z| > 2 → flag as drifted
WARN_THRESHOLD   = 1.5     # sigma: |Z| > 1.5 → warn (yellow)
BASELINE_PATH    = 'output/models/feature_baseline.json'
DRIFT_REPORT     = 'output/feature_drift.json'
DB_PATH          = 'output/data/market_data.db'

# Features known to be regime-sensitive (higher drift tolerance expected)
REGIME_FEATURES = {
    'vix_level', 'vix_roc_5d', 'vix_roc_20d', 'market_stress',
    'spy_ma200_ratio', 'spy_ma50_ratio', 'spy_trend_strength',
}


def _header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


# ── Baseline ─────────────────────────────────────────────────────────────────

def build_baseline_from_history() -> dict:
    """
    Compute feature distribution stats from historical backtest CSV data.
    Uses the honest 2015-2024 dataset as representative training distribution.
    """
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor
    from backtest_honest_2015_2024 import build_dataset, flatten

    _header("BUILDING FEATURE BASELINE FROM HISTORY (2006-2022)")

    bot = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=100_000,
    )
    bot.load_all_stocks()
    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()
    fe = MLFeatureExtractor()

    feats, rets = build_dataset(
        bot, fa_loader, fe, '2006-01-01', '2022-12-31', label='BASELINE'
    )
    X, y, feature_names = flatten(feats, rets)

    df = pd.DataFrame(X, columns=feature_names)

    baseline = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        baseline[col] = {
            'mean':  float(s.mean()),
            'std':   float(s.std()),
            'p5':    float(s.quantile(0.05)),
            'p25':   float(s.quantile(0.25)),
            'p75':   float(s.quantile(0.75)),
            'p95':   float(s.quantile(0.95)),
            'count': int(len(s)),
        }

    os.makedirs('output/models', exist_ok=True)
    with open(BASELINE_PATH, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"  Baseline computed for {len(baseline)} features")
    print(f"  Saved to {BASELINE_PATH}")
    return baseline


def load_baseline() -> dict:
    if not os.path.exists(BASELINE_PATH):
        print(f"  No baseline found at {BASELINE_PATH}")
        print(f"  Run with --rebuild-baseline first.")
        return {}
    with open(BASELINE_PATH) as f:
        return json.load(f)


# ── Current distribution from MarketDB ───────────────────────────────────────

def get_current_distributions(days: int = 30) -> pd.DataFrame:
    """
    Pull last N days of features from MarketDatabase.
    Returns DataFrame: rows=tickers×dates, cols=features.
    """
    from src.data.market_db import MarketDatabase

    db = MarketDatabase(DB_PATH)
    stats = db.get_snapshot_stats()

    if stats['daily_features'] == 0:
        return pd.DataFrame()

    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()

    df = db.get_training_data(start_date, end_date)
    return df


# ── Drift computation ─────────────────────────────────────────────────────────

def compute_drift(baseline: dict, current_df: pd.DataFrame) -> list:
    """
    Compare current feature distributions to baseline.
    Returns list of drift records sorted by |Z-score|.
    """
    records = []

    for feature, bstat in baseline.items():
        if feature not in current_df.columns:
            continue
        s = current_df[feature].dropna()
        if len(s) < 5:
            continue

        cur_mean = s.mean()
        cur_std  = s.std()

        # Z-score of current mean relative to training distribution
        if bstat['std'] > 0:
            z_mean = (cur_mean - bstat['mean']) / bstat['std']
        else:
            z_mean = 0.0

        # Std ratio (current volatility vs training volatility)
        std_ratio = cur_std / bstat['std'] if bstat['std'] > 0 else 1.0

        is_regime = feature in REGIME_FEATURES
        effective_threshold = DRIFT_THRESHOLD * (1.5 if is_regime else 1.0)

        drifted = abs(z_mean) > effective_threshold
        warned  = abs(z_mean) > WARN_THRESHOLD and not drifted

        records.append({
            'feature':     feature,
            'z_score':     round(z_mean, 3),
            'abs_z':       round(abs(z_mean), 3),
            'train_mean':  round(bstat['mean'], 4),
            'cur_mean':    round(cur_mean, 4),
            'train_std':   round(bstat['std'], 4),
            'cur_std':     round(cur_std, 4),
            'std_ratio':   round(std_ratio, 3),
            'cur_n':       int(len(s)),
            'drifted':     drifted,
            'warned':      warned,
            'regime_feat': is_regime,
        })

    records.sort(key=lambda x: -x['abs_z'])
    return records


# ── Report ────────────────────────────────────────────────────────────────────

def print_drift_report(records: list, top_n: int = 20):
    drifted = [r for r in records if r['drifted']]
    warned  = [r for r in records if r['warned']]
    ok      = [r for r in records if not r['drifted'] and not r['warned']]

    _header(f"FEATURE DRIFT REPORT  ({date.today()})")

    print(f"\n  Features checked  : {len(records)}")
    print(f"  DRIFTED (|Z|>{DRIFT_THRESHOLD:.0f}) : {len(drifted)}")
    print(f"  WARNED  (|Z|>{WARN_THRESHOLD:.0f}) : {len(warned)}")
    print(f"  OK                : {len(ok)}")

    if drifted:
        print(f"\n  DRIFTED FEATURES (require attention):")
        print(f"  {'Feature':<30} {'Z-score':>8} {'Train mean':>12} {'Cur mean':>12}  Regime?")
        print(f"  {'-'*70}")
        for r in drifted:
            flag = 'yes' if r['regime_feat'] else ''
            print(f"  {r['feature']:<30} {r['z_score']:>+8.3f} "
                  f"{r['train_mean']:>12.4f} {r['cur_mean']:>12.4f}  {flag}")

    if warned:
        print(f"\n  WARNING FEATURES (monitor):")
        print(f"  {'Feature':<30} {'Z-score':>8} {'Train mean':>12} {'Cur mean':>12}")
        print(f"  {'-'*58}")
        for r in warned:
            print(f"  {r['feature']:<30} {r['z_score']:>+8.3f} "
                  f"{r['train_mean']:>12.4f} {r['cur_mean']:>12.4f}")

    print(f"\n  Top {min(top_n, len(records))} features by drift magnitude:")
    print(f"  {'Feature':<30} {'|Z|':>6} {'Std ratio':>10}  Status")
    print(f"  {'-'*60}")
    for r in records[:top_n]:
        status = 'DRIFT' if r['drifted'] else ('WARN' if r['warned'] else 'ok')
        print(f"  {r['feature']:<30} {r['abs_z']:>6.3f} {r['std_ratio']:>10.3f}  {status}")


def print_verdict(records: list):
    drifted = [r for r in records if r['drifted']]
    non_regime_drifted = [r for r in drifted if not r['regime_feat']]

    _header("VERDICT")

    if len(non_regime_drifted) == 0:
        print(f"\n  No significant non-regime feature drift detected.")
        print(f"  Model predictions are likely stable.")
    elif len(non_regime_drifted) <= 3:
        print(f"\n  Minor drift in {len(non_regime_drifted)} non-regime feature(s).")
        print(f"  Monitor but no immediate action required.")
        for r in non_regime_drifted:
            print(f"    - {r['feature']}: Z={r['z_score']:+.2f}")
    else:
        print(f"\n  SIGNIFICANT DRIFT detected in {len(non_regime_drifted)} non-regime features.")
        print(f"  Recommendation: retrain model with recent data.")
        print(f"  Run: python retrain_model.py")
        for r in non_regime_drifted[:5]:
            print(f"    - {r['feature']}: Z={r['z_score']:+.2f}")

    regime_drifted = [r for r in drifted if r['regime_feat']]
    if regime_drifted:
        print(f"\n  Regime features drifted (expected during market transitions):")
        for r in regime_drifted:
            direction = 'up' if r['z_score'] > 0 else 'down'
            print(f"    - {r['feature']}: {direction} {abs(r['z_score']):.1f} sigma")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(rebuild_baseline: bool = False, days: int = 30):

    _header(f"FEATURE DRIFT DETECTOR  |  last {days} days vs training baseline")

    # Step 1: baseline
    if rebuild_baseline:
        baseline = build_baseline_from_history()
    else:
        baseline = load_baseline()
        if not baseline:
            print("  Run with --rebuild-baseline to create it first.")
            return

    print(f"\n  Baseline features : {len(baseline)}")

    # Step 2: current distributions from MarketDB
    print(f"  Loading last {days} days from MarketDatabase...")
    current_df = get_current_distributions(days=days)

    if current_df.empty:
        print(f"\n  No live data in MarketDatabase yet.")
        print(f"  The DB collects data each time the live bot runs.")
        print(f"  Re-run after a few weeks of live operation.")
        return

    print(f"  Live records loaded : {len(current_df)} rows, {len(current_df.columns)} cols")

    # Step 3: compute drift
    records = compute_drift(baseline, current_df)

    if not records:
        print("  No matching features between baseline and live data.")
        return

    # Step 4: report
    print_drift_report(records)
    print_verdict(records)

    # Step 5: save JSON report
    report = {
        'report_date': date.today().isoformat(),
        'days_of_live_data': days,
        'baseline_features': len(baseline),
        'checked_features': len(records),
        'drifted': [r for r in records if r['drifted']],
        'warned': [r for r in records if r['warned']],
        'top_20': records[:20],
    }
    os.makedirs('output', exist_ok=True)
    with open(DRIFT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {DRIFT_REPORT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Drift Detection')
    parser.add_argument('--rebuild-baseline', action='store_true',
                        help='Recompute baseline from historical data')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of live data to compare (default: 30)')
    args = parser.parse_args()
    run(rebuild_baseline=args.rebuild_baseline, days=args.days)
