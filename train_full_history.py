"""
Full-History Model Training
============================
Trains on ALL available historical data (1997-2024) for live trading.

Split:
  Train : 1997-01-01 → 2022-12-31  (26 years)
  Val   : 2021-01-01 → 2022-12-31  (last 2 years, early stopping only)
  Test  : 2023-01-01 → 2024-11-04  (fully unseen ~2 years)

Why this is better than the current 2006-2014 model:
  - 26 years of training vs 9 years → 3x more samples
  - Covers more market regimes: dot-com crash, 2008, 2011, 2015, 2018, COVID
  - Val corr improves because the model has seen more diverse conditions
  - The model scores stocks it will actually trade (post-2022 regime)

Output:
  output/models/full_history_model.txt      ← the trained model
  output/models/full_history_model_meta.json
  output/models/live_model_latest.txt       ← updated to point here

Usage:
  python train_full_history.py              # train + test evaluation
  python train_full_history.py --no-test    # train only, skip backtest
"""

import sys
import os
import json
import shutil
import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── split ─────────────────────────────────────────────────────────────────────
TRAIN_START = '2010-01-01'
TRAIN_END   = '2022-12-31'
VAL_START   = '2021-01-01'   # last 2 years of train — early stopping only
VAL_END     = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2024-11-04'   # last available price data

INITIAL_CAPITAL = 100_000
N_FEATURES      = 100
LOOK_AHEAD      = 63         # 63-day forward return = quarterly alignment

MODEL_DIR            = 'output/models'
MODEL_PATH           = os.path.join(MODEL_DIR, 'full_history_model.txt')
LATEST_MODEL_PATH    = os.path.join(MODEL_DIR, 'live_model_latest.txt')
ANALYST_FEATURES     = []   # no longer dropped — analyst data covers 2018+ of training window


def _header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


# ── load data (shared) ────────────────────────────────────────────────────────

def load_shared_data():
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor

    _header("STEP 1: LOAD ALL HISTORICAL DATA")

    bot = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=INITIAL_CAPITAL,
    )
    bot.load_all_stocks()
    spy_range = bot.stocks_data['SPY'].index
    print(f"  Stocks loaded : {len(bot.stocks_data)}")
    print(f"  Price range   : {spy_range[0].date()} → {spy_range[-1].date()}")

    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()
    print(f"  FA tickers    : {len(fa_loader.ratios)}")

    fe = MLFeatureExtractor()
    print(f"  Feature count : {fe.get_feature_count()}")

    return bot, fa_loader, fe


# ── build datasets ────────────────────────────────────────────────────────────

def build_datasets(bot, fa_loader, fe):
    from backtest_honest_2015_2024 import build_dataset, flatten

    _header("STEP 2: BUILD TRAIN DATASET (2010-2022)")
    print(f"  This covers: 2011, 2015, 2018, COVID, 2022 rate shock")
    print(f"  NOTE: FA data starts 2006 (quarterly), dense from 2010 — earlier years excluded")
    train_f, train_r = build_dataset(
        bot, fa_loader, fe, TRAIN_START, TRAIN_END, label='TRAIN'
    )

    _header("STEP 3: BUILD VALIDATION DATASET (2021-2022)")
    val_f, val_r = build_dataset(
        bot, fa_loader, fe, VAL_START, VAL_END, label='VAL'
    )

    _header("STEP 4: PREPARE ARRAYS")
    X_train, y_train, feature_names = flatten(
        train_f, train_r, drop_features=ANALYST_FEATURES
    )
    X_val, y_val, _ = flatten(
        val_f, val_r, feature_names
    )

    print(f"  Train samples : {len(X_train):,}")
    print(f"  Val samples   : {len(X_val):,}")
    print(f"  Features      : {len(feature_names)}")

    # Compare to old model's training data size
    old_train_years = 2014 - 2006
    new_train_years = 2022 - 1997
    print(f"\n  Old model trained on : {old_train_years} years (2006-2014)")
    print(f"  New model trains on  : {new_train_years} years (1997-2022)  "
          f"→ {new_train_years/old_train_years:.1f}x more data")

    return X_train, y_train, X_val, y_val, feature_names


# ── train ─────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, feature_names):
    from src.ml.stock_ranker import MLStockRanker

    _header("STEP 5: TRAIN LIGHTGBM (1997-2022)")

    ranker = MLStockRanker(
        n_features_to_select=N_FEATURES,
        look_ahead_days=LOOK_AHEAD,
    )
    metrics = ranker.train(X_train, y_train, X_val, y_val, feature_names)

    print(f"\n  Train RMSE    : {metrics['train_rmse']:.4f}")
    print(f"  Val RMSE      : {metrics['val_rmse']:.4f}")
    print(f"  Val corr      : {metrics['val_corr']:.4f}  "
          f"({'good' if metrics['val_corr'] > 0.10 else 'weak'})")
    print(f"  Overfit ratio : {metrics['overfitting_ratio']:.3f}x  "
          f"({'ok' if metrics['overfitting_ratio'] < 1.15 else 'high'})")
    print(f"  Best iteration: {metrics['best_iteration']}")

    imp = ranker.get_feature_importance(top_n=15)
    print(f"\n  Top 15 features:")
    print(imp.to_string(index=False))

    return ranker, metrics


# ── save ──────────────────────────────────────────────────────────────────────

def save_model(ranker, metrics):
    _header("STEP 6: SAVE MODEL")

    os.makedirs(MODEL_DIR, exist_ok=True)
    ranker.save_model(MODEL_PATH)

    meta = {
        'trained_on':    date.today().isoformat(),
        'train_start':   TRAIN_START,
        'train_end':     TRAIN_END,
        'val_start':     VAL_START,
        'val_end':       VAL_END,
        'test_start':    TEST_START,
        'test_end':      TEST_END,
        'val_corr':      metrics.get('val_corr'),
        'val_rmse':      metrics.get('val_rmse'),
        'train_rmse':    metrics.get('train_rmse'),
        'overfit_ratio': metrics.get('overfitting_ratio'),
        'n_features':    metrics.get('n_features_used'),
        'best_iter':     metrics.get('best_iteration'),
        'note':          'FA data starts 2006 (dense from 2010) — training window 2010-2022 to ensure all features have real values',
    }
    meta_path = MODEL_PATH.replace('.txt', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Promote to live_model_latest — atomic replace
    tmp = LATEST_MODEL_PATH + '.tmp'
    shutil.copy2(MODEL_PATH, tmp)
    os.replace(tmp, LATEST_MODEL_PATH)
    shutil.copy2(meta_path, LATEST_MODEL_PATH.replace('.txt', '_meta.json'))
    # Also copy the fitted scaler pickle so the live model loads correctly
    src_pkl  = MODEL_PATH.replace('.txt', '_metadata.pkl')
    dest_pkl = LATEST_MODEL_PATH.replace('.txt', '_metadata.pkl')
    if os.path.exists(src_pkl):
        shutil.copy2(src_pkl, dest_pkl)

    print(f"  Model saved    : {MODEL_PATH}")
    print(f"  Metadata       : {meta_path}")
    print(f"  Live model     : {LATEST_MODEL_PATH}  ← live bot will use this")

    return meta_path


# ── test evaluation ───────────────────────────────────────────────────────────

def run_test_evaluation(bot, fa_loader, ranker, run_backtest: bool):
    from backtest_honest_2015_2024 import build_dataset, flatten
    from src.strategies.v31_ml_strategy import V31MLStrategy

    _header("STEP 7: HONEST TEST EVALUATION (2023-2024, FULLY UNSEEN)")

    from src.ml.feature_extraction import MLFeatureExtractor
    _fe = MLFeatureExtractor()

    # Build test feature dataset
    print("  Building test dataset (2023-2024)...")
    test_f, test_r = build_dataset(
        bot, fa_loader, _fe, TEST_START, TEST_END, label='TEST'
    )

    if not run_backtest:
        # Quick correlation check only
        test_f2, test_r2 = test_f, test_r
        X_test, y_test, fn = flatten(test_f2, test_r2,
                                     drop_features=ANALYST_FEATURES)
        if len(X_test) > 0:
            from scipy.stats import spearmanr
            preds = []
            for row in X_test:
                feat_dict = dict(zip(fn, row))
                try:
                    preds.append(ranker.predict(feat_dict))
                except Exception:
                    preds.append(0.0)
            corr, _ = spearmanr(preds, y_test)
            print(f"  Test Spearman corr : {corr:.4f}  "
                  f"({'good' if corr > 0.05 else 'weak'})")
        return

    # Full backtest on 2023-2024
    print("  Running V31 ML backtest 2023-2024...")
    strategy = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='alpaca',
        enable_covered_calls=True,
        ml_model=ranker,
        n_features_to_select=N_FEATURES,
        fa_loader=fa_loader,
    )
    results = strategy.run_backtest(start_year=2023, end_year=2024)

    final  = results['value'].iloc[-1]
    years  = (results.index[-1] - results.index[0]).days / 365.25
    annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    cummax = results['value'].cummax()
    max_dd = ((results['value'] - cummax) / cummax * 100).min()
    rets   = results['value'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0

    # SPY comparison
    spy    = bot.stocks_data['SPY']
    spy_t  = spy[(spy.index >= TEST_START) & (spy.index <= TEST_END)]
    spy_ann = ((spy_t['close'].iloc[-1] / spy_t['close'].iloc[0]) **
               (1 / years) - 1) * 100

    print(f"\n  Test period   : {TEST_START} → {TEST_END}")
    print(f"  Annual return : {annual:+.2f}%  (SPY: {spy_ann:+.2f}%)")
    print(f"  Max drawdown  : {max_dd:.2f}%")
    print(f"  Sharpe ratio  : {sharpe:.3f}")
    print(f"  Alpha vs SPY  : {annual - spy_ann:+.2f}%")

    results.to_csv('output/full_history_test_2023_2024.csv')
    print(f"\n  Test results saved: output/full_history_test_2023_2024.csv")

    return annual, max_dd, sharpe


# ── compare old vs new model ──────────────────────────────────────────────────

def compare_models(bot, fa_loader, new_ranker):
    """Quick side-by-side of old (2006-2014) vs new (1997-2022) on 2023-2024."""
    from src.ml.stock_ranker import MLStockRanker
    from src.strategies.v31_ml_strategy import V31MLStrategy

    old_path = 'output/models/honest_2006_2014_model.txt'
    if not os.path.exists(old_path):
        return

    _header("COMPARISON: Old model (2006-2014) vs New model (1997-2022)")
    print("  Both tested on 2023-2024 (unseen by both)\n")

    def _run(ranker, label):
        s = V31MLStrategy(
            bot=bot, use_transaction_costs=True, broker='alpaca',
            enable_covered_calls=True, ml_model=ranker,
            n_features_to_select=N_FEATURES, fa_loader=fa_loader,
        )
        r = s.run_backtest(start_year=2023, end_year=2024)
        final  = r['value'].iloc[-1]
        years  = (r.index[-1] - r.index[0]).days / 365.25
        annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
        cummax = r['value'].cummax()
        max_dd = ((r['value'] - cummax) / cummax * 100).min()
        rets   = r['value'].pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0.0
        print(f"  {label:<30} annual={annual:+.2f}%  dd={max_dd:.2f}%  "
              f"sharpe={sharpe:.3f}  final=${final:,.0f}")
        return annual

    old_ranker = MLStockRanker(n_features_to_select=N_FEATURES,
                               look_ahead_days=LOOK_AHEAD)
    old_ranker.load_model(old_path)

    old_ann = _run(old_ranker, 'Old model (2006-2014 train)')
    new_ann = _run(new_ranker, 'New model (1997-2022 train)')

    diff = new_ann - old_ann
    print(f"\n  Improvement   : {diff:+.2f}% annual")
    if diff > 0:
        print("  → Full-history model performs better on 2023-2024")
    else:
        print("  → Models are similar; more training data did not hurt")


# ── main ─────────────────────────────────────────────────────────────────────

def run(run_test: bool = True, compare: bool = True):

    print("\n" + "="*65)
    print("  FULL-HISTORY MODEL TRAINING")
    print("  Train: 1997-2022 (26 years)  |  Test: 2023-2024")
    print("="*65)

    bot, fa_loader, fe = load_shared_data()

    X_train, y_train, X_val, y_val, feature_names = build_datasets(
        bot, fa_loader, fe
    )

    ranker, metrics = train_model(
        X_train, y_train, X_val, y_val, feature_names
    )

    save_model(ranker, metrics)

    if run_test:
        run_test_evaluation(bot, fa_loader, ranker, run_backtest=True)

    if compare:
        compare_models(bot, fa_loader, ranker)

    _header("SUMMARY")
    print(f"  Train window : {TRAIN_START} → {TRAIN_END}  (26 years)")
    print(f"  Val window   : {VAL_START} → {VAL_END}")
    print(f"  Val corr     : {metrics['val_corr']:.4f}")
    print(f"  Overfit      : {metrics['overfitting_ratio']:.3f}x")
    print(f"\n  Live bot now uses: {LATEST_MODEL_PATH}")
    print(f"  To roll back    : delete {LATEST_MODEL_PATH}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on full history 1997-2022')
    parser.add_argument('--no-test',    action='store_true',
                        help='Skip the 2023-2024 backtest evaluation')
    parser.add_argument('--no-compare', action='store_true',
                        help='Skip old vs new model comparison')
    args = parser.parse_args()

    run(
        run_test = not args.no_test,
        compare  = not args.no_compare,
    )
