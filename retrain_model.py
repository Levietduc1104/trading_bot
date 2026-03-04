"""
Rolling Model Retrain
=====================
Retrains the LightGBM model on a rolling window of data from two sources:

  1. MarketDatabase (output/data/market_data.db)
     Daily feature snapshots collected by the live bot since deployment.
     Used as the primary source once enough data has accumulated.

  2. Historical backtest data (sp500_data/stock_data_1990_2024 + FMP fundamentals)
     Used to backfill training data before the DB has sufficient history,
     and to always provide a minimum training window.

Rolling window logic
--------------------
  train_years  : how many years of data to train on (default 3)
  val_months   : final N months of train window used for early-stopping only
  gap_days     : gap between train end and today (avoid lookahead, default 63 = 1 quarter)

  Example today=2027-03-01, train_years=3, gap_days=63:
    train : 2024-01-01 → 2027-01-07  (≈3 years)
    val   : 2026-07-01 → 2027-01-07  (last 6 months of train)
    live  : model scores stocks from 2027-01-07 onward

Model versioning
----------------
  Each retrain saves:
    output/models/live_model_YYYYMMDD.txt       ← the new model
    output/models/live_model_YYYYMMDD_meta.json ← metrics + window used
    output/models/live_model_latest.txt         ← symlink / copy of latest

  v31_live_trading.py reads output/models/live_model_latest.txt if it
  exists, otherwise falls back to honest_2006_2014_model.txt.

Minimum data requirement
------------------------
  MIN_TRAIN_SAMPLES = 2000 (roughly 2 years × 40 rebalance dates × 25 stocks/date)
  If the DB has fewer samples, the script backfills from historical data.

Usage
-----
  python retrain_model.py                      # auto rolling window
  python retrain_model.py --train-years 5      # use 5-year window
  python retrain_model.py --from-db-only       # only use MarketDatabase rows
  python retrain_model.py --dry-run            # print what would happen, no save
  python retrain_model.py --eval               # also run quick backtest after retrain
"""

import sys
import os
import json
import shutil
import argparse
import logging
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR           = 'output/models'
LATEST_MODEL_PATH   = os.path.join(MODEL_DIR, 'live_model_latest.txt')
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, 'honest_2006_2014_model.txt')
HIST_DATA_DIR       = 'sp500_data/stock_data_1990_2024'

# ── retrain parameters ────────────────────────────────────────────────────────
DEFAULT_TRAIN_YEARS  = 3       # rolling window length
DEFAULT_VAL_MONTHS   = 6       # tail of train window used for early stopping
DEFAULT_GAP_DAYS     = 63      # gap between train end and today (1 quarter)
MIN_TRAIN_SAMPLES    = 2_000   # minimum rows before we trust the model
N_FEATURES           = 60      # top-N features to select
LOOK_AHEAD_DAYS      = 63      # 63-day forward return = quarterly alignment

ANALYST_FEATURES     = ['analyst_eps_revision', 'analyst_rev_revision']


# ── helpers ───────────────────────────────────────────────────────────────────

def _header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def _date_window(train_years: int, val_months: int, gap_days: int):
    """Return (train_start, train_end, val_start) as ISO strings."""
    today      = date.today()
    train_end  = today - timedelta(days=gap_days)
    train_start = date(train_end.year - train_years, train_end.month, train_end.day)
    val_start   = date(train_end.year, train_end.month, train_end.day) - \
                  timedelta(days=val_months * 30)
    return (
        train_start.isoformat(),
        train_end.isoformat(),
        val_start.isoformat(),
    )


# ── source 1: MarketDatabase ──────────────────────────────────────────────────

def _load_from_db(train_start: str, train_end: str) -> tuple:
    """
    Pull feature rows from MarketDatabase for [train_start, train_end].
    Returns (features_dict, returns_dict) in the same format as build_dataset().

    features_dict : {date: {ticker: {feature_name: value}}}
    returns_dict  : {date: {ticker: forward_return_pct}}

    Forward returns are computed from daily_prices:
      fwd_return = (close[date+63] / close[date] - 1) * 100
    """
    from src.data.market_db import MarketDatabase
    db = MarketDatabase()

    _header("SOURCE 1: MarketDatabase")
    stats = db.get_snapshot_stats()
    print(f"  DB rows — features: {stats['daily_features']:,}  "
          f"prices: {stats['daily_prices']:,}  "
          f"date range: {stats['date_range']}")

    df_feat = db.get_training_data(train_start, train_end)
    if df_feat.empty:
        print("  No rows in DB for this window.")
        return {}, {}

    print(f"  Loaded {len(df_feat):,} feature rows from DB")

    # Load prices for forward-return computation
    conn = db._conn
    df_prices = pd.read_sql_query(
        "SELECT date, ticker, close FROM daily_prices "
        "WHERE date >= ? ORDER BY date",
        conn,
        params=(train_start,),
        parse_dates=['date'],
    )
    if df_prices.empty:
        print("  No price rows in DB.")
        return {}, {}

    # Build price lookup: {ticker: Series indexed by date}
    price_lookup = {}
    for ticker, grp in df_prices.groupby('ticker'):
        price_lookup[ticker] = grp.set_index('date')['close'].sort_index()

    # Identify quarterly rebalance dates present in the DB
    df_feat['date'] = pd.to_datetime(df_feat['date'])
    all_dates = sorted(df_feat['date'].unique())
    rebal_dates = [
        d for d in all_dates
        if d.month in (1, 4, 7, 10) and 7 <= d.day <= 15
    ]
    print(f"  Quarterly rebalance dates in DB: {len(rebal_dates)}")

    feature_cols = [c for c in df_feat.columns
                    if c not in ('date', 'ticker') and c not in ANALYST_FEATURES]

    features_dict: dict = {}
    returns_dict:  dict = {}
    total = 0

    for rebal_date in rebal_dates:
        rows = df_feat[df_feat['date'] == rebal_date]
        date_feat = {}
        date_ret  = {}

        for _, row in rows.iterrows():
            ticker = row['ticker']
            if ticker not in price_lookup:
                continue

            prices = price_lookup[ticker]
            future = prices[prices.index > rebal_date]
            if len(future) < 63:
                continue

            cur_price    = prices[prices.index <= rebal_date].iloc[-1] \
                           if not prices[prices.index <= rebal_date].empty else None
            if cur_price is None or cur_price == 0:
                continue

            fwd_price  = float(future.iloc[62])
            fwd_return = (fwd_price / float(cur_price) - 1) * 100

            if np.isnan(fwd_return) or np.isinf(fwd_return):
                continue

            feat = {c: float(row[c]) if pd.notna(row[c]) else 0.0
                    for c in feature_cols}
            date_feat[ticker] = feat
            date_ret[ticker]  = fwd_return
            total += 1

        if date_feat:
            features_dict[rebal_date] = date_feat
            returns_dict[rebal_date]  = date_ret

    print(f"  DB samples with forward returns: {total:,} "
          f"across {len(features_dict)} dates")
    return features_dict, returns_dict


# ── source 2: historical backtest data ───────────────────────────────────────

def _load_from_history(train_start: str, train_end: str) -> tuple:
    """
    Build feature+return dataset from the historical CSV price files and
    FMP fundamentals, using the same build_dataset() function as the
    original training pipeline.
    """
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor
    from backtest_honest_2015_2024 import build_dataset

    _header("SOURCE 2: Historical data (price CSVs + FMP fundamentals)")

    bot = PortfolioRotationBot(
        data_dir=HIST_DATA_DIR,
        initial_capital=100_000,
    )
    bot.load_all_stocks()
    print(f"  {len(bot.stocks_data)} stocks loaded from {HIST_DATA_DIR}")

    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    fe = MLFeatureExtractor()

    features_dict, returns_dict = build_dataset(
        bot, fa_loader, fe,
        train_start, train_end,
        label='ROLLING_TRAIN',
    )
    return features_dict, returns_dict, bot


# ── merge two sources ─────────────────────────────────────────────────────────

def _merge(fd1: dict, rd1: dict, fd2: dict, rd2: dict) -> tuple:
    """Merge two (features_dict, returns_dict) pairs. fd1 wins on date conflict."""
    merged_f = {**fd2, **fd1}   # fd1 (DB) overwrites historical on same date
    merged_r = {**rd2, **rd1}
    return merged_f, merged_r


# ── flatten to numpy arrays ───────────────────────────────────────────────────

def _flatten(features_dict: dict, returns_dict: dict,
             feature_names: list = None) -> tuple:
    """Same logic as backtest_honest_2015_2024.flatten()."""
    X, y = [], []
    fn = feature_names
    drop_set = set(ANALYST_FEATURES)

    for date_key, tickers in features_dict.items():
        for ticker, feat in tickers.items():
            if fn is None:
                fn = sorted(k for k in feat.keys() if k not in drop_set)
            try:
                vals = [float(feat.get(f, 0) or 0) for f in fn]
            except (TypeError, ValueError):
                continue
            if any(np.isnan(v) or np.isinf(v) for v in vals):
                continue
            ret = returns_dict[date_key].get(ticker)
            if ret is None or np.isnan(ret) or np.isinf(ret):
                continue
            X.append(vals)
            y.append(float(ret))

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            fn or [])


# ── train ─────────────────────────────────────────────────────────────────────

def _train(X_train, y_train, X_val, y_val, feature_names):
    from src.ml.stock_ranker import MLStockRanker
    ranker = MLStockRanker(n_features_to_select=N_FEATURES,
                           look_ahead_days=LOOK_AHEAD_DAYS)
    metrics = ranker.train(X_train, y_train, X_val, y_val, feature_names)
    return ranker, metrics


# ── save model + metadata ─────────────────────────────────────────────────────

def _save(ranker, metrics, train_start, train_end, val_start,
          n_train, n_val, sources, dry_run: bool):
    today_str  = date.today().strftime('%Y%m%d')
    model_path = os.path.join(MODEL_DIR, f'live_model_{today_str}.txt')
    meta_path  = model_path.replace('.txt', '_meta.json')

    meta = {
        'trained_on':    date.today().isoformat(),
        'train_start':   train_start,
        'train_end':     train_end,
        'val_start':     val_start,
        'n_train':       n_train,
        'n_val':         n_val,
        'sources':       sources,
        'val_corr':      metrics.get('val_corr'),
        'val_rmse':      metrics.get('val_rmse'),
        'train_rmse':    metrics.get('train_rmse'),
        'overfit_ratio': metrics.get('overfitting_ratio'),
        'n_features':    metrics.get('n_features_used'),
        'best_iter':     metrics.get('best_iteration'),
    }

    if dry_run:
        print(f"\n  [DRY RUN] Would save model → {model_path}")
        print(f"  [DRY RUN] Metadata: {json.dumps(meta, indent=4)}")
        return None

    os.makedirs(MODEL_DIR, exist_ok=True)
    ranker.save_model(model_path)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Copy to live_model_latest.txt (atomic replacement)
    tmp = LATEST_MODEL_PATH + '.tmp'
    shutil.copy2(model_path, tmp)
    os.replace(tmp, LATEST_MODEL_PATH)
    # Copy metadata too
    shutil.copy2(meta_path,
                 LATEST_MODEL_PATH.replace('.txt', '_meta.json'))

    print(f"\n  Model saved  : {model_path}")
    print(f"  Metadata     : {meta_path}")
    print(f"  Latest link  : {LATEST_MODEL_PATH}")
    return model_path


# ── optional quick eval ───────────────────────────────────────────────────────

def _quick_eval(ranker, train_end: str):
    """
    Run a short backtest on the 90 days immediately after train_end
    to get a quick out-of-sample sanity check.
    Prints: hit-rate (fraction of top-5 predictions that beat SPY).
    Does NOT run a full strategy backtest (too slow for a routine retrain).
    """
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor

    _header("QUICK EVAL: 90-day out-of-sample hit rate")

    try:
        bot = PortfolioRotationBot(data_dir=HIST_DATA_DIR, initial_capital=100_000)
        bot.load_all_stocks()
        fa_loader = HistoricalFADataLoader()
        fa_loader.load_all()
        fe = MLFeatureExtractor()

        eval_date = pd.Timestamp(train_end)
        # Find one rebalance date just after train_end
        all_dates = bot.stocks_data['SPY'].index
        future    = all_dates[all_dates > eval_date]
        if future.empty:
            print("  No dates after train_end in historical data — skipping eval.")
            return

        eval_ts = future[0]
        scores  = {}
        for ticker in list(bot.stocks_data.keys())[:100]:   # sample 100 tickers
            df_h = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= eval_ts]
            if len(df_h) < 120:
                continue
            fa = fa_loader.get_fa_data_at_date(ticker, eval_ts.isoformat()[:10])
            feat = fe.extract_features(ticker, eval_ts, bot, fa, fa_loader=fa_loader)
            if feat is None:
                continue
            try:
                scores[ticker] = ranker.predict(feat)
            except Exception:
                pass

        if not scores:
            print("  Could not score any tickers — skipping eval.")
            return

        top5 = sorted(scores, key=scores.get, reverse=True)[:5]

        # Check 63-day forward return for top-5 vs SPY
        spy_f = bot.stocks_data['SPY'][bot.stocks_data['SPY'].index > eval_ts]
        if len(spy_f) < 63:
            print("  Not enough future data for eval.")
            return
        spy_ret = (spy_f.iloc[62]['close'] / bot.stocks_data['SPY'][
            bot.stocks_data['SPY'].index <= eval_ts].iloc[-1]['close'] - 1) * 100

        hits = 0
        print(f"\n  Eval date: {eval_ts.date()}  SPY +63d: {spy_ret:+.2f}%")
        print(f"  {'Ticker':<8} {'ML Score':>10} {'Fwd Ret':>9} {'Beat SPY':>9}")
        print(f"  {'-'*40}")
        for t in top5:
            df_h = bot.stocks_data[t][bot.stocks_data[t].index <= eval_ts]
            df_f = bot.stocks_data[t][bot.stocks_data[t].index > eval_ts]
            if len(df_f) < 63:
                continue
            fwd = (df_f.iloc[62]['close'] / df_h.iloc[-1]['close'] - 1) * 100
            beat = fwd > spy_ret
            hits += int(beat)
            print(f"  {t:<8} {scores[t]:>+10.3f} {fwd:>+8.2f}% {'YES' if beat else 'no':>9}")

        print(f"\n  Hit rate: {hits}/{len(top5)} top-5 picks beat SPY in next quarter")

    except Exception as e:
        print(f"  Quick eval failed: {e}")


# ── update live trading script to use new model ───────────────────────────────

def _patch_live_script(model_path: str, dry_run: bool):
    """
    Update MODEL_PATH in v31_live_trading.py to point at the new model.
    Only patches if the file still references the old path.
    """
    live_script = 'src/live_trading/v31_live_trading.py'
    if not os.path.exists(live_script):
        return

    with open(live_script) as f:
        content = f.read()

    # Replace the MODEL_PATH constant to use live_model_latest
    new_line   = f"MODEL_PATH       = '{LATEST_MODEL_PATH}'"
    if LATEST_MODEL_PATH in content:
        print(f"  Live script already points to {LATEST_MODEL_PATH}")
        return

    import re
    patched = re.sub(
        r"MODEL_PATH\s*=\s*'[^']*'",
        new_line,
        content,
        count=1,
    )

    if patched == content:
        print("  Could not patch MODEL_PATH in live script — update manually.")
        return

    if dry_run:
        print(f"  [DRY RUN] Would patch {live_script}: MODEL_PATH → {LATEST_MODEL_PATH}")
        return

    with open(live_script, 'w') as f:
        f.write(patched)
    print(f"  Patched {live_script}: MODEL_PATH → {LATEST_MODEL_PATH}")


# ── main ─────────────────────────────────────────────────────────────────────

def run(train_years: int = DEFAULT_TRAIN_YEARS,
        val_months:  int = DEFAULT_VAL_MONTHS,
        gap_days:    int = DEFAULT_GAP_DAYS,
        from_db_only: bool = False,
        dry_run:      bool = False,
        do_eval:      bool = False):

    train_start, train_end, val_start = _date_window(train_years, val_months, gap_days)

    _header("ROLLING MODEL RETRAIN")
    print(f"  Train window : {train_start}  →  {train_end}  ({train_years} years)")
    print(f"  Val window   : {val_start}  →  {train_end}  ({val_months} months)")
    print(f"  Gap          : {gap_days} days (1 quarter)")
    print(f"  DB-only      : {from_db_only}")
    print(f"  Dry run      : {dry_run}")

    # ── collect training data ─────────────────────────────────────────────────
    db_feat, db_ret  = _load_from_db(train_start, train_end)
    db_samples = sum(len(v) for v in db_feat.values())
    print(f"\n  DB samples available: {db_samples:,}  (need {MIN_TRAIN_SAMPLES:,} minimum)")

    hist_bot = None
    if from_db_only:
        if db_samples < MIN_TRAIN_SAMPLES:
            print(f"\n  ERROR: only {db_samples} DB samples — "
                  f"need {MIN_TRAIN_SAMPLES}. "
                  f"Remove --from-db-only to backfill from historical data.")
            return
        feat_dict, ret_dict = db_feat, db_ret
        sources = ['market_db']
    else:
        hist_feat, hist_ret, hist_bot = _load_from_history(train_start, train_end)
        hist_samples = sum(len(v) for v in hist_feat.values())
        print(f"  Historical samples: {hist_samples:,}")

        feat_dict, ret_dict = _merge(db_feat, db_ret, hist_feat, hist_ret)
        total_samples = sum(len(v) for v in feat_dict.values())
        print(f"  Combined samples  : {total_samples:,}  "
              f"(DB={db_samples:,}  hist={hist_samples:,})")
        sources = ['market_db', 'historical'] if db_samples > 0 else ['historical']

    # ── split into train / val ────────────────────────────────────────────────
    train_fd = {d: v for d, v in feat_dict.items()
                if str(d)[:10] < val_start}
    val_fd   = {d: v for d, v in feat_dict.items()
                if val_start <= str(d)[:10] <= train_end}
    train_rd = {d: v for d, v in ret_dict.items()
                if str(d)[:10] < val_start}
    val_rd   = {d: v for d, v in ret_dict.items()
                if val_start <= str(d)[:10] <= train_end}

    X_train, y_train, feature_names = _flatten(train_fd, train_rd)
    X_val,   y_val,   _             = _flatten(val_fd,   val_rd,   feature_names)

    _header("DATASET SUMMARY")
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Val samples   : {len(X_val):,}")
    print(f"  Features      : {len(feature_names)}")

    if len(X_train) < MIN_TRAIN_SAMPLES:
        print(f"\n  WARNING: only {len(X_train)} train samples — "
              f"model quality may be poor.")
        print(f"  Keep running the live bot to accumulate more DB snapshots.")
        if len(X_train) < 200:
            print("  Too few samples to train. Aborting.")
            return

    if len(X_val) == 0:
        print("\n  WARNING: no validation samples — using train tail as val.")
        split = max(1, int(len(X_train) * 0.8))
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

    # ── train ─────────────────────────────────────────────────────────────────
    _header("TRAINING LIGHTGBM")
    ranker, metrics = _train(X_train, y_train, X_val, y_val, feature_names)

    print(f"\n  Train RMSE    : {metrics['train_rmse']:.4f}")
    print(f"  Val RMSE      : {metrics['val_rmse']:.4f}")
    print(f"  Val corr      : {metrics['val_corr']:.4f}")
    print(f"  Overfit ratio : {metrics['overfitting_ratio']:.3f}x  (< 1.15 = good)")
    print(f"  Best iteration: {metrics['best_iteration']}")
    print(f"  Features used : {metrics.get('n_features_used')}")

    imp = ranker.get_feature_importance(top_n=10)
    print(f"\n  Top 10 features:")
    print(imp.to_string(index=False))

    # ── quality gate ──────────────────────────────────────────────────────────
    if metrics['overfitting_ratio'] > 1.5:
        print("\n  WARNING: overfit ratio > 1.5 — model is overfitting.")
        print("  Consider increasing train_years or adding more DB data.")
    if metrics['val_corr'] < 0.05:
        print("\n  WARNING: val_corr < 0.05 — model has weak predictive signal.")
        if not dry_run:
            ans = input("  Save model anyway? [y/N]: ").strip().lower()
            if ans != 'y':
                print("  Aborted. Existing model unchanged.")
                return

    # ── save ──────────────────────────────────────────────────────────────────
    model_path = _save(ranker, metrics,
                       train_start, train_end, val_start,
                       len(X_train), len(X_val),
                       sources, dry_run)

    if model_path and not dry_run:
        _patch_live_script(model_path, dry_run)

    # ── optional eval ─────────────────────────────────────────────────────────
    if do_eval:
        _quick_eval(ranker, train_end)

    _header("DONE")
    if not dry_run and model_path:
        print(f"  New model active: {LATEST_MODEL_PATH}")
        print(f"  To roll back    : delete {LATEST_MODEL_PATH}")
        print(f"                    live bot will fall back to {FALLBACK_MODEL_PATH}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rolling model retrain')
    parser.add_argument('--train-years', type=int, default=DEFAULT_TRAIN_YEARS,
                        help=f'Years of training data (default {DEFAULT_TRAIN_YEARS})')
    parser.add_argument('--val-months', type=int, default=DEFAULT_VAL_MONTHS,
                        help=f'Validation window in months (default {DEFAULT_VAL_MONTHS})')
    parser.add_argument('--gap-days', type=int, default=DEFAULT_GAP_DAYS,
                        help=f'Gap between train end and today (default {DEFAULT_GAP_DAYS})')
    parser.add_argument('--from-db-only', action='store_true',
                        help='Use only MarketDatabase rows, no historical backfill')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would happen without saving anything')
    parser.add_argument('--eval', action='store_true',
                        help='Run 90-day out-of-sample hit-rate check after training')
    args = parser.parse_args()

    run(
        train_years   = args.train_years,
        val_months    = args.val_months,
        gap_days      = args.gap_days,
        from_db_only  = args.from_db_only,
        dry_run       = args.dry_run,
        do_eval       = args.eval,
    )
