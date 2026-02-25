"""
HONEST BACKTEST: Train 2006-2014 → Test 2015-2024
==================================================
Split:
  Train   : 2006-01-01 to 2014-12-31  (9 years, never touched in test)
  Val     : 2013-01-01 to 2014-12-31  (last 2 years of train, for early stopping only)
  TEST    : 2015-01-01 to 2024-12-31  (10 years, fully unseen)

The 2015-2024 window is NEVER touched during training or feature selection.
This is the single honest evaluation of the ML model.

Features (146 total):
  - 20 technical (momentum, volatility, RSI, etc.)
  - 111 fundamental (ratios + key metrics from FMP historical)
  - 15 premium (earnings beat %, analyst revisions, revenue growth,
                FCF margin, debt changes — from FMP Ultimate download)

Changes v2:
  - Target: 63-day forward returns (aligns with quarterly rebalancing)
  - Early stopping: 150 rounds (was 50 — model was underfitting)
  - Feature count: 60 (was 50 — allow more premium features)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000
TRAIN_START = '2006-01-01'
TRAIN_END   = '2014-12-31'
VAL_START   = '2013-01-01'   # last 2 years of train — for early stopping only
VAL_END     = '2014-12-31'
TEST_START  = '2015-01-01'
TEST_END    = '2024-12-31'


def build_dataset(bot, fa_loader, feature_extractor, start_date, end_date, label=''):
    """Extract features + forward returns for all quarterly rebalance dates in [start, end]."""
    from src.ml.feature_extraction import MLFeatureExtractor

    logger.info(f"Building {label} dataset: {start_date} → {end_date}")

    if 'SPY' in bot.stocks_data:
        all_dates = bot.stocks_data['SPY'].index
    else:
        all_dates = list(bot.stocks_data.values())[0].index

    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

    # Quarterly rebalance dates: Jan/Apr/Jul/Oct, days 7-15
    rebalance_dates = []
    for d in all_dates:
        if d.month in [1, 4, 7, 10] and 7 <= d.day <= 15:
            if not rebalance_dates or (d.year != rebalance_dates[-1].year or
                                       d.month != rebalance_dates[-1].month):
                rebalance_dates.append(d)

    logger.info(f"  {len(rebalance_dates)} rebalance dates found")

    features_dict = {}
    returns_dict  = {}
    total = 0

    for i, date in enumerate(rebalance_dates):
        if i % 10 == 0:
            logger.info(f"  Processing {i}/{len(rebalance_dates)} dates...")

        date_features = {}
        date_returns  = {}

        for ticker in bot.stocks_data.keys():
            df_hist = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_hist) < 120:
                continue

            fa_data  = fa_loader.get_fa_data_at_date(ticker, date)
            features = feature_extractor.extract_features(ticker, date, bot, fa_data,
                                                          fa_loader=fa_loader)
            if features is None:
                continue

            df_future = bot.stocks_data[ticker][bot.stocks_data[ticker].index > date]
            if len(df_future) >= 63:
                cur_price    = df_hist.iloc[-1]['close']
                future_price = df_future.iloc[62]['close']
                fwd_return   = (future_price / cur_price - 1) * 100
                date_features[ticker] = features
                date_returns[ticker]  = fwd_return
                total += 1

        if date_features:
            features_dict[date] = date_features
            returns_dict[date]  = date_returns

    logger.info(f"  Done: {total} samples across {len(features_dict)} dates")
    return features_dict, returns_dict


def flatten(features_dict, returns_dict, feature_names=None, drop_features=None):
    """Convert nested {date: {ticker: ...}} dicts to X, y arrays."""
    X, y = [], []
    fn = feature_names
    drop_set = set(drop_features or [])

    for date, tickers in features_dict.items():
        for ticker, feat in tickers.items():
            if fn is None:
                fn = sorted(k for k in feat.keys() if k not in drop_set)
            try:
                vals = [float(feat.get(f, 0) or 0) for f in fn]
            except (TypeError, ValueError):
                continue
            if any(np.isnan(v) or np.isinf(v) for v in vals):
                continue
            ret = returns_dict[date].get(ticker)
            if ret is None or np.isnan(ret) or np.isinf(ret):
                continue
            X.append(vals)
            y.append(float(ret))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), fn


def run():
    print("\n" + "="*80)
    print("HONEST BACKTEST: TRAIN 2006-2014 | TEST 2015-2024")
    print("="*80 + "\n")

    # ── Load price data ────────────────────────────────────────────────────────
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    logger.info("Loading price data...")
    bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024',
                               initial_capital=INITIAL_CAPITAL)
    bot.load_all_stocks()
    logger.info(f"  {len(bot.stocks_data)} stocks loaded")

    # ── Load FA data ───────────────────────────────────────────────────────────
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    logger.info("Loading FA data (historical + premium)...")
    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    # ── Feature extractor ──────────────────────────────────────────────────────
    from src.ml.feature_extraction import MLFeatureExtractor
    feature_extractor = MLFeatureExtractor()
    logger.info(f"  Feature count: {feature_extractor.get_feature_count()}")

    # ── Build training dataset (2006-2014) ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Build training dataset (2006-2014)")
    print("="*60)
    train_feat, train_ret = build_dataset(bot, fa_loader, feature_extractor,
                                          TRAIN_START, TRAIN_END, label='TRAIN')

    # ── Build validation dataset (2013-2014) ──────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Build validation dataset (2013-2014, early stopping only)")
    print("="*60)
    val_feat, val_ret = build_dataset(bot, fa_loader, feature_extractor,
                                      VAL_START, VAL_END, label='VAL')

    # ── Flatten to arrays ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Prepare arrays")
    print("="*60)
    # Analyst estimates only exist from ~2019; zero during training → pure noise.
    # Drop them from training features so they don't consume feature slots.
    ANALYST_FEATURES = ['analyst_eps_revision', 'analyst_rev_revision']

    X_train, y_train, feature_names = flatten(train_feat, train_ret,
                                               drop_features=ANALYST_FEATURES)
    X_val,   y_val,   _             = flatten(val_feat,   val_ret,   feature_names)

    print(f"  Train samples : {len(X_train):,}")
    print(f"  Val samples   : {len(X_val):,}")
    print(f"  Features      : {len(feature_names)}")

    # Show breakdown of premium vs non-premium features
    premium_feats = [f for f in feature_names if f in {
        'earnings_beat_pct', 'revenue_surprise_pct', 'eps_beat_3q_avg', 'eps_trend_3q',
        'revenue_growth_yoy', 'revenue_growth_qoq', 'gross_margin_change',
        'net_income_growth_yoy', 'fcf_margin', 'fcf_growth_yoy',
        'debt_change_yoy', 'cash_change_yoy', 'capex_to_revenue'
    }]
    print(f"  Premium features present: {len(premium_feats)}/13 (analyst features excluded from training)")
    print(f"  Target: 63-day forward return (quarterly alignment)")

    # ── Train ML model ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Train LightGBM (2006-2014 train, 2013-2014 val)")
    print("="*60)
    from src.ml.stock_ranker import MLStockRanker
    ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
    metrics = ml_ranker.train(X_train, y_train, X_val, y_val, feature_names)

    print(f"\n  Train RMSE : {metrics['train_rmse']:.4f}")
    print(f"  Val RMSE   : {metrics['val_rmse']:.4f}")
    print(f"  Val corr   : {metrics['val_corr']:.4f}")
    print(f"  Overfit ratio: {metrics['overfitting_ratio']:.3f}x")
    print(f"  Best iteration: {metrics['best_iteration']}")

    # Show top features
    imp = ml_ranker.get_feature_importance(top_n=15)
    print("\n  Top 15 features:")
    print(imp.to_string(index=False))

    # Save model
    os.makedirs('output/models', exist_ok=True)
    model_path = 'output/models/honest_2006_2014_model.txt'
    ml_ranker.save_model(model_path)
    print(f"\n  Model saved to: {model_path}")

    # ── Backtest 2015-2024 ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Honest backtest 2015-2024 (FULLY UNSEEN)")
    print("="*60)

    from src.strategies.v31_ml_strategy import V31MLStrategy
    from src.strategies.v31_tier2_growth_scoring import V31Tier2GrowthScoringStrategy

    # Phase 1A baseline
    print("\nRunning Phase 1A baseline (2015-2024)...")
    strategy_1a = V31Tier2GrowthScoringStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='alpaca',
        enable_covered_calls=True,
        momentum_weight=0.50,
        growth_weight=0.50
    )
    results_1a = strategy_1a.run_backtest(start_year=2015, end_year=2024)
    print("  Phase 1A done")

    # Phase ML
    print("\nRunning Phase ML (2015-2024)...")
    strategy_ml = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='alpaca',
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50
    )
    results_ml = strategy_ml.run_backtest(start_year=2015, end_year=2024)
    print("  Phase ML done")

    # ── Metrics ────────────────────────────────────────────────────────────────
    def calc_metrics(results, name):
        final  = results['value'].iloc[-1]
        years  = (results.index[-1] - results.index[0]).days / 365.25
        annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
        cummax = results['value'].cummax()
        dd     = ((results['value'] - cummax) / cummax * 100).min()
        rets   = results['value'].pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        return {'name': name, 'annual': annual, 'sharpe': sharpe, 'max_dd': dd, 'final': final}

    m1a = calc_metrics(results_1a, 'Phase 1A')
    mml = calc_metrics(results_ml, 'Phase ML')

    imp_annual = mml['annual'] - m1a['annual']
    imp_sharpe = mml['sharpe'] - m1a['sharpe']

    print("\n" + "="*70)
    print("HONEST RESULTS: 2015-2024 (10 YEARS, FULLY UNSEEN)")
    print("="*70)
    print(f"\n{'Metric':<25} {'Phase 1A':>15} {'Phase ML':>15} {'Improvement':>15}")
    print("-"*70)
    print(f"{'Annual Return':<25} {m1a['annual']:>14.2f}% {mml['annual']:>14.2f}% {imp_annual:>+14.2f}%")
    print(f"{'Sharpe Ratio':<25} {m1a['sharpe']:>15.3f} {mml['sharpe']:>15.3f} {imp_sharpe:>+15.3f}")
    print(f"{'Max Drawdown':<25} {m1a['max_dd']:>14.2f}% {mml['max_dd']:>14.2f}%")
    print(f"{'Final Value':<25} ${m1a['final']:>14,.0f} ${mml['final']:>14,.0f}")
    print()

    if imp_annual > 2.0:
        verdict = "STRONG - ML adds meaningful alpha on truly unseen data"
    elif imp_annual > 0.5:
        verdict = "MODEST - ML provides small honest improvement"
    elif imp_annual > -0.5:
        verdict = "NEUTRAL - ML roughly matches Phase 1A"
    else:
        verdict = "NEGATIVE - ML underperforms Phase 1A on unseen data"
    print(f"Verdict: {verdict}")

    print("\nModel quality:")
    print(f"  Val RMSE / Train RMSE = {metrics['overfitting_ratio']:.3f}x  (< 1.15 = good)")
    print(f"  Val correlation       = {metrics['val_corr']:.4f}")

    # Save results
    os.makedirs('output', exist_ok=True)
    results_1a.to_csv('output/honest_1a_2015_2024.csv')
    results_ml.to_csv('output/honest_ml_2015_2024.csv')

    summary = pd.DataFrame([m1a, mml])
    summary.to_csv('output/honest_summary_2015_2024.csv', index=False)
    print("\nResults saved to output/honest_*_2015_2024.csv")
    print("="*70)

    return mml, m1a, metrics


if __name__ == '__main__':
    run()
