"""
Walk-Forward Validation: V31 ML Strategy (with regime filter + ML confidence weights)
======================================================================================
Methodology:
  - Fixed train window: 2006-01-01 to (test_year - 1)-12-31
  - Test window:        test_year-01-01 to test_year-12-31
  - Roll test_year from 2013 to 2024 (12 rounds)

Each round:
  1. Build feature dataset from training window only
  2. Train LightGBM on training data (val = last 2 years of train)
  3. Run V31MLStrategy on test year (no future data leakage)
  4. Run V31Tier2 (Phase 1A) on same test year as baseline
  5. Compare metrics

All 12 test-year results are aggregated at the end.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000
TRAIN_FIXED_START = '2006-01-01'
ANALYST_FEATURES  = ['analyst_eps_revision', 'analyst_rev_revision']


# ── helpers ────────────────────────────────────────────────────────────────────

def build_dataset(bot, fa_loader, feature_extractor, start_date, end_date, label=''):
    """Extract features + 63-day forward returns for all quarterly rebalance dates."""
    if 'SPY' in bot.stocks_data:
        all_dates = bot.stocks_data['SPY'].index
    else:
        all_dates = list(bot.stocks_data.values())[0].index

    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

    rebalance_dates = []
    for d in all_dates:
        if d.month in [1, 4, 7, 10] and 7 <= d.day <= 15:
            if not rebalance_dates or (d.year != rebalance_dates[-1].year or
                                       d.month != rebalance_dates[-1].month):
                rebalance_dates.append(d)

    logger.info(f"  [{label}] {len(rebalance_dates)} rebalance dates")

    features_dict, returns_dict = {}, {}
    total = 0

    for i, date in enumerate(rebalance_dates):
        date_features, date_returns = {}, {}
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
                cur   = df_hist.iloc[-1]['close']
                fut   = df_future.iloc[62]['close']
                fwd   = (fut / cur - 1) * 100
                date_features[ticker] = features
                date_returns[ticker]  = fwd
                total += 1
        if date_features:
            features_dict[date] = date_features
            returns_dict[date]  = date_returns

    logger.info(f"  [{label}] {total} samples across {len(features_dict)} dates")
    return features_dict, returns_dict


def flatten(features_dict, returns_dict, feature_names=None, drop_features=None):
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


def calc_metrics(results, name, initial_capital=INITIAL_CAPITAL):
    v      = results['value']
    final  = v.iloc[-1]
    start  = v.iloc[0]
    years  = (v.index[-1] - v.index[0]).days / 365.25
    annual = ((final / start) ** (1 / years) - 1) * 100 if years > 0 else 0
    cummax = v.cummax()
    dd     = ((v - cummax) / cummax * 100).min()
    rets   = v.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    return dict(name=name, annual=annual, sharpe=sharpe, max_dd=dd,
                final=final, start=start)


# ── main ───────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION: V31 ML  (regime filter + ML confidence weights)")
    print(f"Train: {TRAIN_FIXED_START} → (test_year-1)  |  Test: each year 2013-2024")
    print("="*80 + "\n")

    # ── Load data once (shared across all rounds) ──────────────────────────────
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor

    logger.info("Loading price data...")
    bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024',
                               initial_capital=INITIAL_CAPITAL)
    bot.load_all_stocks()
    logger.info(f"  {len(bot.stocks_data)} stocks loaded")

    logger.info("Loading FA data...")
    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    feature_extractor = MLFeatureExtractor()

    rows = []          # per-year results
    eq_ml_parts = []   # equity curve segments for stitching
    eq_1a_parts = []

    test_years = list(range(2013, 2025))   # 12 rounds

    for test_year in test_years:
        train_end_str = f'{test_year - 1}-12-31'
        val_start_str = f'{test_year - 2}-01-01'   # last 2 years of train for early stopping
        test_start_str = f'{test_year}-01-01'
        test_end_str   = f'{test_year}-12-31'

        print("\n" + "="*70)
        print(f"ROUND: Train {TRAIN_FIXED_START}–{train_end_str}  |  Test {test_year}")
        print("="*70)

        # ── Build training data ─────────────────────────────────────────────
        train_feat, train_ret = build_dataset(
            bot, fa_loader, feature_extractor,
            TRAIN_FIXED_START, train_end_str, label=f'train→{test_year-1}'
        )
        val_feat, val_ret = build_dataset(
            bot, fa_loader, feature_extractor,
            val_start_str, train_end_str, label=f'val→{test_year-1}'
        )

        X_train, y_train, feature_names = flatten(train_feat, train_ret,
                                                   drop_features=ANALYST_FEATURES)
        X_val,   y_val,   _             = flatten(val_feat,   val_ret,   feature_names)

        if len(X_train) < 500:
            logger.warning(f"  Too few training samples ({len(X_train)}), skipping {test_year}")
            continue

        # ── Train ML ────────────────────────────────────────────────────────
        from src.ml.stock_ranker import MLStockRanker
        ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
        metrics_train = ml_ranker.train(X_train, y_train, X_val, y_val, feature_names)
        logger.info(f"  val_corr={metrics_train['val_corr']:.3f}  "
                    f"overfit={metrics_train['overfitting_ratio']:.2f}x")

        # ── Run V31 ML on test year ─────────────────────────────────────────
        from src.strategies.v31_ml_strategy import V31MLStrategy
        strategy_ml = V31MLStrategy(
            bot=bot,
            use_transaction_costs=True,
            broker='interactive_brokers',
            enable_covered_calls=True,
            ml_model=ml_ranker,
            n_features_to_select=50
        )
        # Run only the test year — start a year earlier so trailing stop / peak
        # tracking has some warmup, but we evaluate only from test_start
        results_ml_full = strategy_ml.run_backtest(
            start_year=test_year - 1, end_year=test_year
        )
        results_ml = results_ml_full[results_ml_full.index >= test_start_str]

        # ── Run Phase 1A baseline on test year ─────────────────────────────
        from src.strategies.v31_tier2_growth_scoring import V31Tier2GrowthScoringStrategy
        strategy_1a = V31Tier2GrowthScoringStrategy(
            bot=bot,
            use_transaction_costs=True,
            broker='interactive_brokers',
            enable_covered_calls=True,
            momentum_weight=0.50,
            growth_weight=0.50
        )
        results_1a_full = strategy_1a.run_backtest(
            start_year=test_year - 1, end_year=test_year
        )
        results_1a = results_1a_full[results_1a_full.index >= test_start_str]

        if len(results_ml) < 10 or len(results_1a) < 10:
            logger.warning(f"  Insufficient test data for {test_year}, skipping")
            continue

        # ── Metrics ─────────────────────────────────────────────────────────
        mml = calc_metrics(results_ml, f'ML-{test_year}')
        m1a = calc_metrics(results_1a, f'1A-{test_year}')

        improvement = mml['annual'] - m1a['annual']
        winner = 'ML' if improvement > 0.5 else ('1A' if improvement < -0.5 else 'tie')

        print(f"  {test_year}: ML={mml['annual']:+.1f}%  1A={m1a['annual']:+.1f}%  "
              f"diff={improvement:+.1f}%  dd_ml={mml['max_dd']:.1f}%  "
              f"Sharpe_ml={mml['sharpe']:.2f}  [{winner}]")

        rows.append(dict(
            test_year   = test_year,
            train_end   = test_year - 1,
            ml_annual   = mml['annual'],
            ml_max_dd   = mml['max_dd'],
            ml_sharpe   = mml['sharpe'],
            annual_1a   = m1a['annual'],
            dd_1a       = m1a['max_dd'],
            sharpe_1a   = m1a['sharpe'],
            improvement = improvement,
            winner      = winner,
            val_corr    = metrics_train['val_corr'],
            overfit     = metrics_train['overfitting_ratio'],
        ))

        eq_ml_parts.append(results_ml[['value']])
        eq_1a_parts.append(results_1a[['value']])

    # ── Aggregate summary ──────────────────────────────────────────────────────
    if not rows:
        print("No results collected.")
        return

    df = pd.DataFrame(rows)
    os.makedirs('output/walk_forward', exist_ok=True)
    df.to_csv('output/walk_forward/wf_v31_ml_results.csv', index=False)

    ml_wins  = (df['winner'] == 'ML').sum()
    ties     = (df['winner'] == 'tie').sum()
    a1_wins  = (df['winner'] == '1A').sum()
    n        = len(df)
    avg_imp  = df['improvement'].mean()
    avg_ml   = df['ml_annual'].mean()
    avg_1a   = df['annual_1a'].mean()
    avg_dd   = df['ml_max_dd'].mean()
    avg_sh   = df['ml_sharpe'].mean()

    print("\n" + "="*70)
    print("WALK-FORWARD SUMMARY  (12 out-of-sample years, 2013-2024)")
    print("="*70)
    print(f"\n{'Year':<6} {'ML':>8} {'1A':>8} {'Diff':>8}  {'Winner':<6}  {'Sharpe_ML':>10}  {'DD_ML':>8}")
    print("-"*65)
    for _, r in df.iterrows():
        print(f"  {int(r.test_year):<4}  {r.ml_annual:>7.1f}% {r.annual_1a:>7.1f}% "
              f"{r.improvement:>+7.1f}%  {r.winner:<6}  {r.ml_sharpe:>10.2f}  "
              f"{r.ml_max_dd:>7.1f}%")
    print("-"*65)
    print(f"  {'AVG':<4}  {avg_ml:>7.1f}% {avg_1a:>7.1f}% {avg_imp:>+7.1f}%")
    print(f"\n  ML wins: {ml_wins}/{n}   1A wins: {a1_wins}/{n}   ties: {ties}/{n}")
    print(f"  Avg ML annual: {avg_ml:.1f}%  |  Avg 1A annual: {avg_1a:.1f}%")
    print(f"  Avg ML Sharpe: {avg_sh:.2f}  |  Avg ML max DD: {avg_dd:.1f}%")
    print(f"\nResults saved to: output/walk_forward/wf_v31_ml_results.csv")
    print("="*70)

    return df


if __name__ == '__main__':
    run()
