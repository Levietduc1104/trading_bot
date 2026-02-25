"""
Rebalance Frequency Comparison: Daily vs Monthly vs Quarterly
==============================================================
Each frequency uses its own properly-trained ML model:
  - Daily     : model trained to predict  5-day forward returns
  - Monthly   : model trained to predict 21-day forward returns
  - Quarterly : model trained to predict 63-day forward returns (original)

Train: 2006-2014 | Test: 2015-2024 (fully unseen)
Uses only local data — no network calls.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 100_000
TRAIN_START = '2006-01-01'
TRAIN_END   = '2014-12-31'
VAL_START   = '2013-01-01'
VAL_END     = '2014-12-31'
TEST_START  = '2015-01-01'
TEST_END    = '2024-12-31'

ANALYST_FEATURES = ['analyst_eps_revision', 'analyst_rev_revision']

# Config per frequency
FREQ_CONFIG = {
    'daily':     {'look_ahead': 5,  'min_history': 30,  'label': 'Daily (5-day target)'},
    'monthly':   {'look_ahead': 21, 'min_history': 60,  'label': 'Monthly (21-day target)'},
    'quarterly': {'look_ahead': 63, 'min_history': 120, 'label': 'Quarterly (63-day target)'},
}


# ── Rebalance date generators ────────────────────────────────────────────────

def get_rebalance_dates(all_dates, frequency: str):
    """Return list of dates at which a rebalance should occur."""
    if frequency == 'quarterly':
        result = []
        for d in all_dates:
            if d.month in [1, 4, 7, 10] and 7 <= d.day <= 15:
                if not result or (d.year != result[-1].year or d.month != result[-1].month):
                    result.append(d)
        return result

    if frequency == 'monthly':
        result = []
        for d in all_dates:
            if not result or (d.year != result[-1].year or d.month != result[-1].month):
                result.append(d)
        return result

    if frequency == 'daily':
        return list(all_dates)

    raise ValueError(f"Unknown frequency: {frequency}")


def _should_rebalance(date, last_rebalance, frequency: str) -> bool:
    if last_rebalance is None:
        return True
    if frequency == 'daily':
        return True
    if frequency == 'monthly':
        return date.year != last_rebalance.year or date.month != last_rebalance.month
    if frequency == 'quarterly':
        return (date.month in [1, 4, 7, 10] and 7 <= date.day <= 15 and
                (date.year != last_rebalance.year or date.month != last_rebalance.month))
    raise ValueError(f"Unknown frequency: {frequency}")


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_dataset(bot, fa_loader, feature_extractor,
                  start_date, end_date, frequency, look_ahead, min_history, label=''):
    """Build features + forward returns at the correct rebalance dates."""
    logger.info(f"Building {label} dataset [{frequency}]: {start_date} → {end_date}")

    ref = 'SPY' if 'SPY' in bot.stocks_data else list(bot.stocks_data.keys())[0]
    all_dates = bot.stocks_data[ref].index
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

    rebalance_dates = get_rebalance_dates(all_dates, frequency)
    logger.info(f"  {len(rebalance_dates)} {frequency} rebalance dates in window")

    features_dict = {}
    returns_dict  = {}
    total = 0

    for i, date in enumerate(rebalance_dates):
        if i % max(1, len(rebalance_dates) // 10) == 0:
            logger.info(f"  Processing {i}/{len(rebalance_dates)} dates...")

        date_features = {}
        date_returns  = {}

        for ticker in bot.stocks_data:
            df_hist = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_hist) < min_history:
                continue

            fa_data  = fa_loader.get_fa_data_at_date(ticker, date)
            features = feature_extractor.extract_features(
                ticker, date, bot, fa_data, fa_loader=fa_loader
            )
            if features is None:
                continue

            df_future = bot.stocks_data[ticker][bot.stocks_data[ticker].index > date]
            if len(df_future) >= look_ahead:
                cur_price    = df_hist.iloc[-1]['close']
                future_price = df_future.iloc[look_ahead - 1]['close']
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


# ── Backtest runner ──────────────────────────────────────────────────────────

def run_backtest(bot, fa_loader, ml_ranker, frequency: str, start_year=2015, end_year=2024):
    """Run backtest for a given frequency using its trained model."""
    from src.strategies.v31_ml_strategy import V31MLStrategy

    strategy = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=False,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
    )

    ref = 'SPY' if 'SPY' in bot.stocks_data else list(bot.stocks_data.keys())[0]
    all_dates = bot.stocks_data[ref].index
    all_dates = all_dates[
        (all_dates >= f'{start_year}-01-01') & (all_dates <= f'{end_year}-12-31')
    ]

    portfolio_values = []
    holdings = {}
    cash = strategy.initial_capital
    last_rebalance = None
    strategy.trade_history = []
    strategy.total_costs = 0.0
    strategy.num_rebalances = 0

    for date in all_dates:
        # Trailing stops
        for ticker in list(holdings.keys()):
            df_h = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_h) > 0:
                cur_price = df_h.iloc[-1]['close']
                holdings[ticker]['peak_price'] = max(
                    holdings[ticker].get('peak_price', cur_price), cur_price
                )
                if strategy.check_trailing_stop(ticker, cur_price, holdings):
                    shares = holdings[ticker]['shares']
                    proceeds = shares * cur_price
                    cost = strategy.calculate_trade_cost(ticker, shares, cur_price, date)
                    cash += proceeds - cost
                    strategy.total_costs += cost
                    strategy.trade_history.append({
                        'date': date, 'ticker': ticker, 'action': 'SELL',
                        'reason': 'trailing_stop', 'shares': shares,
                        'price': cur_price, 'value': proceeds, 'cost': cost
                    })
                    del holdings[ticker]

        # Rebalance check
        if not _should_rebalance(date, last_rebalance, frequency):
            # Just value the portfolio
            stocks_val = sum(
                h['shares'] * bot.stocks_data[t][bot.stocks_data[t].index <= date].iloc[-1]['close']
                for t, h in holdings.items()
                if len(bot.stocks_data[t][bot.stocks_data[t].index <= date]) > 0
            )
            portfolio_values.append({
                'date': date, 'value': cash + stocks_val,
                'cash': cash, 'stocks_value': stocks_val
            })
            continue

        last_rebalance = date
        strategy.num_rebalances += 1

        # Liquidate
        for ticker in list(holdings.keys()):
            df_h = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_h) > 0:
                shares = holdings[ticker]['shares']
                price = df_h.iloc[-1]['close']
                proceeds = shares * price
                cost = strategy.calculate_trade_cost(ticker, shares, price, date)
                cash += proceeds - cost
                strategy.total_costs += cost
                strategy.trade_history.append({
                    'date': date, 'ticker': ticker, 'action': 'SELL',
                    'reason': 'rebalance', 'shares': shares,
                    'price': price, 'value': proceeds, 'cost': cost
                })
        holdings = {}

        # Risk filters
        vix = 20
        if bot.vix_data is not None:
            vix_at = bot.vix_data[bot.vix_data.index <= date]
            if len(vix_at) > 0:
                vix = vix_at.iloc[-1]['close']

        port_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
        invest = (cash
                  * (1 - strategy.get_vix_cash_reserve(vix))
                  * strategy.get_portfolio_dd_multiplier(port_df)
                  * strategy.get_regime_multiplier(date))

        # Score and select stocks
        megacaps = strategy.identify_megacaps(date, strategy.config['num_top_megacaps'])
        mega_scores = {t: s for t in megacaps if t in bot.stocks_data
                       for s in [strategy.score_stock_ml(t, date)] if s is not None}
        top_mega = sorted(mega_scores.items(), key=lambda x: x[1], reverse=True)[
            :strategy.config['num_megacap']]

        mom_scores = {}
        for t in bot.stocks_data:
            if t in megacaps:
                continue
            df_h = bot.stocks_data[t][bot.stocks_data[t].index <= date]
            if len(df_h) >= 100:
                try:
                    s = strategy.score_stock_ml(t, date)
                    if s is not None:
                        mom_scores[t] = s
                except Exception:
                    pass
        top_mom = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)[
            :strategy.config['num_momentum']]

        mega_amt = invest * strategy.config['megacap_allocation']
        mom_amt  = invest * (1 - strategy.config['megacap_allocation'])

        def _buy(picks, amount, reason):
            nonlocal cash
            if not picks:
                return
            base = strategy.position_sizer.calculate_enhanced_positions(picks, date, bot, amount)
            allocs = strategy._apply_ml_confidence_weights(picks, base, amount)
            for ticker, allocation in allocs.items():
                df_h = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_h) > 0:
                    price = df_h.iloc[-1]['close']
                    shares = allocation / price
                    cost = strategy.calculate_trade_cost(ticker, shares, price, date)
                    cash -= allocation + cost
                    strategy.total_costs += cost
                    holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                    strategy.trade_history.append({
                        'date': date, 'ticker': ticker, 'action': 'BUY',
                        'reason': reason, 'shares': shares,
                        'price': price, 'value': allocation, 'cost': cost
                    })

        _buy(top_mega, mega_amt, 'rebalance_megacap')
        _buy(top_mom,  mom_amt,  'rebalance_momentum')

        # Portfolio value
        stocks_val = sum(
            h['shares'] * bot.stocks_data[t][bot.stocks_data[t].index <= date].iloc[-1]['close']
            for t, h in holdings.items()
            if len(bot.stocks_data[t][bot.stocks_data[t].index <= date]) > 0
        )
        portfolio_values.append({
            'date': date, 'value': cash + stocks_val,
            'cash': cash, 'stocks_value': stocks_val
        })

    return (pd.DataFrame(portfolio_values).set_index('date'),
            len(strategy.trade_history),
            strategy.total_costs,
            strategy.num_rebalances)


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(results, name):
    final  = results['value'].iloc[-1]
    years  = (results.index[-1] - results.index[0]).days / 365.25
    annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    cummax = results['value'].cummax()
    dd     = ((results['value'] - cummax) / cummax * 100).min()
    rets   = results['value'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    vol    = rets.std() * np.sqrt(252) * 100
    return {'name': name, 'annual_return': annual, 'sharpe': sharpe,
            'max_drawdown': dd, 'volatility': vol, 'final_value': final}


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 80)
    print("REBALANCE FREQUENCY COMPARISON: DAILY vs MONTHLY vs QUARTERLY")
    print("Each frequency trains its own ML model with matching forward-return target")
    print("Train: 2006-2014  |  Test: 2015-2024 (fully unseen)")
    print("=" * 80 + "\n")

    # Load data (shared across all runs)
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.feature_extraction import MLFeatureExtractor
    from src.ml.stock_ranker import MLStockRanker

    print("Loading price data (468 stocks)...")
    bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024',
                               initial_capital=INITIAL_CAPITAL)
    bot.load_all_stocks()
    print(f"  {len(bot.stocks_data)} stocks loaded")

    print("Loading FA data (local only)...")
    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    feature_extractor = MLFeatureExtractor()

    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    all_results = {}
    all_meta    = {}

    frequencies = ['quarterly', 'monthly', 'daily']

    for freq in frequencies:
        cfg = FREQ_CONFIG[freq]
        look_ahead  = cfg['look_ahead']
        min_history = cfg['min_history']

        print(f"\n{'=' * 60}")
        print(f"FREQUENCY: {cfg['label']}")
        print(f"{'=' * 60}")

        model_path = f"output/models/model_{freq}.txt"

        # ── Train or load model ───────────────────────────────────────────
        if os.path.exists(model_path):
            print(f"  Loading cached model: {model_path}")
            ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=look_ahead)
            ml_ranker.load_model(model_path)
        else:
            print(f"  Building train dataset ({freq}, {look_ahead}-day target)...")
            train_feat, train_ret = build_dataset(
                bot, fa_loader, feature_extractor,
                TRAIN_START, TRAIN_END, freq, look_ahead, min_history, label='TRAIN'
            )
            print(f"  Building val dataset...")
            val_feat, val_ret = build_dataset(
                bot, fa_loader, feature_extractor,
                VAL_START, VAL_END, freq, look_ahead, min_history, label='VAL'
            )

            X_train, y_train, feature_names = flatten(
                train_feat, train_ret, drop_features=ANALYST_FEATURES
            )
            X_val, y_val, _ = flatten(val_feat, val_ret, feature_names)

            print(f"  Train: {len(X_train):,} samples | Val: {len(X_val):,} | Features: {len(feature_names)}")

            ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=look_ahead)
            metrics = ml_ranker.train(X_train, y_train, X_val, y_val, feature_names)

            print(f"  Train RMSE: {metrics['train_rmse']:.4f} | "
                  f"Val RMSE: {metrics['val_rmse']:.4f} | "
                  f"Val corr: {metrics['val_corr']:.4f} | "
                  f"Overfit: {metrics['overfitting_ratio']:.3f}x")

            ml_ranker.save_model(model_path)
            print(f"  Model saved: {model_path}")

        # ── Backtest 2015-2024 ────────────────────────────────────────────
        print(f"  Running backtest 2015-2024...")
        results, n_trades, total_costs, n_rebalances = run_backtest(
            bot, fa_loader, ml_ranker, freq, start_year=2015, end_year=2024
        )
        print(f"  Done — {n_rebalances} rebalances | {n_trades} trades | "
              f"${total_costs:,.0f} costs")

        m = calc_metrics(results, cfg['label'])
        m['n_trades']     = n_trades
        m['n_rebalances'] = n_rebalances
        m['total_costs']  = total_costs

        all_results[freq] = results
        all_meta[freq]    = m

        results.to_csv(f'output/rebal_{freq}_2015_2024.csv')

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS: 2015-2024 (10 YEARS, FULLY UNSEEN DATA)")
    print("=" * 80)

    w = 16
    print(f"\n{'Metric':<28}" + "".join(f"{f.capitalize():>{w}}" for f in frequencies))
    print("-" * (28 + w * len(frequencies)))

    rows = [
        ('Annual Return (%)',  'annual_return',  '.2f'),
        ('Sharpe Ratio',       'sharpe',         '.3f'),
        ('Max Drawdown (%)',   'max_drawdown',   '.2f'),
        ('Volatility (%)',     'volatility',     '.2f'),
        ('Final Value ($)',    'final_value',    ',.0f'),
        ('# Rebalances',       'n_rebalances',   ',d'),
        ('# Trades',           'n_trades',       ',d'),
        ('Total Costs ($)',    'total_costs',    ',.0f'),
    ]

    for label, key, fmt in rows:
        vals = [f"{all_meta[f][key]:{fmt}}" for f in frequencies]
        print(f"{label:<28}" + "".join(f"{v:>{w}}" for v in vals))

    print()
    print("Cost drag per rebalance:")
    for freq in frequencies:
        m = all_meta[freq]
        avg = m['total_costs'] / max(m['n_rebalances'], 1)
        print(f"  {freq.capitalize():<12}: ${avg:,.0f}/rebalance  "
              f"({m['n_rebalances']} rebalances, ${m['total_costs']:,.0f} total)")

    best_sharpe = max(frequencies, key=lambda f: all_meta[f]['sharpe'])
    best_return = max(frequencies, key=lambda f: all_meta[f]['annual_return'])
    print(f"\nBest risk-adjusted (Sharpe): {best_sharpe.upper()}")
    print(f"Best absolute return:        {best_return.upper()}")

    # Save summary
    pd.DataFrame([all_meta[f] for f in frequencies]).to_csv(
        'output/rebalance_frequency_comparison.csv', index=False
    )
    print("\nSummary saved: output/rebalance_frequency_comparison.csv")
    print("Portfolio CSVs: output/rebal_<freq>_2015_2024.csv")
    print("=" * 80)

    return all_results, all_meta


if __name__ == '__main__':
    run()
