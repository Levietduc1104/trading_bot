"""
Test 2: Position Count  3 megacaps + 2 momentum (baseline) vs 8 megacaps + 7 momentum (new)
Uses pre-trained model — no retraining needed.
More stocks = more diversification, more ML picks utilized.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.WARNING)

INITIAL_CAPITAL = 100_000
MODEL_PATH = 'output/models/honest_2006_2014_model.txt'


def calc_metrics(results, name):
    final  = results['value'].iloc[-1]
    years  = (results.index[-1] - results.index[0]).days / 365.25
    annual = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    cummax = results['value'].cummax()
    dd     = ((results['value'] - cummax) / cummax * 100).min()
    rets   = results['value'].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    return {'name': name, 'annual': annual, 'sharpe': sharpe, 'max_dd': dd, 'final': final}


def run_with_config(bot, fa_loader, ml_ranker,
                    num_megacap, num_momentum, num_top_megacaps,
                    min_pos, max_pos, label):
    """Run backtest with custom position count config."""
    from src.strategies.v31_ml_strategy import V31MLStrategy
    from src.strategies.enhanced_position_sizing import EnhancedPositionSizer

    strategy = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
    )
    # Override position count config
    strategy.config['num_megacap']      = num_megacap
    strategy.config['num_momentum']     = num_momentum
    strategy.config['num_top_megacaps'] = num_top_megacaps

    # Override position sizer constraints to match new count
    strategy.position_sizer = EnhancedPositionSizer(config={
        'volatility_weight': 0.40,
        'momentum_weight': 0.30,
        'mean_reversion_weight': 0.20,
        'correlation_weight': 0.10,
        'vol_lookback': 20,
        'momentum_lookback': 60,
        'mean_reversion_lookback': 20,
        'min_position_size': min_pos,
        'max_position_size': max_pos,
    })

    results = strategy.run_backtest(start_year=2015, end_year=2024)
    return results, strategy.total_costs


def run():
    print("\n" + "=" * 70)
    print("TEST 2: Position Count  3+2 (baseline) vs 8+7 (new)")
    print("More stocks = more ML picks, better diversification")
    print("=" * 70)

    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.data.historical_fa_data_loader import HistoricalFADataLoader
    from src.ml.stock_ranker import MLStockRanker

    print("\nLoading data...")
    bot = PortfolioRotationBot(data_dir='sp500_data/stock_data_1990_2024',
                               initial_capital=INITIAL_CAPITAL)
    bot.load_all_stocks()
    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    ml_ranker = MLStockRanker(n_features_to_select=60, look_ahead_days=63)
    ml_ranker.load_model(MODEL_PATH)
    print("  Model loaded\n")

    experiments = [
        # label,               mega, mom, top_mega, min_pos, max_pos
        ('3+2 baseline (5)',   3,    2,   7,        0.08,    0.22),
        ('8+7 new (15)',       8,    7,   15,       0.05,    0.15),
    ]

    all_metrics = []
    for label, mega, mom, top_mega, min_pos, max_pos in experiments:
        print(f"Running {label}...")
        r, costs = run_with_config(
            bot, fa_loader, ml_ranker,
            mega, mom, top_mega, min_pos, max_pos, label
        )
        m = calc_metrics(r, label)
        m['costs'] = costs
        m['n_stocks'] = mega + mom
        all_metrics.append(m)
        print(f"  Annual: {m['annual']:.2f}%  Sharpe: {m['sharpe']:.3f}  "
              f"MaxDD: {m['max_dd']:.2f}%  Final: ${m['final']:,.0f}")

    b0, b1 = all_metrics
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'3+2 (5 stocks)':>18} {'8+7 (15 stocks)':>18} {'Change':>12}")
    print("-" * 70)
    rows = [
        ('Annual Return (%)', 'annual', '.2f'),
        ('Sharpe Ratio',      'sharpe', '.3f'),
        ('Max Drawdown (%)',  'max_dd', '.2f'),
        ('Final Value ($)',   'final',  ',.0f'),
        ('Total Costs ($)',   'costs',  ',.0f'),
    ]
    for label, key, fmt in rows:
        v0, v1 = b0[key], b1[key]
        diff = v1 - v0
        sign = '+' if diff > 0 else ''
        print(f"{label:<25} {v0:>18{fmt}} {v1:>18{fmt}} {sign}{diff:>11{fmt}}")
    print("=" * 70)

    delta = b1['annual'] - b0['annual']
    if delta > 1.0:
        verdict = f"IMPROVEMENT: +{delta:.2f}% annual — adopt 8+7 config"
    elif delta > 0:
        verdict = f"SMALL GAIN: +{delta:.2f}% — marginal"
    else:
        verdict = f"NO IMPROVEMENT: {delta:.2f}% — keep 3+2"
    print(f"\nVerdict: {verdict}")

    os.makedirs('output', exist_ok=True)
    pd.DataFrame(all_metrics).to_csv('output/test2_position_count.csv', index=False)
    print("Saved: output/test2_position_count.csv")


if __name__ == '__main__':
    run()
