"""
Test 1: ML Blend 0.40 (baseline) vs 0.65 (new)
Uses pre-trained model — no retraining needed.
blend only affects how much weight ML scores get in position sizing.
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


def run_with_blend(bot, fa_loader, ml_ranker, blend_value: float):
    """Run backtest with a specific blend value by monkey-patching."""
    from src.strategies.v31_ml_strategy import V31MLStrategy

    strategy = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
    )

    # Patch _apply_ml_confidence_weights to use our blend value
    _orig = strategy._apply_ml_confidence_weights.__func__
    strategy._apply_ml_confidence_weights = lambda ts, ba, ta, blend=blend_value: _orig(
        strategy, ts, ba, ta, blend=blend_value
    )

    results = strategy.run_backtest(start_year=2015, end_year=2024)
    return results, strategy.total_costs


def run():
    print("\n" + "=" * 70)
    print("TEST 1: ML Blend Weight  0.40 (baseline) vs 0.65 (new)")
    print("Higher blend = ML scores drive more of the position sizing")
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

    experiments = [('Blend 0.40 (baseline)', 0.40), ('Blend 0.65 (new)', 0.65)]
    all_metrics = []

    for label, blend in experiments:
        print(f"Running {label}...")
        r, costs = run_with_blend(bot, fa_loader, ml_ranker, blend)
        m = calc_metrics(r, label)
        m['costs'] = costs
        all_metrics.append(m)
        print(f"  Annual: {m['annual']:.2f}%  Sharpe: {m['sharpe']:.3f}  "
              f"MaxDD: {m['max_dd']:.2f}%  Final: ${m['final']:,.0f}")

    # Print comparison table
    b0, b1 = all_metrics
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'Blend 0.40':>18} {'Blend 0.65':>18} {'Change':>12}")
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

    # Verdict
    delta = b1['annual'] - b0['annual']
    if delta > 1.0:
        verdict = f"IMPROVEMENT: +{delta:.2f}% annual — keep blend=0.65"
    elif delta > 0:
        verdict = f"SMALL GAIN: +{delta:.2f}% — marginal improvement"
    else:
        verdict = f"NO IMPROVEMENT: {delta:.2f}% — revert to 0.40"
    print(f"\nVerdict: {verdict}")

    # Save results
    os.makedirs('output', exist_ok=True)
    pd.DataFrame(all_metrics).to_csv('output/test1_blend_comparison.csv', index=False)
    print("Saved: output/test1_blend_comparison.csv")


if __name__ == '__main__':
    run()
