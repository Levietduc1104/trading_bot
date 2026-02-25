"""
Test 3: Combined improvements so far
  Baseline : blend=0.40, broker=interactive_brokers
  Improved : blend=0.65, broker=alpaca (no commission, correct fees)
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


def run_strategy(bot, fa_loader, ml_ranker, broker, blend):
    from src.strategies.v31_ml_strategy import V31MLStrategy
    strategy = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker=broker,
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
    )
    _orig = strategy._apply_ml_confidence_weights.__func__
    strategy._apply_ml_confidence_weights = lambda ts, ba, ta, **kw: _orig(
        strategy, ts, ba, ta, blend=blend
    )
    results = strategy.run_backtest(start_year=2015, end_year=2024)
    return results, strategy.total_costs


def run():
    print("\n" + "=" * 70)
    print("TEST 3: Baseline vs All Improvements Combined")
    print("  Baseline : blend=0.40 + IBKR commissions ($0.0035/share)")
    print("  Improved : blend=0.65 + Alpaca ($0 commission, correct fees)")
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
    print("  Data loaded\n")

    experiments = [
        ('Baseline (blend=0.40, IBKR)', 'interactive_brokers', 0.40),
        ('Improved (blend=0.65, Alpaca)', 'alpaca', 0.65),
    ]

    all_metrics = []
    for label, broker, blend in experiments:
        print(f"Running {label}...")
        r, costs = run_strategy(bot, fa_loader, ml_ranker, broker, blend)
        m = calc_metrics(r, label)
        m['costs'] = costs
        all_metrics.append(m)
        print(f"  Annual: {m['annual']:.2f}%  Sharpe: {m['sharpe']:.3f}  "
              f"MaxDD: {m['max_dd']:.2f}%  Final: ${m['final']:,.0f}  Costs: ${costs:,.0f}")

    b0, b1 = all_metrics
    print("\n" + "=" * 70)
    print(f"{'Metric':<28} {'Baseline':>18} {'Improved':>18} {'Change':>10}")
    print("-" * 70)
    for label, key, fmt in [
        ('Annual Return (%)',  'annual', '.2f'),
        ('Sharpe Ratio',       'sharpe', '.3f'),
        ('Max Drawdown (%)',   'max_dd', '.2f'),
        ('Final Value ($)',    'final',  ',.0f'),
        ('Total Costs ($)',    'costs',  ',.0f'),
    ]:
        v0, v1 = b0[key], b1[key]
        diff = v1 - v0
        sign = '+' if diff > 0 else ''
        print(f"{label:<28} {v0:>18{fmt}} {v1:>18{fmt}} {sign}{diff:>9{fmt}}")
    print("=" * 70)

    delta = b1['annual'] - b0['annual']
    print(f"\nTotal improvement: {'+' if delta>0 else ''}{delta:.2f}% annual return")

    os.makedirs('output', exist_ok=True)
    pd.DataFrame(all_metrics).to_csv('output/test3_combined.csv', index=False)
    print("Saved: output/test3_combined.csv")


if __name__ == '__main__':
    run()
