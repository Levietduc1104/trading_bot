"""
Test 4: Risk Filter Loosening
Tests three variants against the current improved baseline:
  A. Looser VIX reserves only
  B. Higher DD multiplier floor only
  C. Higher regime floor only
  D. All three combined
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


def patch_strategy(strategy, vix_table=None, dd_table=None, regime_floor=None):
    """Patch risk filter methods on the strategy instance."""
    import types

    if vix_table is not None:
        def new_vix_reserve(self, vix):
            for threshold, reserve in vix_table:
                if vix < threshold:
                    return reserve
            return vix_table[-1][1]
        strategy.get_vix_cash_reserve = types.MethodType(new_vix_reserve, strategy)

    if dd_table is not None:
        def new_dd_multiplier(self, portfolio_df):
            if portfolio_df is None or len(portfolio_df) < 2:
                return 1.0
            peak = portfolio_df['value'].cummax().iloc[-1]
            current = portfolio_df['value'].iloc[-1]
            dd = (current - peak) / peak
            for threshold, mult in dd_table:
                if dd > threshold:
                    return mult
            return dd_table[-1][1]
        strategy.get_portfolio_dd_multiplier = types.MethodType(new_dd_multiplier, strategy)

    if regime_floor is not None:
        import src.strategies.v31_ml_strategy as v31mod
        orig_regime = strategy.get_regime_multiplier.__func__
        def new_regime(self, date):
            val = orig_regime(self, date)
            return max(val, regime_floor)
        strategy.get_regime_multiplier = types.MethodType(new_regime, strategy)


# Current (baseline) risk tables
CURRENT_VIX = [
    (15, 0.05), (20, 0.10), (25, 0.20), (30, 0.35), (35, 0.50), (999, 0.70)
]
CURRENT_DD = [
    (-0.05, 1.0), (-0.10, 0.90), (-0.15, 0.75), (-0.20, 0.50), (-999, 0.25)
]

# Looser VIX: cut reserves roughly in half
LOOSER_VIX = [
    (15, 0.03), (20, 0.05), (25, 0.10), (30, 0.15), (35, 0.25), (999, 0.40)
]

# Higher DD floor: reduce less during drawdowns
LOOSER_DD = [
    (-0.05, 1.0), (-0.10, 0.95), (-0.15, 0.85), (-0.20, 0.70), (-999, 0.50)
]


def run_experiment(bot, fa_loader, ml_ranker, label,
                   vix_table=None, dd_table=None, regime_floor=None):
    from src.strategies.v31_ml_strategy import V31MLStrategy
    strategy = V31MLStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='alpaca',
        enable_covered_calls=True,
        ml_model=ml_ranker,
        n_features_to_select=50,
        fa_loader=fa_loader,
    )
    patch_strategy(strategy, vix_table=vix_table, dd_table=dd_table, regime_floor=regime_floor)
    results = strategy.run_backtest(start_year=2015, end_year=2024)
    return results, strategy.total_costs


def run():
    print("\n" + "=" * 75)
    print("TEST 4: Risk Filter Loosening")
    print("Baseline uses blend=0.65 + Alpaca (current best)")
    print("=" * 75)

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
        # label,                          vix,         dd,         regime_floor
        ('Baseline (current filters)',    CURRENT_VIX, CURRENT_DD, 0.20),
        ('A. Looser VIX only',            LOOSER_VIX,  CURRENT_DD, 0.20),
        ('B. Looser DD only',             CURRENT_VIX, LOOSER_DD,  0.20),
        ('C. Higher regime floor (0.40)', CURRENT_VIX, CURRENT_DD, 0.40),
        ('D. All three combined',         LOOSER_VIX,  LOOSER_DD,  0.40),
    ]

    all_metrics = []
    for label, vix, dd, rf in experiments:
        print(f"Running: {label}...")
        r, costs = run_experiment(bot, fa_loader, ml_ranker, label, vix, dd, rf)
        m = calc_metrics(r, label)
        m['costs'] = costs
        all_metrics.append(m)
        print(f"  Annual: {m['annual']:.2f}%  Sharpe: {m['sharpe']:.3f}  "
              f"MaxDD: {m['max_dd']:.2f}%  Final: ${m['final']:,.0f}")

    # Print table
    base = all_metrics[0]
    print("\n" + "=" * 80)
    print(f"{'Variant':<32} {'Annual%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Final$':>12} {'vs Base':>8}")
    print("-" * 80)
    for m in all_metrics:
        delta = m['annual'] - base['annual']
        sign  = '+' if delta >= 0 else ''
        tag   = f"{sign}{delta:.2f}%" if m['name'] != base['name'] else '(base)'
        print(f"{m['name']:<32} {m['annual']:>7.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_dd']:>7.2f}% ${m['final']:>11,.0f} {tag:>8}")
    print("=" * 80)

    # Verdict
    best = max(all_metrics[1:], key=lambda x: x['sharpe'])
    print(f"\nBest risk-adjusted variant: {best['name']}")
    print(f"  Annual: {best['annual']:.2f}% vs baseline {base['annual']:.2f}% "
          f"({'+' if best['annual']>base['annual'] else ''}{best['annual']-base['annual']:.2f}%)")
    print(f"  Sharpe: {best['sharpe']:.3f} vs {base['sharpe']:.3f}")
    print(f"  Max DD: {best['max_dd']:.2f}% vs {base['max_dd']:.2f}%")

    os.makedirs('output', exist_ok=True)
    pd.DataFrame(all_metrics).to_csv('output/test4_risk_filters.csv', index=False)
    print("\nSaved: output/test4_risk_filters.csv")


if __name__ == '__main__':
    run()
