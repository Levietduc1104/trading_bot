"""
V10 Test: Inverse Volatility Position Weighting

Compares position weighting methods:
1. V8 VIX + Equal Weighting (current best)
2. V10 VIX + Inverse Volatility Weighting (new)

Formula: weight_i = (momentum_score_i / volatility_i) / sum(score_j / vol_j)
- Higher momentum score → larger position
- Higher volatility → smaller position
- No hindsight bias (uses only historical data)
"""

import sys
sys.path.append('src/backtest')
import pandas as pd
import numpy as np
import logging
from portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_backtest(name, bot, **kwargs):
    """Run a single backtest with given parameters"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Running: {name}")
    logger.info("=" * 70)

    portfolio_df = bot.backtest_with_bear_protection(
        top_n=10,
        use_vix_regime=True,  # Always use VIX (best regime filter)
        **kwargs
    )

    # Calculate metrics from portfolio_df
    final_value = portfolio_df['value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

    # Max drawdown
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = {}
    for year in portfolio_df_copy['year'].unique():
        year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    negative_years = sum(1 for r in yearly_returns.values() if r < 0)

    return {
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'final_value': final_value,
        'negative_years': negative_years,
        'total_years': len(yearly_returns),
        'yearly_returns': yearly_returns
    }


def print_comparison(v8_results, v10_results):
    """Print comparison table of results"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS COMPARISON: Equal Weight vs Inverse Volatility Weight")
    logger.info("=" * 70)

    # Header
    logger.info(f"{'Metric':<30} {'V8 (Equal)':<15} {'V10 (InvVol)':<15} {'Change':<15}")
    logger.info("-" * 70)

    # Metrics
    metrics = [
        ('Annual Return', 'annual_return', '%'),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Sharpe Ratio', 'sharpe', ''),
        ('Final Value', 'final_value', '$'),
        ('Negative Years', 'negative_years', ''),
    ]

    for metric_name, metric_key, unit in metrics:
        v8_val = v8_results[metric_key]
        v10_val = v10_results[metric_key]
        diff = v10_val - v8_val

        if unit == '$':
            logger.info(f"{metric_name:<30} ${v8_val:>12,.0f}  ${v10_val:>12,.0f}  ${diff:>+12,.0f}")
        elif unit == '%':
            logger.info(f"{metric_name:<30} {v8_val:>13.1f}%  {v10_val:>13.1f}%  {diff:>+13.1f}%")
        elif metric_key == 'negative_years':
            logger.info(f"{metric_name:<30} {v8_val:>13}/{v8_results['total_years']:<3} {v10_val:>13}/{v10_results['total_years']:<3} {diff:>+13.0f}")
        else:
            logger.info(f"{metric_name:<30} {v8_val:>14.2f}  {v10_val:>14.2f}  {diff:>+14.2f}")

    logger.info("-" * 70)

    # Analysis
    logger.info("")
    logger.info("PERFORMANCE IMPROVEMENT ANALYSIS:")
    logger.info("-" * 70)

    ann_diff = v10_results['annual_return'] - v8_results['annual_return']
    dd_diff = v10_results['max_drawdown'] - v8_results['max_drawdown']
    sharpe_diff = v10_results['sharpe'] - v8_results['sharpe']
    value_diff = v10_results['final_value'] - v8_results['final_value']

    logger.info(f"\nAnnual Return:  {ann_diff:+.2f}% ({v10_results['annual_return']:.1f}% vs {v8_results['annual_return']:.1f}%)")
    if ann_diff >= 0.5:
        logger.info("  ✅ STRONG improvement in returns")
    elif ann_diff > 0:
        logger.info("  ⚠️  Modest improvement in returns")
    else:
        logger.info("  ❌ Returns declined")

    logger.info(f"\nMax Drawdown:   {dd_diff:+.2f}% ({v10_results['max_drawdown']:.1f}% vs {v8_results['max_drawdown']:.1f}%)")
    if dd_diff > 0:
        logger.info(f"  ⚠️  Drawdown WORSE by {dd_diff:.1f}%")
    elif dd_diff < -2:
        logger.info(f"  ✅ STRONG drawdown improvement by {abs(dd_diff):.1f}%")
    elif dd_diff < 0:
        logger.info(f"  ⚠️  Modest drawdown improvement by {abs(dd_diff):.1f}%")
    else:
        logger.info("  ➡️  Drawdown unchanged")

    logger.info(f"\nSharpe Ratio:   {sharpe_diff:+.3f} ({v10_results['sharpe']:.2f} vs {v8_results['sharpe']:.2f})")
    if sharpe_diff >= 0.10:
        logger.info("  ✅ STRONG risk-adjusted improvement")
    elif sharpe_diff > 0:
        logger.info("  ⚠️  Modest risk-adjusted improvement")
    else:
        logger.info("  ❌ Risk-adjusted returns declined")

    logger.info(f"\nFinal Value:    ${value_diff:+,.0f} (${v10_results['final_value']:,.0f} vs ${v8_results['final_value']:,.0f})")

    # Overall verdict
    logger.info("")
    logger.info("=" * 70)
    logger.info("VERDICT:")
    logger.info("=" * 70)

    if ann_diff >= 0.5 and sharpe_diff >= 0.05:
        logger.info("✅ SUCCESS! Inverse volatility weighting improves both returns AND risk-adjusted performance")
        logger.info("   RECOMMENDATION: Adopt V10 as new production strategy")
    elif sharpe_diff >= 0.10 and dd_diff < -2:
        logger.info("✅ SUCCESS! Better risk-adjusted returns and lower drawdown")
        logger.info("   RECOMMENDATION: Adopt V10 for better risk management")
    elif ann_diff > 0 and sharpe_diff > 0:
        logger.info("⚠️  MODEST IMPROVEMENT. V10 is slightly better but not dramatically")
        logger.info("   RECOMMENDATION: Consider adopting if you prefer smoother returns")
    else:
        logger.info("❌ NO IMPROVEMENT. Stick with V8 equal weighting")
        logger.info("   RECOMMENDATION: Keep V8 as production strategy")

    logger.info("")


def main():
    logger.info("=" * 70)
    logger.info("V10: INVERSE VOLATILITY POSITION WEIGHTING TEST")
    logger.info("=" * 70)
    logger.info("Testing position sizing methods:")
    logger.info("  V8: VIX Regime + Equal Weighting (current)")
    logger.info("  V10: VIX Regime + Inverse Volatility Weighting (new)")
    logger.info("")
    logger.info("Formula: weight_i = (score_i / volatility_i) / sum(score_j / vol_j)")
    logger.info("  - Higher momentum score → larger position")
    logger.info("  - Higher volatility → smaller position")
    logger.info("")

    # Initialize bot
    logger.info("Loading data...")
    bot = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
    bot.prepare_data()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Test 1: V8 Equal Weighting (baseline)
    v8_results = run_backtest(
        "V8: VIX Regime + Equal Weighting (Baseline)",
        bot,
        use_inverse_vol_weighting=False
    )

    # Test 2: V10 Inverse Volatility Weighting
    v10_results = run_backtest(
        "V10: VIX Regime + Inverse Volatility Weighting (NEW)",
        bot,
        use_inverse_vol_weighting=True
    )

    # Print comparison
    print_comparison(v8_results, v10_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
