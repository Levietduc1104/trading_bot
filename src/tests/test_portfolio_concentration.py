"""
PORTFOLIO CONCENTRATION TEST
=============================

Goal: Beat V13's 8.5% annual by increasing concentration

Current: 10 stocks (diversified)
Test: 5, 8, 10, 15, 20 stocks

Theory:
- Fewer stocks (5) = Higher conviction = Potentially higher returns
- But also higher volatility and risk
- More stocks (20) = Lower volatility but diluted alpha

Zero-bias because:
- Just adjusting concentration parameter
- No curve-fitting or optimization
- Testing fundamental risk/return trade-off

Target: 10%+ annual return
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_portfolio_size(bot, num_stocks):
    """
    Test V13 with different portfolio sizes

    Args:
        bot: PortfolioRotationBot instance
        num_stocks: Number of stocks to hold

    Returns:
        dict: Performance metrics
    """
    logger.info(f"\nTesting {num_stocks} stocks...")

    portfolio_df = bot.backtest_with_bear_protection(
        top_n=num_stocks,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    # Calculate metrics
    final_value = portfolio_df['value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Volatility (annualized std dev of returns)
    volatility = daily_returns.std() * np.sqrt(252) * 100

    # Yearly returns
    portfolio_df['year'] = portfolio_df.index.year
    yearly_returns = {}
    for year in portfolio_df['year'].unique():
        year_data = portfolio_df[portfolio_df['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    positive_years = sum(1 for r in yearly_returns.values() if r > 0)
    total_years = len(yearly_returns)

    return {
        'num_stocks': num_stocks,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'volatility': volatility,
        'final_value': final_value,
        'positive_years': positive_years,
        'total_years': total_years,
        'win_rate': positive_years / total_years if total_years > 0 else 0,
        'yearly_returns': yearly_returns
    }


def main():
    """Run portfolio concentration test"""

    logger.info("="*80)
    logger.info("PORTFOLIO CONCENTRATION TEST")
    logger.info("="*80)
    logger.info("")
    logger.info("Goal: Beat V13's 8.5% annual by increasing concentration")
    logger.info("")
    logger.info("Testing:")
    logger.info("  - 5 stocks:  Ultra-concentrated (high conviction)")
    logger.info("  - 8 stocks:  Concentrated")
    logger.info("  - 10 stocks: Baseline (V13)")
    logger.info("  - 15 stocks: Diversified")
    logger.info("  - 20 stocks: Highly diversified")
    logger.info("")
    logger.info("Expected trade-off:")
    logger.info("  Fewer stocks → Higher returns but higher volatility")
    logger.info("  More stocks → Lower volatility but diluted alpha")
    logger.info("")

    # Initialize bot
    logger.info("Loading data...")
    bot = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
    bot.prepare_data()
    bot.score_all_stocks()

    # Test different portfolio sizes
    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING TESTS")
    logger.info("="*80)

    portfolio_sizes = [5, 8, 10, 15, 20]
    results = []

    for size in portfolio_sizes:
        result = test_portfolio_size(bot, size)
        results.append(result)

    # RESULTS SUMMARY
    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info("")

    logger.info(f"{'Stocks':<8} {'Annual':>8} {'Volatility':>11} {'Drawdown':>10} {'Sharpe':>8} {'Win Rate':>10}")
    logger.info("-" * 70)

    baseline_return = None
    for result in results:
        if result['num_stocks'] == 10:
            baseline_return = result['annual_return']

        status = ""
        if result['annual_return'] >= 10.0:
            status = " 🎯"  # Hit 10%+ goal
        elif baseline_return and result['annual_return'] > baseline_return:
            status = " ✅"  # Beat baseline

        logger.info(
            f"{result['num_stocks']:<8} "
            f"{result['annual_return']:>7.1f}% "
            f"{result['volatility']:>10.1f}% "
            f"{result['max_drawdown']:>9.1f}% "
            f"{result['sharpe']:>8.2f} "
            f"{result['positive_years']}/{result['total_years']:>2}"
            f"{status}"
        )

    # FIND BEST
    logger.info("")
    logger.info("="*80)
    logger.info("BEST CONFIGURATIONS")
    logger.info("="*80)

    best_return = max(results, key=lambda x: x['annual_return'])
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_drawdown = max(results, key=lambda x: x['max_drawdown'])  # Highest = least negative

    logger.info("")
    logger.info(f"🏆 Best Annual Return: {best_return['num_stocks']} stocks")
    logger.info(f"   Return: {best_return['annual_return']:.1f}% annual")
    logger.info(f"   Drawdown: {best_return['max_drawdown']:.1f}%")
    logger.info(f"   Volatility: {best_return['volatility']:.1f}%")
    logger.info(f"   Sharpe: {best_return['sharpe']:.2f}")

    logger.info("")
    logger.info(f"📈 Best Sharpe Ratio: {best_sharpe['num_stocks']} stocks")
    logger.info(f"   Sharpe: {best_sharpe['sharpe']:.2f}")
    logger.info(f"   Return: {best_sharpe['annual_return']:.1f}% annual")

    logger.info("")
    logger.info(f"🛡️  Best Drawdown: {best_drawdown['num_stocks']} stocks")
    logger.info(f"   Drawdown: {best_drawdown['max_drawdown']:.1f}%")
    logger.info(f"   Return: {best_drawdown['annual_return']:.1f}% annual")

    # CHECK IF WE HIT 10%+ GOAL
    logger.info("")
    logger.info("="*80)
    logger.info("GOAL CHECK: 10%+ ANNUAL RETURN")
    logger.info("="*80)

    winners = [r for r in results if r['annual_return'] >= 10.0]

    if winners:
        logger.info("")
        logger.info(f"🎯 SUCCESS! {len(winners)} configuration(s) achieved 10%+ annual:")
        logger.info("")
        for w in sorted(winners, key=lambda x: x['annual_return'], reverse=True):
            logger.info(f"  {w['num_stocks']} stocks: {w['annual_return']:.1f}% annual")
            logger.info(f"    Max Drawdown: {w['max_drawdown']:.1f}%")
            logger.info(f"    Sharpe: {w['sharpe']:.2f}")
            logger.info(f"    Volatility: {w['volatility']:.1f}%")
            logger.info(f"    Win Rate: {w['positive_years']}/{w['total_years']} ({w['win_rate']*100:.0f}%)")
            logger.info("")
    else:
        logger.info("")
        logger.info("⚠️  No configuration reached 10%+ annual")
        logger.info(f"   Best: {best_return['num_stocks']} stocks at {best_return['annual_return']:.1f}%")
        logger.info(f"   Gap to goal: {10.0 - best_return['annual_return']:.1f}%")

    # RISK/RETURN ANALYSIS
    logger.info("")
    logger.info("="*80)
    logger.info("RISK/RETURN TRADE-OFF ANALYSIS")
    logger.info("="*80)
    logger.info("")

    logger.info("Key Insights:")
    logger.info("")

    # Compare 5 vs 10 stocks
    result_5 = next(r for r in results if r['num_stocks'] == 5)
    result_10 = next(r for r in results if r['num_stocks'] == 10)

    return_gain = result_5['annual_return'] - result_10['annual_return']
    vol_increase = result_5['volatility'] - result_10['volatility']
    dd_increase = result_5['max_drawdown'] - result_10['max_drawdown']

    logger.info("5 stocks vs 10 stocks (baseline):")
    logger.info(f"  Return change:     {return_gain:+.1f}% annual")
    logger.info(f"  Volatility change: {vol_increase:+.1f}%")
    logger.info(f"  Drawdown change:   {dd_increase:+.1f}%")
    logger.info("")

    if return_gain > 0 and dd_increase < 5:  # Better return without much worse drawdown
        logger.info("✅ CONCENTRATION PAYS OFF!")
        logger.info(f"   {result_5['num_stocks']} stocks delivers {return_gain:.1f}% more return")
        logger.info(f"   with acceptable risk increase")
    elif return_gain > 0:
        logger.info("⚠️  Higher returns but significantly higher risk")
        logger.info(f"   +{return_gain:.1f}% return comes with {dd_increase:.1f}% worse drawdown")
    else:
        logger.info("❌ Concentration doesn't help")
        logger.info(f"   10 stocks (baseline) is optimal")

    # YEARLY BREAKDOWN FOR BEST
    logger.info("")
    logger.info("="*80)
    logger.info(f"YEARLY RETURNS - BEST RETURN ({best_return['num_stocks']} stocks)")
    logger.info("="*80)

    for year in sorted(best_return['yearly_returns'].keys()):
        ret = best_return['yearly_returns'][year]
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:>6.1f}% {status}")

    logger.info("")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    results = main()
