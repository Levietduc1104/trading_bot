"""
COMPREHENSIVE STRATEGY COMPARISON: V13 → V14 → V15 → V16 → V17

Compares all recent strategy versions to show evolution and improvements.

V13: Momentum-Strength Weighting + Drawdown Control (BASELINE)
V14: V13 + Rebalance Bands (let winners run)
V15: V13 + Multi-Timeframe Momentum Ensemble
V16: V13 + Sector Rotation
V17: V16 + Correlation Diversification
"""

import sys
import logging
from datetime import datetime
sys.path.insert(0, 'src')

from backtest.portfolio_bot_demo import PortfolioRotationBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("           COMPREHENSIVE STRATEGY COMPARISON: V13 → V17           ")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # Initialize bot
    bot = PortfolioRotationBot(
        data_dir='sp500_data/daily',
        initial_capital=100000
    )

    # Load and prepare data
    logger.info("Loading data...")
    bot.prepare_data()
    logger.info(f"✅ Loaded {len(bot.stocks_data)} stocks")
    logger.info("")

    # Store results
    results = {}

    # ========================================================================
    # V13: MOMENTUM-STRENGTH WEIGHTING + DRAWDOWN CONTROL (BASELINE)
    # ========================================================================
    logger.info("="*80)
    logger.info("RUNNING V13: Momentum-Strength Weighting + Drawdown Control")
    logger.info("="*80)
    logger.info("Features:")
    logger.info("  - VIX-based regime detection")
    logger.info("  - Adaptive weighting (equal in calm, inverse-vol in stress)")
    logger.info("  - Momentum-strength position sizing (momentum/vol ratio)")
    logger.info("  - Portfolio-level drawdown control")
    logger.info("")

    portfolio_v13 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    results['V13'] = calculate_metrics(portfolio_v13, bot.initial_capital)
    logger.info(f"✅ V13: Annual {results['V13']['annual']:.1f}%, Sharpe {results['V13']['sharpe']:.2f}, Max DD {results['V13']['max_dd']:.1f}%")
    logger.info("")

    # ========================================================================
    # V14: V13 + REBALANCE BANDS
    # ========================================================================
    logger.info("="*80)
    logger.info("RUNNING V14: V13 + Rebalance Bands (Let Winners Run)")
    logger.info("="*80)
    logger.info("New Feature:")
    logger.info("  + Rebalance bands (only when drift > 25% or 90 days)")
    logger.info("")

    portfolio_v14 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_drawdown_control=True,
        use_rebalance_bands=True,
        rebalance_drift_threshold=0.25,
        trading_fee_pct=0.001
    )

    results['V14'] = calculate_metrics(portfolio_v14, bot.initial_capital)
    logger.info(f"✅ V14: Annual {results['V14']['annual']:.1f}%, Sharpe {results['V14']['sharpe']:.2f}, Max DD {results['V14']['max_dd']:.1f}%")
    logger.info("")

    # ========================================================================
    # V15: V13 + MULTI-TIMEFRAME MOMENTUM
    # ========================================================================
    logger.info("="*80)
    logger.info("RUNNING V15: V13 + Multi-Timeframe Momentum Ensemble")
    logger.info("="*80)
    logger.info("New Feature:")
    logger.info("  + Multi-timeframe momentum (3m, 6m, 9m, 12m weighted)")
    logger.info("")

    portfolio_v15 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_multi_timeframe_momentum=True,
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    results['V15'] = calculate_metrics(portfolio_v15, bot.initial_capital)
    logger.info(f"✅ V15: Annual {results['V15']['annual']:.1f}%, Sharpe {results['V15']['sharpe']:.2f}, Max DD {results['V15']['max_dd']:.1f}%")
    logger.info("")

    # ========================================================================
    # V16: V13 + SECTOR ROTATION
    # ========================================================================
    logger.info("="*80)
    logger.info("RUNNING V16: V13 + Sector Rotation")
    logger.info("="*80)
    logger.info("New Feature:")
    logger.info("  + Sector rotation (6-month sector momentum multiplier)")
    logger.info("")

    portfolio_v16 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_sector_rotation=True,
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    results['V16'] = calculate_metrics(portfolio_v16, bot.initial_capital)
    logger.info(f"✅ V16: Annual {results['V16']['annual']:.1f}%, Sharpe {results['V16']['sharpe']:.2f}, Max DD {results['V16']['max_dd']:.1f}%")
    logger.info("")

    # ========================================================================
    # V17: V16 + CORRELATION DIVERSIFICATION
    # ========================================================================
    logger.info("="*80)
    logger.info("RUNNING V17: V16 + Correlation Diversification")
    logger.info("="*80)
    logger.info("New Feature:")
    logger.info("  + Correlation-based diversification among candidates")
    logger.info("")

    portfolio_v17 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_sector_rotation=True,
        use_correlation_diversification=True,
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    results['V17'] = calculate_metrics(portfolio_v17, bot.initial_capital)
    logger.info(f"✅ V17: Annual {results['V17']['annual']:.1f}%, Sharpe {results['V17']['sharpe']:.2f}, Max DD {results['V17']['max_dd']:.1f}%")
    logger.info("")

    # ========================================================================
    # COMPREHENSIVE COMPARISON
    # ========================================================================
    logger.info("="*80)
    logger.info("                    COMPREHENSIVE COMPARISON                    ")
    logger.info("="*80)
    logger.info("")

    # Performance table
    logger.info("PERFORMANCE SUMMARY:")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<10} {'Annual':<10} {'Sharpe':<10} {'Max DD':<12} {'Neg Years':<12} {'Final Value':<15}")
    logger.info("-" * 80)

    for version in ['V13', 'V14', 'V15', 'V16', 'V17']:
        r = results[version]
        logger.info(f"{version:<10} {r['annual']:>7.1f}%   {r['sharpe']:>7.2f}    {r['max_dd']:>8.1f}%    {r['neg_years']:>3}/{r['total_years']:<6}  ${r['final']:>13,.0f}")

    logger.info("-" * 80)
    logger.info("")

    # Find winners
    best_annual = max(results.items(), key=lambda x: x[1]['annual'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    best_dd = max(results.items(), key=lambda x: x[1]['max_dd'])  # Less negative is better
    best_neg = min(results.items(), key=lambda x: x[1]['neg_years'])

    logger.info("🏆 CATEGORY WINNERS:")
    logger.info("")
    logger.info(f"  🥇 Best Annual Return:  {best_annual[0]} ({best_annual[1]['annual']:.1f}%)")
    logger.info(f"  🥇 Best Sharpe Ratio:   {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    logger.info(f"  🥇 Best Max Drawdown:   {best_dd[0]} ({best_dd[1]['max_dd']:.1f}%)")
    logger.info(f"  🥇 Fewest Neg Years:    {best_neg[0]} ({best_neg[1]['neg_years']}/{best_neg[1]['total_years']})")
    logger.info("")

    # Improvements vs V13 baseline
    logger.info("📊 IMPROVEMENTS VS V13 BASELINE:")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<10} {'Annual Δ':<12} {'Sharpe Δ':<12} {'Max DD Δ':<12} {'Extra Wealth':<15}")
    logger.info("-" * 80)

    baseline = results['V13']
    for version in ['V14', 'V15', 'V16', 'V17']:
        r = results[version]
        annual_delta = r['annual'] - baseline['annual']
        sharpe_delta = r['sharpe'] - baseline['sharpe']
        dd_delta = r['max_dd'] - baseline['max_dd']
        wealth_delta = r['final'] - baseline['final']

        logger.info(f"{version:<10} {annual_delta:>+7.1f}%     {sharpe_delta:>+7.2f}      {dd_delta:>+7.1f}%     ${wealth_delta:>+12,.0f}")

    logger.info("-" * 80)
    logger.info("")

    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics for a portfolio"""
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100

    # Max drawdown
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_copy = portfolio_df.copy()
    portfolio_copy['year'] = portfolio_copy.index.year
    yearly_returns = {}
    for year in portfolio_copy['year'].unique():
        year_data = portfolio_copy[portfolio_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    neg_years = sum(1 for r in yearly_returns.values() if r < 0)

    return {
        'final': final_value,
        'total': total_return,
        'annual': annual_return,
        'max_dd': max_drawdown,
        'sharpe': sharpe,
        'neg_years': neg_years,
        'total_years': len(yearly_returns),
        'yearly': yearly_returns
    }


if __name__ == "__main__":
    main()
