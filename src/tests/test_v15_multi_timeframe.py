"""
V15 MULTI-TIMEFRAME MOMENTUM ENSEMBLE TEST

Compares V13 (Single 9-Month Momentum) vs V15 (Multi-Timeframe Ensemble)

V15 uses weighted combination of multiple momentum periods:
- 0.15 × 3-month return  (short-term surge)
- 0.25 × 6-month return  (medium-term trend)
- 0.35 × 9-month return  (primary signal)
- 0.25 × 12-month return (long-term confirmation)

Expected improvements:
- More robust momentum signal
- Better CAGR (+0.5-0.8%)
- Lower drawdown (-1 to -2%)
- Reduced whipsaw in trend transitions
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
    logger.info("           V15 MULTI-TIMEFRAME MOMENTUM ENSEMBLE COMPARISON           ")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("                           STEP 1: LOADING DATA                           ")
    logger.info("="*80)

    # Initialize bot
    bot = PortfolioRotationBot(
        data_dir='sp500_data/daily',
        initial_capital=100000
    )

    # Load and prepare data
    logger.info(f"Loading {len(bot.stocks_data)} stocks...")
    bot.prepare_data()
    logger.info(f"✅ Loaded {len(bot.stocks_data)} stocks")

    # ========================================================================
    # V13: SINGLE 9-MONTH MOMENTUM (BASELINE)
    # ========================================================================
    logger.info("="*80)
    logger.info("              STEP 2: RUNNING V13 (SINGLE 9-MONTH MOMENTUM)              ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V13 Single-Period Momentum (9-month)")
    logger.info("="*80)
    logger.info("Momentum Signal: Single 9-month return")
    logger.info("Multi-Timeframe: DISABLED")

    portfolio_v13 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_multi_timeframe_momentum=False,  # V15: Disabled for baseline
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    # Calculate V13 metrics
    v13_final = portfolio_v13['value'].iloc[-1]
    v13_total_return = (v13_final / bot.initial_capital - 1) * 100
    years = (portfolio_v13.index[-1] - portfolio_v13.index[0]).days / 365.25
    v13_annual = ((v13_final / bot.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_v13['value'].cummax()
    drawdown = (portfolio_v13['value'] - cummax) / cummax * 100
    v13_max_dd = drawdown.min()

    daily_returns = portfolio_v13['value'].pct_change().dropna()
    v13_sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_v13_copy = portfolio_v13.copy()
    portfolio_v13_copy['year'] = portfolio_v13_copy.index.year
    v13_yearly = {}
    for year in portfolio_v13_copy['year'].unique():
        year_data = portfolio_v13_copy[portfolio_v13_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            v13_yearly[year] = year_return

    v13_neg_years = sum(1 for r in v13_yearly.values() if r < 0)

    logger.info("")
    logger.info(f"✅ V13: Annual {v13_annual:.1f}%, Sharpe {v13_sharpe:.2f}, Max DD {v13_max_dd:.1f}%, Neg Years {v13_neg_years}/{len(v13_yearly)}")

    # ========================================================================
    # V15: MULTI-TIMEFRAME MOMENTUM ENSEMBLE
    # ========================================================================
    logger.info("="*80)
    logger.info("         STEP 3: RUNNING V15 (MULTI-TIMEFRAME MOMENTUM ENSEMBLE)         ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V15 Multi-Timeframe Momentum Ensemble")
    logger.info("="*80)
    logger.info("Momentum Signal: Weighted ensemble of 3m, 6m, 9m, 12m")
    logger.info("Weights: 15% (3m) + 25% (6m) + 35% (9m) + 25% (12m)")
    logger.info("Multi-Timeframe: ENABLED")

    portfolio_v15 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_multi_timeframe_momentum=True,  # V15: Multi-timeframe ensemble ✨
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    # Calculate V15 metrics
    v15_final = portfolio_v15['value'].iloc[-1]
    v15_total_return = (v15_final / bot.initial_capital - 1) * 100
    v15_annual = ((v15_final / bot.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_v15['value'].cummax()
    drawdown = (portfolio_v15['value'] - cummax) / cummax * 100
    v15_max_dd = drawdown.min()

    daily_returns = portfolio_v15['value'].pct_change().dropna()
    v15_sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_v15_copy = portfolio_v15.copy()
    portfolio_v15_copy['year'] = portfolio_v15_copy.index.year
    v15_yearly = {}
    for year in portfolio_v15_copy['year'].unique():
        year_data = portfolio_v15_copy[portfolio_v15_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            v15_yearly[year] = year_return

    v15_neg_years = sum(1 for r in v15_yearly.values() if r < 0)

    logger.info("")
    logger.info(f"✅ V15: Annual {v15_annual:.1f}%, Sharpe {v15_sharpe:.2f}, Max DD {v15_max_dd:.1f}%, Neg Years {v15_neg_years}/{len(v15_yearly)}")

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    logger.info("="*80)
    logger.info("                        COMPARISON COMPLETE                        ")
    logger.info("="*80)
    logger.info("")
    logger.info("🏆 WINNER DETERMINATION:")
    logger.info("")

    # Determine winners
    best_annual = "V15" if v15_annual > v13_annual else "V13"
    best_sharpe = "V15" if v15_sharpe > v13_sharpe else "V13"
    best_dd = "V15" if v15_max_dd > v13_max_dd else "V13"  # Higher (less negative) is better
    best_neg = "V15" if v15_neg_years < v13_neg_years else "V13"

    logger.info(f"  🥇 Best Annual Return:  {best_annual} " +
                f"(V13: {v13_annual:.1f}%, V15: {v15_annual:.1f}%)")
    logger.info(f"  🥇 Best Sharpe Ratio:   {best_sharpe} " +
                f"(V13: {v13_sharpe:.2f}, V15: {v15_sharpe:.2f})")
    logger.info(f"  🥇 Best Max Drawdown:   {best_dd} " +
                f"(V13: {v13_max_dd:.1f}%, V15: {v15_max_dd:.1f}%)")
    logger.info(f"  🥇 Fewest Neg Years:    {best_neg} " +
                f"(V13: {v13_neg_years}/{len(v13_yearly)}, V15: {v15_neg_years}/{len(v15_yearly)})")
    logger.info("")

    # Improvement metrics
    annual_improvement = v15_annual - v13_annual
    sharpe_improvement = v15_sharpe - v13_sharpe
    dd_improvement = v15_max_dd - v13_max_dd  # Less negative is better
    wealth_improvement = v15_final - v13_final

    logger.info("📊 V15 IMPROVEMENTS:")
    logger.info("")
    logger.info(f"  Annual Return:  {annual_improvement:+.1f}% " +
                f"({'📈 Better' if annual_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Sharpe Ratio:   {sharpe_improvement:+.2f} " +
                f"({'📈 Better' if sharpe_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Max Drawdown:   {dd_improvement:+.1f}% " +
                f"({'📈 Better' if dd_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Negative Years: {v15_neg_years - v13_neg_years:+d} " +
                f"({'📈 Better' if v15_neg_years < v13_neg_years else '📉 Worse' if v15_neg_years > v13_neg_years else '➡️  Same'})")
    logger.info(f"  Extra Wealth:   ${wealth_improvement:+,.0f} " +
                f"({'📈 Better' if wealth_improvement > 0 else '📉 Worse'})")
    logger.info("")

    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
