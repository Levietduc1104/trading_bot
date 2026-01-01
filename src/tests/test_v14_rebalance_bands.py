"""
V14 REBALANCE BANDS TEST

Compares V13 (Monthly Rebalance) vs V14 (V13 + Rebalance Bands)

V14 adds rebalance bands to let winners run:
- Only rebalances when position weights drift > threshold (default 25%)
- OR when max days exceeded (default 90 days)
- Reduces unnecessary trading and transaction costs

Expected improvements:
- Lower turnover (fewer trades)
- Better CAGR (let winners compound)
- Lower transaction costs
- More tax efficient
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
    logger.info("                 V14 REBALANCE BANDS COMPARISON                 ")
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
    # V13: MOMENTUM + DRAWDOWN (BASELINE - MONTHLY REBALANCE)
    # ========================================================================
    logger.info("="*80)
    logger.info("              STEP 2: RUNNING V13 (BASELINE - MONTHLY REBALANCE)              ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V13 Monthly Rebalance")
    logger.info("="*80)
    logger.info("Rebalancing: Monthly (mid-month day 7-10)")
    logger.info("Rebalance Bands: DISABLED")

    portfolio_v13 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_drawdown_control=True,
        use_rebalance_bands=False,  # V14: Disabled for baseline
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
    # V14: V13 + REBALANCE BANDS
    # ========================================================================
    logger.info("="*80)
    logger.info("         STEP 3: RUNNING V14 (V13 + REBALANCE BANDS)         ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V14 Rebalance Bands")
    logger.info("="*80)
    logger.info("Rebalancing: Only when weight drift > 25% OR 90 days exceeded")
    logger.info("Rebalance Bands: ENABLED (let winners run)")

    portfolio_v14 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_drawdown_control=True,
        use_rebalance_bands=True,  # V14: Rebalance bands ✨
        rebalance_drift_threshold=0.25,  # 25% drift threshold
        trading_fee_pct=0.001
    )

    # Calculate V14 metrics
    v14_final = portfolio_v14['value'].iloc[-1]
    v14_total_return = (v14_final / bot.initial_capital - 1) * 100
    v14_annual = ((v14_final / bot.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_v14['value'].cummax()
    drawdown = (portfolio_v14['value'] - cummax) / cummax * 100
    v14_max_dd = drawdown.min()

    daily_returns = portfolio_v14['value'].pct_change().dropna()
    v14_sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_v14_copy = portfolio_v14.copy()
    portfolio_v14_copy['year'] = portfolio_v14_copy.index.year
    v14_yearly = {}
    for year in portfolio_v14_copy['year'].unique():
        year_data = portfolio_v14_copy[portfolio_v14_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            v14_yearly[year] = year_return

    v14_neg_years = sum(1 for r in v14_yearly.values() if r < 0)

    logger.info("")
    logger.info(f"✅ V14: Annual {v14_annual:.1f}%, Sharpe {v14_sharpe:.2f}, Max DD {v14_max_dd:.1f}%, Neg Years {v14_neg_years}/{len(v14_yearly)}")

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
    best_annual = "V14" if v14_annual > v13_annual else "V13"
    best_sharpe = "V14" if v14_sharpe > v13_sharpe else "V13"
    best_dd = "V14" if v14_max_dd > v13_max_dd else "V13"  # Higher (less negative) is better
    best_neg = "V14" if v14_neg_years < v13_neg_years else "V13"

    logger.info(f"  🥇 Best Annual Return:  {best_annual} " +
                f"(V13: {v13_annual:.1f}%, V14: {v14_annual:.1f}%)")
    logger.info(f"  🥇 Best Sharpe Ratio:   {best_sharpe} " +
                f"(V13: {v13_sharpe:.2f}, V14: {v14_sharpe:.2f})")
    logger.info(f"  🥇 Best Max Drawdown:   {best_dd} " +
                f"(V13: {v13_max_dd:.1f}%, V14: {v14_max_dd:.1f}%)")
    logger.info(f"  🥇 Fewest Neg Years:    {best_neg} " +
                f"(V13: {v13_neg_years}/{len(v13_yearly)}, V14: {v14_neg_years}/{len(v14_yearly)})")
    logger.info("")

    # Improvement metrics
    annual_improvement = v14_annual - v13_annual
    sharpe_improvement = v14_sharpe - v13_sharpe
    dd_improvement = v14_max_dd - v13_max_dd  # Less negative is better

    logger.info("📊 V14 IMPROVEMENTS:")
    logger.info("")
    logger.info(f"  Annual Return:  {annual_improvement:+.1f}% " +
                f"({'📈 Better' if annual_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Sharpe Ratio:   {sharpe_improvement:+.2f} " +
                f"({'📈 Better' if sharpe_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Max Drawdown:   {dd_improvement:+.1f}% " +
                f"({'📈 Better' if dd_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Negative Years: {v14_neg_years - v13_neg_years:+d} " +
                f"({'📈 Better' if v14_neg_years < v13_neg_years else '📉 Worse' if v14_neg_years > v13_neg_years else '➡️  Same'})")
    logger.info("")

    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
