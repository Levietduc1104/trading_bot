"""
V16 SECTOR ROTATION TEST

Compares V13 (Stock Momentum Only) vs V16 (V13 + Sector Rotation)

V16 adds sector momentum layer:
- Calculates 6-month momentum for each sector
- Applies sector multiplier to stock weights:
  * Strong sectors (>+15%): 1.3x weight multiplier
  * Weak sectors (<-10%): 0.8x weight multiplier
- Overweights stocks in leading sectors
- Underweights stocks in lagging sectors

Expected improvements:
- Higher CAGR (+0.5-1.0%)
- Same or better drawdown
- Captures sector rotation trends
- Academically proven (Moskowitz & Grinblatt 1999)
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
    logger.info("                 V16 SECTOR ROTATION COMPARISON                 ")
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
    # V13: STOCK MOMENTUM ONLY (BASELINE)
    # ========================================================================
    logger.info("="*80)
    logger.info("              STEP 2: RUNNING V13 (STOCK MOMENTUM ONLY)              ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V13 Stock Momentum Only")
    logger.info("="*80)
    logger.info("Position Weighting: Momentum/volatility ratio")
    logger.info("Sector Rotation: DISABLED")

    portfolio_v13 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_sector_rotation=False,  # V16: Disabled for baseline
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
    # V16: STOCK MOMENTUM + SECTOR ROTATION
    # ========================================================================
    logger.info("="*80)
    logger.info("         STEP 3: RUNNING V16 (V13 + SECTOR ROTATION)         ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V16 Sector Rotation")
    logger.info("="*80)
    logger.info("Position Weighting: (Momentum/vol) × Sector Multiplier")
    logger.info("Sector Rotation: ENABLED (6-month sector momentum)")
    logger.info("  - Strong sectors (>+10%): 1.2-1.4x weight multiplier")
    logger.info("  - Weak sectors (<-10%): 0.6-0.8x weight multiplier")

    portfolio_v16 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_sector_rotation=True,  # V16: Sector rotation ✨
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    # Calculate V16 metrics
    v16_final = portfolio_v16['value'].iloc[-1]
    v16_total_return = (v16_final / bot.initial_capital - 1) * 100
    v16_annual = ((v16_final / bot.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_v16['value'].cummax()
    drawdown = (portfolio_v16['value'] - cummax) / cummax * 100
    v16_max_dd = drawdown.min()

    daily_returns = portfolio_v16['value'].pct_change().dropna()
    v16_sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_v16_copy = portfolio_v16.copy()
    portfolio_v16_copy['year'] = portfolio_v16_copy.index.year
    v16_yearly = {}
    for year in portfolio_v16_copy['year'].unique():
        year_data = portfolio_v16_copy[portfolio_v16_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            v16_yearly[year] = year_return

    v16_neg_years = sum(1 for r in v16_yearly.values() if r < 0)

    logger.info("")
    logger.info(f"✅ V16: Annual {v16_annual:.1f}%, Sharpe {v16_sharpe:.2f}, Max DD {v16_max_dd:.1f}%, Neg Years {v16_neg_years}/{len(v16_yearly)}")

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
    best_annual = "V16" if v16_annual > v13_annual else "V13"
    best_sharpe = "V16" if v16_sharpe > v13_sharpe else "V13"
    best_dd = "V16" if v16_max_dd > v13_max_dd else "V13"  # Higher (less negative) is better
    best_neg = "V16" if v16_neg_years < v13_neg_years else "V13"

    logger.info(f"  🥇 Best Annual Return:  {best_annual} " +
                f"(V13: {v13_annual:.1f}%, V16: {v16_annual:.1f}%)")
    logger.info(f"  🥇 Best Sharpe Ratio:   {best_sharpe} " +
                f"(V13: {v13_sharpe:.2f}, V16: {v16_sharpe:.2f})")
    logger.info(f"  🥇 Best Max Drawdown:   {best_dd} " +
                f"(V13: {v13_max_dd:.1f}%, V16: {v16_max_dd:.1f}%)")
    logger.info(f"  🥇 Fewest Neg Years:    {best_neg} " +
                f"(V13: {v13_neg_years}/{len(v13_yearly)}, V16: {v16_neg_years}/{len(v16_yearly)})")
    logger.info("")

    # Improvement metrics
    annual_improvement = v16_annual - v13_annual
    sharpe_improvement = v16_sharpe - v13_sharpe
    dd_improvement = v16_max_dd - v13_max_dd  # Less negative is better
    wealth_improvement = v16_final - v13_final

    logger.info("📊 V16 IMPROVEMENTS:")
    logger.info("")
    logger.info(f"  Annual Return:  {annual_improvement:+.1f}% " +
                f"({'📈 Better' if annual_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Sharpe Ratio:   {sharpe_improvement:+.2f} " +
                f"({'📈 Better' if sharpe_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Max Drawdown:   {dd_improvement:+.1f}% " +
                f"({'📈 Better' if dd_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Negative Years: {v16_neg_years - v13_neg_years:+d} " +
                f"({'📈 Better' if v16_neg_years < v13_neg_years else '📉 Worse' if v16_neg_years > v13_neg_years else '➡️  Same'})")
    logger.info(f"  Extra Wealth:   ${wealth_improvement:+,.0f} " +
                f"({'📈 Better' if wealth_improvement > 0 else '📉 Worse'})")
    logger.info("")

    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
