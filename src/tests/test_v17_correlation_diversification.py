"""
V17 CORRELATION-BASED DIVERSIFICATION TEST

Compares V16 (Sector Rotation) vs V17 (V16 + Correlation Diversification)

V17 adds correlation-based diversification within candidate group:
- Measures each candidate's correlation to OTHER candidates (not existing holdings)
- Applies weight multipliers based on average correlation to the group:
  * Low correlation (<0.4): 1.15x weight bonus (prefer uncorrelated stocks)
  * Medium correlation (0.4-0.6): 1.0x neutral
  * High correlation (0.6-0.8): 0.90x penalty (avoid clustered picks)
  * Very high correlation (>0.8): 0.75x penalty (strong penalty for redundancy)
- Builds naturally diversified portfolios by preferring low-correlation combinations

Why this approach works better:
- Compares candidates against each other (same time window, apples-to-apples)
- Doesn't depend on what we currently hold (works from first rebalance)
- Reduces sector/factor concentration risk
- Academic evidence: lower correlation = better risk-adjusted returns

Expected improvements:
- Better risk-adjusted returns (higher Sharpe ratio)
- Lower portfolio volatility
- Improved drawdown protection through diversification
- More sector balance without explicit sector constraints
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
    logger.info("           V17 CORRELATION-BASED DIVERSIFICATION COMPARISON           ")
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
    # V16: SECTOR ROTATION (BASELINE)
    # ========================================================================
    logger.info("="*80)
    logger.info("              STEP 2: RUNNING V16 (SECTOR ROTATION BASELINE)              ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V16 Sector Rotation")
    logger.info("="*80)
    logger.info("Position Weighting: (Momentum/vol) × Sector Multiplier")
    logger.info("Correlation Diversification: DISABLED")

    portfolio_v16 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_sector_rotation=True,
        use_correlation_diversification=False,  # V17: Disabled for baseline
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    # Calculate V16 metrics
    v16_final = portfolio_v16['value'].iloc[-1]
    v16_total_return = (v16_final / bot.initial_capital - 1) * 100
    years = (portfolio_v16.index[-1] - portfolio_v16.index[0]).days / 365.25
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
    # V17: V16 + CORRELATION DIVERSIFICATION
    # ========================================================================
    logger.info("="*80)
    logger.info("         STEP 3: RUNNING V17 (V16 + CORRELATION DIVERSIFICATION)         ")
    logger.info("="*80)
    logger.info("")
    logger.info("="*80)
    logger.info("Running: V17 Correlation-Based Diversification")
    logger.info("="*80)
    logger.info("Position Weighting: (Momentum/vol) × Sector × Correlation Penalty")
    logger.info("Correlation Diversification: ENABLED (60-day lookback)")
    logger.info("  - Low correlation (<0.4): 1.15x bonus")
    logger.info("  - Medium correlation (0.4-0.6): 1.0x neutral")
    logger.info("  - High correlation (0.6-0.8): 0.90x penalty")
    logger.info("  - Very high correlation (>0.8): 0.75x penalty")

    portfolio_v17 = bot.backtest_with_bear_protection(
        top_n=10,
        rebalance_freq='M',
        use_vix_regime=True,
        use_adaptive_weighting=True,
        use_momentum_weighting=True,
        use_sector_rotation=True,
        use_correlation_diversification=True,  # V17: Correlation diversification ✨
        use_drawdown_control=True,
        trading_fee_pct=0.001
    )

    # Calculate V17 metrics
    v17_final = portfolio_v17['value'].iloc[-1]
    v17_total_return = (v17_final / bot.initial_capital - 1) * 100
    v17_annual = ((v17_final / bot.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_v17['value'].cummax()
    drawdown = (portfolio_v17['value'] - cummax) / cummax * 100
    v17_max_dd = drawdown.min()

    daily_returns = portfolio_v17['value'].pct_change().dropna()
    v17_sharpe = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_v17_copy = portfolio_v17.copy()
    portfolio_v17_copy['year'] = portfolio_v17_copy.index.year
    v17_yearly = {}
    for year in portfolio_v17_copy['year'].unique():
        year_data = portfolio_v17_copy[portfolio_v17_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            v17_yearly[year] = year_return

    v17_neg_years = sum(1 for r in v17_yearly.values() if r < 0)

    logger.info("")
    logger.info(f"✅ V17: Annual {v17_annual:.1f}%, Sharpe {v17_sharpe:.2f}, Max DD {v17_max_dd:.1f}%, Neg Years {v17_neg_years}/{len(v17_yearly)}")

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
    best_annual = "V17" if v17_annual > v16_annual else "V16"
    best_sharpe = "V17" if v17_sharpe > v16_sharpe else "V16"
    best_dd = "V17" if v17_max_dd > v16_max_dd else "V16"  # Higher (less negative) is better
    best_neg = "V17" if v17_neg_years < v16_neg_years else "V16"

    logger.info(f"  🥇 Best Annual Return:  {best_annual} " +
                f"(V16: {v16_annual:.1f}%, V17: {v17_annual:.1f}%)")
    logger.info(f"  🥇 Best Sharpe Ratio:   {best_sharpe} " +
                f"(V16: {v16_sharpe:.2f}, V17: {v17_sharpe:.2f})")
    logger.info(f"  🥇 Best Max Drawdown:   {best_dd} " +
                f"(V16: {v16_max_dd:.1f}%, V17: {v17_max_dd:.1f}%)")
    logger.info(f"  🥇 Fewest Neg Years:    {best_neg} " +
                f"(V16: {v16_neg_years}/{len(v16_yearly)}, V17: {v17_neg_years}/{len(v17_yearly)})")
    logger.info("")

    # Improvement metrics
    annual_improvement = v17_annual - v16_annual
    sharpe_improvement = v17_sharpe - v16_sharpe
    dd_improvement = v17_max_dd - v16_max_dd  # Less negative is better
    wealth_improvement = v17_final - v16_final

    logger.info("📊 V17 IMPROVEMENTS:")
    logger.info("")
    logger.info(f"  Annual Return:  {annual_improvement:+.1f}% " +
                f"({'📈 Better' if annual_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Sharpe Ratio:   {sharpe_improvement:+.2f} " +
                f"({'📈 Better' if sharpe_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Max Drawdown:   {dd_improvement:+.1f}% " +
                f"({'📈 Better' if dd_improvement > 0 else '📉 Worse'})")
    logger.info(f"  Negative Years: {v17_neg_years - v16_neg_years:+d} " +
                f"({'📈 Better' if v17_neg_years < v16_neg_years else '📉 Worse' if v17_neg_years > v16_neg_years else '➡️  Same'})")
    logger.info(f"  Extra Wealth:   ${wealth_improvement:+,.0f} " +
                f"({'📈 Better' if wealth_improvement > 0 else '📉 Worse'})")
    logger.info("")

    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
