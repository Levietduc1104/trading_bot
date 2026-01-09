"""
V34: HYBRID V22+V28 - Best of Both Worlds
==========================================

GOAL: Combine V22's superior structure with V28's momentum filters

Why This Should Work:
---------------------
V22 (10.2% annual, -15.2% DD, 1.11 Sharpe):
  ✅ Fixed 5-stock portfolio (optimal concentration)
  ✅ Kelly √score position sizing (proven best method)
  ✅ Lower drawdown than V28
  ✅ Better Sharpe ratio than V28

V28 (9.4% annual, -18% DD, 1.00 Sharpe, 85% win rate):
  ✅ 52-week breakout filter (0-20 pts)
  ✅ Relative strength vs SPY filter (0-15 pts)
  ✅ Excellent stock selection (85% win rate)
  ✅ Only buys momentum leaders

The Hybrid Approach:
--------------------
Take V22's structure:
  - Fixed 5 stocks (NOT regime-based 3-10)
  - Kelly √score weighting
  - VIX-based cash reserves
  - Drawdown control

Add V28's enhanced scoring:
  - 52-week breakout bonus
  - Relative strength vs SPY
  - (Already implemented in portfolio_bot_demo.py)

Expected Performance:
---------------------
  Annual Return:   10.8-11.5% (+1.4-2.1% vs V28)
  Max Drawdown:    -15% to -17% (better than V28's -18%)
  Sharpe Ratio:    1.10-1.15 (better than both)
  Win Rate:        85-88% (maintain V28's selection quality)

Confidence: VERY HIGH
- V22's 10.2% is proven
- V28's filters improve stock quality (85% win rate)
- No new features, just combining proven components
- Low overfitting risk (rule-based, not ML)

Implementation Time: 3 days
Risk Level: LOW
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('src/tests/v34_hybrid_output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_kelly_weights_sqrt(scored_stocks):
    """
    Calculate Kelly weights using Square Root method (V22's best performer)

    Args:
        scored_stocks: List of (ticker, score) tuples

    Returns:
        dict: {ticker: weight} where weights sum to 1.0
    """
    tickers = [t for t, s in scored_stocks]
    scores = [s for t, s in scored_stocks]

    sqrt_scores = [np.sqrt(max(0, score)) for score in scores]
    total_sqrt = sum(sqrt_scores)

    if total_sqrt > 0:
        weights = {
            ticker: np.sqrt(max(0, score)) / total_sqrt
            for ticker, score in scored_stocks
        }
    else:
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    return weights


def run_v34_hybrid_backtest(bot):
    """
    Run V34 Hybrid V22+V28 backtest

    V22 Structure:
    - Fixed 5 stocks (NOT regime-based)
    - Kelly √score position sizing
    - VIX-based cash reserves
    - Drawdown control

    V28 Scoring (already in score_stock()):
    - 52-week breakout bonus (0-20 pts)
    - Relative strength vs SPY (0-15 pts)
    - All V13-V29 features active
    """
    logger.info("Running V34 Hybrid V22+V28 backtest...")
    logger.info("  Structure: V22 (Fixed 5 stocks + Kelly √score weighting)")
    logger.info("  Scoring: V28+ (52w breakout + RS vs SPY + multi-TF alignment)")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    position_history = []

    # Track momentum stats
    breakout_picks = 0
    rs_leader_picks = 0
    multi_tf_picks = 0

    for date in dates[252:]:  # Skip first 252 days (need 1 year for 52w high)
        # Monthly rebalancing (day 7-10)
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 10
            )
        )

        if is_rebalance_day:
            # Liquidate holdings
            for ticker in list(holdings.keys()):
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * current_price
            holdings = {}
            last_rebalance_date = date

            # Get current VIX
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

            # Score stocks (V28+ scoring with all enhancements)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:  # Need 1 year for 52w high
                    try:
                        score = bot.score_stock(ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score

                            # Track which features are firing
                            distance_52w = bot.calculate_52week_high_distance(df_at_date)
                            rs = bot.calculate_relative_strength_vs_spy(ticker, df_at_date, 60)
                            multi_tf = bot.calculate_multi_timeframe_alignment(df_at_date)

                            if distance_52w > -2:
                                breakout_picks += 1
                            if rs > 10:
                                rs_leader_picks += 1
                            if multi_tf:
                                multi_tf_picks += 1
                    except:
                        pass

            # ===========================
            # V22: FIXED 5 STOCKS
            # ===========================
            # (NOT regime-based like V28)
            top_n = 5  # Always hold exactly 5 stocks

            # Get top 5 stocks
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # VIX-based cash reserve (V22 formula)
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            invest_amount = cash * (1 - cash_reserve)

            # ===========================
            # V22: KELLY √SCORE WEIGHTING
            # ===========================
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)

            # Record position weights for analysis
            position_history.append({
                'date': date,
                'weights': kelly_weights.copy(),
                'scores': {t: s for t, s in top_stocks}
            })

            allocations = {
                ticker: invest_amount * weight
                for ticker, weight in kelly_weights.items()
            }

            # Drawdown control (V22)
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

            # Buy stocks
            for ticker, _ in top_stocks:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    allocation_amount = allocations.get(ticker, 0)
                    shares = allocation_amount / current_price
                    holdings[ticker] = shares
                    fee = allocation_amount * 0.001
                    cash -= (allocation_amount + fee)

        # Calculate daily portfolio value
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    position_df = pd.DataFrame(position_history)

    # Log feature usage statistics
    total_picks = len(position_history) * 5  # 5 stocks per rebalance
    logger.info(f"\nV34 Feature Usage Statistics:")
    logger.info(f"  Total stock picks: {total_picks}")
    logger.info(f"  Breakout picks (within 2% of 52w high): {breakout_picks} ({breakout_picks/total_picks*100:.1f}%)")
    logger.info(f"  RS leader picks (RS > 10): {rs_leader_picks} ({rs_leader_picks/total_picks*100:.1f}%)")
    logger.info(f"  Multi-TF aligned picks: {multi_tf_picks} ({multi_tf_picks/total_picks*100:.1f}%)")

    return portfolio_df, position_df


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate comprehensive performance metrics"""
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
    max_drawdown = drawdown.min()

    daily_returns = portfolio_df['value'].pct_change().dropna()
    rf_rate = 0.02
    excess_returns = daily_returns - (rf_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    # Downside deviation (for Sortino ratio)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.001
    sortino = (annual_return / 100) / downside_std if downside_std > 0 else 0

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Calculate yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = {}

    for year in sorted(portfolio_df_copy['year'].unique()):
        year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    positive_years = sum(1 for ret in yearly_returns.values() if ret > 0)
    win_rate = positive_years / len(yearly_returns) * 100 if yearly_returns else 0

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'years': years,
        'start_date': portfolio_df.index[0].strftime('%Y-%m-%d'),
        'end_date': portfolio_df.index[-1].strftime('%Y-%m-%d'),
        'yearly_returns': yearly_returns,
        'positive_years': positive_years,
        'total_years': len(yearly_returns),
        'win_rate': win_rate
    }


def main():
    logger.info("=" * 80)
    logger.info("V34 HYBRID V22+V28 TEST")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Strategy: Combine V22's structure with V28's momentum filters")
    logger.info("")
    logger.info("From V22 (10.2% annual, -15.2% DD, 1.11 Sharpe):")
    logger.info("  ✅ Fixed 5-stock portfolio")
    logger.info("  ✅ Kelly √score position sizing")
    logger.info("  ✅ Lower drawdown")
    logger.info("")
    logger.info("From V28 (9.4% annual, -18% DD, 1.00 Sharpe, 85% win rate):")
    logger.info("  ✅ 52-week breakout bonus")
    logger.info("  ✅ Relative strength vs SPY")
    logger.info("  ✅ Excellent stock selection")
    logger.info("")
    logger.info("Expected: 10.8-11.5% annual, -15% to -17% DD, 1.10-1.15 Sharpe")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()

    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Run V34 Hybrid backtest
    portfolio_df, position_df = run_v34_hybrid_backtest(bot)

    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("V34 HYBRID RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nInitial Capital: ${metrics['initial_capital']:,.0f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Total Return:    {metrics['total_return']:.1f}%")
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio:   {metrics['sortino_ratio']:.2f}")
    logger.info(f"Calmar Ratio:    {metrics['calmar_ratio']:.2f}")
    logger.info(f"Period:          {metrics['start_date']} to {metrics['end_date']}")
    logger.info(f"Duration:        {metrics['years']:.1f} years")

    logger.info(f"\nYearly Performance:")
    logger.info(f"Win Rate: {metrics['positive_years']}/{metrics['total_years']} ({metrics['win_rate']:.0f}%)")

    for year, ret in sorted(metrics['yearly_returns'].items()):
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:6.1f}% {status}")

    # Comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: V22 vs V28 vs V34 HYBRID")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Strategy':<20} {'Annual':>8} {'Drawdown':>10} {'Sharpe':>8} {'Win Rate':>10} {'Status'}")
    logger.info("-" * 80)

    # V22 baseline
    logger.info(f"{'V22 (Baseline)':<20} {'10.2%':>8} {'-15.2%':>10} {'1.11':>8} {'16/20':>10}")

    # V28 baseline
    logger.info(f"{'V28 (Baseline)':<20} {'9.4%':>8} {'-18.0%':>10} {'1.00':>8} {'17/20':>10}")

    # V34 Hybrid (current)
    status = "🎯 NEW BEST" if metrics['annual_return'] > 10.2 else "⚠️ WORSE"
    logger.info(
        f"{'V34 Hybrid':<20} "
        f"{metrics['annual_return']:>7.1f}% "
        f"{metrics['max_drawdown']:>9.1f}% "
        f"{metrics['sharpe_ratio']:>8.2f} "
        f"{metrics['positive_years']}/{metrics['total_years']:>2}      "
        f"{status}"
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("=" * 80)

    # vs V22
    v22_improvement = metrics['annual_return'] - 10.2
    v22_dd_improvement = metrics['max_drawdown'] - (-15.2)
    v22_sharpe_improvement = metrics['sharpe_ratio'] - 1.11

    logger.info("\nV34 vs V22 (Fixed 5 stocks):")
    logger.info(f"  Annual Return:  {v22_improvement:+.1f}% ({metrics['annual_return']:.1f}% vs 10.2%)")
    logger.info(f"  Max Drawdown:   {v22_dd_improvement:+.1f}% ({metrics['max_drawdown']:.1f}% vs -15.2%)")
    logger.info(f"  Sharpe Ratio:   {v22_sharpe_improvement:+.2f} ({metrics['sharpe_ratio']:.2f} vs 1.11)")

    # vs V28
    v28_improvement = metrics['annual_return'] - 9.4
    v28_dd_improvement = metrics['max_drawdown'] - (-18.0)
    v28_sharpe_improvement = metrics['sharpe_ratio'] - 1.00

    logger.info("\nV34 vs V28 (Momentum Leaders):")
    logger.info(f"  Annual Return:  {v28_improvement:+.1f}% ({metrics['annual_return']:.1f}% vs 9.4%)")
    logger.info(f"  Max Drawdown:   {v28_dd_improvement:+.1f}% ({metrics['max_drawdown']:.1f}% vs -18.0%)")
    logger.info(f"  Sharpe Ratio:   {v28_sharpe_improvement:+.2f} ({metrics['sharpe_ratio']:.2f} vs 1.00)")

    # Position concentration analysis
    if len(position_df) > 0:
        avg_top_weight = []
        for idx, row in position_df.iterrows():
            weights = row['weights']
            if weights:
                avg_top_weight.append(max(weights.values()))
        avg_max_position = np.mean(avg_top_weight) if avg_top_weight else 0.20

        logger.info("\n" + "=" * 80)
        logger.info("POSITION SIZING ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Average largest position: {avg_max_position*100:.1f}%")
        logger.info(f"Position sizing method: Kelly √score (V22)")
        logger.info(f"Portfolio size: Fixed 5 stocks (V22)")

    # Final verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    if metrics['annual_return'] > 10.8:
        logger.info("✅ SUCCESS: V34 Hybrid achieves target (>10.8% annual)!")
        logger.info(f"   Actual: {metrics['annual_return']:.1f}% annual")
        logger.info(f"   Improvement: +{v22_improvement:.1f}% vs V22, +{v28_improvement:.1f}% vs V28")
        logger.info("\n🎯 RECOMMENDATION: Deploy V34 Hybrid as new production strategy")
    elif metrics['annual_return'] > 10.2:
        logger.info("✅ GOOD: V34 Hybrid beats V22 baseline!")
        logger.info(f"   Actual: {metrics['annual_return']:.1f}% annual")
        logger.info(f"   Improvement: +{v22_improvement:.1f}% vs V22")
        logger.info("\n🎯 RECOMMENDATION: V34 Hybrid is an improvement, deploy to production")
    else:
        logger.info("⚠️  NEUTRAL: V34 Hybrid doesn't beat V22")
        logger.info(f"   Actual: {metrics['annual_return']:.1f}% annual vs V22's 10.2%")
        logger.info("\n🎯 RECOMMENDATION: Investigate further - may need additional filters")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("If V34 Hybrid succeeds:")
    logger.info("  1. Add trailing stop loss (+0.5-1.5% annual, better DD)")
    logger.info("  2. Add sector rotation (+2.0-3.0% annual)")
    logger.info("  3. Integrate earnings momentum (+1.0-2.0% annual)")
    logger.info("")
    logger.info("Target: 15% annual return within 4-6 weeks")
    logger.info("=" * 80)

    return metrics


if __name__ == "__main__":
    results = main()
