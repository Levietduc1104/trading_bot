"""
V28: MOMENTUM LEADERS - 52-Week Breakout + Relative Strength
==============================================================

PROBLEM: V27 doesn't filter for momentum leadership
- Current system buys any stock with positive momentum
- Doesn't distinguish between market leaders and laggards
- Missing the "breakout to new highs" momentum effect

SOLUTION: Add two proven momentum factors

52-Week Breakout (Factor 4):
----------------------------
- Stocks near 52-week highs (within 2%) get +20 points
- Stocks within 5% of high get +10 points
- Rationale: Institutional accumulation drives stocks to new highs
- Academic backing: Momentum effect (Jagadeesh & Titman, 1993)

Relative Strength vs SPY (Factor 5):
-----------------------------------
- Must outperform SPY over 60-day period
- Strong outperformance (>10%) gets +15 points
- Underperformance (<-5%) gets 20% penalty
- Rationale: Only buy market leaders, not sector sympathy plays

Why This Works:
---------------
- Catches trends EARLY (breakouts) not late (already trending)
- Filters out weak stocks that are just "less bad" than others
- No look-ahead bias: calculated from historical OHLCV only
- Proven momentum effects with academic backing
- Simple, rule-based logic (no ML overfitting)

Expected Impact:
----------------
- Annual Return: +3-5% improvement (V27: 8.5% → V28: 12-14%)
- Win Rate: +2-5% improvement (better stock selection)
- Sharpe Ratio: Maintain or slightly improve
- Risk: LOW (well-established momentum principles)

Data Requirements:
------------------
- 52-week high: Calculated from rolling 252-day high (daily OHLCV)
- Relative strength: Calculated from stock vs SPY returns (daily OHLCV)
- NO fundamental data required (avoids look-ahead bias)

Test Type: INDEPENDENT - Tests current scoring with V28 enhancements
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
        logging.FileHandler('src/tests/v28_momentum_output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_kelly_weights_sqrt(scored_stocks):
    """Calculate Kelly weights (Square Root method) - V22"""
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


def run_v28_backtest(bot):
    """
    Run V28 backtest with momentum leader selection

    V28 adds two momentum factors to V27:
    1. 52-week breakout bonus (0-20 points)
    2. Relative strength vs SPY (0-15 points)

    All V27 features remain:
    - Regime-based portfolio sizing (3-10 stocks)
    - Kelly position sizing
    - VIX-based cash reserves
    - Drawdown control
    """
    logger.info("Running V28 backtest (Momentum Leaders)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    # Track momentum stats
    breakout_picks = 0  # Stocks within 2% of 52w high
    near_breakout_picks = 0  # Stocks within 5-10% of 52w high
    rs_leader_picks = 0  # Stocks with RS > 10
    rs_laggard_filtered = 0  # Stocks filtered for RS < -5

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

            # Score stocks (V28 scoring with momentum enhancements)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:  # Need 1 year for 52w high
                    try:
                        score = bot.score_stock(ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score

                            # Track momentum stats
                            distance_52w = bot.calculate_52week_high_distance(df_at_date)
                            rs = bot.calculate_relative_strength_vs_spy(ticker, df_at_date, 60)

                            if distance_52w > -2:
                                breakout_picks += 1
                            elif distance_52w > -10:
                                near_breakout_picks += 1

                            if rs > 10:
                                rs_leader_picks += 1
                        elif score == 0:
                            # Check if filtered by relative strength
                            df_temp = df_at_date
                            rs = bot.calculate_relative_strength_vs_spy(ticker, df_temp, 60)
                            if rs < -5:
                                rs_laggard_filtered += 1
                    except:
                        pass

            # V27: Determine portfolio size based on regime
            top_n = bot.determine_portfolio_size(date)

            # Get top N stocks (dynamic based on regime)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # VIX-based cash reserve
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            invest_amount = cash * (1 - cash_reserve)

            # Kelly position sizing
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)

            allocations = {
                ticker: invest_amount * weight
                for ticker, weight in kelly_weights.items()
            }

            # Drawdown control
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

    # Log momentum statistics
    logger.info(f"\nV28 Momentum Selection Statistics:")
    logger.info(f"  Breakout picks (within 2% of 52w high): {breakout_picks}")
    logger.info(f"  Near-breakout picks (2-10% from high): {near_breakout_picks}")
    logger.info(f"  RS leaders picked (RS > 10): {rs_leader_picks}")
    logger.info(f"  RS laggards filtered (RS < -5): {rs_laggard_filtered}")

    return portfolio_df


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics"""
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

    # Calculate yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = {}

    for year in sorted(portfolio_df_copy['year'].unique()):
        year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'years': years,
        'start_date': portfolio_df.index[0].strftime('%Y-%m-%d'),
        'end_date': portfolio_df.index[-1].strftime('%Y-%m-%d'),
        'yearly_returns': yearly_returns
    }


def main():
    logger.info("=" * 80)
    logger.info("V28 TEST: MOMENTUM LEADERS (52-Week Breakout + Relative Strength)")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()

    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Run V28 backtest
    portfolio_df = run_v28_backtest(bot)

    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("V28 RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nInitial Capital: ${metrics['initial_capital']:,.0f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Total Return:    {metrics['total_return']:.1f}%")
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Period:          {metrics['start_date']} to {metrics['end_date']}")
    logger.info(f"Duration:        {metrics['years']:.1f} years")

    logger.info("\nYearly Returns:")
    positive_years = sum(1 for ret in metrics['yearly_returns'].values() if ret > 0)
    total_years = len(metrics['yearly_returns'])
    logger.info(f"Win Rate: {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)")

    for year, ret in sorted(metrics['yearly_returns'].items()):
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:6.1f}% {status}")

    logger.info("\n" + "=" * 80)
    logger.info("V28 Strategy Configuration:")
    logger.info("=" * 80)
    logger.info("  NEW V28 Momentum Factors:")
    logger.info("    - 52-week breakout bonus (0-20 pts)")
    logger.info("    - Relative strength vs SPY (0-15 pts)")
    logger.info("  V27 Features (maintained):")
    logger.info("    - Regime-based portfolio sizing (3-10 stocks)")
    logger.info("    - Kelly position sizing")
    logger.info("    - VIX-based cash reserves")
    logger.info("    - Drawdown control")
    logger.info("=" * 80)

    logger.info("\n✅ V28 test complete!")


if __name__ == "__main__":
    main()
