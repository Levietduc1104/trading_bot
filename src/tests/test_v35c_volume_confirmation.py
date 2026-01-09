"""
V35c: TEST 3 - VOLUME CONFIRMATION ONLY
=========================================

Testing ONLY the volume confirmation bonus.

Change from V28:
----------------
BEFORE (V28): Volume not used in scoring
AFTER (V35c): Award +10 pts if 20-day volume > 60-day volume by 20%+

Logic:
------
- Calculate average volume over last 20 days
- Calculate average volume over last 60 days
- If 20d_volume > 60d_volume * 1.2 → +10 bonus points
- Indicates institutional accumulation / increased interest

Why This Should Work:
---------------------
- Volume confirms price moves (price + volume = stronger signal)
- Institutional buying drives sustainable trends
- Well-established technical analysis principle

Expected Impact:
----------------
- Annual Return: +0.2-0.5% (better stock selection)
- Max Drawdown: Similar or slightly better
- Win Rate: +1-2% (fewer false signals)

Overfitting Risk: VERY LOW (20% threshold is standard)
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def calculate_kelly_weights_sqrt(scored_stocks):
    """Calculate Kelly weights using Square Root method"""
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


def calculate_volume_confirmation(df):
    """
    V35c: Volume Confirmation

    Awards bonus if recent volume significantly higher than historical average.

    Args:
        df: Stock dataframe

    Returns:
        int: Bonus points (0 or 10)
    """
    if len(df) < 60:
        return 0

    volume_20 = df['volume'].tail(20).mean()
    volume_60 = df['volume'].tail(60).mean()

    # 20% increase in volume = institutional interest
    if volume_20 > volume_60 * 1.2:
        return 10  # Bonus for volume confirmation

    return 0


def score_stock_v35c_with_volume(bot, ticker, df):
    """
    V35c: Score stock with volume confirmation ONLY

    ONLY CHANGE: Add +10 pts for volume confirmation
    Everything else identical to V28 scoring
    """
    if len(df) < 252:
        return 0

    # Use bot's existing score_stock method
    score = bot.score_stock(ticker, df)

    if score == 0:
        return 0  # Already disqualified

    # Add volume confirmation bonus
    volume_bonus = calculate_volume_confirmation(df)
    score += volume_bonus

    return score


def run_v35c_backtest(bot):
    """Run V35c backtest with volume confirmation"""
    logger.info("Running V35c backtest (Volume Confirmation)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    # Track volume confirmation stats
    volume_confirmed_picks = 0
    total_picks = 0

    for date in dates[252:]:
        # Monthly rebalancing
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

            # Get VIX
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

            # Score stocks with volume confirmation
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:
                    try:
                        score = score_stock_v35c_with_volume(bot, ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score
                    except:
                        pass

            # V28 regime-based portfolio sizing
            top_n = bot.determine_portfolio_size(date)

            # Get top N stocks
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # Track volume confirmation in top picks
            total_picks += len(top_stocks)
            for ticker, score in top_stocks:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) >= 60:
                    vol_bonus = calculate_volume_confirmation(df_at_date)
                    if vol_bonus > 0:
                        volume_confirmed_picks += 1

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

    # Log statistics
    vol_confirmation_rate = (volume_confirmed_picks / total_picks * 100) if total_picks > 0 else 0
    logger.info(f"\nV35c Volume Confirmation Statistics:")
    logger.info(f"  Picks with volume confirmation: {volume_confirmed_picks}/{total_picks} ({vol_confirmation_rate:.1f}%)")

    return portfolio_df


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics"""
    final_value = portfolio_df['value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_df['value'].cummax()
    drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
    max_drawdown = drawdown.min()

    daily_returns = portfolio_df['value'].pct_change().dropna()
    rf_rate = 0.02
    excess_returns = daily_returns - (rf_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = {}

    for year in sorted(portfolio_df_copy['year'].unique()):
        year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    positive_years = sum(1 for ret in yearly_returns.values() if ret > 0)

    return {
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'final_value': final_value,
        'yearly_returns': yearly_returns,
        'positive_years': positive_years,
        'total_years': len(yearly_returns)
    }


def main():
    logger.info("=" * 80)
    logger.info("V35c TEST: VOLUME CONFIRMATION ONLY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Change: +10 pts if 20d volume > 60d volume by 20%+")
    logger.info("Everything else: Identical to V28")
    logger.info("")
    logger.info("Expected: +0.2-0.5% annual")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Run V35c backtest
    portfolio_df = run_v35c_backtest(bot)
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Win Rate:        {metrics['positive_years']}/{metrics['total_years']} ({metrics['positive_years']/metrics['total_years']*100:.0f}%)")

    logger.info("\nYearly Returns:")
    for year, ret in sorted(metrics['yearly_returns'].items()):
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:6.1f}% {status}")

    # Comparison to V28
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON VS V28 BASELINE")
    logger.info("=" * 80)
    logger.info(f"{'Strategy':<25} {'Annual':>8} {'Drawdown':>10} {'Sharpe':>8} {'Win Rate':>10}")
    logger.info("-" * 70)
    logger.info(f"{'V28 (Baseline)':<25} {'9.3%':>8} {'-17.7%':>10} {'0.99':>8} {'17/20':>10}")

    improvement = metrics['annual_return'] - 9.3
    dd_improvement = metrics['max_drawdown'] - (-17.7)
    sharpe_improvement = metrics['sharpe_ratio'] - 0.99

    logger.info(
        f"{'V35c (Volume Confirm)':<25} "
        f"{metrics['annual_return']:>7.1f}% "
        f"{metrics['max_drawdown']:>9.1f}% "
        f"{metrics['sharpe_ratio']:>8.2f} "
        f"{metrics['positive_years']}/{metrics['total_years']:>2}"
    )

    logger.info("")
    logger.info("Delta from V28:")
    logger.info(f"  Annual Return:  {improvement:+.1f}%")
    logger.info(f"  Max Drawdown:   {dd_improvement:+.1f}%")
    logger.info(f"  Sharpe Ratio:   {sharpe_improvement:+.2f}")

    # Verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)
    if improvement >= 0.2:
        logger.info(f"✅ SUCCESS: Volume confirmation improves returns by +{improvement:.1f}%")
        logger.info("   This feature WORKS - include in final strategy")
    elif improvement > 0:
        logger.info(f"⚠️  MARGINAL: Tiny improvement of +{improvement:.1f}%")
        logger.info("   Borderline - may or may not include")
    else:
        logger.info(f"❌ NO IMPROVEMENT: Returns worse by {improvement:.1f}%")
        logger.info("   Skip this feature")

    return metrics


if __name__ == "__main__":
    results = main()
