"""
V35b: TEST 2 - TRAILING STOP LOSS ONLY
========================================

Testing ONLY the trailing stop loss mechanism.

Change from V28:
----------------
BEFORE (V28): Hold stocks for full month until rebalance
AFTER (V35b): Sell if price drops 12% from peak, redeploy immediately

Logic:
------
- Track peak price for each holding daily
- If current_price < peak * 0.88 → SELL (hit 12% trailing stop)
- Immediately redeploy cash to next-best ranked stock
- On monthly rebalance, refresh all positions normally

Why This Should Work:
---------------------
- Asymmetric risk: Cut losers at -12%, let winners run indefinitely
- Redeploys capital from weak stocks to strong stocks mid-month
- Reduces drawdowns (exit deteriorating positions early)
- Academic backing: Trend-following / momentum strategies

Expected Impact:
----------------
- Annual Return: +0.5-1.5% (better capital deployment)
- Max Drawdown: -3% to -5% improvement (cut losses early)
- Win/Loss Ratio: 2.5-3.5x (vs current ~2x)

Overfitting Risk: VERY LOW (12% is standard, no optimization)
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


def run_v35b_backtest_with_trailing_stops(bot):
    """
    Run V35b backtest with trailing stop loss ONLY

    Everything else identical to V28
    """
    logger.info("Running V35b backtest (Trailing Stop Loss)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    # Track peak prices for trailing stops
    peak_prices = {}

    # Track ranked candidates for mid-month redeployment
    current_ranked_stocks = []
    current_vix = 20

    # Statistics
    trailing_stop_triggers = 0
    mid_month_redeployments = 0

    for date in dates[252:]:
        # =================================
        # DAILY TRAILING STOP CHECK
        # =================================
        for ticker in list(holdings.keys()):
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']

                # Update peak price
                if ticker not in peak_prices:
                    peak_prices[ticker] = current_price
                else:
                    peak_prices[ticker] = max(peak_prices[ticker], current_price)

                # Check 12% trailing stop
                if current_price < peak_prices[ticker] * 0.88:
                    # TRAILING STOP HIT
                    sell_value = holdings[ticker] * current_price
                    cash += sell_value
                    del holdings[ticker]
                    del peak_prices[ticker]
                    trailing_stop_triggers += 1

                    # Redeploy to next-best stock (if we have ranked list)
                    if current_ranked_stocks and len(holdings) < 10:
                        # Find next best stock not currently held
                        for candidate_ticker, candidate_score in current_ranked_stocks:
                            if candidate_ticker not in holdings:
                                # Buy this stock
                                df_candidate = bot.stocks_data[candidate_ticker][
                                    bot.stocks_data[candidate_ticker].index <= date
                                ]
                                if len(df_candidate) > 0:
                                    candidate_price = df_candidate.iloc[-1]['close']

                                    # Use same VIX-based cash reserve
                                    if current_vix < 30:
                                        cash_reserve = 0.05 + (current_vix - 10) * 0.005
                                    else:
                                        cash_reserve = 0.15 + (current_vix - 30) * 0.0125
                                    cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

                                    available_cash = cash * (1 - cash_reserve)
                                    allocation = available_cash * 0.15  # Conservative allocation

                                    if allocation > 0 and cash > allocation:
                                        shares = allocation / candidate_price
                                        holdings[candidate_ticker] = shares
                                        peak_prices[candidate_ticker] = candidate_price
                                        fee = allocation * 0.001
                                        cash -= (allocation + fee)
                                        mid_month_redeployments += 1
                                        break

        # =================================
        # MONTHLY REBALANCING
        # =================================
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 10
            )
        )

        if is_rebalance_day:
            # Liquidate all holdings
            for ticker in list(holdings.keys()):
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * current_price
            holdings = {}
            peak_prices = {}
            last_rebalance_date = date

            # Get VIX
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                current_vix = vix_at_date.iloc[-1]['close']
            else:
                current_vix = 20

            # Score all stocks (use V28 scoring)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:
                    try:
                        score = bot.score_stock(ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score
                    except:
                        pass

            # V28 regime-based portfolio sizing
            top_n = bot.determine_portfolio_size(date)

            # Get top N stocks and store for mid-month redeployment
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            current_ranked_stocks = [(t, s) for t, s in ranked if s > 0]  # Store all ranked
            top_stocks = current_ranked_stocks[:top_n]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # VIX-based cash reserve
            if current_vix < 30:
                cash_reserve = 0.05 + (current_vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (current_vix - 30) * 0.0125
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
                    peak_prices[ticker] = current_price  # Initialize peak
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
    logger.info(f"\nV35b Trailing Stop Statistics:")
    logger.info(f"  Trailing stop triggers: {trailing_stop_triggers}")
    logger.info(f"  Mid-month redeployments: {mid_month_redeployments}")

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
    logger.info("V35b TEST: TRAILING STOP LOSS ONLY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Change: 12% trailing stop loss from peak")
    logger.info("Everything else: Identical to V28")
    logger.info("")
    logger.info("Expected: +0.5-1.5% annual, -3% to -5% better drawdown")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Run V35b backtest
    portfolio_df = run_v35b_backtest_with_trailing_stops(bot)
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
        f"{'V35b (Trailing Stops)':<25} "
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
    if improvement >= 0.5:
        logger.info(f"✅ SUCCESS: Trailing stops improve returns by +{improvement:.1f}%")
        if dd_improvement > 1:
            logger.info(f"✅ BONUS: Drawdown improved by +{dd_improvement:.1f}%")
        logger.info("   This feature WORKS - include in final strategy")
    elif improvement > 0:
        logger.info(f"⚠️  PARTIAL: Small improvement of +{improvement:.1f}%")
        logger.info("   Consider including if drawdown improved")
    else:
        logger.info(f"❌ NO IMPROVEMENT: Returns worse by {improvement:.1f}%")
        logger.info("   Skip this feature")

    return metrics


if __name__ == "__main__":
    results = main()
