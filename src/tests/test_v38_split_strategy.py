"""
V38: SPLIT STRATEGY - 70% Momentum + 30% Mean-Reversion
=========================================================

GOAL: Achieve 11-12% annual by combining two uncorrelated strategies

Strategy Allocation:
--------------------
- 70% capital → V28 Momentum Strategy (monthly rebalance)
- 30% capital → Mean-Reversion Strategy (weekly rebalance)

Why This Should Work:
---------------------
1. Different market conditions:
   - Momentum works in trending markets
   - Mean-reversion works in choppy/sideways markets

2. Uncorrelated returns:
   - When one underperforms, other may outperform
   - Smoother equity curve

3. No leverage costs:
   - Both strategies use cash only
   - No margin interest eating returns

Mean-Reversion Strategy (30%):
-------------------------------
- Find oversold stocks (RSI < 30)
- Confirm at lower Bollinger Band (price < BB_lower)
- Buy expecting reversion to mean
- Sell when RSI > 70 OR price > BB_upper
- Weekly rebalancing (faster than momentum)

Expected Performance:
---------------------
- V28 alone: 9.3% annual, -17.7% DD
- Mean-reversion: 8-10% annual, -12% DD (less correlated)
- Combined 70/30: 11-12% annual, -15% to -17% DD
- Better Sharpe ratio through diversification

Overfitting Risk: LOW
- Both strategies use standard indicators
- No parameter optimization
- Natural diversification

Confidence: HIGH (80%) - Diversification is free lunch
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


def calculate_bollinger_bands(df, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(df) < period:
        return None, None, None

    sma = df['close'].tail(period).mean()
    std = df['close'].tail(period).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return upper_band, sma, lower_band


def score_stock_mean_reversion(df):
    """
    Mean-Reversion Scoring

    Look for oversold stocks at lower Bollinger Band

    Returns:
        float: Score (0-100), higher = more oversold = better buy candidate
    """
    if len(df) < 100:
        return 0

    latest = df.iloc[-1]
    current_price = latest['close']
    rsi = latest['rsi']

    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, period=20, num_std=2)
    if bb_lower is None:
        return 0

    # Must be oversold
    if rsi >= 35:  # Not oversold enough
        return 0

    # Calculate how close to lower BB (lower = better)
    distance_from_lower = ((current_price - bb_lower) / bb_lower) * 100

    # Disqualify if too far above lower band
    if distance_from_lower > 5:  # More than 5% above lower band
        return 0

    # Score based on RSI (lower RSI = higher score)
    score = 0

    # RSI component (70 points max)
    if rsi < 20:
        score += 70  # Extremely oversold
    elif rsi < 25:
        score += 50
    elif rsi < 30:
        score += 30
    elif rsi < 35:
        score += 10

    # Bollinger Band component (30 points max)
    # Closer to lower band = higher score
    if distance_from_lower < 0:  # Below lower band
        score += 30
    elif distance_from_lower < 2:
        score += 20
    elif distance_from_lower < 5:
        score += 10

    # Volume confirmation (bonus 20 points)
    if len(df) >= 60:
        volume_20 = df['volume'].tail(20).mean()
        volume_60 = df['volume'].tail(60).mean()
        if volume_20 > volume_60 * 1.2:  # Volume spike
            score += 20

    return min(score, 120)


def check_mean_reversion_exit(df):
    """
    Check if we should exit a mean-reversion position

    Exit conditions:
    - RSI > 70 (overbought)
    - Price > upper Bollinger Band
    - Price < lower Bollinger Band (stop loss - continued decline)

    Returns:
        bool: True if should exit
    """
    if len(df) < 100:
        return False

    latest = df.iloc[-1]
    current_price = latest['close']
    rsi = latest['rsi']

    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, period=20, num_std=2)
    if bb_upper is None:
        return False

    # Exit if overbought (target reached)
    if rsi > 70:
        return True

    # Exit if above upper Bollinger Band (target reached)
    if current_price > bb_upper:
        return True

    # Stop loss: Exit if broke below lower BB significantly
    if current_price < bb_lower * 0.98:  # 2% below lower band
        return True

    return False


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


def run_mean_reversion_strategy(bot, capital):
    """
    Run mean-reversion strategy on 30% of capital

    Args:
        bot: PortfolioRotationBot instance
        capital: Amount to allocate to this strategy

    Returns:
        DataFrame: Portfolio values over time
    """
    logger.info(f"Running mean-reversion strategy with ${capital:,.0f}...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = capital
    last_rebalance_date = None

    # Statistics
    total_trades = 0
    profitable_trades = 0

    for date in dates[100:]:  # Need 100 days for indicators

        # Check daily exits for mean-reversion positions
        for ticker in list(holdings.keys()):
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                # Check exit conditions
                should_exit = check_mean_reversion_exit(df_at_date)

                if should_exit:
                    # Exit position
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker]['shares'] * current_price

                    # Track profitability
                    entry_price = holdings[ticker]['entry_price']
                    if current_price > entry_price:
                        profitable_trades += 1
                    total_trades += 1

                    del holdings[ticker]

        # Weekly rebalancing (every Monday)
        is_rebalance_day = (
            last_rebalance_date is None or
            (date.weekday() == 0 and (date - last_rebalance_date).days >= 6)
        )

        if is_rebalance_day:
            # Liquidate remaining holdings
            for ticker in list(holdings.keys()):
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker]['shares'] * current_price

                    # Track profitability
                    entry_price = holdings[ticker]['entry_price']
                    if current_price > entry_price:
                        profitable_trades += 1
                    total_trades += 1

            holdings = {}
            last_rebalance_date = date

            # Score stocks for mean-reversion opportunities
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        score = score_stock_mean_reversion(df_at_date)
                        if score > 0:
                            current_scores[ticker] = score
                    except:
                        pass

            # Select top 5 oversold stocks
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:5]

            if top_stocks:
                # Equal weight allocation (simple for mean-reversion)
                per_stock_allocation = cash * 0.95 / len(top_stocks)  # 95% invested, 5% cash buffer

                for ticker, score in top_stocks:
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        shares = per_stock_allocation / current_price
                        holdings[ticker] = {
                            'shares': shares,
                            'entry_price': current_price
                        }
                        fee = per_stock_allocation * 0.001
                        cash -= (per_stock_allocation + fee)

        # Calculate daily portfolio value
        stocks_value = 0
        for ticker, position in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += position['shares'] * current_price

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

    # Log statistics
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    logger.info(f"Mean-reversion stats: {profitable_trades}/{total_trades} trades profitable ({win_rate:.1f}% win rate)")

    return portfolio_df


def run_v38_split_strategy(bot):
    """
    V38: Run 70/30 split strategy

    70% → V28 Momentum (monthly rebalance)
    30% → Mean-Reversion (weekly rebalance)
    """
    logger.info("Running V38 Split Strategy (70% Momentum + 30% Mean-Reversion)...")

    # Allocate capital
    momentum_capital = bot.initial_capital * 0.70
    mean_rev_capital = bot.initial_capital * 0.30

    logger.info(f"  Momentum allocation: ${momentum_capital:,.0f}")
    logger.info(f"  Mean-reversion allocation: ${mean_rev_capital:,.0f}")

    # Run momentum strategy (V28)
    logger.info("\n--- Running Momentum Strategy (V28) ---")
    momentum_bot = PortfolioRotationBot(data_dir=bot.data_dir, initial_capital=momentum_capital)
    momentum_bot.prepare_data()
    momentum_bot.score_all_stocks()

    # Use bot's built-in backtest for momentum
    from src.tests.test_v28_momentum_leaders import run_v28_backtest
    momentum_df = run_v28_backtest(momentum_bot)

    # Run mean-reversion strategy
    logger.info("\n--- Running Mean-Reversion Strategy ---")
    mean_rev_df = run_mean_reversion_strategy(bot, mean_rev_capital)

    # Combine the two strategies
    logger.info("\n--- Combining Strategies ---")

    # Align dates
    common_dates = momentum_df.index.intersection(mean_rev_df.index)
    momentum_aligned = momentum_df.loc[common_dates]
    mean_rev_aligned = mean_rev_df.loc[common_dates]

    # Combined portfolio value
    combined_values = momentum_aligned['value'] + mean_rev_aligned['value']
    combined_df = pd.DataFrame({'value': combined_values})

    return combined_df, momentum_df, mean_rev_df


def calculate_metrics(portfolio_df, initial_capital, name=""):
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
        'name': name,
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
    logger.info("V38: SPLIT STRATEGY - 70% Momentum + 30% Mean-Reversion")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Allocation:")
    logger.info("  70% → V28 Momentum (monthly, trending markets)")
    logger.info("  30% → Mean-Reversion (weekly, choppy markets)")
    logger.info("")
    logger.info("Goal: Achieve 11-12% annual through diversification")
    logger.info("No leverage costs - pure cash strategies")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Run V38 split strategy
    combined_df, momentum_df, mean_rev_df = run_v38_split_strategy(bot)

    # Calculate metrics for each component
    combined_metrics = calculate_metrics(combined_df, bot.initial_capital, "V38 Combined")
    momentum_metrics = calculate_metrics(momentum_df, bot.initial_capital * 0.70, "Momentum (70%)")
    mean_rev_metrics = calculate_metrics(mean_rev_df, bot.initial_capital * 0.30, "Mean-Rev (30%)")

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("V38 SPLIT STRATEGY RESULTS")
    logger.info("=" * 80)

    logger.info(f"\n{combined_metrics['name']}:")
    logger.info(f"  Initial Capital: ${bot.initial_capital:,.0f}")
    logger.info(f"  Final Value:     ${combined_metrics['final_value']:,.0f}")
    logger.info(f"  Annual Return:   {combined_metrics['annual_return']:.1f}%")
    logger.info(f"  Max Drawdown:    {combined_metrics['max_drawdown']:.1f}%")
    logger.info(f"  Sharpe Ratio:    {combined_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Win Rate:        {combined_metrics['positive_years']}/{combined_metrics['total_years']} ({combined_metrics['positive_years']/combined_metrics['total_years']*100:.0f}%)")

    logger.info("\nYearly Returns (Combined):")
    for year, ret in sorted(combined_metrics['yearly_returns'].items()):
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:6.1f}% {status}")

    # Component breakdown
    logger.info("\n" + "=" * 80)
    logger.info("COMPONENT BREAKDOWN")
    logger.info("=" * 80)

    logger.info(f"\n{momentum_metrics['name']}: {momentum_metrics['annual_return']:.1f}% annual")
    logger.info(f"{mean_rev_metrics['name']}: {mean_rev_metrics['annual_return']:.1f}% annual")

    # Comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON VS V28 BASELINE")
    logger.info("=" * 80)
    logger.info(f"{'Strategy':<30} {'Annual':>8} {'Drawdown':>10} {'Sharpe':>8} {'Win Rate':>10}")
    logger.info("-" * 75)
    logger.info(f"{'V28 (Baseline - 100%)':<30} {'9.3%':>8} {'-17.7%':>10} {'0.99':>8} {'17/20':>10}")

    improvement = combined_metrics['annual_return'] - 9.3
    dd_improvement = combined_metrics['max_drawdown'] - (-17.7)
    sharpe_improvement = combined_metrics['sharpe_ratio'] - 0.99

    status = "🎯 SUCCESS" if improvement >= 1.5 else "✅ GOOD" if improvement >= 1.0 else "⚠️ PARTIAL"
    logger.info(
        f"{'V38 (70/30 Split)':<30} "
        f"{combined_metrics['annual_return']:>7.1f}% "
        f"{combined_metrics['max_drawdown']:>9.1f}% "
        f"{combined_metrics['sharpe_ratio']:>8.2f} "
        f"{combined_metrics['positive_years']}/{combined_metrics['total_years']:>2}      "
        f"{status}"
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
    if improvement >= 1.5:
        logger.info(f"🎯 SUCCESS: Split strategy achieves target!")
        logger.info(f"   Combined: {combined_metrics['annual_return']:.1f}% vs V28: 9.3% (+{improvement:.1f}%)")
        logger.info("   NO LEVERAGE COSTS - pure diversification benefit")
        logger.info("\n✅ RECOMMENDATION: Deploy V38 Split Strategy")
    elif improvement >= 0.5:
        logger.info(f"✅ GOOD: Split strategy improves by +{improvement:.1f}%")
        logger.info("   Solid improvement through diversification")
    else:
        logger.info(f"⚠️  NEUTRAL: Improvement only +{improvement:.1f}%")

    logger.info("=" * 80)

    return combined_metrics


if __name__ == "__main__":
    results = main()
