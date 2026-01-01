"""
V22: KELLY POSITION SIZING
===========================

Goal: Instead of equal-weight positions, size positions based on CONVICTION (score)

Core Idea:
- High score (120+) → Larger position (30%)
- Medium score (80-100) → Medium position (20%)
- Low score (60-80) → Smaller position (10%)

Why it should work:
- If our scoring works, higher scores should outperform
- Kelly Criterion says: bet more when edge is higher
- Concentrating capital in best ideas should boost returns

Test 4 position sizing methods:
1. Simple Tiered: Fixed tiers (30/25/20/15/10%)
2. Proportional: Weight ∝ score
3. Exponential: Weight ∝ score² (aggressive)
4. Sqrt: Weight ∝ √score (conservative)
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def calculate_kelly_weights(scores, method='simple'):
    """
    Calculate position weights based on scores using different Kelly methods

    Args:
        scores: List of (ticker, score) tuples for top 5 stocks
        method: 'simple', 'proportional', 'exponential', 'sqrt'

    Returns:
        dict: {ticker: weight} where weights sum to 1.0
    """
    tickers = [t for t, s in scores]
    score_values = [s for t, s in scores]

    if method == 'simple':
        # Fixed tiers based on rank
        tier_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        weights = {ticker: tier_weights[i] for i, ticker in enumerate(tickers)}

    elif method == 'proportional':
        # Weight proportional to score
        total_score = sum(score_values)
        if total_score > 0:
            weights = {ticker: score / total_score for ticker, score in scores}
        else:
            # Fallback to equal weight
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    elif method == 'exponential':
        # Weight proportional to score²
        squared_scores = [s**2 for s in score_values]
        total_squared = sum(squared_scores)
        if total_squared > 0:
            weights = {ticker: (score**2) / total_squared for ticker, score in scores}
        else:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    elif method == 'sqrt':
        # Weight proportional to √score (more conservative)
        sqrt_scores = [np.sqrt(max(0, s)) for s in score_values]
        total_sqrt = sum(sqrt_scores)
        if total_sqrt > 0:
            weights = {ticker: np.sqrt(max(0, score)) / total_sqrt for ticker, score in scores}
        else:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def run_v22_backtest(bot, kelly_method='simple'):
    """
    V22: Backtest with Kelly position sizing

    All V13 features remain:
    - VIX regime detection
    - Momentum-strength weighting for candidate selection
    - Drawdown control
    - Dynamic cash reserves

    NEW: Position sizes based on conviction (score)
    """
    logger.info(f"Kelly method: {kelly_method}")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    position_history = []

    for date in dates[100:]:
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

            # Score stocks (V13 scoring)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            # Get top 5 stocks
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:5]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # Calculate VIX-based cash reserve (V8)
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            # Calculate invest amount
            invest_amount = cash * (1 - cash_reserve)

            # 🆕 KELLY POSITION SIZING
            kelly_weights = calculate_kelly_weights(top_stocks, method=kelly_method)

            # Record position weights for analysis
            position_history.append({
                'date': date,
                'weights': kelly_weights.copy(),
                'scores': {t: s for t, s in top_stocks}
            })

            # Calculate allocations with Kelly weights
            allocations = {
                ticker: invest_amount * weight
                for ticker, weight in kelly_weights.items()
            }

            # Apply V12 drawdown control
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

            # Buy stocks
            top_tickers = [t for t, s in top_stocks]
            for ticker in top_tickers:
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

    return portfolio_df, position_df


def main():
    """Test V22 with different Kelly position sizing methods"""
    logger.info("="*80)
    logger.info("V22: KELLY POSITION SIZING")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy:")
    logger.info("  Base: V13 (Momentum + Drawdown Control + VIX Regime)")
    logger.info("  NEW: Size positions by conviction (score)")
    logger.info("")
    logger.info("Testing 4 position sizing methods:")
    logger.info("  1. Simple Tiered: 30/25/20/15/10% by rank")
    logger.info("  2. Proportional: weight ∝ score")
    logger.info("  3. Exponential: weight ∝ score² (aggressive)")
    logger.info("  4. Sqrt: weight ∝ √score (conservative)")
    logger.info("")
    logger.info("Goal: Beat V13's 9.8% annual return")
    logger.info("")

    # Test different Kelly methods
    kelly_methods = [
        ('simple', 'Simple Tiered'),
        ('proportional', 'Proportional'),
        ('exponential', 'Exponential'),
        ('sqrt', 'Square Root')
    ]

    results = []

    for method, label in kelly_methods:
        logger.info("="*80)
        logger.info(f"Testing: {label}")
        logger.info("")

        bot = PortfolioRotationBot(
            data_dir='sp500_data/daily',
            initial_capital=100000
        )

        bot.prepare_data()
        bot.score_all_stocks()

        portfolio_df, position_df = run_v22_backtest(bot, kelly_method=method)

        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        realized_vol = daily_returns.std() * np.sqrt(252) * 100

        # Yearly returns
        portfolio_df['year'] = portfolio_df.index.year
        yearly_returns = {}
        for year in portfolio_df['year'].unique():
            year_data = portfolio_df[portfolio_df['year'] == year]
            if len(year_data) > 1:
                year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
                yearly_returns[year] = year_return

        positive_years = sum(1 for r in yearly_returns.values() if r > 0)
        total_years = len(yearly_returns)

        # Analyze position concentration
        if len(position_df) > 0:
            # Average weight of top position
            avg_top_weight = []
            for idx, row in position_df.iterrows():
                weights = row['weights']
                if weights:
                    avg_top_weight.append(max(weights.values()))
            avg_max_position = np.mean(avg_top_weight) if avg_top_weight else 0.20
        else:
            avg_max_position = 0.20

        results.append({
            'method': method,
            'label': label,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'realized_vol': realized_vol,
            'positive_years': positive_years,
            'total_years': total_years,
            'yearly_returns': yearly_returns,
            'final_value': final_value,
            'avg_max_position': avg_max_position
        })

        logger.info(f"  Annual: {annual_return:.1f}%, DD: {max_drawdown:.1f}%, Sharpe: {sharpe:.2f}")
        logger.info(f"  Avg largest position: {avg_max_position*100:.1f}%")
        logger.info("")

    # RESULTS SUMMARY
    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS SUMMARY - V22 KELLY POSITION SIZING")
    logger.info("="*80)
    logger.info("")

    logger.info(f"{'Method':<20} {'Annual':>8} {'Drawdown':>10} {'Sharpe':>8} {'Max Pos':>9} {'Win Rate':>10} {'Status'}")
    logger.info("-" * 90)

    for r in results:
        status = "🎯" if r['annual_return'] > 9.8 else ""
        logger.info(
            f"{r['label']:<20} "
            f"{r['annual_return']:>7.1f}% "
            f"{r['max_drawdown']:>9.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['avg_max_position']*100:>8.1f}% "
            f"{r['positive_years']}/{r['total_years']:>2}"
            f"  {status}"
        )

    # Best result
    best = max(results, key=lambda x: x['annual_return'])

    logger.info("")
    logger.info("="*80)
    logger.info("BEST KELLY METHOD")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Method: {best['label']}")
    logger.info(f"Annual Return: {best['annual_return']:.1f}%")
    logger.info(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio: {best['sharpe']:.2f}")
    logger.info(f"Win Rate: {best['positive_years']}/{best['total_years']} ({best['positive_years']/best['total_years']*100:.0f}%)")
    logger.info(f"Final Value: ${best['final_value']:,.0f}")
    logger.info(f"Avg Largest Position: {best['avg_max_position']*100:.1f}%")
    logger.info("")

    # Comparison to V13
    logger.info("="*80)
    logger.info("COMPARISON: V13 vs V22")
    logger.info("="*80)
    logger.info("")
    logger.info("V13 (Equal Weight - 5 stocks):")
    logger.info("  Annual: 9.8%, DD: -19.1%, Sharpe: 1.07")
    logger.info("")
    logger.info(f"V22 ({best['label']}):")
    logger.info(f"  Annual: {best['annual_return']:.1f}%, DD: {best['max_drawdown']:.1f}%, Sharpe: {best['sharpe']:.2f}")
    logger.info("")

    improvement = best['annual_return'] - 9.8
    if improvement > 0:
        logger.info(f"V22 BEATS V13 by +{improvement:.1f}% annual!")
        logger.info("")
        logger.info("="*80)
        logger.info("SUCCESS: KELLY POSITION SIZING IMPROVES RETURNS!")
        logger.info("="*80)
        logger.info(f"New baseline for leverage testing: {best['annual_return']:.1f}% annual")
        logger.info(f"Best method: {best['label']}")
    else:
        logger.info(f"V13 is still better by {-improvement:.1f}%")
        logger.info("Kelly sizing didn't add value - equal weight is optimal")
        logger.info("This means our scoring doesn't differentiate well enough")

    # Yearly breakdown for best
    logger.info("")
    logger.info("="*80)
    logger.info(f"YEARLY RETURNS - BEST ({best['label']})")
    logger.info("="*80)

    for year in sorted(best['yearly_returns'].keys()):
        ret = best['yearly_returns'][year]
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:>6.1f}% {status}")

    logger.info("")
    logger.info("="*80)
    logger.info("NEXT STEP: Add leverage to best Kelly method")
    logger.info("="*80)

    return results, best


if __name__ == '__main__':
    results, best_method = main()
