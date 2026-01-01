"""
V21: TACTICAL PORTFOLIO SIZE (Adaptive Concentration)
======================================================

Goal: Beat V13's 9.8% by varying portfolio concentration based on market conditions

Core Idea:
- VIX < 15 (very calm): 3 stocks (maximum concentration - capture alpha)
- VIX 15-25 (normal): 5 stocks (current optimal)
- VIX > 25 (stressed): 7 stocks (more diversification - reduce risk)

Why it should work:
1. Calm markets: Few stocks sufficient, concentration captures more alpha
2. Normal markets: 5 stocks is balanced (proven optimal)
3. Volatile markets: More stocks reduce idiosyncratic risk

Expected:
- 10.0-10.5% annual return (vs 9.8% for fixed 5 stocks)
- Similar or better Sharpe ratio
- Potentially lower max drawdown (diversify in stress)
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


def get_tactical_portfolio_size(vix_value, vix_thresholds=(15, 25)):
    """
    Determine portfolio size based on VIX level

    Args:
        vix_value: Current VIX reading
        vix_thresholds: (calm_threshold, stress_threshold)

    Returns:
        int: Number of stocks to hold (3, 5, or 7)
    """
    calm_threshold, stress_threshold = vix_thresholds

    if vix_value < calm_threshold:
        return 3  # Very calm - maximum concentration
    elif vix_value < stress_threshold:
        return 5  # Normal - balanced
    else:
        return 7  # Stressed - maximum diversification


def run_tactical_portfolio_backtest(bot, vix_thresholds=(15, 25)):
    """
    V21: Backtest with tactical portfolio sizing

    All V13 features remain:
    - VIX regime detection
    - Momentum-strength weighting
    - Drawdown control
    - Dynamic cash reserves

    NEW: Portfolio size adapts to VIX
    """
    logger.info(f"VIX thresholds: < {vix_thresholds[0]} → 3 stocks, {vix_thresholds[0]}-{vix_thresholds[1]} → 5 stocks, > {vix_thresholds[1]} → 7 stocks")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    portfolio_size_history = []

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
                vix = 20  # Default to normal conditions

            # TACTICAL PORTFOLIO SIZE based on VIX
            target_portfolio_size = get_tactical_portfolio_size(vix, vix_thresholds)

            portfolio_size_history.append({
                'date': date,
                'vix': vix,
                'portfolio_size': target_portfolio_size
            })

            # Score stocks (V13 scoring)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            # Get top N stocks (N varies by VIX)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [t for t, s in ranked if s > 0][:target_portfolio_size]

            # Calculate VIX-based cash reserve (V8)
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            # Calculate invest amount
            invest_amount = cash * (1 - cash_reserve)

            # Calculate allocations with momentum weighting (V13)
            if top_stocks:
                momentum_weights = {}
                for ticker in top_stocks:
                    weight = bot.calculate_momentum_strength_weight(ticker, date, lookback_months=9)
                    if weight > 0:
                        momentum_weights[ticker] = weight

                total_weight = sum(momentum_weights.values())
                if total_weight > 0:
                    allocations = {
                        ticker: (weight / total_weight) * invest_amount
                        for ticker, weight in momentum_weights.items()
                    }
                else:
                    allocation_per_stock = invest_amount / len(top_stocks)
                    allocations = {ticker: allocation_per_stock for ticker in top_stocks}
            else:
                allocations = {}

            # Apply V12 drawdown control
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

            # Buy stocks
            for ticker in top_stocks:
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
    size_df = pd.DataFrame(portfolio_size_history).set_index('date')

    return portfolio_df, size_df


def main():
    """Test V21 with tactical portfolio sizing"""
    logger.info("="*80)
    logger.info("V21: TACTICAL PORTFOLIO SIZE (ADAPTIVE CONCENTRATION)")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy:")
    logger.info("  Base: V13 (Momentum + Drawdown Control + VIX Regime)")
    logger.info("  NEW: Vary portfolio size (3-7 stocks) based on VIX")
    logger.info("")
    logger.info("Rules (CALIBRATED for VIX proxy):")
    logger.info("  VIX < 37:  3 stocks (calm - 20th percentile)")
    logger.info("  VIX 37-61: 5 stocks (normal - 20th-80th percentile)")
    logger.info("  VIX > 61:  7 stocks (stress - 80th percentile)")
    logger.info("")
    logger.info("Note: VIX is proxy from SPY 30-day volatility, range ~25-142")
    logger.info("")
    logger.info("Goal: Beat V13's 9.8% annual return")
    logger.info("")

    # Test different VIX threshold combinations (CALIBRATED for VIX proxy)
    threshold_sets = [
        (37, 61, "Balanced 20/60/20"),
        (35, 65, "More calm (25/50/25)"),
        (39, 58, "More balanced (15/70/15)"),
        (40, 60, "Moderate (10/80/10)"),
        (43, 61, "Conservative (50/30/20)")
    ]

    results = []

    for thresholds in threshold_sets:
        calm_thresh, stress_thresh, label = thresholds
        logger.info("="*80)
        logger.info(f"Testing: {label}")
        logger.info(f"  < {calm_thresh} VIX → 3 stocks")
        logger.info(f"  {calm_thresh}-{stress_thresh} VIX → 5 stocks")
        logger.info(f"  > {stress_thresh} VIX → 7 stocks")
        logger.info("")

        bot = PortfolioRotationBot(
            data_dir='sp500_data/daily',
            initial_capital=100000
        )

        bot.prepare_data()
        bot.score_all_stocks()

        portfolio_df, size_df = run_tactical_portfolio_backtest(
            bot,
            vix_thresholds=(calm_thresh, stress_thresh)
        )

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

        # Portfolio size distribution
        size_counts = size_df['portfolio_size'].value_counts().sort_index()
        total_rebalances = len(size_df)
        size_pct = {size: (count/total_rebalances*100) for size, count in size_counts.items()}

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

        results.append({
            'label': label,
            'thresholds': (calm_thresh, stress_thresh),
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'realized_vol': realized_vol,
            'size_distribution': size_pct,
            'positive_years': positive_years,
            'total_years': total_years,
            'yearly_returns': yearly_returns,
            'final_value': final_value
        })

        logger.info(f"  Annual: {annual_return:.1f}%, DD: {max_drawdown:.1f}%, Sharpe: {sharpe:.2f}")
        logger.info(f"  Portfolio size distribution:")
        for size in sorted(size_counts.index):
            pct = size_pct.get(size, 0)
            logger.info(f"    {size} stocks: {pct:.1f}% of time")
        logger.info("")

    # RESULTS SUMMARY
    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS SUMMARY - V21 TACTICAL PORTFOLIO SIZE")
    logger.info("="*80)
    logger.info("")

    logger.info(f"{'Configuration':<25} {'Annual':>8} {'Drawdown':>10} {'Sharpe':>8} {'Win Rate':>10} {'Status'}")
    logger.info("-" * 80)

    for r in results:
        status = "🎯" if r['annual_return'] > 9.8 else ""
        logger.info(
            f"{r['label']:<25} "
            f"{r['annual_return']:>7.1f}% "
            f"{r['max_drawdown']:>9.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['positive_years']}/{r['total_years']:>2}"
            f"  {status}"
        )

    # Best result
    best = max(results, key=lambda x: x['annual_return'])

    logger.info("")
    logger.info("="*80)
    logger.info("BEST CONFIGURATION")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Thresholds: {best['label']}")
    logger.info(f"  VIX < {best['thresholds'][0]}: 3 stocks")
    logger.info(f"  VIX {best['thresholds'][0]}-{best['thresholds'][1]}: 5 stocks")
    logger.info(f"  VIX > {best['thresholds'][1]}: 7 stocks")
    logger.info("")
    logger.info(f"Annual Return: {best['annual_return']:.1f}%")
    logger.info(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio: {best['sharpe']:.2f}")
    logger.info(f"Realized Volatility: {best['realized_vol']:.1f}%")
    logger.info(f"Win Rate: {best['positive_years']}/{best['total_years']} ({best['positive_years']/best['total_years']*100:.0f}%)")
    logger.info(f"Final Value: ${best['final_value']:,.0f}")
    logger.info("")
    logger.info("Portfolio Size Distribution:")
    for size in sorted(best['size_distribution'].keys()):
        pct = best['size_distribution'][size]
        logger.info(f"  {size} stocks: {pct:.1f}% of time")

    # Comparison to V13
    logger.info("")
    logger.info("="*80)
    logger.info("COMPARISON: V13 vs V21")
    logger.info("="*80)
    logger.info("")
    logger.info("V13 (Fixed 5 stocks):")
    logger.info("  Annual: 9.8%, DD: -19.1%, Sharpe: 1.07")
    logger.info("")
    logger.info(f"V21 ({best['label']}):")
    logger.info(f"  Annual: {best['annual_return']:.1f}%, DD: {best['max_drawdown']:.1f}%, Sharpe: {best['sharpe']:.2f}")
    logger.info("")

    improvement = best['annual_return'] - 9.8
    if improvement > 0:
        logger.info(f"V21 BEATS V13 by +{improvement:.1f}% annual\!")
        logger.info("")
        logger.info("="*80)
        logger.info("SUCCESS: TACTICAL PORTFOLIO SIZE IMPROVES RETURNS\!")
        logger.info("="*80)
        logger.info(f"New annual return: {best['annual_return']:.1f}%")
        logger.info(f"Improvement: +{improvement:.1f}% vs V13")
    else:
        logger.info(f"V13 is still better by {-improvement:.1f}%")
        logger.info("Recommendation: Stick with V13 (fixed 5 stocks)")

    # Yearly breakdown for best
    logger.info("")
    logger.info("="*80)
    logger.info(f"YEARLY RETURNS - BEST ({best['label']})")
    logger.info("="*80)

    for year in sorted(best['yearly_returns'].keys()):
        ret = best['yearly_returns'][year]
        status = "+" if ret > 0 else ""
        logger.info(f"  {year}: {ret:>6.1f}% {status}")

    logger.info("")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    results = main()
