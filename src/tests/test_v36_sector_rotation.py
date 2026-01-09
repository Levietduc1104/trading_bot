"""
V36: SECTOR ROTATION
=====================

GOAL: Improve V28 (9.3% annual) by trading ONLY top-performing sectors

Strategy:
---------
1. Calculate 60-day momentum for each of 11 sectors
2. Select TOP 3 sectors each month
3. Apply V28 stock selection ONLY within those 3 sectors
4. Rebalance monthly

Why This Should Work:
---------------------
- Sector momentum premium (Moskowitz & Grinblatt, 2003)
  Academic finding: 2-4% annual excess return
- Economic intuition: Sectors rotate leadership (Tech → Financials → Energy...)
- Concentrates capital in sectors with institutional flows
- Diversifies across sectors without dilution

Expected Impact:
----------------
- Annual Return: +2.0-3.0% (V28: 9.3% → V36: 11-12%)
- Max Drawdown: Similar or slightly better (sector diversification)
- Win Rate: Maintain 85%+

Overfitting Risk: VERY LOW
- Fixed rule: Top 3 sectors (no optimization)
- 60-day lookback (standard momentum period)
- Academic backing

Confidence: HIGH (75%) - Well-established academic finding
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot, SECTOR_MAP

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


def calculate_sector_momentum(bot, date, periods=60):
    """
    Calculate 60-day momentum for each sector

    Args:
        bot: PortfolioRotationBot instance
        date: Current date
        periods: Lookback period (60 days = ~3 months)

    Returns:
        dict: {sector: momentum_pct}
    """
    sector_returns = defaultdict(list)

    # Calculate returns for each stock
    for ticker, df in bot.stocks_data.items():
        df_at_date = df[df.index <= date]

        if len(df_at_date) < periods:
            continue

        # Get sector for this ticker
        sector = SECTOR_MAP.get(ticker, 'Unknown')
        if sector == 'Unknown':
            continue

        # Calculate 60-day return
        current_price = df_at_date.iloc[-1]['close']
        past_price = df_at_date.iloc[-periods]['close']
        stock_return = ((current_price / past_price) - 1) * 100

        sector_returns[sector].append(stock_return)

    # Average returns within each sector
    sector_momentum = {}
    for sector, returns in sector_returns.items():
        if returns:
            sector_momentum[sector] = np.mean(returns)
        else:
            sector_momentum[sector] = -999  # No data

    return sector_momentum


def get_top_sectors(sector_momentum, top_n=3):
    """
    Get top N sectors by momentum

    Args:
        sector_momentum: dict of {sector: momentum_pct}
        top_n: Number of sectors to select (default: 3)

    Returns:
        list: Top N sector names
    """
    # Sort by momentum
    ranked_sectors = sorted(
        sector_momentum.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Get top N
    top_sectors = [sector for sector, momentum in ranked_sectors[:top_n] if momentum > -999]

    return top_sectors


def run_v36_sector_rotation_backtest(bot, top_n_sectors=3):
    """
    V36: Sector Rotation Backtest

    Changes from V28:
    - Calculate sector momentum monthly
    - Select top 3 sectors
    - ONLY trade stocks in those sectors
    - Everything else identical to V28
    """
    logger.info(f"Running V36 Sector Rotation backtest (Top {top_n_sectors} sectors)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    # Track sector selection history
    sector_selection_history = []

    for date in dates[252:]:  # Need 252 days for calculations
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

            # =================================
            # V36: SECTOR MOMENTUM SELECTION
            # =================================
            sector_momentum = calculate_sector_momentum(bot, date, periods=60)
            top_sectors = get_top_sectors(sector_momentum, top_n=top_n_sectors)

            # Log sector selection
            sector_selection_history.append({
                'date': date,
                'top_sectors': top_sectors,
                'momentum': {sector: sector_momentum.get(sector, 0) for sector in top_sectors}
            })

            # Score stocks (V28 scoring) ONLY in top sectors
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                # Check if ticker is in a top sector
                ticker_sector = SECTOR_MAP.get(ticker, 'Unknown')
                if ticker_sector not in top_sectors:
                    continue  # Skip stocks not in top sectors

                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:
                    try:
                        score = bot.score_stock(ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score
                    except:
                        pass

            # V28 regime-based portfolio sizing
            top_n_stocks = bot.determine_portfolio_size(date)

            # Get top N stocks (from top sectors only)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n_stocks]

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

    # Analyze sector rotation patterns
    sector_counts = defaultdict(int)
    for entry in sector_selection_history:
        for sector in entry['top_sectors']:
            sector_counts[sector] += 1

    logger.info(f"\nV36 Sector Rotation Statistics:")
    logger.info(f"  Total rebalances: {len(sector_selection_history)}")
    logger.info(f"\n  Sector selection frequency:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(sector_selection_history) * 100
        logger.info(f"    {sector:<25}: {count:>3} times ({pct:>5.1f}%)")

    # Show recent sector selections
    logger.info(f"\n  Recent sector selections (last 6 months):")
    for entry in sector_selection_history[-6:]:
        sectors_str = ", ".join(entry['top_sectors'])
        logger.info(f"    {entry['date'].strftime('%Y-%m')}: {sectors_str}")

    return portfolio_df, sector_selection_history


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

    # Downside deviation (for Sortino)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.001
    sortino = (annual_return / 100) / downside_std if downside_std > 0 else 0

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
        'sortino_ratio': sortino,
        'final_value': final_value,
        'yearly_returns': yearly_returns,
        'positive_years': positive_years,
        'total_years': len(yearly_returns)
    }


def main():
    logger.info("=" * 80)
    logger.info("V36: SECTOR ROTATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Strategy:")
    logger.info("  1. Calculate 60-day momentum for 11 sectors")
    logger.info("  2. Select TOP 3 sectors")
    logger.info("  3. Apply V28 stock selection WITHIN those sectors only")
    logger.info("")
    logger.info("Academic Backing: Moskowitz & Grinblatt (2003)")
    logger.info("Expected: +2.0-3.0% annual (V28: 9.3% → V36: 11-12%)")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")
    logger.info(f"Sectors: {len(set(SECTOR_MAP.values()))} unique sectors")

    # Run V36 backtest
    portfolio_df, sector_history = run_v36_sector_rotation_backtest(bot, top_n_sectors=3)
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("V36 RESULTS")
    logger.info("=" * 80)
    logger.info(f"Initial Capital: ${bot.initial_capital:,.0f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio:   {metrics['sortino_ratio']:.2f}")
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

    status = "🎯 SUCCESS" if improvement >= 2.0 else "✅ GOOD" if improvement >= 1.0 else "⚠️ PARTIAL" if improvement > 0 else "❌ WORSE"
    logger.info(
        f"{'V36 (Sector Rotation)':<25} "
        f"{metrics['annual_return']:>7.1f}% "
        f"{metrics['max_drawdown']:>9.1f}% "
        f"{metrics['sharpe_ratio']:>8.2f} "
        f"{metrics['positive_years']}/{metrics['total_years']:>2}      "
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
    if improvement >= 2.0:
        logger.info(f"🎯 SUCCESS: Sector rotation WORKS! +{improvement:.1f}% annual improvement")
        logger.info(f"   V36: {metrics['annual_return']:.1f}% vs V28: 9.3%")
        logger.info("\n✅ RECOMMENDATION: Deploy V36 Sector Rotation as new production strategy")
        logger.info("   Ready to push toward 12-15% with additional enhancements")
    elif improvement >= 1.0:
        logger.info(f"✅ GOOD: Sector rotation helps by +{improvement:.1f}%")
        logger.info("   Solid improvement - consider deployment")
    elif improvement > 0:
        logger.info(f"⚠️  PARTIAL: Small improvement of +{improvement:.1f}%")
        logger.info("   May not be worth added complexity")
    else:
        logger.info(f"❌ NO IMPROVEMENT: Returns worse by {improvement:.1f}%")
        logger.info("   Sector rotation didn't work - try different approach")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    if improvement >= 1.0:
        logger.info("If deploying V36:")
        logger.info("  - Consider adding bull market filter (better drawdown)")
        logger.info("  - Test different top_n_sectors (2 vs 3 vs 4)")
        logger.info("  - Combine with earnings momentum overlay")
        logger.info("")
        logger.info("Target: 12-15% annual return")
    else:
        logger.info("Alternative approaches:")
        logger.info("  - Bull market filter (200-day MA)")
        logger.info("  - Machine learning (XGBoost ranking)")
        logger.info("  - Different sector selection criteria")
    logger.info("=" * 80)

    return metrics


if __name__ == "__main__":
    results = main()
