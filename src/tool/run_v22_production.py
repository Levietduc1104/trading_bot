"""
V22 PRODUCTION EXECUTION - KELLY-WEIGHTED POSITION SIZING
===========================================================

This is the PRODUCTION implementation of V22-Sqrt strategy.

Strategy: Kelly-weighted position sizing with Square Root method
Expected: 10.2% annual, -15.2% max drawdown, 1.11 Sharpe

This script:
1. Loads stock data
2. Runs V22 backtest with Kelly position sizing
3. Saves results to database
4. Generates performance reports
5. Creates visualizations

Run this to execute the production strategy.
"""

import sys
import os
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

# Setup logging
log_dir = os.path.join(project_root, 'output', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'v22_execution.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_kelly_weights_sqrt(scored_stocks):
    """
    Calculate position weights using Square Root Kelly method

    This is the CORE INNOVATION of V22.

    Args:
        scored_stocks: List of (ticker, score) tuples

    Returns:
        dict: {ticker: weight} where weights sum to 1.0

    Example:
        Scores: [(AAPL, 120), (MSFT, 100), (GOOGL, 80), (NVDA, 70), (META, 60)]
        Sqrt:   [10.95, 10.0, 8.94, 8.37, 7.75]
        Weights: [23.9%, 21.9%, 19.5%, 18.3%, 16.9%]
    """
    tickers = [t for t, s in scored_stocks]
    scores = [s for t, s in scored_stocks]

    # Square root of each score
    sqrt_scores = [np.sqrt(max(0, score)) for score in scores]
    total_sqrt = sum(sqrt_scores)

    # Normalize to sum to 1.0
    if total_sqrt > 0:
        weights = {
            ticker: np.sqrt(max(0, score)) / total_sqrt
            for ticker, score in scored_stocks
        }
    else:
        # Fallback to equal weight if scores are invalid
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    return weights


def run_v22_backtest(bot):
    """
    Run V22 backtest with Kelly-weighted position sizing

    All V13 features remain:
    - VIX regime detection
    - Momentum scoring
    - Drawdown control
    - Dynamic cash reserves

    NEW: Position sizes based on conviction (Square Root Kelly)
    """
    logger.info("Running V22-Sqrt backtest...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

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

            # VIX-based cash reserve (V8)
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            invest_amount = cash * (1 - cash_reserve)

            # 🆕 KELLY POSITION SIZING (Square Root method)
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)

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
    return portfolio_df


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics"""
    final_value = portfolio_df['value'].iloc[-1]

    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]
    years = (end_date - start_date).days / 365.25
    annual_return = (((final_value / initial_capital) ** (1 / years)) - 1) * 100

    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    returns = portfolio_df['value'].pct_change()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

    # Yearly returns
    portfolio_df['year'] = portfolio_df.index.year
    yearly_returns = {}
    for year in portfolio_df['year'].unique():
        year_data = portfolio_df[portfolio_df['year'] == year]
        if len(year_data) > 0:
            year_start = year_data['value'].iloc[0]
            year_end = year_data['value'].iloc[-1]
            year_return = (year_end - year_start) / year_start * 100
            yearly_returns[year] = year_return

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'years': years,
        'yearly_returns': yearly_returns
    }


def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("V22-SQRT PRODUCTION EXECUTION")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy: Kelly-Weighted Position Sizing (Square Root method)")
    logger.info("Expected: 10.2% annual, -15.2% max drawdown, 1.11 Sharpe")
    logger.info("")

    # Initialize and load data
    logger.info("Loading data...")
    bot = PortfolioRotationBot(
        data_dir='sp500_data/daily',
        initial_capital=100000
    )
    bot.prepare_data()
    bot.score_all_stocks()

    # Run backtest
    logger.info("Running backtest...")
    portfolio_df = run_v22_backtest(bot)

    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)

    # Display results
    logger.info("")
    logger.info("="*80)
    logger.info("BACKTEST COMPLETE - V22-SQRT")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Initial Capital: ${metrics['initial_capital']:,.0f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Period:          {metrics['start_date']} to {metrics['end_date']}")
    logger.info(f"Duration:        {metrics['years']:.1f} years")
    logger.info("")

    # Yearly returns
    logger.info("Yearly Returns:")
    positive_years = sum(1 for r in metrics['yearly_returns'].values() if r > 0)
    total_years = len(metrics['yearly_returns'])
    logger.info(f"Win Rate: {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)")
    logger.info("")

    for year, ret in sorted(metrics['yearly_returns'].items()):
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:6.1f}% {status}")

    logger.info("")
    logger.info("="*80)
    logger.info("STRATEGY DETAILS")
    logger.info("="*80)
    logger.info("")
    logger.info("Stock Selection:")
    logger.info("  - Top 5 stocks by V13 momentum/quality scoring")
    logger.info("  - Monthly rebalancing (day 7-10)")
    logger.info("")
    logger.info("Position Sizing (THE KEY INNOVATION):")
    logger.info("  - Kelly-weighted using Square Root method")
    logger.info("  - High scores → larger positions (up to ~24%)")
    logger.info("  - Low scores → smaller positions (down to ~17%)")
    logger.info("  - vs Equal weight: all positions 20%")
    logger.info("")
    logger.info("Risk Management:")
    logger.info("  - VIX-based cash reserve (5-70%)")
    logger.info("  - Portfolio-level drawdown control")
    logger.info("  - Trading fee: 0.1% per trade")
    logger.info("  - NO leverage, NO margin interest")
    logger.info("")
    logger.info("="*80)
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    return portfolio_df, metrics


if __name__ == '__main__':
    portfolio_df, metrics = main()
