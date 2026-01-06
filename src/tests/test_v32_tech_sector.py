"""
V32: TECH SECTOR MOMENTUM LEADERS
===================================

PROBLEM: V28 applies to all S&P 500 sectors
- Some sectors are low-growth (utilities, consumer staples)
- Dilutes returns by including slow-growth sectors
- Tech sector historically outperforms by 3-5% annually

SOLUTION: Apply V28 ONLY to Technology sector stocks

Key Changes:
-------------
1. **Tech Sector Filter** (V32 NEW):
   - Only trade stocks in Technology sector
   - 69 tech stocks (vs 473 total S&P 500)
   - Higher growth = higher momentum returns

2. **All V28 Features Maintained**:
   - 52-week breakout bonus
   - Relative strength vs SPY
   - Multi-timeframe alignment
   - Regime-based portfolio sizing (3-10 stocks)
   - Kelly position sizing
   - VIX cash reserves
   - Drawdown control

Why This Works:
---------------
- Tech sector premium: 14-16% annual (vs S&P 500's 10%)
- Higher growth companies = stronger momentum
- Tech has highest earnings growth
- V28 momentum filters work BEST in high-growth sectors
- Simple sector filter (no overfitting)

Expected Impact:
----------------
- Annual Return: **13-16%** (vs V28's 9.4%) ⭐ +3-7% improvement!
- Max Drawdown: **-25% to -35%** (vs V28's -18%) - higher volatility
- Sharpe Ratio: **0.9-1.1** (vs V28's 1.26) - lower but acceptable
- Win Rate: **75-80%** (vs V28's 85%)
- Volatility: Higher (tech is more volatile)

Trade-offs:
-----------
- Sector concentration risk (all eggs in tech basket)
- Higher drawdowns (tech crashes harder)
- Correlation risk (tech stocks move together)
- But much higher returns!

Historical Evidence:
-------------------
- Tech sector CAGR (2000-2024): ~14%
- S&P 500 CAGR: ~10%
- Tech premium: ~4% annually
- V28 should capture this premium

Test Type: INDEPENDENT - Tests V32 tech sector concentration
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('src/tests/v32_tech_sector_output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_tech_stocks(metadata_dir):
    """Load list of technology sector stocks from metadata"""
    tech_stocks = []

    metadata_path = os.path.join(project_root, metadata_dir)

    for filename in os.listdir(metadata_path):
        if filename.endswith('.json'):
            filepath = os.path.join(metadata_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if data.get('sector') == 'Technology':
                        ticker = data.get('ticker')
                        if ticker:
                            tech_stocks.append(ticker)
            except Exception as e:
                continue

    return tech_stocks


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


def run_v32_backtest(bot, tech_stocks):
    """
    Run V32 backtest - Tech Sector Only

    V32 = V28 applied to technology sector only

    All V28 features maintained:
    - 52-week breakout bonus
    - Relative strength vs SPY
    - Multi-timeframe alignment
    - Regime-based portfolio sizing
    - Kelly position sizing
    - VIX cash reserves
    - Drawdown control
    """
    logger.info(f"Running V32 backtest (Tech Sector Only: {len(tech_stocks)} stocks)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    # Track tech stock stats
    tech_picks = 0
    total_picks = 0

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

            # V32: Score ONLY tech stocks (filter by sector)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                # TECH FILTER: Only score tech stocks
                if ticker not in tech_stocks:
                    continue

                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:
                    try:
                        score = bot.score_stock(ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score
                            tech_picks += 1
                            total_picks += 1
                    except:
                        pass

            # V27: Determine portfolio size based on regime
            top_n = bot.determine_portfolio_size(date)

            # Get top N stocks (from tech stocks only)
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

    logger.info(f"\nV32 Tech Sector Statistics:")
    logger.info(f"  Total tech stock picks: {tech_picks}")
    logger.info(f"  Average tech stocks per period: {tech_picks/total_picks if total_picks > 0 else 0:.1f}")

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
    logger.info("V32 TEST: TECH SECTOR MOMENTUM LEADERS")
    logger.info("=" * 80)

    # Load tech stocks from metadata
    logger.info("\nLoading technology sector stocks...")
    tech_stocks = load_tech_stocks('sp500_data/metadata')
    logger.info(f"Found {len(tech_stocks)} technology stocks")
    logger.info(f"Tech stocks: {', '.join(sorted(tech_stocks)[:10])}... (showing first 10)")

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading ALL stock data...")
    bot.prepare_data()
    bot.score_all_stocks()

    logger.info(f"Loaded {len(bot.stocks_data)} total stocks")
    logger.info(f"Will trade only {len(tech_stocks)} tech stocks ({len(tech_stocks)/len(bot.stocks_data)*100:.1f}% of universe)")

    # Run V32 backtest
    portfolio_df = run_v32_backtest(bot, tech_stocks)

    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, bot.initial_capital)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("V32 RESULTS (Tech Sector Momentum Leaders)")
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
    logger.info("V32 vs V28 COMPARISON")
    logger.info("=" * 80)
    logger.info("  V28 (All Sectors):     9.4% annual, -18.0% DD, 1.26 Sharpe, 85% win")
    logger.info(f"  V32 (Tech Only):      {metrics['annual_return']:.1f}% annual, {metrics['max_drawdown']:.1f}% DD, {metrics['sharpe_ratio']:.2f} Sharpe, {positive_years/total_years*100:.0f}% win")
    logger.info("")
    logger.info("  Strategy Configuration:")
    logger.info("    - Sector: Technology ONLY (69 stocks)")
    logger.info("    - All V28 features maintained:")
    logger.info("      * 52-week breakout bonus")
    logger.info("      * Relative strength vs SPY")
    logger.info("      * Multi-timeframe alignment")
    logger.info("      * Regime-based portfolio sizing")
    logger.info("      * Kelly position sizing")
    logger.info("=" * 80)

    logger.info("\n✅ V32 test complete!")


if __name__ == "__main__":
    main()
