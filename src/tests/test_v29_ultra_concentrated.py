"""
V29: ULTRA-CONCENTRATED MOMENTUM + MULTI-TIMEFRAME ALIGNMENT
=============================================================

PROBLEM: V28 still uses 3-10 stock portfolio (regime-based)
- This diversification limits upside potential
- Multi-timeframe filtering could allow safer concentration

SOLUTION: Fixed 2-3 stock portfolio + Multi-Timeframe Alignment

Key Changes:
-------------
1. **Portfolio Size**: FIXED 2-3 stocks (vs V28's dynamic 3-10)
2. **Multi-Timeframe Alignment** (V29 NEW):
   - Daily: EMA-89
   - Weekly: 100-day MA (~20 weeks)
   - Monthly: 200-day MA (~10 months)
   - Bonus: +20 points if ALL timeframes aligned

3. **All V28 Features Maintained**:
   - 52-week breakout bonus
   - Relative strength vs SPY
   - Kelly position sizing
   - VIX cash reserves
   - Drawdown control

Why This Works:
---------------
- Multi-timeframe filters out weak trends (stronger stocks)
- Safer to concentrate when quality is higher
- No margin fees (unlike leverage)
- Simple, rule-based (no overfitting)

Expected Impact:
----------------
- Annual Return: **15-18%** (vs V28's 9.4%)
- Max Drawdown: **-25% to -35%** (vs V28's -18%)
- Win Rate: **70-75%** (lower due to concentration)
- Sharpe Ratio: ~0.8-0.9 (lower but acceptable for higher returns)

Trade-offs:
-----------
- Higher returns come with higher volatility
- Fewer stocks = more concentrated risk
- But multi-TF filtering should reduce false signals

Test Type: INDEPENDENT - Tests V29 concentration strategy
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
        logging.FileHandler('src/tests/v29_concentration_output.log'),
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


def run_v29_backtest(bot, top_n=3):
    """
    Run V29 backtest with ultra-concentration + multi-timeframe alignment

    V29 NEW features:
    1. Multi-timeframe alignment (0-20 points bonus)
    2. FIXED 2-3 stock portfolio (maximum concentration)

    V28 features (maintained):
    - 52-week breakout bonus
    - Relative strength vs SPY
    - Kelly position sizing
    - VIX cash reserves
    - Drawdown control

    Args:
        bot: PortfolioRotationBot instance
        top_n: Number of stocks to hold (2 or 3 for ultra-concentration)
    """
    logger.info(f"Running V29 backtest (Ultra-Concentrated: Top {top_n} stocks)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    # Track multi-TF stats
    multi_tf_picks = 0  # Stocks with all timeframes aligned
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

            # Score stocks (V29 scoring with multi-timeframe bonus)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 252:  # Need 1 year for 52w high
                    try:
                        score = bot.score_stock(ticker, df_at_date)
                        if score > 0:
                            current_scores[ticker] = score

                            # Track multi-TF stats
                            multi_tf_aligned = bot.calculate_multi_timeframe_alignment(df_at_date)
                            if multi_tf_aligned:
                                multi_tf_picks += 1
                            total_picks += 1
                    except:
                        pass

            # V29: FIXED top N stocks (ultra-concentration)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

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

            # Kelly position sizing (V22)
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)

            allocations = {
                ticker: invest_amount * weight
                for ticker, weight in kelly_weights.items()
            }

            # Drawdown control (V12)
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

    # Log multi-TF statistics
    multi_tf_pct = (multi_tf_picks / total_picks * 100) if total_picks > 0 else 0
    logger.info(f"\nV29 Multi-Timeframe Statistics:")
    logger.info(f"  Stocks with all TF aligned: {multi_tf_picks}/{total_picks} ({multi_tf_pct:.1f}%)")

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
    logger.info("V29 TEST: ULTRA-CONCENTRATED + MULTI-TIMEFRAME ALIGNMENT")
    logger.info("=" * 80)

    # Initialize bot
    data_dir = os.path.join(project_root, 'sp500_data', 'daily')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)

    logger.info("\nLoading data...")
    bot.prepare_data()
    bot.score_all_stocks()

    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    # Test with different concentration levels
    for top_n in [3, 2]:
        logger.info("\n" + "=" * 80)
        logger.info(f"TESTING: TOP {top_n} STOCKS")
        logger.info("=" * 80)

        # Run V29 backtest
        portfolio_df = run_v29_backtest(bot, top_n=top_n)

        # Calculate metrics
        metrics = calculate_metrics(portfolio_df, bot.initial_capital)

        # Display results
        logger.info("\n" + "-" * 80)
        logger.info(f"V29 RESULTS (Top {top_n} Stocks)")
        logger.info("-" * 80)
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
    logger.info("V29 Strategy Configuration:")
    logger.info("=" * 80)
    logger.info("  NEW V29 Features:")
    logger.info("    - Multi-timeframe alignment bonus (0-20 pts)")
    logger.info("    - FIXED 2-3 stock portfolio (ultra-concentration)")
    logger.info("  V28 Features (maintained):")
    logger.info("    - 52-week breakout bonus")
    logger.info("    - Relative strength vs SPY")
    logger.info("    - Kelly position sizing")
    logger.info("    - VIX cash reserves")
    logger.info("    - Drawdown control")
    logger.info("=" * 80)

    logger.info("\n✅ V29 test complete!")


if __name__ == "__main__":
    main()
