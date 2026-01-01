"""
V18: PROGRESSIVE CASH REDEPLOYMENT

Problem: Cash stays idle too long after stress (discrete regime flips)
Solution: Redeploy cash progressively based on continuous market signals

Key Changes from V13:
1. Replace binary VIX thresholds with smooth continuous function
2. Incorporate multiple recovery signals:
   - Volatility declining (VIX improvement)
   - Drawdown improving (equity curve recovery)
   - Trend stabilizing (market strength)
3. Cash reserve adjusts gradually, not in steps

Why this boosts profit:
- Earlier participation in recoveries
- No leverage needed
- Drawdown protection maintained
- More responsive to changing conditions

Zero-bias because:
- Uses only historical data (backward-looking)
- Continuous functions (no optimization)
- Same fundamental rules, just smoother
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ProgressiveCashBot(PortfolioRotationBot):
    """V18: Progressive Cash Redeployment Strategy"""

    def calculate_progressive_cash_reserve(self, date, portfolio_values_df=None):
        """
        V18: PROGRESSIVE CASH REDEPLOYMENT

        Calculates cash reserve using continuous functions of market conditions.
        Cash is redeployed gradually as markets improve across multiple dimensions.

        Three signals (each 0-1, where 1 = bullish, 0 = bearish):
        1. Volatility Signal: VIX declining (smooth function, not threshold)
        2. Recovery Signal: Drawdown improving (equity curve healing)
        3. Trend Signal: Market trend strengthening (SPY vs MA200)

        Cash Reserve Formula:
            cash_reserve = base_cash * (1 - composite_signal)

        Where:
            base_cash = 0.70 (70% maximum in extreme stress)
            composite_signal = weighted average of 3 signals

        This creates smooth transitions:
        - All bullish (signal = 1.0) → 0% cash (fully invested)
        - All bearish (signal = 0.0) → 70% cash (maximum defense)
        - Mixed signals → 0-70% cash (proportional)

        Args:
            date: Current date for calculation
            portfolio_values_df: DataFrame with portfolio history (for recovery signal)

        Returns:
            float: Cash reserve percentage (0.0 to 0.70)
        """
        # Signal 1: VOLATILITY SIGNAL (VIX-based, continuous)
        volatility_signal = self._calculate_volatility_signal(date)

        # Signal 2: RECOVERY SIGNAL (drawdown improvement)
        recovery_signal = self._calculate_recovery_signal(portfolio_values_df)

        # Signal 3: TREND SIGNAL (market strength)
        trend_signal = self._calculate_trend_signal(date)

        # Weighted composite (emphasize volatility and trend)
        # Weights: 40% volatility, 30% trend, 30% recovery
        composite_signal = (
            0.40 * volatility_signal +
            0.30 * trend_signal +
            0.30 * recovery_signal
        )

        # Convert to cash reserve (inverse relationship)
        # Signal = 1.0 (all bullish) → 0% cash
        # Signal = 0.0 (all bearish) → 70% cash
        # Signal = 0.5 (mixed) → 35% cash
        base_cash = 0.70
        cash_reserve = base_cash * (1 - composite_signal)

        # Ensure bounds
        cash_reserve = np.clip(cash_reserve, 0.0, 0.70)

        return cash_reserve

    def _calculate_volatility_signal(self, date):
        """
        Signal 1: Volatility declining (smooth VIX function)

        Instead of: VIX < 30 = aggressive, VIX >= 30 = defensive
        Use smooth curve: VIX inversely maps to signal strength

        Formula: signal = 1 - ((VIX - 20) / 60)^0.8

        Mapping:
        - VIX = 20 (calm) → signal = 1.0 (very bullish)
        - VIX = 35 (moderate) → signal = 0.7
        - VIX = 50 (high) → signal = 0.4
        - VIX = 80 (panic) → signal = 0.0 (very bearish)

        Returns:
            float: 0.0 (bearish) to 1.0 (bullish)
        """
        # Get VIX value
        if self.vix_data is None:
            return 0.5  # Neutral if no VIX data

        vix_at_date = self.vix_data[self.vix_data.index <= date]
        if len(vix_at_date) == 0:
            return 0.5

        vix = vix_at_date.iloc[-1]['close']

        # Smooth mapping: lower VIX = higher signal
        # Normalize VIX to 0-1 range (20 to 80)
        vix_normalized = (vix - 20) / 60
        vix_normalized = np.clip(vix_normalized, 0, 1)

        # Apply power curve for smoother transitions
        signal = 1 - (vix_normalized ** 0.8)

        return np.clip(signal, 0.0, 1.0)

    def _calculate_recovery_signal(self, portfolio_values_df):
        """
        Signal 2: Drawdown improving (equity curve healing)

        Measures how much the portfolio has recovered from recent drawdown.
        This creates progressive redeployment as drawdowns heal.

        Formula: signal = 1 - (current_drawdown / -20%)

        Mapping:
        - 0% drawdown (at peak) → signal = 1.0 (fully recovered)
        - -5% drawdown → signal = 0.75
        - -10% drawdown → signal = 0.5
        - -20% drawdown or worse → signal = 0.0 (deep drawdown)

        Returns:
            float: 0.0 (deep drawdown) to 1.0 (recovered)
        """
        if portfolio_values_df is None or len(portfolio_values_df) < 2:
            return 1.0  # Default to recovered if no history

        # Calculate current drawdown
        current_value = portfolio_values_df['value'].iloc[-1]
        peak_value = portfolio_values_df['value'].max()

        drawdown_pct = ((current_value - peak_value) / peak_value) * 100

        # Map drawdown to signal (0% = 1.0, -20% = 0.0)
        # Normalize to 0-1 range
        signal = 1 - (drawdown_pct / -20)

        return np.clip(signal, 0.0, 1.0)

    def _calculate_trend_signal(self, date):
        """
        Signal 3: Trend stabilizing (market strength)

        Reuses the existing market_trend_strength calculation but interprets
        it as a signal (already returns 0-1).

        Returns:
            float: 0.0 (bearish) to 1.0 (bullish)
        """
        return self.calculate_market_trend_strength(date)


def run_v18_test():
    """Test V18 Progressive Cash Redeployment"""
    logger.info("="*80)
    logger.info("V18: PROGRESSIVE CASH REDEPLOYMENT TEST")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy:")
    logger.info("  - Progressive cash redeployment (not binary regime flips)")
    logger.info("  - Three continuous signals: volatility, recovery, trend")
    logger.info("  - Cash adjusts smoothly as markets improve")
    logger.info("  - Earlier participation in recoveries")
    logger.info("")

    # Initialize bot
    bot = ProgressiveCashBot(data_dir='sp500_data/daily', initial_capital=100000)

    logger.info("Loading data...")
    bot.prepare_data()
    bot.score_all_stocks()

    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING V18 BACKTEST")
    logger.info("="*80)

    # Create custom backtest that uses progressive cash
    portfolio_df = run_progressive_cash_backtest(bot)

    # Calculate metrics
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / bot.initial_capital - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

    # Max drawdown
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Sharpe
    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Yearly returns
    portfolio_df['year'] = portfolio_df.index.year
    yearly_returns = {}
    for year in portfolio_df['year'].unique():
        year_data = portfolio_df[portfolio_df['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
            yearly_returns[year] = year_return

    all_years_positive = all(r > 0 for r in yearly_returns.values())

    logger.info("")
    logger.info("="*80)
    logger.info("V18 RESULTS")
    logger.info("="*80)
    logger.info(f"Initial Capital: ${bot.initial_capital:,.0f}")
    logger.info(f"Final Value:     ${final_value:,.0f}")
    logger.info(f"Total Return:    {total_return:.1f}%")
    logger.info(f"Annual Return:   {annual_return:.1f}%")
    logger.info(f"Max Drawdown:    {max_drawdown:.1f}%")
    logger.info(f"Sharpe Ratio:    {sharpe:.2f}")
    logger.info("")
    logger.info("Yearly Returns:")
    for year in sorted(yearly_returns.keys()):
        ret = yearly_returns[year]
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:>6.1f}% {status}")

    logger.info("")
    if annual_return >= 20 and all_years_positive:
        logger.info(f"🎯 GOAL ACHIEVED! {annual_return:.1f}% annual + all years positive!")
    elif all_years_positive:
        logger.info(f"✅ All years positive! Annual: {annual_return:.1f}%")
    else:
        logger.info(f"⚠️  Annual: {annual_return:.1f}%, Some years negative")

    logger.info("")
    logger.info("="*80)

    return portfolio_df


def run_progressive_cash_backtest(bot):
    """
    Custom backtest using progressive cash redeployment

    Similar to backtest_with_bear_protection but uses the new
    calculate_progressive_cash_reserve method
    """
    logger.info("Starting progressive cash backtest...")

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

            # V18: PROGRESSIVE CASH CALCULATION
            portfolio_df_so_far = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            cash_reserve = bot.calculate_progressive_cash_reserve(date, portfolio_df_so_far)

            # Score stocks
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            # Get top N stocks
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [t for t, s in ranked if s > 0][:10]

            # Calculate allocations with momentum weighting
            invest_amount = cash * (1 - cash_reserve)

            if top_stocks:
                # Use momentum-strength weighting
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
                    # Fallback to equal
                    allocation_per_stock = invest_amount / len(top_stocks)
                    allocations = {ticker: allocation_per_stock for ticker in top_stocks}
            else:
                allocations = {}

            # Apply drawdown control
            if portfolio_df_so_far is not None and len(portfolio_df_so_far) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df_so_far)
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

    return pd.DataFrame(portfolio_values).set_index('date')


if __name__ == '__main__':
    portfolio_df = run_v18_test()

    logger.info("")
    logger.info("="*80)
    logger.info("COMPARISON: V13 vs V18")
    logger.info("="*80)
    logger.info("")
    logger.info("V13 (Previous): Binary VIX thresholds")
    logger.info("  - VIX < 30: 5-15% cash")
    logger.info("  - VIX >= 30: 30-70% cash")
    logger.info("  - Problem: Cash stays idle after stress")
    logger.info("")
    logger.info("V18 (New): Progressive cash redeployment")
    logger.info("  - Continuous volatility signal (smooth VIX curve)")
    logger.info("  - Recovery signal (drawdown healing)")
    logger.info("  - Trend signal (market strength)")
    logger.info("  - Benefit: Earlier participation in recoveries")
    logger.info("")
    logger.info("="*80)
