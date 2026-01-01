"""
V19: ASYMMETRIC PROGRESSIVE REDEPLOYMENT

Problem: V13 holds cash too long, V18 redeploys too early
Solution: Fast to defend (binary), slow to redeploy (progressive + confirmation)

Key Innovation: ASYMMETRIC CASH MANAGEMENT
1. Defense (raising cash): FAST & BINARY
   - VIX spikes → immediate cash raise
   - No confirmation needed (protect capital NOW)

2. Offense (lowering cash): SLOW & PROGRESSIVE
   - Require confirmation: 5+ consecutive days of improvement
   - Check momentum: Only redeploy if stocks show positive momentum
   - Progressive: Lower cash gradually, not all at once

3. Momentum confirmation gate:
   - Count how many top stocks have positive momentum
   - Only redeploy if >70% show strength
   - Avoid redeploying into weak rallies

Why this works:
- Preserves V13's strong defense (quick cash raise)
- Avoids V18's early redeployment problem (confirmation required)
- Only participates in confirmed recoveries (momentum gate)
- No leverage, pure risk management
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


class AsymmetricCashBot(PortfolioRotationBot):
    """V19: Asymmetric Progressive Redeployment"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track market state history for confirmation
        self.market_state_history = []

    def calculate_asymmetric_cash_reserve(self, date, portfolio_values_df=None, top_stocks=None):
        """
        V19: ASYMMETRIC CASH RESERVE CALCULATION

        Fast to defend (binary VIX), slow to redeploy (progressive + confirmation)

        Algorithm:
        1. Check if we should RAISE cash (defense) - FAST/BINARY
           - VIX > 50: Immediate 70% cash
           - VIX > 40: Immediate 50% cash
           - VIX > 30: Immediate 30% cash

        2. Check if we can LOWER cash (offense) - SLOW/PROGRESSIVE
           - Requires 5+ days of consecutive improvement
           - Requires 70%+ of stocks showing positive momentum
           - Lower cash progressively (10% per rebalance period)

        3. Hold current level if neither condition met

        Args:
            date: Current date
            portfolio_values_df: Portfolio history for drawdown tracking
            top_stocks: List of current top stocks for momentum check

        Returns:
            float: Cash reserve percentage (0.0 to 0.70)
        """
        # Get current VIX
        if self.vix_data is None:
            return 0.30  # Default defensive if no VIX

        vix_at_date = self.vix_data[self.vix_data.index <= date]
        if len(vix_at_date) == 0:
            return 0.30

        current_vix = vix_at_date.iloc[-1]['close']

        # Get previous cash reserve (or default if first run)
        previous_cash = self._get_previous_cash_reserve()

        # STEP 1: CHECK DEFENSE TRIGGERS (FAST/BINARY)
        # If VIX spikes, immediately raise cash
        if current_vix > 50:
            new_cash = 0.70  # Panic mode
            if new_cash > previous_cash:
                self._update_market_state(date, 'DEFENSE', current_vix, new_cash)
                return new_cash
        elif current_vix > 40:
            new_cash = 0.50  # High stress
            if new_cash > previous_cash:
                self._update_market_state(date, 'DEFENSE', current_vix, new_cash)
                return new_cash
        elif current_vix > 30:
            new_cash = 0.30  # Moderate stress
            if new_cash > previous_cash:
                self._update_market_state(date, 'DEFENSE', current_vix, new_cash)
                return new_cash

        # STEP 2: CHECK OFFENSE TRIGGERS (SLOW/PROGRESSIVE)
        # Only lower cash if conditions are met
        can_lower_cash = self._check_offense_conditions(
            date, current_vix, portfolio_values_df, top_stocks
        )

        if can_lower_cash:
            # Lower cash progressively (reduce by 10% of current level)
            reduction = 0.10
            new_cash = max(0.05, previous_cash * (1 - reduction))
            self._update_market_state(date, 'OFFENSE', current_vix, new_cash)
            return new_cash

        # STEP 3: HOLD CURRENT LEVEL
        # No trigger to raise or lower - maintain
        self._update_market_state(date, 'HOLD', current_vix, previous_cash)
        return previous_cash

    def _check_offense_conditions(self, date, current_vix, portfolio_values_df, top_stocks):
        """
        Check if conditions are met to LOWER cash (offensive redeployment)

        Requires ALL three conditions:
        1. Confirmation: 5+ consecutive days of VIX declining
        2. Momentum: 70%+ of top stocks show positive momentum
        3. Recovery: Portfolio not in deep drawdown (< -15%)

        Returns:
            bool: True if can lower cash, False otherwise
        """
        # Condition 1: VIX CONFIRMATION (5 consecutive days declining)
        vix_confirmed = self._check_vix_confirmation(date, days_required=5)

        # Condition 2: MOMENTUM CONFIRMATION (70%+ stocks strong)
        momentum_confirmed = self._check_momentum_confirmation(date, top_stocks, threshold=0.70)

        # Condition 3: RECOVERY (not in deep drawdown)
        recovery_confirmed = self._check_recovery_confirmation(portfolio_values_df, max_drawdown=-15)

        # All three must be true
        return vix_confirmed and momentum_confirmed and recovery_confirmed

    def _check_vix_confirmation(self, date, days_required=5):
        """
        Check if VIX has been declining for N consecutive days

        Args:
            date: Current date
            days_required: Number of consecutive days needed (default 5)

        Returns:
            bool: True if VIX declining for N days
        """
        if self.vix_data is None:
            return False

        vix_at_date = self.vix_data[self.vix_data.index <= date]

        if len(vix_at_date) < days_required + 1:
            return False

        # Get last N+1 VIX values
        recent_vix = vix_at_date['close'].tail(days_required + 1).values

        # Check if declining (each day lower than previous)
        declining_count = 0
        for i in range(1, len(recent_vix)):
            if recent_vix[i] < recent_vix[i-1]:
                declining_count += 1
            else:
                declining_count = 0  # Reset if not declining

        return declining_count >= days_required

    def _check_momentum_confirmation(self, date, top_stocks, threshold=0.70):
        """
        Check if enough top stocks show positive momentum

        Args:
            date: Current date
            top_stocks: List of top stock tickers
            threshold: Minimum fraction that must be positive (0.70 = 70%)

        Returns:
            bool: True if >= threshold of stocks have positive momentum
        """
        if not top_stocks or len(top_stocks) == 0:
            return False

        positive_count = 0
        total_count = 0

        for ticker in top_stocks:
            if ticker not in self.stocks_data:
                continue

            df = self.stocks_data[ticker]
            df_at_date = df[df.index <= date]

            if len(df_at_date) < 21:  # Need at least 1 month
                continue

            # Check 20-day momentum (1 month)
            current_price = df_at_date['close'].iloc[-1]
            price_20d_ago = df_at_date['close'].iloc[-21]
            momentum = (current_price / price_20d_ago - 1) * 100

            if momentum > 2:  # Positive momentum (>2% gain)
                positive_count += 1
            total_count += 1

        if total_count == 0:
            return False

        positive_fraction = positive_count / total_count
        return positive_fraction >= threshold

    def _check_recovery_confirmation(self, portfolio_values_df, max_drawdown=-15):
        """
        Check if portfolio is not in deep drawdown

        Args:
            portfolio_values_df: Portfolio history
            max_drawdown: Maximum acceptable drawdown (e.g., -15%)

        Returns:
            bool: True if drawdown is better than max_drawdown
        """
        if portfolio_values_df is None or len(portfolio_values_df) < 2:
            return True  # No drawdown if no history

        current_value = portfolio_values_df['value'].iloc[-1]
        peak_value = portfolio_values_df['value'].max()

        drawdown_pct = ((current_value - peak_value) / peak_value) * 100

        return drawdown_pct > max_drawdown

    def _get_previous_cash_reserve(self):
        """Get previous cash reserve from history, or default"""
        if not self.market_state_history:
            return 0.30  # Default moderate starting level
        return self.market_state_history[-1]['cash_reserve']

    def _update_market_state(self, date, action, vix, cash_reserve):
        """Record market state for tracking"""
        self.market_state_history.append({
            'date': date,
            'action': action,
            'vix': vix,
            'cash_reserve': cash_reserve
        })


def run_asymmetric_backtest(bot):
    """
    Custom backtest using asymmetric cash redeployment
    """
    logger.info("Starting asymmetric progressive backtest...")

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

            # V19: ASYMMETRIC CASH CALCULATION
            portfolio_df_so_far = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            cash_reserve = bot.calculate_asymmetric_cash_reserve(
                date,
                portfolio_df_so_far,
                top_stocks
            )

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


def run_v19_test():
    """Test V19 Asymmetric Progressive Redeployment"""
    logger.info("="*80)
    logger.info("V19: ASYMMETRIC PROGRESSIVE REDEPLOYMENT")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy Components:")
    logger.info("  1. DEFENSE (Fast/Binary):")
    logger.info("     - VIX > 50 → 70% cash IMMEDIATELY")
    logger.info("     - VIX > 40 → 50% cash IMMEDIATELY")
    logger.info("     - VIX > 30 → 30% cash IMMEDIATELY")
    logger.info("")
    logger.info("  2. OFFENSE (Slow/Progressive):")
    logger.info("     - Requires VIX declining 5+ consecutive days")
    logger.info("     - Requires 70%+ stocks show positive momentum")
    logger.info("     - Requires portfolio not in deep drawdown")
    logger.info("     - Lowers cash 10% per period if conditions met")
    logger.info("")
    logger.info("  3. MOMENTUM CONFIRMATION:")
    logger.info("     - Only redeploy if top stocks strong")
    logger.info("     - Avoids redeploying into weak rallies")
    logger.info("")

    # Initialize bot
    bot = AsymmetricCashBot(data_dir='sp500_data/daily', initial_capital=100000)

    logger.info("Loading data...")
    bot.prepare_data()
    bot.score_all_stocks()

    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING V19 BACKTEST")
    logger.info("="*80)

    portfolio_df = run_asymmetric_backtest(bot)

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
    logger.info("V19 RESULTS")
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

    # Analyze cash deployment actions
    logger.info("")
    logger.info("="*80)
    logger.info("CASH DEPLOYMENT ANALYSIS")
    logger.info("="*80)

    defense_count = sum(1 for s in bot.market_state_history if s['action'] == 'DEFENSE')
    offense_count = sum(1 for s in bot.market_state_history if s['action'] == 'OFFENSE')
    hold_count = sum(1 for s in bot.market_state_history if s['action'] == 'HOLD')

    logger.info(f"Defense actions (cash raised): {defense_count}")
    logger.info(f"Offense actions (cash lowered): {offense_count}")
    logger.info(f"Hold actions (maintained):     {hold_count}")
    logger.info(f"Total rebalance periods:       {len(bot.market_state_history)}")

    logger.info("")
    logger.info("="*80)
    logger.info("COMPARISON: V13 vs V18 vs V19")
    logger.info("="*80)
    logger.info("")
    logger.info("V13: Binary VIX (baseline)")
    logger.info("  - Annual: 8.5%, Drawdown: -18.5%, Sharpe: 1.26")
    logger.info("  - Problem: Cash stays idle too long")
    logger.info("")
    logger.info("V18: Fully progressive (failed)")
    logger.info("  - Annual: 5.8%, Drawdown: -25.1%, Sharpe: 0.71")
    logger.info("  - Problem: Redeploys too early (catches falling knives)")
    logger.info("")
    logger.info(f"V19: Asymmetric (current)")
    logger.info(f"  - Annual: {annual_return:.1f}%, Drawdown: {max_drawdown:.1f}%, Sharpe: {sharpe:.2f}")
    logger.info(f"  - Combines fast defense + slow confirmed offense")
    logger.info("")
    logger.info("="*80)

    return portfolio_df


if __name__ == '__main__':
    portfolio_df = run_v19_test()
