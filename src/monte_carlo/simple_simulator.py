"""
Simple Monte Carlo Simulator - Step 2
======================================

This is the SIMPLEST version of a Monte Carlo simulator.
It runs your V28 backtest with different parameters and collects results.

How it works:
1. Load your trading bot (PortfolioRotationBot)
2. Run backtest with custom parameters
3. Calculate performance metrics
4. Return results as a dictionary
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot


class SimpleSimulator:
    """
    Simple Monte Carlo Simulator

    This simulator runs your V28 strategy with different parameters
    and returns the results for comparison.
    """

    def __init__(self, data_dir='sp500_data/daily', initial_capital=100000):
        """
        Initialize the simulator

        Args:
            data_dir: Path to your stock data
            initial_capital: Starting capital (default $100,000)
        """
        self.data_dir = os.path.join(project_root, data_dir)
        self.initial_capital = initial_capital
        self.results = []  # Will store all simulation results

        print(f"✓ Simulator initialized")
        print(f"  Data: {data_dir}")
        print(f"  Capital: ${initial_capital:,}")


    def run_single_simulation(self,
                             sim_id,
                             portfolio_size=None,
                             kelly_exponent=0.5,
                             vix_multiplier=1.0,
                             fee_pct=0.001):
        """
        Run ONE simulation with specific parameters

        This is the core function that:
        1. Creates a trading bot
        2. Loads data and scores stocks
        3. Runs backtest with your parameters
        4. Calculates metrics
        5. Returns results

        Args:
            sim_id: Simulation number (for tracking)
            portfolio_size: Number of stocks (None = dynamic based on VIX)
            kelly_exponent: Position sizing power (0.5 = sqrt, 1.0 = linear)
            vix_multiplier: Multiply VIX cash reserve by this (1.0 = normal)
            fee_pct: Trading fee percentage (0.001 = 0.1%)

        Returns:
            Dictionary with results:
            {
                'sim_id': 1,
                'annual_return': 9.4,
                'max_drawdown': -18.0,
                'sharpe_ratio': 1.00,
                ...
            }
        """

        print(f"\nSimulation #{sim_id}")
        print(f"  Parameters: portfolio={portfolio_size}, kelly={kelly_exponent}, vix_mult={vix_multiplier}, fee={fee_pct}")

        # STEP 1: Initialize the trading bot
        bot = PortfolioRotationBot(
            data_dir=self.data_dir,
            initial_capital=self.initial_capital
        )

        # STEP 2: Load data and score stocks
        bot.prepare_data()
        bot.score_all_stocks()

        # STEP 3: Run the backtest with custom parameters
        portfolio_df = self._run_backtest(
            bot=bot,
            portfolio_size=portfolio_size,
            kelly_exponent=kelly_exponent,
            vix_multiplier=vix_multiplier,
            fee_pct=fee_pct
        )

        # STEP 4: Calculate performance metrics
        metrics = self._calculate_metrics(portfolio_df)

        # STEP 5: Add metadata
        metrics['sim_id'] = sim_id
        metrics['parameters'] = {
            'portfolio_size': portfolio_size,
            'kelly_exponent': kelly_exponent,
            'vix_multiplier': vix_multiplier,
            'fee_pct': fee_pct
        }

        # STEP 6: Store result
        self.results.append(metrics)

        print(f"  Results: {metrics['annual_return']:.2f}% return, {metrics['max_drawdown']:.2f}% drawdown, {metrics['sharpe_ratio']:.2f} Sharpe")

        return metrics


    def _run_backtest(self, bot, portfolio_size, kelly_exponent, vix_multiplier, fee_pct):
        """
        Run the V28 backtest with custom parameters

        This function is similar to your existing backtest but with
        modifications to test different parameters.

        Returns:
            DataFrame with daily portfolio values
        """

        # Get date range
        first_ticker = list(bot.stocks_data.keys())[0]
        dates = bot.stocks_data[first_ticker].index

        # Storage
        portfolio_values = []
        holdings = {}
        cash = bot.initial_capital
        last_rebalance_date = None

        # Loop through each trading day
        for date in dates[100:]:  # Start after 100 days for indicator warmup

            # Check if it's time to rebalance (monthly, days 7-10)
            is_rebalance_day = (
                last_rebalance_date is None or
                (
                    (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                    7 <= date.day <= 10
                )
            )

            if is_rebalance_day:
                # REBALANCE: Sell everything
                for ticker in list(holdings.keys()):
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        cash += holdings[ticker] * current_price
                holdings = {}
                last_rebalance_date = date

                # Get VIX for this date
                vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
                if vix_at_date is not None and len(vix_at_date) > 0:
                    vix = vix_at_date.iloc[-1]['close']
                else:
                    vix = 20  # Default if no VIX data

                # Score all stocks at this date
                current_scores = {}
                for ticker, df in bot.stocks_data.items():
                    df_at_date = df[df.index <= date]
                    if len(df_at_date) >= 100:  # Need enough data for indicators
                        try:
                            current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                        except:
                            pass

                # Determine how many stocks to hold
                if portfolio_size is None:
                    # Use dynamic sizing based on VIX (V28/V27 feature)
                    top_n = bot.determine_portfolio_size(date)
                else:
                    # Use fixed portfolio size (for testing)
                    top_n = portfolio_size

                # Select top N stocks
                ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
                top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

                if not top_stocks:
                    portfolio_values.append({'date': date, 'value': cash})
                    continue

                # Calculate VIX-based cash reserve (with multiplier for testing)
                if vix < 30:
                    cash_reserve = 0.05 + (vix - 10) * 0.005
                else:
                    cash_reserve = 0.15 + (vix - 30) * 0.0125
                cash_reserve = np.clip(cash_reserve * vix_multiplier, 0.05, 0.70)

                invest_amount = cash * (1 - cash_reserve)

                # Kelly position sizing with custom exponent
                kelly_weights = self._calculate_kelly_weights(top_stocks, kelly_exponent)

                allocations = {
                    ticker: invest_amount * weight
                    for ticker, weight in kelly_weights.items()
                }

                # Apply drawdown control
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
                        fee = allocation_amount * fee_pct
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

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
        portfolio_df = portfolio_df.rename(columns={'value': 'portfolio_value'})

        return portfolio_df


    def _calculate_kelly_weights(self, scored_stocks, exponent):
        """
        Calculate Kelly position weights

        Args:
            scored_stocks: List of (ticker, score) tuples
            exponent: Power to raise scores to (0.5 = sqrt, 1.0 = linear)

        Returns:
            Dictionary of {ticker: weight}
        """
        tickers = [t for t, s in scored_stocks]
        scores = [s for t, s in scored_stocks]

        # Apply exponent (0.5 = sqrt for V28, but we can test other values)
        weighted_scores = [max(0, score) ** exponent for score in scores]
        total_weighted = sum(weighted_scores)

        if total_weighted > 0:
            weights = {
                ticker: (max(0, score) ** exponent) / total_weighted
                for ticker, score in scored_stocks
            }
        else:
            # Fallback to equal weights
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

        return weights


    def _calculate_metrics(self, portfolio_df):
        """
        Calculate performance metrics from portfolio values

        Args:
            portfolio_df: DataFrame with portfolio_value column

        Returns:
            Dictionary with metrics:
            {
                'annual_return': 9.4,
                'max_drawdown': -18.0,
                'sharpe_ratio': 1.00,
                ...
            }
        """

        # Basic values
        initial_value = self.initial_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value * 100

        # Time period
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        years = (end_date - start_date).days / 365.25

        # Annual return (CAGR)
        annual_return = (((final_value / initial_value) ** (1 / years)) - 1) * 100

        # Max drawdown
        cummax = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Daily returns
        returns = portfolio_df['portfolio_value'].pct_change().dropna()

        # Sharpe ratio (annualized)
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = (returns.mean() / downside_std) * (252 ** 0.5) if downside_std > 0 else 0

        # Calmar ratio (return / max drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'years': years,
            'start_date': str(start_date.date()),
            'end_date': str(end_date.date())
        }
