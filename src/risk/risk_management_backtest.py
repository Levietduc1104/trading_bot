"""
Add risk management to existing monthly strategy:
1. Stop-loss: Exit if stock drops X% from entry
2. Take-profit: Exit if stock gains Y%
3. Position sizing: Allocate based on volatility
"""

# Add this method to PortfolioRotationBot class

def backtest_with_risk_management(self, top_n=10, stop_loss_pct=5, take_profit_pct=15,
                                  cash_reserve=0.20, use_volatility_sizing=False):
    """
    Backtest with risk management:
    - Stop-loss: Sell if stock drops stop_loss_pct% from entry
    - Take-profit: Sell if stock gains take_profit_pct%
    - Optional: Size positions by volatility (risky stocks = smaller position)

    Args:
        top_n: Number of stocks to hold
        stop_loss_pct: Sell if stock drops this % (e.g., 5 = sell at -5%)
        take_profit_pct: Sell if stock gains this % (e.g., 15 = sell at +15%)
        cash_reserve: % to keep in cash
        use_volatility_sizing: True = allocate less to volatile stocks
    """

    import logging
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("BACKTEST WITH RISK MANAGEMENT")
    logger.info(f"Stop-Loss: {stop_loss_pct}% | Take-Profit: {take_profit_pct}%")
    logger.info(f"Cash Reserve: {cash_reserve*100}% | Vol Sizing: {use_volatility_sizing}")
    logger.info("="*80)

    # Get dates
    first_ticker = list(self.stocks_data.keys())[0]
    all_dates = self.stocks_data[first_ticker].index

    # Track portfolio
    cash = self.initial_capital
    holdings = {}  # {ticker: {'shares': X, 'entry_price': Y, 'entry_date': Z}}
    portfolio_values = []

    # Monthly rebalance dates
    rebalance_dates = all_dates[::21]

    for date_idx, date in enumerate(all_dates[100:], 100):

        # CHECK RISK MANAGEMENT DAILY (stop-loss & take-profit)
        tickers_to_sell = []
        for ticker in list(holdings.keys()):
            if ticker in self.stocks_data:
                current_price = self.stocks_data[ticker].loc[date, 'close']
                entry_price = holdings[ticker]['entry_price']

                # Calculate gain/loss %
                pnl_pct = ((current_price / entry_price) - 1) * 100

                # STOP-LOSS: Sell if loss exceeds limit
                if pnl_pct < -stop_loss_pct:
                    shares = holdings[ticker]['shares']
                    cash += shares * current_price
                    tickers_to_sell.append(ticker)
                    logger.info(f"  🛑 STOP-LOSS: {ticker} at {date.date()} | Loss: {pnl_pct:.1f}%")

                # TAKE-PROFIT: Sell if gain exceeds target
                elif pnl_pct > take_profit_pct:
                    shares = holdings[ticker]['shares']
                    cash += shares * current_price
                    tickers_to_sell.append(ticker)
                    logger.info(f"  💰 TAKE-PROFIT: {ticker} at {date.date()} | Gain: {pnl_pct:.1f}%")

        # Remove sold positions
        for ticker in tickers_to_sell:
            del holdings[ticker]

        # MONTHLY REBALANCE
        if date in rebalance_dates:
            # Liquidate remaining positions
            for ticker in list(holdings.keys()):
                if ticker in self.stocks_data:
                    current_price = self.stocks_data[ticker].loc[date, 'close']
                    cash += holdings[ticker]['shares'] * current_price
            holdings = {}

            # Score stocks at this date
            current_scores = {}
            for ticker, df in self.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = self.score_stock(ticker, df_at_date)
                    except:
                        pass

            # Get top N stocks
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [t for t, s in ranked[:top_n]]

            # Calculate allocation
            invest_amount = cash * (1 - cash_reserve)

            if use_volatility_sizing and len(top_stocks) > 0:
                # VOLATILITY-BASED POSITION SIZING
                # Lower volatility = larger position
                volatilities = {}
                for ticker in top_stocks:
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    daily_returns = df_at_date['close'].pct_change().tail(60).dropna()
                    vol = daily_returns.std() * 100  # Daily volatility %
                    volatilities[ticker] = vol

                # Inverse volatility weighting
                inv_vols = {t: 1/v if v > 0 else 1 for t, v in volatilities.items()}
                total_inv_vol = sum(inv_vols.values())

                # Allocate proportionally
                for ticker in top_stocks:
                    weight = inv_vols[ticker] / total_inv_vol
                    allocation = invest_amount * weight

                    current_price = self.stocks_data[ticker].loc[date, 'close']
                    shares = allocation / current_price
                    holdings[ticker] = {
                        'shares': shares,
                        'entry_price': current_price,
                        'entry_date': date
                    }
                    cash -= allocation

            else:
                # EQUAL WEIGHTING
                if len(top_stocks) > 0:
                    per_stock = invest_amount / len(top_stocks)

                    for ticker in top_stocks:
                        current_price = self.stocks_data[ticker].loc[date, 'close']
                        shares = per_stock / current_price
                        holdings[ticker] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'entry_date': date
                        }
                        cash -= per_stock

        # Calculate portfolio value
        stocks_value = 0
        for ticker, position in holdings.items():
            if ticker in self.stocks_data:
                current_price = self.stocks_data[ticker].loc[date, 'close']
                stocks_value += position['shares'] * current_price

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    # Convert to DataFrame
    import pandas as pd
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

    # Calculate metrics
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / self.initial_capital - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100

    # Max drawdown
    import numpy as np
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = portfolio_df_copy.groupby('year')['value'].apply(
        lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 0 else 0
    )

    logger.info("="*80)
    logger.info("RISK-MANAGED BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
    logger.info(f"Final Value: ${final_value:,.0f}")
    logger.info(f"Total Return: {total_return:.1f}%")
    logger.info(f"Annual Return: {annual_return:.1f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info("")
    logger.info("YEARLY RETURNS:")
    logger.info("-" * 80)

    all_positive = True
    for year, ret in yearly_returns.items():
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:>7.1f}%  {status}")
        if ret <= 0:
            all_positive = False

    logger.info("="*80)

    if all_positive and annual_return >= 20:
        logger.info("🎯 SUCCESS: 20%+ annual return + ALL years positive!")
    elif all_positive:
        logger.info(f"✅ All years positive, but {annual_return:.1f}% < 20% target")
    elif annual_return >= 20:
        logger.info(f"⚠️  20%+ return achieved, but some years had losses")
    else:
        logger.info("⚠️  Goals not met")

    logger.info("="*80)

    return portfolio_df


# To add this to the class, save this file and we'll integrate it
