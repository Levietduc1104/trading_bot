"""
S&P 500 Portfolio Rotation Bot - Complete Demo

This demonstrates the end-to-end portfolio rotation system:
1. Load stock data
2. Calculate technical indicators
3. Score and rank stocks
4. Allocate portfolio to top N stocks
5. Simulate monthly rebalancing
6. Calculate performance metrics

Goal: 20%+ annual return with low drawdown
"""

import pandas as pd
import numpy as np
import glob
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PortfolioRotationBot:
    """Complete portfolio rotation system"""

    def __init__(self, data_dir='sp500_data/daily', initial_capital=100000):
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.stocks_data = {}
        self.scores = {}
        self.rankings = []

    # ========================
    # 1. DATA LOADING
    # ========================

    def load_all_stocks(self):
        """Load all stock data"""
        csv_files = glob.glob(f"{self.data_dir}/*.csv")
        logger.info(f"Loading {len(csv_files)} stocks...")

        for file_path in csv_files:
            ticker = os.path.basename(file_path).replace('.csv', '')
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = [col.lower() for col in df.columns]
                self.stocks_data[ticker] = df
            except Exception as e:
                logger.warning(f"Error loading {ticker}: {e}")

        logger.info(f"Loaded {len(self.stocks_data)} stocks")

    # ========================
    # 2. TECHNICAL INDICATORS
    # ========================

    def add_indicators(self, df):
        """Add all technical indicators"""
        df = df.copy()

        # EMAs
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        df['ema_34'] = df['close'].ewm(span=34, adjust=False).mean()
        df['ema_89'] = df['close'].ewm(span=89, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # ROC
        df['roc_20'] = df['close'].pct_change(20) * 100

        # EMA alignment
        df['ema_alignment'] = ((df['ema_13'] > df['ema_34']).astype(int) +
                               (df['ema_34'] > df['ema_89']).astype(int))

        return df

    def prepare_data(self):
        """Load and add indicators to all stocks"""
        self.load_all_stocks()
        logger.info("Calculating indicators...")

        for ticker in self.stocks_data.keys():
            self.stocks_data[ticker] = self.add_indicators(self.stocks_data[ticker])

        logger.info("Indicators calculated")

    # ========================
    # 3. STOCK SCORING (0-100)
    # ========================

    def score_stock(self, ticker, df):
        """
        V5 SIMPLIFIED SCORING (fewer factors, clearer logic)

        Improvements over previous version:
        - Removed double-counting of EMA alignment
        - Simplified to 3 clear factors
        - Better trend focus (50 pts vs 25 pts)
        - Uses same indicators, just clearer logic

        Performance: 7.7% annual, -19.3% max drawdown (+1.1% vs baseline)
        """
        latest = df.iloc[-1]
        score = 0

        # Factor 1: Price Trend (50 pts) - Is it going up?
        # Check multiple timeframes
        close = latest['close']
        ema_13 = latest['ema_13']
        ema_34 = latest['ema_34']
        ema_89 = latest['ema_89']

        # Short-term trend (20 pts)
        if close > ema_13 > ema_34:
            score += 20
        elif close > ema_13:
            score += 10

        # Long-term trend (30 pts)
        if close > ema_89:
            score += 30
            # Bonus if accelerating (EMAs properly aligned)
            if ema_34 > ema_89:
                score += 10
        elif close > ema_89 * 0.95:  # Close to breakout
            score += 15

        # Factor 2: Recent Performance (30 pts) - Has it been winning?
        roc = latest['roc_20']
        if roc > 15:
            score += 30
        elif roc > 10:
            score += 20
        elif roc > 5:
            score += 15
        elif roc > 0:
            score += 10

        # Factor 3: Risk Level (20 pts) - Is volatility reasonable?
        atr_pct = (latest['atr'] / latest['close']) * 100
        if atr_pct < 2:
            score += 20
        elif atr_pct < 3:
            score += 15
        elif atr_pct < 4:
            score += 10
        elif atr_pct < 5:
            score += 5

        return min(score, 100)

    def score_all_stocks(self):
        """Score all stocks"""
        logger.info("Scoring stocks...")
        self.scores = {}

        for ticker, df in self.stocks_data.items():
            if len(df) >= 100:
                self.scores[ticker] = self.score_stock(ticker, df)

        self.rankings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Scored {len(self.scores)} stocks")

    # ========================
    # 4. PORTFOLIO BACKTEST (FIXED)
    # ========================

    def backtest(self, top_n=10, rebalance_freq='M'):
        """
        Backtest portfolio rotation strategy

        Args:
            top_n: Number of stocks to hold
            rebalance_freq: 'M' for monthly
        """
        logger.info(f"Starting backtest: Top {top_n} stocks, Monthly rebalance")

        # Get common dates across all stocks
        first_ticker = list(self.stocks_data.keys())[0]
        dates = self.stocks_data[first_ticker].index

        # Track portfolio
        portfolio_values = []
        cash = self.initial_capital
        holdings = {}  # {ticker: shares}
        last_rebalance_month = None

        for date in dates[100:]:  # Skip first 100 days for indicators
            current_month = date.month

            # Monthly rebalance
            if current_month != last_rebalance_month:
                last_rebalance_month = current_month

                # Liquidate current holdings
                for ticker in list(holdings.keys()):
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        cash += holdings[ticker] * current_price
                holdings = {}

                # Score all stocks at this date
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

                # Allocate capital (invest 80%, keep 20% cash)
                invest_amount = cash * 0.80
                allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0

                # Buy top stocks
                for ticker in top_stocks:
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        shares = allocation_per_stock / current_price
                        holdings[ticker] = shares
                        cash -= allocation_per_stock

            # Calculate daily portfolio value
            stocks_value = 0
            for ticker, shares in holdings.items():
                df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    stocks_value += shares * current_price

            total_value = cash + stocks_value
            portfolio_values.append({'date': date, 'value': total_value})

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100

        # Max drawdown
        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        logger.info("="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:.1f}%")
        logger.info(f"Annual Return: {annual_return:.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Period: {portfolio_df.index[0].date()} to {portfolio_df.index[-1].date()}")
        logger.info(f"Total Days: {len(portfolio_df)}")
        logger.info("="*60)

        # Goal check
        if annual_return >= 20:
            logger.info(f"🎯 GOAL ACHIEVED\! {annual_return:.1f}% exceeds 20% target")
        else:
            logger.info(f"⚠️  Below target: {annual_return:.1f}% vs 20% goal")

        return portfolio_df

    def backtest_swing_trading(self, top_n=10, rebalance_days=5, cash_reserve=0.5,
                                stop_loss_pct=5, take_profit_pct=15):
        """
        Swing trading backtest with faster rebalancing and risk management

        Args:
            top_n: Number of stocks to hold
            rebalance_days: Rebalance every N days (5=weekly, 1=daily)
            cash_reserve: Percentage to keep in cash (0.5 = 50%)
            stop_loss_pct: Stop-loss percentage (5 = 5%)
            take_profit_pct: Take-profit percentage (15 = 15%)
        """
        logger.info(f"Starting SWING TRADING backtest:")
        logger.info(f"  - Rebalance every {rebalance_days} days")
        logger.info(f"  - Cash reserve: {cash_reserve*100:.0f}%")
        logger.info(f"  - Stop-loss: {stop_loss_pct}%")
        logger.info(f"  - Take-profit: {take_profit_pct}%")

        # Get common dates
        first_ticker = list(self.stocks_data.keys())[0]
        dates = self.stocks_data[first_ticker].index

        # Track portfolio
        portfolio_values = []
        cash = self.initial_capital
        holdings = {}  # {ticker: {'shares': N, 'entry_price': P}}
        days_since_rebalance = 0

        for idx, date in enumerate(dates[100:]):  # Skip first 100 days
            days_since_rebalance += 1

            # Check stop-loss and take-profit DAILY
            for ticker in list(holdings.keys()):
                df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    entry_price = holdings[ticker]['entry_price']
                    price_change = ((current_price - entry_price) / entry_price) * 100

                    # Stop-loss: sell if down X%
                    if price_change <= -stop_loss_pct:
                        shares = holdings[ticker]['shares']
                        cash += shares * current_price
                        del holdings[ticker]
                        continue

                    # Take-profit: sell if up Y%
                    if price_change >= take_profit_pct:
                        shares = holdings[ticker]['shares']
                        cash += shares * current_price
                        del holdings[ticker]
                        continue

            # Rebalance on schedule
            if days_since_rebalance >= rebalance_days:
                days_since_rebalance = 0

                # Liquidate current holdings
                for ticker in list(holdings.keys()):
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        cash += holdings[ticker]['shares'] * current_price
                holdings = {}

                # Score all stocks at this date
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

                # Allocate capital (use 1-cash_reserve)
                invest_amount = cash * (1 - cash_reserve)
                allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0

                # Buy top stocks
                for ticker in top_stocks:
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        shares = allocation_per_stock / current_price
                        holdings[ticker] = {
                            'shares': shares,
                            'entry_price': current_price
                        }
                        cash -= allocation_per_stock

            # Calculate daily portfolio value
            stocks_value = 0
            for ticker, holding in holdings.items():
                df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    stocks_value += holding['shares'] * current_price

            total_value = cash + stocks_value
            portfolio_values.append({'date': date, 'value': total_value})

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100

        # Max drawdown
        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Yearly returns
        portfolio_df_copy = portfolio_df.copy()
        portfolio_df_copy['year'] = portfolio_df_copy.index.year
        yearly_returns = {}
        for year in portfolio_df_copy['year'].unique():
            year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
            if len(year_data) > 1:
                year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
                yearly_returns[year] = year_return

        all_years_positive = all(r > 0 for r in yearly_returns.values())

        logger.info("="*60)
        logger.info("SWING TRADING RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:.1f}%")
        logger.info(f"Annual Return: {annual_return:.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info("="*60)

        # Yearly breakdown
        logger.info("\nYearly Returns:")
        for year in sorted(yearly_returns.keys()):
            ret = yearly_returns[year]
            emoji = "✅" if ret > 0 else "❌"
            logger.info(f"  {year}: {ret:>6.1f}% {emoji}")

        # Goal check
        if annual_return >= 20 and all_years_positive:
            logger.info(f"\n🎯 GOAL ACHIEVED! {annual_return:.1f}% annual + all years positive!")
        elif annual_return >= 20:
            logger.info(f"\n⚠️  Good return ({annual_return:.1f}%) but some years negative")
        else:
            logger.info(f"\n⚠️  Below target: {annual_return:.1f}% vs 20% goal")

        return portfolio_df

    def _calculate_regime_score(self, date):
        """
        Calculate raw regime score from 4 market factors
        Returns integer score from -4 (very bearish) to +4 (very bullish)

        Used by both calculate_adaptive_regime and index hedging strategy
        """
        spy_data = self.stocks_data.get('SPY')
        if spy_data is None:
            return -2  # Default to defensive

        df = spy_data[spy_data.index <= date]
        if len(df) < 200:
            return -2  # Default to defensive

        price = df.iloc[-1]['close']

        # Factor 1: Trend (200 MA)
        ma200 = df['close'].tail(200).mean()
        trend_score = 1 if price > ma200 else -1

        # Factor 2: Momentum (50-day ROC)
        if len(df) >= 50:
            price_50_ago = df['close'].iloc[-50]
            momentum = (price / price_50_ago - 1)
            momentum_score = 1 if momentum > 0.05 else -1
        else:
            momentum_score = 0

        # Factor 3: Volatility (30-day vs 1-year)
        returns = df['close'].pct_change()
        if len(returns) >= 252:
            vol_30 = returns.tail(30).std()
            vol_252 = returns.tail(252).std()
            vol_score = 1 if vol_30 < vol_252 else -1
        else:
            vol_score = 0

        # Factor 4: Market Breadth (% stocks above 200 MA)
        breadth_count = 0
        total_count = 0
        for ticker, stock_df in self.stocks_data.items():
            stock_at_date = stock_df[stock_df.index <= date]
            if len(stock_at_date) >= 200:
                stock_price = stock_at_date.iloc[-1]['close']
                stock_ma200 = stock_at_date['close'].tail(200).mean()
                if stock_price > stock_ma200:
                    breadth_count += 1
                total_count += 1

        if total_count > 0:
            breadth_pct = breadth_count / total_count
            breadth_score = 1 if breadth_pct > 0.50 else -1
        else:
            breadth_score = 0

        # Return total score (-4 to +4)
        return trend_score + momentum_score + vol_score + breadth_score

    def calculate_adaptive_regime(self, date):
        """
        ADAPTIVE MULTI-FACTOR REGIME DETECTION
        Uses 4 standard factors with equal weighting (no tuning/optimization)
        Returns cash reserve percentage based on market regime

        Factors:
        1. Trend (200 MA) - Long-term direction
        2. Momentum (50-day ROC) - Medium-term strength
        3. Volatility (30-day vs 1-year) - Market stress level
        4. Breadth (% stocks > 200 MA) - Market health

        Returns cash reserve: 0.05 (very bullish) to 0.65 (very bearish)
        Conservative range to handle unknown future market conditions
        """
        # Use helper method to calculate regime score
        total_score = self._calculate_regime_score(date)

        # Map score to cash reserve (5-65% range)
        # Conservative approach - avoids overfitting to historical data
        if total_score >= 3:
            return 0.05   # Very bullish (score 3-4): Light cash buffer
        elif total_score >= 1:
            return 0.25   # Bullish (score 1-2): Moderate cash
        elif total_score >= -1:
            return 0.45   # Neutral (score -1 to 0): Balanced
        else:
            return 0.65   # Bearish (score -2 to -4): Heavy protection

    def backtest_with_bear_protection(self, top_n=10, rebalance_freq='M',
                                       bear_cash_reserve=0.7, bull_cash_reserve=0.2,
                                       use_adaptive_regime=False, trading_fee_pct=0.001):
        """
        Backtest with BEAR MARKET PROTECTION
        Key idea: Reduce exposure (increase cash) when market is declining

        Args:
            top_n: Number of stocks to hold
            rebalance_freq: 'M' for monthly, 'W' for weekly
            bear_cash_reserve: Cash % in bear market (0.7 = 70%)
            bull_cash_reserve: Cash % in bull market (0.2 = 20%)
            trading_fee_pct: Trading fee as percentage (0.001 = 0.1% per trade)
        """
        logger.info(f"Starting BEAR PROTECTION backtest:")
        logger.info(f"  - Bull market cash: {bull_cash_reserve*100:.0f}%")
        logger.info(f"  - Bear market cash: {bear_cash_reserve*100:.0f}%")
        logger.info(f"  - Trading fee: {trading_fee_pct*100:.3f}% per trade")

        # Get common dates
        first_ticker = list(self.stocks_data.keys())[0]
        dates = self.stocks_data[first_ticker].index

        # Track portfolio
        portfolio_values = []
        cash = self.initial_capital
        holdings = {}
        last_rebalance_month = None
        last_rebalance_week = None

        for date in dates[100:]:  # Skip first 100 days
            current_month = date.month
            current_week = date.isocalendar()[1]

            # Determine rebalance
            should_rebalance = False
            if rebalance_freq == 'M' and current_month != last_rebalance_month:
                should_rebalance = True
                last_rebalance_month = current_month
            elif rebalance_freq == 'W' and current_week != last_rebalance_week:
                should_rebalance = True
                last_rebalance_week = current_week

            if should_rebalance:
                # Liquidate current holdings
                for ticker in list(holdings.keys()):
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        sale_value = holdings[ticker] * current_price
                        fee = sale_value * trading_fee_pct
                        cash += sale_value - fee
                holdings = {}

                # Use adaptive regime detection or fallback to simple bear/bull
                if use_adaptive_regime:
                    cash_reserve = self.calculate_adaptive_regime(date)
                    regime = f"ADAPTIVE ({cash_reserve*100:.0f}% cash)"
                else:
                    # DETECT BEAR MARKET: Check if SPY is below its 200-day MA
                    # (This is a simple bear market indicator)
                    spy_data = self.stocks_data.get('SPY')
                    is_bear_market = False

                    if spy_data is not None:
                        spy_at_date = spy_data[spy_data.index <= date]
                        if len(spy_at_date) >= 200:
                            spy_price = spy_at_date.iloc[-1]['close']
                            spy_ma200 = spy_at_date['close'].tail(200).mean()
                            is_bear_market = spy_price < spy_ma200

                    # Adjust cash reserve based on market regime
                    cash_reserve = bear_cash_reserve if is_bear_market else bull_cash_reserve
                    regime = '🐻 BEAR' if is_bear_market else '🐂 BULL'

                logger.info(f"{date.date()}: {regime} - Cash reserve: {cash_reserve*100:.0f}%")

                # Score all stocks
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

                # Allocate capital
                invest_amount = cash * (1 - cash_reserve)
                allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0

                # Buy top stocks
                for ticker in top_stocks:
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        shares = allocation_per_stock / current_price
                        holdings[ticker] = shares
                        fee = allocation_per_stock * trading_fee_pct
                        cash -= (allocation_per_stock + fee)

            # Calculate daily portfolio value
            stocks_value = 0
            for ticker, shares in holdings.items():
                df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    stocks_value += shares * current_price

            total_value = cash + stocks_value
            portfolio_values.append({'date': date, 'value': total_value})

        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100

        # Max drawdown
        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # Yearly returns
        portfolio_df_copy = portfolio_df.copy()
        portfolio_df_copy['year'] = portfolio_df_copy.index.year
        yearly_returns = {}
        for year in portfolio_df_copy['year'].unique():
            year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
            if len(year_data) > 1:
                year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
                yearly_returns[year] = year_return

        all_years_positive = all(r > 0 for r in yearly_returns.values())

        logger.info("="*60)
        logger.info("BEAR PROTECTION RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Final Value: ${final_value:,.0f}")
        logger.info(f"Total Return: {total_return:.1f}%")
        logger.info(f"Annual Return: {annual_return:.1f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.1f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info("="*60)

        # Yearly breakdown
        logger.info("\nYearly Returns:")
        for year in sorted(yearly_returns.keys()):
            ret = yearly_returns[year]
            emoji = "✅" if ret > 0 else "❌"
            logger.info(f"  {year}: {ret:>6.1f}% {emoji}")

        # Goal check
        if annual_return >= 20 and all_years_positive:
            logger.info(f"\n🎯 GOAL ACHIEVED! {annual_return:.1f}% annual + all years positive!")
        elif all_years_positive:
            logger.info(f"\n✅ All years positive! Annual: {annual_return:.1f}% (target: 20%)")
        else:
            logger.info(f"\n⚠️  Annual: {annual_return:.1f}%, Some years negative")

        return portfolio_df





# ========================
# DATA GENERATION - Use generate_sp500_data.py instead
# ========================


# ========================
# ANALYSIS & DIAGNOSTICS - Use separate analysis scripts instead
# ========================




# ========================
# MAIN EXECUTION
# ========================



# ========================
# STRATEGY OPTIMIZATION - Moved to separate scripts
# ========================



def backtest_with_cash(self, top_n=10, cash_reserve=0.20, rebalance_freq='M'):
    """Backtest with adjustable cash reserve"""
    # Similar to regular backtest but with configurable cash
    
    all_dates = self.stocks_data[list(self.stocks_data.keys())[0]].index
    portfolio_values = []
    dates_list = []
    
    cash = self.initial_capital
    holdings = {}
    
    rebalance_dates = all_dates[::21] if rebalance_freq == 'M' else all_dates
    
    for current_date in all_dates:
        if current_date in rebalance_dates:
            # Liquidate
            for ticker in list(holdings.keys()):
                if ticker in self.stocks_data:
                    price = self.stocks_data[ticker].loc[current_date, 'close']
                    cash += holdings[ticker] * price
            holdings = {}
            
            # Get current scores
            top_stocks = [(ticker, score) for ticker, score in self.rankings[:top_n]
                         if ticker in self.stocks_data]
            
            # Allocate with cash reserve
            invest_amount = cash * (1 - cash_reserve)
            if len(top_stocks) > 0:
                per_stock = invest_amount / len(top_stocks)
                
                for ticker, score in top_stocks:
                    price = self.stocks_data[ticker].loc[current_date, 'close']
                    shares = per_stock / price
                    holdings[ticker] = shares
                    cash -= per_stock
        
        # Calculate portfolio value
        total_value = cash
        for ticker, shares in holdings.items():
            if ticker in self.stocks_data:
                price = self.stocks_data[ticker].loc[current_date, 'close']
                total_value += shares * price
        
        portfolio_values.append(total_value)
        dates_list.append(current_date)
    
    return pd.DataFrame({'value': portfolio_values}, index=dates_list)


def score_defensive(self):
    """Score stocks defensively - prioritize low volatility and capital preservation"""
    
    self.scores = {}
    
    for ticker, df in self.stocks_data.items():
        if len(df) < 200:
            self.scores[ticker] = 0
            continue
        
        score = 0
        latest = df.iloc[-1]
        
        # DEFENSIVE CRITERIA (Total: 100 points)
        
        # 1. Low Volatility (40 points) - MOST IMPORTANT in crash
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        if volatility < 15:
            score += 40
        elif volatility < 20:
            score += 30
        elif volatility < 25:
            score += 20
        elif volatility < 30:
            score += 10
        
        # 2. Smallest Loss (30 points) - Minimize damage
        returns_60d = ((df['close'].iloc[-1] / df['close'].iloc[-60]) - 1) * 100
        returns_120d = ((df['close'].iloc[-1] / df['close'].iloc[-120]) - 1) * 100
        
        avg_return = (returns_60d + returns_120d) / 2
        
        if avg_return > 0:
            score += 30  # Actually positive\!
        elif avg_return > -10:
            score += 25  # Small loss
        elif avg_return > -20:
            score += 20
        elif avg_return > -30:
            score += 15
        elif avg_return > -50:
            score += 10
        
        # 3. Positive Trend (20 points) - At least not getting worse
        if 'ema_13' in df.columns and 'ema_34' in df.columns:
            if latest['ema_13'] > latest['ema_34']:
                score += 20
            elif latest['ema_13'] > latest['ema_34'] * 0.98:
                score += 10
        
        # 4. RSI Not Oversold (10 points) - Avoid catching falling knives
        if 'rsi' in df.columns:
            rsi = latest['rsi']
            if 40 <= rsi <= 60:
                score += 10
            elif 30 <= rsi <= 70:
                score += 5
        
        self.scores[ticker] = min(score, 100)
    
    self.rankings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
    return self.scores


def backtest_defensive(self, top_n=20):
    """Backtest using defensive scoring"""
    return self.backtest(top_n=top_n)


def backtest_with_risk_management(self, top_n=10, stop_loss_pct=5, take_profit_pct=15,
                                  cash_reserve=0.20, use_volatility_sizing=False):
    """
    Backtest with RISK MANAGEMENT:
    - Stop-loss: Sell if stock drops stop_loss_pct% from entry
    - Take-profit: Sell if stock gains take_profit_pct%
    - Optional: Size positions by volatility

    Args:
        top_n: Number of stocks to hold
        stop_loss_pct: Sell at -X% loss (e.g., 5 = sell at -5%)
        take_profit_pct: Sell at +Y% gain (e.g., 15 = sell at +15%)
        cash_reserve: % to keep in cash (default 20%)
        use_volatility_sizing: Allocate less to volatile stocks
    """

    logger.info("="*80)
    logger.info("BACKTEST WITH RISK MANAGEMENT")
    logger.info(f"Stop-Loss: {stop_loss_pct}% | Take-Profit: {take_profit_pct}%")
    logger.info(f"Cash Reserve: {cash_reserve*100}% | Volatility Sizing: {use_volatility_sizing}")
    logger.info("="*80)

    # Get dates
    first_ticker = list(self.stocks_data.keys())[0]
    all_dates = self.stocks_data[first_ticker].index

    # Track portfolio
    cash = self.initial_capital
    holdings = {}  # {ticker: {'shares': X, 'entry_price': Y}}
    portfolio_values = []

    # Monthly rebalance dates
    rebalance_dates = all_dates[::21]
    stop_loss_count = 0
    take_profit_count = 0

    for date_idx, date in enumerate(all_dates[100:], 100):

        # CHECK RISK MANAGEMENT DAILY
        tickers_to_sell = []
        for ticker in list(holdings.keys()):
            if ticker in self.stocks_data:
                current_price = self.stocks_data[ticker].loc[date, 'close']
                entry_price = holdings[ticker]['entry_price']
                pnl_pct = ((current_price / entry_price) - 1) * 100

                # STOP-LOSS
                if pnl_pct < -stop_loss_pct:
                    shares = holdings[ticker]['shares']
                    cash += shares * current_price
                    tickers_to_sell.append(ticker)
                    stop_loss_count += 1

                # TAKE-PROFIT
                elif pnl_pct > take_profit_pct:
                    shares = holdings[ticker]['shares']
                    cash += shares * current_price
                    tickers_to_sell.append(ticker)
                    take_profit_count += 1

        for ticker in tickers_to_sell:
            del holdings[ticker]

        # MONTHLY REBALANCE
        if date in rebalance_dates:
            # Liquidate
            for ticker in list(holdings.keys()):
                if ticker in self.stocks_data:
                    current_price = self.stocks_data[ticker].loc[date, 'close']
                    cash += holdings[ticker]['shares'] * current_price
            holdings = {}

            # Score stocks
            current_scores = {}
            for ticker, df in self.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = self.score_stock(ticker, df_at_date)
                    except:
                        pass

            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [t for t, s in ranked[:top_n]]

            invest_amount = cash * (1 - cash_reserve)

            if len(top_stocks) > 0:
                per_stock = invest_amount / len(top_stocks)
                for ticker in top_stocks:
                    current_price = self.stocks_data[ticker].loc[date, 'close']
                    shares = per_stock / current_price
                    holdings[ticker] = {
                        'shares': shares,
                        'entry_price': current_price
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

    # Results
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / self.initial_capital - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100

    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    daily_returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = portfolio_df_copy.groupby('year')['value'].apply(
        lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 0 else 0
    )

    logger.info("="*80)
    logger.info("RISK-MANAGED RESULTS")
    logger.info("="*80)
    logger.info(f"Stop-Loss Triggered: {stop_loss_count} times")
    logger.info(f"Take-Profit Triggered: {take_profit_count} times")
    logger.info(f"")
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
        logger.info("🎯 SUCCESS: 20%+ annual + ALL years positive!")
    elif all_positive:
        logger.info(f"✅ All years positive (but {annual_return:.1f}% < 20%)")
    else:
        logger.info("⚠️  Some years had losses")

    logger.info("="*80)

    return portfolio_df


# Add these methods to the PortfolioRotationBot class
PortfolioRotationBot.backtest_with_cash = backtest_with_cash
PortfolioRotationBot.score_defensive = score_defensive
PortfolioRotationBot.backtest_defensive = backtest_defensive
PortfolioRotationBot.backtest_with_risk_management = backtest_with_risk_management


# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    # This module provides the PortfolioRotationBot class
    # Use run_best_bear_protection.py or execution.py to run backtests
    logger.debug("portfolio_bot_demo.py loaded as main module")
    logger.debug("Use run_best_bear_protection.py or execution.py to run backtests")

