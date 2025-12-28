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

# S&P 500 Sector Mapping (for V7 sector relative strength)
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'GOOGL': 'Technology', 'GOOG': 'Technology', 'META': 'Technology',
    'AVGO': 'Technology', 'TSLA': 'Technology', 'ORCL': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology',
    'QCOM': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology',
    # Financials
    'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials',
    'BAC': 'Financials', 'WFC': 'Financials', 'MS': 'Financials',
    'GS': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
    # Healthcare
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare',
    'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'PFE': 'Healthcare',
    'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
    # Communication Services
    'NFLX': 'Communication Services', 'DIS': 'Communication Services',
    'CMCSA': 'Communication Services', 'T': 'Communication Services',
    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples',
    'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'SLB': 'Energy', 'EOG': 'Energy', 'MPC': 'Energy',
    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
    'RTX': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
}



class PortfolioRotationBot:
    """Complete portfolio rotation system"""

    def __init__(self, data_dir='sp500_data/daily', initial_capital=100000):
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.stocks_data = {}
        self.scores = {}
        self.rankings = []
        self.sector_map = SECTOR_MAP
        self.vix_data = None  # V8: VIX volatility index for regime detection

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


        # V8: Load VIX data for regime detection
        self.load_vix_data()
        logger.info("Indicators calculated")

    def load_vix_data(self):
        """
        V8: Load VIX volatility index data
        
        VIX is a forward-looking fear indicator (implied volatility)
        Better than lagging indicators like 200-day MA for regime detection
        """
        try:
            vix_path = f"{self.data_dir}/VIX.csv"
            vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
            vix_df.columns = [col.lower() for col in vix_df.columns]
            
            # Drop NaN values from the proxy
            vix_df = vix_df.dropna()
            
            self.vix_data = vix_df
            logger.info(f"Loaded VIX data: {len(vix_df)} days")
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")
            logger.warning("Will fall back to adaptive regime detection")
            self.vix_data = None


    # ========================
    # 3. STOCK SCORING (0-100)
    # ========================

    def score_stock(self, ticker, df):
        """
        V7 STOCK SCORING: 0-150 points

        Combines:
        - V5 base scoring (3 factors: Trend 50pts, Performance 30pts, Risk 20pts)
        - V6 momentum filters (quality gates)
        - V7 sector relative strength (sector leadership bonus)

        Returns:
            float: Score 0-150 (0 = disqualified, higher = better)
        """
        latest = df.iloc[-1]
        score = 0

        # ====================
        # V5 BASE SCORING
        # ====================

        # Factor 1: Price Trend (50 pts) - Is it going up?
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

        # ====================
        # V6 MOMENTUM FILTERS
        # ====================

        # CRITICAL FILTER 1: Must be above long-term trend
        if close < ema_89:
            return 0  # DISQUALIFY - downtrend

        # CRITICAL FILTER 2: Must have positive momentum
        if roc < 2:
            return 0  # DISQUALIFY - weak/negative momentum

        # RSI filters (penalties, not disqualifications)
        rsi = latest['rsi']
        if rsi > 75:
            score *= 0.7  # 30% penalty for overbought
        if rsi < 30:
            score *= 0.5  # 50% penalty for oversold

        # ====================
        # V7 SECTOR BONUS
        # ====================

        sector_bonus = self.calculate_sector_relative_strength(ticker, df)
        score += sector_bonus

        return min(score, 150)  # Cap at 150

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


    def calculate_vix_regime(self, date):
        """
        V8: VIX-BASED REGIME DETECTION
        
        Uses VIX volatility index (forward-looking fear indicator) instead of
        lagging indicators like 200-day MA. VIX reflects market's expectation
        of future volatility, making it a better early warning system.
        
        VIX Regimes (adjusted for our proxy which runs higher than real VIX):
        - VIX < 30: Very low fear -> 5% cash (aggressive)
        - VIX 30-40: Low fear -> 15% cash
        - VIX 40-50: Moderate fear -> 30% cash
        - VIX 50-70: High fear -> 50% cash (defensive)
        - VIX > 70: Panic mode -> 70% cash (very defensive)
        
        Args:
            date: Current date for regime detection
        
        Returns:
            float: Cash reserve percentage (0.05 to 0.70)
        """
        # Fallback to adaptive regime if VIX data not available
        if self.vix_data is None:
            return self.calculate_adaptive_regime(date)
        
        # Get VIX value at this date
        vix_at_date = self.vix_data[self.vix_data.index <= date]
        if len(vix_at_date) == 0:
            return self.calculate_adaptive_regime(date)
        
        vix = vix_at_date.iloc[-1]['close']
        
        # VIX-based regime detection
        # Lower VIX = lower fear = more aggressive (less cash)
        # Higher VIX = higher fear = more defensive (more cash)
        if vix < 30:
            cash_reserve = 0.05  # Very low fear: aggressive
        elif vix < 40:
            cash_reserve = 0.15  # Low fear
        elif vix < 50:
            cash_reserve = 0.30  # Moderate fear
        elif vix < 70:
            cash_reserve = 0.50  # High fear
        else:
            cash_reserve = 0.70  # Panic mode
        
        return cash_reserve

    def calculate_market_trend_strength(self, date):
        """
        MARKET REGIME FILTER (Time-Series Momentum)

        Determines overall market strength using S&P 500 index (SPY).
        Returns a continuous trend strength score instead of binary signal.

        Formula:
            trend_strength = clip((SPY_price / MA200 - 1) / 0.15, 0, 1)

        Where:
            - 0 = weak / bearish market (SPY is 15%+ below MA200)
            - 1 = strong / bullish market (SPY is 15%+ above MA200)
            - 0.5 = neutral (SPY is at MA200)

        This approach:
            - Avoids hindsight bias (uses only data up to current date)
            - Reduces whipsaw effects (continuous vs binary signal)
            - Provides proportional market strength assessment

        Args:
            date: Current date for trend strength calculation

        Returns:
            float: Trend strength from 0.0 (weak/bearish) to 1.0 (strong/bullish)
        """
        # Get SPY data
        spy_data = self.stocks_data.get('SPY')
        if spy_data is None:
            logger.warning("SPY data not available, returning neutral trend strength")
            return 0.5  # Neutral if no data

        # Get data up to current date
        df = spy_data[spy_data.index <= date]
        if len(df) < 200:
            logger.warning(f"Insufficient data for 200-day MA at {date}, returning neutral")
            return 0.5  # Neutral if insufficient data

        # Get current price and 200-day MA
        current_price = df.iloc[-1]['close']
        ma200 = df['close'].tail(200).mean()

        # Calculate trend strength
        # (price / MA200 - 1) gives the percentage deviation from MA200
        # Dividing by 0.15 normalizes it so ±15% deviation maps to [0, 1]
        price_deviation = (current_price / ma200 - 1) / 0.15

        # Clip to [0, 1] range
        trend_strength = np.clip(price_deviation + 1.0, 0.0, 1.0)

        return trend_strength

    def calculate_stock_volatility(self, ticker, date, lookback_days=40):
        """
        Calculate historical volatility for a stock (no hindsight bias)

        Uses standard deviation of returns over lookback period.
        Default 40 days (~2 months) balances recency vs stability.

        Args:
            ticker: Stock ticker symbol
            date: Current date (only uses data up to this date)
            lookback_days: Number of days to calculate volatility (default 40)

        Returns:
            float: Annualized volatility (standard deviation of returns)
                   Returns 0.20 (20%) as default if insufficient data
        """
        if ticker not in self.stocks_data:
            return 0.20  # Default 20% volatility

        df = self.stocks_data[ticker]
        df_at_date = df[df.index <= date]

        if len(df_at_date) < lookback_days + 1:
            return 0.20  # Default if insufficient history

        # Calculate returns over lookback period
        recent_prices = df_at_date['close'].tail(lookback_days + 1)
        returns = recent_prices.pct_change().dropna()

        if len(returns) < 10:  # Need minimum data points
            return 0.20

        # Calculate volatility (std dev of returns)
        volatility = returns.std()

        # Annualize: daily vol * sqrt(252 trading days)
        annualized_vol = volatility * np.sqrt(252)

        # Safety bounds: cap between 5% and 100%
        # (prevents extreme values from breaking allocation)
        annualized_vol = np.clip(annualized_vol, 0.05, 1.0)

        return annualized_vol

    def calculate_portfolio_volatility(self, portfolio_values_df, lookback_days=60):
        """
        V12: Calculate realized portfolio volatility (no hindsight bias)

        Uses rolling window of past portfolio returns to estimate current risk.
        This is NOT predicting returns - just normalizing risk exposure.

        Args:
            portfolio_values_df: DataFrame with 'value' column (portfolio value over time)
            lookback_days: Rolling window for volatility calculation (default 60 days)

        Returns:
            float: Annualized portfolio volatility (standard deviation)
                   Returns None if insufficient data
        """
        if len(portfolio_values_df) < lookback_days + 1:
            return None  # Insufficient history

        # Calculate daily returns
        recent_values = portfolio_values_df['value'].tail(lookback_days + 1)
        returns = recent_values.pct_change().dropna()

        if len(returns) < 30:  # Need minimum 30 days
            return None

        # Calculate volatility (standard deviation of returns)
        daily_vol = returns.std()

        # Annualize: daily vol * sqrt(252 trading days)
        annualized_vol = daily_vol * np.sqrt(252)

        return annualized_vol

    def calculate_momentum_strength_weight(self, ticker, date, lookback_months=9):
        """
        Calculate momentum-strength weight for position sizing (zero-bias)

        Uses momentum/volatility ratio to allocate more capital to strong,
        stable trends and less to weak or volatile stocks.

        Formula: weight ∝ momentum / volatility

        Where:
        - momentum = 6-12 month return (default 9 months = 189 days)
        - volatility = rolling past volatility (annualized)

        Why this works:
        1. Strong trends get more capital (trend following)
        2. High volatility gets less capital (risk control)
        3. Momentum is persistent (academic evidence)
        4. No prediction, just extrapolation

        This is zero-bias because:
        - Uses only historical data (backward-looking)
        - Momentum persistence is empirical fact (Jegadeesh & Titman 1993)
        - Not optimizing or fitting to specific periods

        Args:
            ticker: Stock ticker symbol
            date: Current date (only uses data up to this date)
            lookback_months: Momentum lookback in months (default 9 = 6-12 month range)

        Returns:
            float: Momentum-strength ratio (unnormalized weight)
                   Returns 0 if insufficient data or negative momentum

        Example:
            - Stock A: +30% return, 15% vol → weight = 30/15 = 2.0
            - Stock B: +20% return, 10% vol → weight = 20/10 = 2.0
            - Stock C: +40% return, 40% vol → weight = 40/40 = 1.0
            - Stock D: -5% return (negative) → weight = 0 (exclude)
        """
        if ticker not in self.stocks_data:
            return 0.0

        df = self.stocks_data[ticker]
        df_at_date = df[df.index <= date]

        lookback_days = lookback_months * 21  # Approximate trading days per month

        if len(df_at_date) < lookback_days + 1:
            return 0.0  # Insufficient history

        # Calculate momentum (return over lookback period)
        price_current = df_at_date['close'].iloc[-1]
        price_past = df_at_date['close'].iloc[-lookback_days-1]
        momentum = (price_current / price_past - 1) * 100  # Return as percentage

        # Exclude negative momentum (we only want positive trends)
        if momentum <= 0:
            return 0.0

        # Calculate volatility (same as stock-level vol calculation)
        volatility = self.calculate_stock_volatility(ticker, date, lookback_days=40)

        # Momentum-strength ratio
        # Higher momentum + lower vol = higher weight
        weight = momentum / volatility

        return weight

    def calculate_drawdown_multiplier(self, portfolio_values_df):
        """
        V12: Calculate exposure multiplier based on current drawdown from peak

        **DRAWDOWN CONTROL**: The most powerful risk management tool for cash-only trading.
        Progressively reduces exposure as drawdown increases, preventing drawdown acceleration
        and preserving capital for recovery.

        Rules:
        - Drawdown < 10%:  1.00x exposure (fully invested)
        - Drawdown 10-15%: 0.75x exposure (hold 25% cash)
        - Drawdown 15-20%: 0.50x exposure (hold 50% cash)
        - Drawdown ≥ 20%:  0.25x exposure (hold 75% cash, maximum defense)

        Why this increases long-term return:
        1. Prevents drawdown acceleration (exponential losses)
        2. Preserves capital for recovery (geometric return boost)
        3. Improves Sharpe ratio (lower volatility)
        4. Counter-intuitive: Lower arithmetic return → Higher geometric return

        This is zero-bias because:
        - Uses only historical equity curve (no prediction)
        - Pure risk management (not market timing)
        - Mechanical rule (no optimization)

        Args:
            portfolio_values_df: DataFrame with 'value' column (portfolio value over time)

        Returns:
            float: Exposure multiplier (0.25 to 1.0)

        Example:
            - Current value: $90k, Peak: $100k → DD = 10% → multiplier = 0.75
            - Current value: $80k, Peak: $100k → DD = 20% → multiplier = 0.25
            - Current value: $105k, Peak: $100k → DD = 0% → multiplier = 1.0 (new peak!)
        """
        if len(portfolio_values_df) < 2:
            return 1.0  # Default to fully invested if insufficient history

        # Calculate current drawdown from peak
        current_value = portfolio_values_df['value'].iloc[-1]
        peak_value = portfolio_values_df['value'].max()

        # Drawdown as percentage
        drawdown_pct = ((current_value - peak_value) / peak_value) * 100

        # Apply tiered exposure reduction
        if drawdown_pct >= -10:
            # Less than 10% drawdown: fully invested
            return 1.0
        elif drawdown_pct >= -15:
            # 10-15% drawdown: reduce to 75%
            return 0.75
        elif drawdown_pct >= -20:
            # 15-20% drawdown: reduce to 50%
            return 0.50
        else:
            # 20%+ drawdown: maximum defense (25% invested)
            return 0.25

    # ========================
    # V7 OPTIMIZATION METHODS
    # ========================

    def get_seasonal_cash_multiplier(self, date):
        """
        V7 OPTIMIZATION: Seasonal Patterns
        
        "Sell in May and go away" effect:
        - Nov-Apr: Historically strong months (85% multiplier = more invested)
        - May-Oct: Historically weak months (115% multiplier = more defensive)
        
        Returns:
            float: Multiplier for cash reserve (0.85 = reduce cash, 1.15 = increase cash)
        """
        month = date.month
        
        if month in [11, 12, 1, 2, 3, 4]:  # Winter months (strong season)
            return 0.85  # More aggressive (reduce cash by 15%)
        else:  # Summer months May-Oct (weak season)
            return 1.15  # More defensive (increase cash by 15%)
    
    def should_rebalance_midmonth(self, date, last_rebalance_date):
        """
        V7 OPTIMIZATION: Mid-Month Rebalancing
        
        Rebalances on day 7-10 instead of day 1 to:
        - Avoid institutional window dressing at month-end
        - Get better execution prices
        - Reduce slippage from crowded trades
        
        Returns:
            bool: True if should rebalance
        """
        if last_rebalance_date is None:
            return True
        
        current_month = (date.year, date.month)
        last_month = (last_rebalance_date.year, last_rebalance_date.month)
        
        # New month AND we're on day 7-10
        return current_month != last_month and 7 <= date.day <= 10
    
    def calculate_sector_relative_strength(self, ticker, df_at_date):
        """
        V7 OPTIMIZATION: Sector Relative Strength
        
        Awards bonus points for stocks outperforming their sector peers.
        This ensures we buy sector leaders, not just market leaders.
        
        Args:
            ticker: Stock ticker symbol
            df_at_date: Stock dataframe up to current date
        
        Returns:
            int: Bonus points -10 to +15
        """
        if len(df_at_date) < 60:
            return 0
        
        # Get ticker's sector
        sector = self.sector_map.get(ticker, 'Other')
        
        # Calculate 60-day return for this stock
        ticker_return = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1) * 100
        
        # Calculate average return for sector peers
        sector_returns = []
        for peer_ticker, peer_sector in self.sector_map.items():
            if peer_sector == sector and peer_ticker != ticker:
                if peer_ticker in self.stocks_data:
                    peer_df = self.stocks_data[peer_ticker]
                    peer_at_date = peer_df[peer_df.index <= df_at_date.index[-1]]
                    if len(peer_at_date) >= 60:
                        peer_return = (peer_at_date['close'].iloc[-1] / peer_at_date['close'].iloc[-60] - 1) * 100
                        sector_returns.append(peer_return)
        
        if not sector_returns:
            return 0
        
        # Calculate outperformance vs sector
        sector_avg = np.mean(sector_returns)
        relative_strength = ticker_return - sector_avg
        
        # Award bonus points
        if relative_strength > 10:
            return 15  # Crushing sector
        elif relative_strength > 5:
            return 10  # Strong outperformance
        elif relative_strength > 2:
            return 5   # Modest outperformance
        elif relative_strength < -5:
            return -10  # Lagging sector
        
        return 0


    def backtest_with_bear_protection(self, top_n=10, rebalance_freq='M',
                                       bear_cash_reserve=0.7, bull_cash_reserve=0.2,
                                       use_adaptive_regime=False, use_vix_regime=False,
                                       use_trend_strength=False, use_inverse_vol_weighting=False,
                                       use_adaptive_weighting=False, use_momentum_weighting=False,
                                       use_drawdown_control=False, trading_fee_pct=0.001):
        """
        Backtest with BEAR MARKET PROTECTION + V7 OPTIMIZATIONS + V8 VIX REGIME + V10 INVERSE VOL + V11 HYBRID + V12 DRAWDOWN + V13 MOMENTUM
        - V7: Mid-month rebalancing (day 7-10)
        - V7: Seasonal cash adjustment (winter aggressive, summer defensive)
        - V7: Sector relative strength scoring
        - V8: VIX-based regime detection (forward-looking fear indicator)
        - V9: Market trend strength (time-series momentum, continuous)
        - V10: Inverse volatility position weighting (risk-parity)
        - V11: Adaptive weighting (equal in bull, inverse-vol in bear) ✨
        - V12: Portfolio-level drawdown control (progressive exposure reduction) 🛡️
        - V13: Momentum-strength weighting (momentum/vol ratio instead of equal) 🚀
        - V6: Momentum quality filters (disqualify weak trends)
        Args:
            top_n: Number of stocks to hold
            rebalance_freq: 'M' for monthly, 'W' for weekly
            bear_cash_reserve: Cash % in bear market (0.7 = 70%)
            bull_cash_reserve: Cash % in bull market (0.2 = 20%)
            trading_fee_pct: Trading fee as percentage (0.001 = 0.1% per trade)
            use_vix_regime: Use VIX volatility index for regime (V8)
            use_trend_strength: Use market trend strength for regime (V9)
            use_inverse_vol_weighting: Use inverse volatility position weighting (V10)
            use_adaptive_weighting: Adaptively switch between equal and inverse vol based on VIX (V11)
            use_momentum_weighting: Use momentum-strength weighting instead of equal weight (V13) 🚀
            use_drawdown_control: Use portfolio-level drawdown control (V12) 🛡️
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
        holdings = {}
        cash = self.initial_capital
        last_rebalance_date = None  # V7: Changed from last_rebalance_month

        for date in dates[100:]:  # Skip first 100 days

            # V7: Use mid-month rebalancing (day 7-10)
            should_rebalance = self.should_rebalance_midmonth(date, last_rebalance_date)

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

                # V7: Track rebalance date for mid-month timing
                last_rebalance_date = date

                # Regime detection priority: VIX > Trend Strength > Adaptive > Simple Bear/Bull
                if use_vix_regime:
                    cash_reserve = self.calculate_vix_regime(date)
                    regime = f"VIX ({cash_reserve*100:.0f}% cash)"
                elif use_trend_strength:
                    # V9: Use continuous trend strength to calculate cash reserve
                    trend_strength = self.calculate_market_trend_strength(date)
                    # Convert trend strength (0-1) to cash reserve (70%-5%)
                    # Strong trend (1.0) = 5% cash, Weak trend (0.0) = 70% cash
                    cash_reserve = 0.70 - (trend_strength * 0.65)
                    regime = f"TREND ({trend_strength:.2f} strength, {cash_reserve*100:.0f}% cash)"
                elif use_adaptive_regime:
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

                # V7: Apply seasonal adjustment
                seasonal_multiplier = self.get_seasonal_cash_multiplier(date)
                cash_reserve = max(0.05, min(0.70, cash_reserve * seasonal_multiplier))

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
                top_stocks = [t for t, s in ranked if s > 0][:top_n]  # V7: Filter out disqualified stocks (score=0)

                # V10/V11: Calculate position sizes (equal, inverse vol, or adaptive)
                invest_amount = cash * (1 - cash_reserve)

                if use_adaptive_weighting and top_stocks:
                    # V11: ADAPTIVE WEIGHTING - Switch based on VIX regime
                    # Calm market (VIX < 30) → Equal weighting (max returns)
                    # Stressed market (VIX >= 30) → Inverse vol (risk control)

                    # Get current VIX value
                    if self.vix_data is not None:
                        vix_at_date = self.vix_data[self.vix_data.index <= date]
                        if len(vix_at_date) > 0:
                            current_vix = vix_at_date.iloc[-1]['close']
                        else:
                            current_vix = 50  # Default to defensive
                    else:
                        current_vix = 50  # Default to defensive if no VIX data

                    # Decision threshold: VIX < 30 = calm, VIX >= 30 = stressed
                    if current_vix < 30:
                        # CALM MARKET: Equal or momentum-strength weighting for maximum returns
                        if use_momentum_weighting:
                            # V13: MOMENTUM-STRENGTH WEIGHTING
                            momentum_weights = {}
                            for ticker in top_stocks:
                                weight = self.calculate_momentum_strength_weight(ticker, date, lookback_months=9)
                                if weight > 0:
                                    momentum_weights[ticker] = weight

                            # Normalize weights to sum to invest_amount
                            total_weight = sum(momentum_weights.values())
                            if total_weight > 0:
                                allocations = {
                                    ticker: (weight / total_weight) * invest_amount
                                    for ticker, weight in momentum_weights.items()
                                }
                            else:
                                # Fallback to equal if all weights are zero
                                allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0
                                allocations = {ticker: allocation_per_stock for ticker in top_stocks}
                        else:
                            # V11: EQUAL WEIGHTING (traditional)
                            allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0
                            allocations = {ticker: allocation_per_stock for ticker in top_stocks}
                    else:
                        # STRESSED MARKET: Inverse volatility for risk control
                        risk_adjusted_weights = {}
                        for ticker, score in ranked[:top_n]:
                            if score > 0:
                                volatility = self.calculate_stock_volatility(ticker, date, lookback_days=40)
                                risk_adjusted_weights[ticker] = score / volatility

                        total_risk_adj = sum(risk_adjusted_weights.values())
                        if total_risk_adj > 0:
                            allocations = {
                                ticker: (weight / total_risk_adj) * invest_amount
                                for ticker, weight in risk_adjusted_weights.items()
                            }
                        else:
                            allocations = {ticker: invest_amount / len(top_stocks) for ticker in top_stocks}

                elif use_inverse_vol_weighting and top_stocks:
                    # V10: INVERSE VOLATILITY WEIGHTING (always on)
                    # weight_i = (score_i / vol_i) / sum(score_j / vol_j)
                    # Higher score + lower volatility = larger position

                    # Calculate score/volatility ratios for each stock
                    risk_adjusted_weights = {}
                    for ticker, score in ranked[:top_n]:
                        if score > 0:  # Only qualified stocks
                            volatility = self.calculate_stock_volatility(ticker, date, lookback_days=40)
                            # weight proportional to score/volatility
                            risk_adjusted_weights[ticker] = score / volatility

                    # Normalize weights to sum to 1.0
                    total_risk_adj = sum(risk_adjusted_weights.values())
                    if total_risk_adj > 0:
                        allocations = {
                            ticker: (weight / total_risk_adj) * invest_amount
                            for ticker, weight in risk_adjusted_weights.items()
                        }
                    else:
                        # Fallback to equal weight if calculation fails
                        allocations = {ticker: invest_amount / len(top_stocks) for ticker in top_stocks}
                else:
                    # V8/V13: EQUAL OR MOMENTUM-STRENGTH WEIGHTING (baseline)
                    if use_momentum_weighting:
                        # V13: MOMENTUM-STRENGTH WEIGHTING
                        momentum_weights = {}
                        for ticker in top_stocks:
                            weight = self.calculate_momentum_strength_weight(ticker, date, lookback_months=9)
                            if weight > 0:
                                momentum_weights[ticker] = weight

                        # Normalize weights to sum to invest_amount
                        total_weight = sum(momentum_weights.values())
                        if total_weight > 0:
                            allocations = {
                                ticker: (weight / total_weight) * invest_amount
                                for ticker, weight in momentum_weights.items()
                            }
                        else:
                            # Fallback to equal if all weights are zero
                            allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0
                            allocations = {ticker: allocation_per_stock for ticker in top_stocks}
                    else:
                        # V8: EQUAL WEIGHTING (traditional)
                        allocation_per_stock = invest_amount / len(top_stocks) if top_stocks else 0
                        allocations = {ticker: allocation_per_stock for ticker in top_stocks}

                # V12: PORTFOLIO-LEVEL DRAWDOWN CONTROL
                drawdown_multiplier = 1.0  # Default: no adjustment
                if use_drawdown_control and len(portfolio_values) > 1:
                    # Calculate current drawdown and apply exposure reduction
                    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
                    drawdown_multiplier = self.calculate_drawdown_multiplier(portfolio_df)

                    # Apply drawdown-based exposure reduction
                    # Small drawdown → fully invested (1.0x)
                    # Large drawdown → defensive (0.25x to 0.75x)
                    allocations = {
                        ticker: amount * drawdown_multiplier
                        for ticker, amount in allocations.items()
                    }

                # Buy top stocks with calculated allocations
                for ticker in top_stocks:
                    df_at_date = self.stocks_data[ticker][self.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        current_price = df_at_date.iloc[-1]['close']
                        allocation_amount = allocations.get(ticker, 0)
                        shares = allocation_amount / current_price
                        holdings[ticker] = shares
                        fee = allocation_amount * trading_fee_pct
                        cash -= (allocation_amount + fee)

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

