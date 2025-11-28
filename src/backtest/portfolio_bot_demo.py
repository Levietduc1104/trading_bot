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
        """Score a stock 0-100"""
        latest = df.iloc[-1]
        recent = df.tail(60)

        score = 0

        # Momentum (25 pts)
        rsi = latest['rsi']
        if 40 <= rsi <= 60:
            score += 5
        elif 30 <= rsi <= 70:
            score += 3

        roc = latest['roc_20']
        if roc > 10:
            score += 10
        elif roc > 5:
            score += 7
        elif roc > 0:
            score += 4

        if latest['ema_alignment'] == 2:
            score += 10
        elif latest['ema_alignment'] == 1:
            score += 5

        # Trend (25 pts)
        if latest['ema_alignment'] == 2 and roc > 0:
            score += 25
        elif latest['ema_alignment'] >= 1:
            score += 15

        # Volatility (20 pts) - lower is better
        atr_pct = (latest['atr'] / latest['close']) * 100
        if atr_pct < 2:
            score += 20
        elif atr_pct < 4:
            score += 10

        # Risk/Reward (30 pts)
        returns_60d = (latest['close'] / recent['close'].iloc[0] - 1) * 100
        if returns_60d > 15:
            score += 30
        elif returns_60d > 10:
            score += 20
        elif returns_60d > 5:
            score += 15
        elif returns_60d > 0:
            score += 10

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

    # ========================
    # 5. MAIN EXECUTION
    # ========================

    def run(self):
        """Run complete end-to-end demo"""
        logger.info("="*60)
        logger.info("S&P 500 PORTFOLIO ROTATION BOT")
        logger.info("Goal: 20%+ Annual Return")
        logger.info("="*60)

        # 1. Load data
        self.prepare_data()

        # 2. Score stocks
        self.score_all_stocks()

        # 3. Show rankings
        logger.info("\nTOP 10 STOCKS (Current Rankings):")
        for i, (ticker, score) in enumerate(self.rankings[:10], 1):
            logger.info(f"  {i}. {ticker}: {score:.1f} points")

        logger.info("\nBOTTOM 5 STOCKS:")
        for i, (ticker, score) in enumerate(self.rankings[-5:], 1):
            logger.info(f"  {ticker}: {score:.1f} points")

        # 4. Backtest
        logger.info("\n" + "="*60)
        logger.info("Running Portfolio Backtest...")
        logger.info("="*60)
        portfolio_df = self.backtest(top_n=10)

        return portfolio_df


if __name__ == "__main__":
    bot = PortfolioRotationBot()
    results = bot.run()
