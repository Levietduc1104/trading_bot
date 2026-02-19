"""
V30 Live Trading Bot Adapter
=============================
Adapts V30 backtesting strategy to work with live Alpaca data

This class mimics the backtesting bot structure that V30 strategy expects:
- bot.stocks_data = {ticker: DataFrame with OHLCV}
- bot.initial_capital = starting capital
"""

import sys
import os

# IMPORTANT: Disable proxies BEFORE any network imports
os.environ['NO_PROXY'] = '*'
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from alpaca_broker_enhanced import AlpacaBrokerEnhanced
import alpaca_trade_api as tradeapi
from alpaca_config import (
    ALPACA_PAPER_API_KEY,
    ALPACA_PAPER_SECRET_KEY,
    ALPACA_PAPER_BASE_URL
)


class V30LiveBot:
    """
    Adapter class that makes V30 strategy work with live Alpaca data
    Mimics the backtest bot structure
    """

    def __init__(self, paper_trading=True, lookback_days=252, debug=True):
        """
        Initialize the V30 live trading bot

        Args:
            paper_trading: Use paper trading account
            lookback_days: Days of historical data to fetch (default: 252 = 1 year)
            debug: Enable debug mode with verbose logging
        """
        self.debug = debug
        self.lookback_days = lookback_days

        if self.debug:
            print("\n" + "="*60)
            print("V30 LIVE TRADING BOT - INITIALIZATION")
            print("="*60)

        # Initialize Alpaca broker
        if self.debug:
            print("\n[DEBUG] Connecting to Alpaca...")
        self.broker = AlpacaBrokerEnhanced(paper_trading=paper_trading)

        # Initialize Alpaca API for data fetching
        self.api = tradeapi.REST(
            ALPACA_PAPER_API_KEY,
            ALPACA_PAPER_SECRET_KEY,
            ALPACA_PAPER_BASE_URL,
            api_version='v2'
        )

        # ============================================================
        # ACCOUNT STATUS CHECK (BEFORE TRADING)
        # ============================================================
        if self.debug:
            print("\n" + "="*70)
            print("ACCOUNT STATUS CHECK")
            print("="*70)

        # Get account info
        account = self.broker.get_account_info()
        self.initial_capital = account['portfolio_value']

        if self.debug:
            print(f"\nPortfolio Value:      ${account['portfolio_value']:>12,.2f}")
            print(f"Cash Available:       ${account['cash']:>12,.2f}")
            print(f"Buying Power:         ${account['buying_power']:>12,.2f}")
            
            # Check current positions
            positions = self.broker.get_positions()
            print(f"\nCurrent Positions:    {len(positions)}")
            
            if not positions.empty:
                print("\nYour Positions:")
                for _, pos in positions.iterrows():
                    pnl = float(pos.get('unrealized_pl', 0))
                    pnl_pct = float(pos.get('unrealized_plpc', 0)) * 100
                    print(f"  • {pos['symbol']}: {int(pos['qty'])} shares @ ${float(pos['current_price']):.2f}")
                    print(f"    P/L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            else:
                print("  (No positions - fresh account)")
            
            # Check recent orders
            try:
                orders = self.broker.api.list_orders(status='all', limit=5)
                if orders:
                    print(f"\nRecent Orders:        {len(orders)}")
                    for order in orders[:3]:
                        print(f"  • {order.symbol}: {order.side.upper()} {order.qty} shares - {order.status}")
                else:
                    print(f"\nRecent Orders:        0")
            except:
                pass
            
            print(f"\nLookback Days:        {lookback_days}")
            print("\n" + "="*70)
            print("✅ READY TO START V30 TRADING")
            print("="*70 + "\n")

        # Storage for stock data (mimics backtest bot structure)
        self.stocks_data = {}

    def fetch_stock_universe(self, max_stocks=50):
        """
        Get list of stocks to analyze
        Returns top liquid stocks suitable for V30 strategy

        Args:
            max_stocks: Maximum number of stocks to fetch

        Returns:
            List of ticker symbols
        """
        if self.debug:
            print(f"\n[DEBUG] Fetching stock universe (max {max_stocks} stocks)...")

        # Top liquid mega-cap and large-cap stocks
        # These are suitable for V30's mega-cap + momentum strategy
        top_stocks = [
            # Mega caps (top by market cap and liquidity)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK.B', 'V', 'UNH',
            'JNJ', 'WMT', 'JPM', 'MA', 'PG',
            'HD', 'CVX', 'MRK', 'ABBV', 'KO',
            'PEP', 'COST', 'AVGO', 'ADBE', 'CSCO',
            'MCD', 'ACN', 'TMO', 'NFLX', 'ABT',
            'CRM', 'DHR', 'VZ', 'INTC', 'NKE',
            'AMD', 'TXN', 'QCOM', 'UNP', 'PM',
            'HON', 'NEE', 'BMY', 'RTX', 'INTU',
            'SBUX', 'UPS', 'LOW', 'BA', 'CAT'
        ]

        stocks = top_stocks[:max_stocks]

        if self.debug:
            print(f"[DEBUG] Selected {len(stocks)} stocks")
            print(f"[DEBUG] Stocks: {', '.join(stocks[:10])}{'...' if len(stocks) > 10 else ''}")

        return stocks

    def fetch_historical_data(self, symbols, days=None):
        """
        Fetch historical data from Alpaca for given symbols
        Structures data like backtest bot: {ticker: DataFrame}

        Args:
            symbols: List of ticker symbols
            days: Days of historical data (default: self.lookback_days)

        Returns:
            Dict of {ticker: DataFrame with OHLCV}
        """
        if days is None:
            days = self.lookback_days

        if self.debug:
            print(f"\n[DEBUG] Fetching {days} days of historical data...")
            print(f"[DEBUG] Symbols: {len(symbols)}")

        # Use reasonable date range (Alpaca has data up to ~2024)
        # If system date is future, use 2024-12-31 as end date
        end_date = datetime.now()
        if end_date.year > 2024:
            end_date = datetime(2024, 12, 31)
            if self.debug:
                print(f"[DEBUG] System date is in future, using end date: {end_date.date()}")

        start_date = end_date - timedelta(days=days + 100)  # Extra buffer

        if self.debug:
            print(f"[DEBUG] Date range: {start_date.date()} to {end_date.date()}")

        stocks_data = {}
        failed_symbols = []

        for i, symbol in enumerate(symbols):
            try:
                if self.debug and (i + 1) % 10 == 0:
                    print(f"[DEBUG] Progress: {i + 1}/{len(symbols)} stocks...")

                # Fetch bars from Alpaca
                bars = self.api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='all'  # Adjust for splits/dividends
                ).df

                if bars.empty:
                    if self.debug:
                        print(f"[DEBUG] {symbol}: No data available")
                    failed_symbols.append(symbol)
                    continue

                # Convert to format expected by V30 strategy
                df = pd.DataFrame({
                    'open': bars['open'],
                    'high': bars['high'],
                    'low': bars['low'],
                    'close': bars['close'],
                    'volume': bars['volume']
                })

                # Remove timezone info (V30 expects tz-naive timestamps)
                df.index = pd.to_datetime(df.index).tz_localize(None)

                stocks_data[symbol] = df

                # Rate limiting to avoid hitting Alpaca API limits
                time.sleep(0.05)  # 50ms delay

            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] {symbol}: Error - {str(e)[:50]}")
                failed_symbols.append(symbol)
                continue

        if self.debug:
            print(f"\n[DEBUG] Successfully fetched: {len(stocks_data)}/{len(symbols)} stocks")
            if failed_symbols:
                print(f"[DEBUG] Failed symbols: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")

        return stocks_data

    def load_market_data(self):
        """
        Load all market data needed for V30 strategy
        This populates self.stocks_data
        """
        if self.debug:
            print("\n" + "="*60)
            print("LOADING MARKET DATA")
            print("="*60)

        # Get stock universe
        symbols = self.fetch_stock_universe(max_stocks=50)

        # Fetch historical data
        self.stocks_data = self.fetch_historical_data(symbols, self.lookback_days)

        # Add market indicators (SPY, VIX)
        if self.debug:
            print(f"\n[DEBUG] Fetching market indicators (SPY, VIX)...")

        for indicator in ['SPY', 'VIX']:
            if indicator not in self.stocks_data:
                try:
                    data = self.fetch_historical_data([indicator], self.lookback_days)
                    if data:
                        self.stocks_data.update(data)
                        if self.debug:
                            print(f"[DEBUG] {indicator}: ✓ Loaded")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] {indicator}: ✗ Failed - {str(e)[:50]}")

        if self.debug:
            print(f"\n[DEBUG] Total stocks loaded: {len(self.stocks_data)}")
            if self.stocks_data:
                sample_ticker = list(self.stocks_data.keys())[0]
                sample_df = self.stocks_data[sample_ticker]
                print(f"[DEBUG] Date range: {sample_df.index[0].date()} to {sample_df.index[-1].date()}")
            print("="*60 + "\n")

    def get_current_vix(self):
        """
        Get current VIX value
        Returns VIX if available, otherwise estimates from SPY volatility
        """
        try:
            # Try to get VIX from loaded data
            if 'VIX' in self.stocks_data:
                vix_value = self.stocks_data['VIX']['close'].iloc[-1]
                if self.debug:
                    print(f"[DEBUG] VIX: {vix_value:.2f} (from data)")
                return float(vix_value)
        except:
            pass

        # Fallback: Estimate from SPY volatility
        try:
            if 'SPY' in self.stocks_data:
                spy_df = self.stocks_data['SPY'].tail(20)
                returns = spy_df['close'].pct_change().dropna()
                spy_vol = returns.std() * np.sqrt(252) * 100  # Annualized vol in %
                if self.debug:
                    print(f"[DEBUG] VIX: {spy_vol:.2f} (estimated from SPY)")
                return spy_vol
        except:
            pass

        # Default fallback
        if self.debug:
            print(f"[DEBUG] VIX: 15.00 (default fallback)")
        return 15.0


if __name__ == "__main__":
    # Test the bot adapter
    print("\nTesting V30LiveBot...")
    print("="*60)

    bot = V30LiveBot(paper_trading=True, lookback_days=252, debug=True)

    print("\n✓ Bot initialized successfully!")
    print(f"  Initial capital: ${bot.initial_capital:,.2f}")
    print(f"  Lookback days: {bot.lookback_days}")

    print("\nLoading market data...")
    bot.load_market_data()

    print(f"\n✓ Data loaded successfully!")
    print(f"  Stocks loaded: {len(bot.stocks_data)}")

    if bot.stocks_data:
        print(f"\n  Sample stocks:")
        for i, (ticker, df) in enumerate(list(bot.stocks_data.items())[:5]):
            print(f"    {ticker}: {len(df)} days, latest close: ${df['close'].iloc[-1]:.2f}")

    print("\n" + "="*60)
