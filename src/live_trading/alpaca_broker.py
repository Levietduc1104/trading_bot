"""
Alpaca Broker Integration
Connects your ML trading strategies to Alpaca's API for real-time trading
"""

import sys
import os

# IMPORTANT: Disable proxies BEFORE importing alpaca_trade_api
os.environ['NO_PROXY'] = '*'
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

# Now import after proxy is disabled
import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Import config
from alpaca_config import (
    ALPACA_PAPER_API_KEY,
    ALPACA_PAPER_SECRET_KEY,
    ALPACA_PAPER_BASE_URL,
)


class AlpacaBroker:
    """
    Alpaca broker interface for executing trades from ML strategies
    """

    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca API connection

        Args:
            paper_trading: If True, use paper trading account
        """
        self.paper_trading = paper_trading

        if paper_trading:
            self.api = tradeapi.REST(
                ALPACA_PAPER_API_KEY,
                ALPACA_PAPER_SECRET_KEY,
                ALPACA_PAPER_BASE_URL,
                api_version='v2'
            )
        else:
            raise NotImplementedError("Live trading not configured yet - stay in paper trading!")

        # Verify connection
        try:
            account = self.api.get_account()
            print(f"\n{'='*60}")
            print(f"✓ Connected to Alpaca ({'Paper' if paper_trading else 'Live'} Trading)")
            print(f"{'='*60}")
            print(f"Account Status: {account.status}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"{'='*60}\n")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")

    def get_account_info(self) -> Dict:
        """Get account information"""
        account = self.api.get_account()
        return {
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'buying_power': float(account.buying_power),
            'equity': float(account.equity),
            'pattern_day_trader': account.pattern_day_trader
        }

    def get_positions(self) -> pd.DataFrame:
        """Get current positions"""
        positions = self.api.list_positions()

        if not positions:
            return pd.DataFrame()

        pos_data = []
        for pos in positions:
            pos_data.append({
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc) * 100
            })

        return pd.DataFrame(pos_data)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price:
                prices[symbol] = price
        return prices

    def place_market_order(self, symbol: str, qty: int, side: str = 'buy') -> Optional[str]:
        """
        Place a market order

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"✓ {side.upper()} order placed: {qty} shares of {symbol} (Order ID: {order.id})")
            return order.id
        except Exception as e:
            print(f"✗ Error placing {side} order for {symbol}: {e}")
            return None

    def place_stop_loss_order(self, symbol: str, qty: int, stop_price: float) -> Optional[str]:
        """Place a stop-loss order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='stop',
                time_in_force='gtc',  # Good 'til cancelled
                stop_price=stop_price
            )
            print(f"✓ STOP-LOSS order placed: {qty} shares of {symbol} @ ${stop_price:.2f}")
            return order.id
        except Exception as e:
            print(f"✗ Error placing stop-loss for {symbol}: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            print("✓ All orders cancelled")
        except Exception as e:
            print(f"✗ Error cancelling orders: {e}")

    def get_historical_bars(self, symbol: str, timeframe: str = '1Day',
                           start: str = None, end: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Get historical price data

        Args:
            symbol: Stock ticker
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            limit: Number of bars to fetch
        """
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                limit=limit
            ).df
            return bars
        except Exception as e:
            print(f"Error getting bars for {symbol}: {e}")
            return pd.DataFrame()

    def rebalance_portfolio(self, target_positions: Dict[str, float], cash_reserve: float = 0.0):
        """
        Rebalance portfolio to match target positions

        Args:
            target_positions: Dict of {symbol: target_weight} where weights are 0-1
            cash_reserve: Fraction of portfolio to keep as cash (0-1)

        Example:
            target_positions = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.50}
            cash_reserve = 0.10  # Keep 10% cash
        """
        print(f"\n{'='*60}")
        print("REBALANCING PORTFOLIO")
        print(f"{'='*60}")

        # Get account info
        account = self.get_account_info()
        portfolio_value = account['portfolio_value']

        print(f"\nPortfolio Value: ${portfolio_value:,.2f}")
        print(f"Cash Reserve: {cash_reserve*100:.1f}%")
        print(f"Target Positions: {len(target_positions)}")

        # Get current positions
        current_positions = self.get_positions()
        current_symbols = set(current_positions['symbol'].tolist()) if not current_positions.empty else set()
        target_symbols = set(target_positions.keys())

        print(f"\nCurrent Holdings: {len(current_symbols)}")
        print(f"Target Holdings: {len(target_symbols)}")

        # Get latest prices
        all_symbols = list(current_symbols.union(target_symbols))
        prices = self.get_latest_prices(all_symbols)

        # Step 1: Sell positions not in target
        to_sell = current_symbols - target_symbols
        if to_sell:
            print(f"\n🔴 SELLING {len(to_sell)} positions no longer in target:")
            for symbol in to_sell:
                pos = current_positions[current_positions['symbol'] == symbol].iloc[0]
                qty = int(pos['qty'])
                print(f"  Selling {qty} shares of {symbol}")
                self.place_market_order(symbol, qty, side='sell')
        else:
            print(f"\n✓ No positions to sell")

        # Step 2: Calculate target dollar amounts
        investable_amount = portfolio_value * (1 - cash_reserve)

        print(f"\n🔵 BUYING/ADJUSTING {len(target_positions)} positions:")
        print(f"{'Symbol':<8} {'Target $':>12} {'Target %':>10} {'Shares':>8}")
        print("-" * 50)

        for symbol, target_weight in sorted(target_positions.items(), key=lambda x: x[1], reverse=True):
            target_value = target_weight * portfolio_value

            if symbol not in prices:
                print(f"  {symbol:<8} - ✗ No price available, skipping")
                continue

            price = prices[symbol]
            target_shares = int(target_value / price)

            # Get current shares
            current_shares = 0
            if symbol in current_symbols:
                pos = current_positions[current_positions['symbol'] == symbol].iloc[0]
                current_shares = int(pos['qty'])

            shares_diff = target_shares - current_shares

            print(f"  {symbol:<8} ${target_value:>10,.2f} {target_weight*100:>8.2f}% {target_shares:>8}")

            if shares_diff > 0:
                # Buy more shares
                self.place_market_order(symbol, shares_diff, side='buy')
            elif shares_diff < 0:
                # Sell excess shares
                self.place_market_order(symbol, abs(shares_diff), side='sell')
            # else: shares_diff == 0, position already correct

        print("-" * 50)
        print(f"\n✓ Rebalance orders submitted!")
        print(f"{'='*60}\n")


def test_connection():
    """Test Alpaca connection and display account info"""
    print("\n" + "="*60)
    print("TESTING ALPACA API CONNECTION")
    print("="*60 + "\n")

    try:
        broker = AlpacaBroker(paper_trading=True)

        print("\n" + "="*60)
        print("ACCOUNT INFORMATION")
        print("="*60)

        account_info = broker.get_account_info()
        for key, value in account_info.items():
            if isinstance(value, float):
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value}")

        print("\n" + "="*60)
        print("CURRENT POSITIONS")
        print("="*60)

        positions = broker.get_positions()
        if positions.empty:
            print("No positions")
        else:
            print(positions.to_string())

        print("\n" + "="*60)
        print("TEST SUCCESSFUL!")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ CONNECTION FAILED: {e}\n")
        print("Make sure you:")
        print("1. Added your API keys to alpaca_config.py")
        print("2. Enabled MFA on your Alpaca account")
        print("3. Generated API keys from the Alpaca dashboard")
        return False


if __name__ == "__main__":
    # Test connection
    test_connection()
