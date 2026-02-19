"""
Enhanced Alpaca Broker with Trailing Stop Support
==================================================
Extends AlpacaBroker with trailing stop functionality

Features:
- Native Alpaca trailing stop orders (monitored 24/7 by Alpaca)
- Batch trailing stop setup for entire portfolio
- Stop price monitoring and logging
"""

import sys
import os

# Disable proxies
os.environ['NO_PROXY'] = '*'
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

from typing import Optional, Dict
from alpaca_broker import AlpacaBroker


class AlpacaBrokerEnhanced(AlpacaBroker):
    """
    Enhanced Alpaca broker with trailing stop support

    Trailing stops are monitored 24/7 by Alpaca's servers,
    so your computer doesn't need to be running.
    """

    def place_trailing_stop_order(self, symbol: str, qty: int,
                                  trail_percent: float) -> Optional[str]:
        """
        Place a trailing stop order

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            qty: Number of shares to protect
            trail_percent: Trailing percentage (e.g., 0.15 for 15%)

        Returns:
            Order ID if successful, None otherwise

        Example:
            broker.place_trailing_stop_order('AAPL', 10, trail_percent=0.15)
            # Sells 10 shares of AAPL if price drops 15% from peak
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='trailing_stop',
                trail_percent=trail_percent,
                time_in_force='gtc'  # Good 'til cancelled
            )

            print(f"  ✓ {symbol}: {qty} shares @ {trail_percent*100:.1f}% trail (Order: {order.id[:8]}...)")
            return order.id

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {str(e)[:50]}")
            return None

    def set_trailing_stops_for_portfolio(self, trail_percent: float = 0.15) -> int:
        """
        Set trailing stops for ALL current positions

        Args:
            trail_percent: Trailing percentage (default 0.15 = 15%)

        Returns:
            Number of trailing stops successfully placed

        Example:
            broker.set_trailing_stops_for_portfolio(trail_percent=0.15)
            # Sets 15% trailing stops on all positions
        """
        print(f"\n{'='*70}")
        print("SETTING TRAILING STOPS FOR ALL POSITIONS")
        print(f"{'='*70}\n")

        # Get current positions
        positions = self.get_positions()

        if positions.empty:
            print("⚠️  No positions found - nothing to protect")
            print("    Run rebalancing first to create positions\n")
            return 0

        print(f"Protecting {len(positions)} positions with {trail_percent*100:.1f}% trailing stops:\n")

        success_count = 0

        for _, pos in positions.iterrows():
            symbol = pos['symbol']
            qty = int(pos['qty'])

            # Place trailing stop order
            order_id = self.place_trailing_stop_order(symbol, qty, trail_percent)

            if order_id:
                success_count += 1

        print(f"\n{'='*70}")
        print(f"✅ Successfully set {success_count}/{len(positions)} trailing stops")
        print(f"   Alpaca monitors 24/7 - stops will trigger automatically")
        print(f"{'='*70}\n")

        return success_count

    def cancel_all_trailing_stops(self) -> int:
        """
        Cancel all pending trailing stop orders

        Returns:
            Number of orders cancelled
        """
        try:
            orders = self.api.list_orders(status='open')
            trailing_stops = [o for o in orders if o.type == 'trailing_stop']

            if not trailing_stops:
                print("No trailing stop orders to cancel")
                return 0

            print(f"\nCancelling {len(trailing_stops)} trailing stop orders...")

            for order in trailing_stops:
                try:
                    self.api.cancel_order(order.id)
                    print(f"  ✓ Cancelled: {order.symbol}")
                except Exception as e:
                    print(f"  ✗ Failed {order.symbol}: {e}")

            print(f"✓ Cancelled {len(trailing_stops)} stops\n")
            return len(trailing_stops)

        except Exception as e:
            print(f"✗ Error: {e}")
            return 0

    def get_trailing_stop_status(self) -> Dict:
        """Get status of all trailing stop orders"""
        try:
            orders = self.api.list_orders(status='open')
            trailing_stops = [o for o in orders if o.type == 'trailing_stop']

            status = {
                'total': len(trailing_stops),
                'orders': []
            }

            for order in trailing_stops:
                status['orders'].append({
                    'symbol': order.symbol,
                    'qty': int(order.qty),
                    'trail_percent': float(order.trail_percent) if hasattr(order, 'trail_percent') else None,
                    'submitted_at': order.submitted_at,
                    'order_id': order.id
                })

            return status

        except Exception as e:
            print(f"Error: {e}")
            return {'total': 0, 'orders': []}

    def display_trailing_stop_status(self):
        """Display trailing stop status"""
        print(f"\n{'='*70}")
        print("TRAILING STOP STATUS")
        print(f"{'='*70}\n")

        status = self.get_trailing_stop_status()

        if status['total'] == 0:
            print("⚠️  No active trailing stops\n")
            print("Recommendation after rebalancing:")
            print("  broker.set_trailing_stops_for_portfolio(trail_percent=0.15)\n")
            return

        print(f"Active Trailing Stops: {status['total']}\n")

        for order in status['orders']:
            trail_pct = f"{order['trail_percent']*100:.1f}%" if order['trail_percent'] else "N/A"
            submitted = order['submitted_at'].strftime('%Y-%m-%d %H:%M')

            print(f"  {order['symbol']:6} - {order['qty']:>4} shares @ {trail_pct} trail (since {submitted})")

        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING ENHANCED ALPACA BROKER - TRAILING STOPS")
    print("="*70 + "\n")

    try:
        broker = AlpacaBrokerEnhanced(paper_trading=True)

        # Display current positions
        print("\n1. Current Positions:")
        positions = broker.get_positions()

        if positions.empty:
            print("   No positions - run v30_yahoo_alpaca.py first\n")
        else:
            for _, pos in positions.iterrows():
                print(f"   {pos['symbol']:6} - {int(pos['qty']):>4} shares @ ${pos['current_price']:.2f}")

        # Display trailing stop status
        print("\n2. Trailing Stop Status:")
        broker.display_trailing_stop_status()

        print("="*70)
        print("✓ TEST COMPLETE")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}\n")

    def fetch_historical_data(self, symbol: str, days: int = 7):
        """
        Fetch historical OHLCV data from Alpaca
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            days: Number of days to fetch
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        from datetime import datetime, timedelta
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+2)  # Extra days for safety
            
            bars = self.api.get_bars(
                symbol,
                "1Day",
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            ).df
            
            if bars.empty:
                return None
            
            # Rename columns to match V30 expectations
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Keep only needed columns
            bars = bars[['open', 'high', 'low', 'close', 'volume']]
            
            # Remove timezone if present
            if bars.index.tz is not None:
                bars.index = bars.index.tz_localize(None)
            
            return bars
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
