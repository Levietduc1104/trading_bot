"""
Daily Position Monitor (Optional)
=================================
Monitors positions and displays status

This script is OPTIONAL - Alpaca handles trailing stops automatically.
This just provides daily visibility into your positions.

Run daily at 3:55 PM (5 min before market close):
  55 15 * * 1-5 python3 monitor_positions.py >> logs/daily_monitor.log
"""

from alpaca_broker_enhanced import AlpacaBrokerEnhanced
from datetime import datetime

print(f"\n{'='*80}")
print(f"POSITION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

try:
    broker = AlpacaBrokerEnhanced(paper_trading=True)

    # Get account info
    account = broker.get_account_info()
    positions = broker.get_positions()

    print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
    print(f"Cash: ${account['cash']:,.2f}")
    print(f"Buying Power: ${account['buying_power']:,.2f}\n")

    if positions.empty:
        print("No positions currently held\n")
    else:
        print(f"{'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P/L %':>8} {'Stop Cushion':>14}")
        print("-" * 80)

        for _, pos in positions.iterrows():
            symbol = pos['symbol']
            qty = int(pos['qty'])
            entry = pos['avg_entry_price']
            current = pos['current_price']
            pl_pct = pos['unrealized_plpc']

            # Calculate stop cushion (how far from 15% trailing stop)
            peak = max(entry, current)
            stop_price = peak * 0.85
            cushion = ((current - stop_price) / current) * 100

            # Status indicator
            if pl_pct > 5:
                status = "🟢"
            elif pl_pct < -10:
                status = "🔴"
            else:
                status = "🟡"

            print(f"{status} {symbol:<6} {qty:>6} ${entry:>9.2f} ${current:>9.2f} "
                  f"{pl_pct:>7.2f}% {cushion:>13.1f}%")

        print("-" * 80)

        # Summary
        total_value = positions['market_value'].sum()
        total_pl = positions['unrealized_pl'].sum()
        total_pl_pct = (total_pl / (total_value - total_pl)) * 100

        print(f"\nTotal P/L: ${total_pl:,.2f} ({total_pl_pct:+.2f}%)\n")

    # Display trailing stop status
    broker.display_trailing_stop_status()

    print(f"{'='*80}\n")

except Exception as e:
    print(f"\n✗ Error: {e}\n")
    print("Make sure Alpaca connection is working")
    print("Run: python3 test_alpaca_connection.py\n")
