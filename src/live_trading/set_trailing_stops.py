#\!/usr/bin/env python3
"""
Quick script to set 15% trailing stops on all current positions
"""

from alpaca_broker_enhanced import AlpacaBrokerEnhanced

print("\n" + "="*70)
print("SETTING TRAILING STOPS FOR EXISTING POSITIONS")
print("="*70 + "\n")

try:
    broker = AlpacaBrokerEnhanced(paper_trading=True)
    
    # Set 15% trailing stops on all positions
    success_count = broker.set_trailing_stops_for_portfolio(trail_percent=0.15)
    
    if success_count > 0:
        print("\n" + "="*70)
        print("SUCCESS\! Your positions are now protected.")
        print("="*70)
        print("\nWhat happens next:")
        print("  • Alpaca monitors your positions 24/7")
        print("  • If any stock drops 15% from its peak, it sells automatically")
        print("  • You'll receive email notifications when stops trigger")
        print("  • Check status anytime: python3 alpaca_broker_enhanced.py")
        print("\n" + "="*70 + "\n")
    else:
        print("\n⚠️  No positions to protect")
        
except Exception as e:
    print(f"\n❌ Error: {e}\n")
    print("Make sure you're in the src/live_trading directory")
