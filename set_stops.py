import sys
sys.path.insert(0, 'src/live_trading')
from alpaca_broker_enhanced import AlpacaBrokerEnhanced
b = AlpacaBrokerEnhanced(paper_trading=True)
b.cancel_all_trailing_stops()
n = b.set_trailing_stops_for_portfolio(trail_percent=0.15)
print(f'{n} trailing stops set')
