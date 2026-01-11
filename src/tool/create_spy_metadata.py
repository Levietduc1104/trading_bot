"""
Create metadata for SPY and SPYEW indices
"""
import pandas as pd
import json

print("Creating metadata for SPY and SPYEW...")

# Load SPY data
spy_df = pd.read_csv('../../sp500_data/daily/SPY.csv')
spy_df['date'] = pd.to_datetime(spy_df['date'])

# Calculate metrics
current_price = spy_df['close'].iloc[-1]
previous_close = spy_df['close'].iloc[-2]
start_price = spy_df['close'].iloc[0]
total_return = ((current_price / start_price) - 1) * 100

# Calculate volatility
daily_returns = spy_df['close'].pct_change().dropna()
annual_volatility = daily_returns.std() * (252 ** 0.5) * 100

spy_metadata = {
    'ticker': 'SPY',
    'sector': 'Index - Market Cap Weighted S&P 500',
    'current_price': round(current_price, 2),
    'previous_close': round(previous_close, 2),
    'open': round(spy_df['open'].iloc[-1], 2),
    'day_low': round(spy_df['low'].iloc[-1], 2),
    'day_high': round(spy_df['high'].iloc[-1], 2),
    'day_range': f"{spy_df['low'].iloc[-1]:.2f} - {spy_df['high'].iloc[-1]:.2f}",
    '52_week_low': round(spy_df['low'].tail(252).min(), 2),
    '52_week_high': round(spy_df['high'].tail(252).max(), 2),
    '52_week_range': f"{spy_df['low'].tail(252).min():.2f} - {spy_df['high'].tail(252).max():.2f}",
    'volume': int(spy_df['volume'].iloc[-1]),
    'avg_volume': int(spy_df['volume'].tail(63).mean()),
    'market_cap': 0,
    'market_cap_display': 'Index (470 stocks)',
    'beta': 1.0,
    'pe_ratio': 'N/A',
    'eps': 'N/A',
    'dividend_yield': 0,
    'annual_dividend': 0,
    'forward_dividend': 0,
    'ex_dividend_date': 'N/A',
    'earnings_date': 'N/A',
    'target_est': 'N/A',
    'shares_outstanding': 0,
    'total_return_pct': round(total_return, 1),
    'annual_volatility_pct': round(annual_volatility, 1),
    'start_price': round(start_price, 2)
}

with open('../../sp500_data/metadata/SPY.json', 'w') as f:
    json.dump(spy_metadata, f, indent=2)

print(f'✅ Created SPY metadata')
print(f'   Total Return: {total_return:.1f}%')
print(f'   Annual Volatility: {annual_volatility:.1f}%')

# SPYEW
spyew_df = pd.read_csv('../../sp500_data/daily/SPYEW.csv')
spyew_df['date'] = pd.to_datetime(spyew_df['date'])

current_price = spyew_df['close'].iloc[-1]
previous_close = spyew_df['close'].iloc[-2]
start_price = spyew_df['close'].iloc[0]
total_return = ((current_price / start_price) - 1) * 100

daily_returns = spyew_df['close'].pct_change().dropna()
annual_volatility = daily_returns.std() * (252 ** 0.5) * 100

spyew_metadata = {
    'ticker': 'SPYEW',
    'sector': 'Index - Equal Weighted S&P 500',
    'current_price': round(current_price, 2),
    'previous_close': round(previous_close, 2),
    'open': round(spyew_df['open'].iloc[-1], 2),
    'day_low': round(spyew_df['low'].iloc[-1], 2),
    'day_high': round(spyew_df['high'].iloc[-1], 2),
    'day_range': f"{spyew_df['low'].iloc[-1]:.2f} - {spyew_df['high'].iloc[-1]:.2f}",
    '52_week_low': round(spyew_df['low'].tail(252).min(), 2),
    '52_week_high': round(spyew_df['high'].tail(252).max(), 2),
    '52_week_range': f"{spyew_df['low'].tail(252).min():.2f} - {spyew_df['high'].tail(252).max():.2f}",
    'volume': int(spyew_df['volume'].iloc[-1]),
    'avg_volume': int(spyew_df['volume'].tail(63).mean()),
    'market_cap': 0,
    'market_cap_display': 'Index (470 stocks)',
    'beta': 1.0,
    'pe_ratio': 'N/A',
    'eps': 'N/A',
    'dividend_yield': 0,
    'annual_dividend': 0,
    'forward_dividend': 0,
    'ex_dividend_date': 'N/A',
    'earnings_date': 'N/A',
    'target_est': 'N/A',
    'shares_outstanding': 0,
    'total_return_pct': round(total_return, 1),
    'annual_volatility_pct': round(annual_volatility, 1),
    'start_price': round(start_price, 2)
}

with open('../../sp500_data/metadata/SPYEW.json', 'w') as f:
    json.dump(spyew_metadata, f, indent=2)

print(f'✅ Created SPYEW metadata')
print(f'   Total Return: {total_return:.1f}%')
print(f'   Annual Volatility: {annual_volatility:.1f}%')
