"""Create VIX proxy from SPY volatility"""
import pandas as pd
import numpy as np

# Load SPY data
spy = pd.read_csv('sp500_data/daily/SPY.csv', index_col=0, parse_dates=True)
spy.columns = [c.lower() for c in spy.columns]

# Calculate 30-day rolling volatility (annualized)
returns = spy['close'].pct_change()
rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100

# VIX proxy: scale to typical VIX range (10-80)
vix_proxy = rolling_vol

# Create VIX-like dataframe
vix_df = pd.DataFrame({
    'open': vix_proxy,
    'high': vix_proxy * 1.05,
    'low': vix_proxy * 0.95,
    'close': vix_proxy,
    'volume': 0
}, index=spy.index)

# Save
vix_df.to_csv('sp500_data/daily/VIX.csv')

print(f"Created VIX proxy with {len(vix_df)} days")
print(f"\nVIX proxy stats:")
print(f"Mean: {vix_proxy.mean():.2f}")
print(f"Min: {vix_proxy.min():.2f}")
print(f"Max: {vix_proxy.max():.2f}")
print("\nSample:")
print(vix_df.tail())
