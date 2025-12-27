"""Download VIX historical data"""
import yfinance as yf
import pandas as pd

print("Downloading VIX data...")
vix = yf.download('^VIX', start='2004-01-01', end='2024-12-31', progress=False)
print(f"Downloaded {len(vix)} days of VIX data")

# Save to CSV
vix.to_csv('sp500_data/daily/^VIX.csv')
print("Saved to sp500_data/daily/^VIX.csv")

# Show sample
print("\nSample VIX data:")
print(vix.tail())
print(f"\nVIX range: {vix['Close'].min():.2f} - {vix['Close'].max():.2f}")
