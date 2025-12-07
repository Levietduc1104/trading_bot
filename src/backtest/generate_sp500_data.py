#\!/usr/bin/env python3
"""
Generate S&P 500 stock data from 2005 to present for all stocks
Fixed version - no flat price periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_stock_data(ticker, start_year=2005):
    """Generate realistic stock data from start_year to present"""
    
    np.random.seed(hash(ticker) % (2**32))
    
    # Generate varying parameters for different stocks
    base_price = np.random.uniform(20, 200)
    annual_growth = np.random.uniform(0.08, 0.20)  # 8-20% annual growth
    volatility = np.random.uniform(0.015, 0.035)  # Daily volatility
    
    # Some stocks have different patterns
    if hash(ticker) % 3 == 0:  # High growth stocks
        annual_growth = np.random.uniform(0.15, 0.30)
        volatility = np.random.uniform(0.025, 0.045)
    elif hash(ticker) % 3 == 1:  # Stable stocks
        annual_growth = np.random.uniform(0.05, 0.12)
        volatility = np.random.uniform(0.01, 0.025)
    
    start_date = datetime(start_year, 1, 3)
    end_date = datetime(2024, 10, 3)
    
    current_date = start_date
    price = base_price
    data = []
    
    # Calculate daily drift
    trading_days_per_year = 252
    daily_drift = (1 + annual_growth) ** (1/trading_days_per_year) - 1
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Market event adjustments (temporary volatility spikes, NOT price multipliers)
        volatility_multiplier = 1.0
        drift_adjustment = 0.0
        
        # Financial crisis 2008 (higher volatility, negative drift)
        if datetime(2008, 9, 1) <= current_date <= datetime(2009, 3, 1):
            volatility_multiplier = 2.5
            drift_adjustment = -0.003  # Temporary negative drift
        # COVID crash 2020 (brief spike)
        elif datetime(2020, 2, 20) <= current_date <= datetime(2020, 3, 23):
            volatility_multiplier = 3.5
            drift_adjustment = -0.005
        # COVID recovery 2020-2021 (higher growth)
        elif datetime(2020, 4, 1) <= current_date <= datetime(2021, 6, 30):
            volatility_multiplier = 1.8
            drift_adjustment = 0.002
        
        # Generate daily return with drift
        adjusted_drift = daily_drift + drift_adjustment
        daily_return = adjusted_drift + (volatility * volatility_multiplier) * np.random.randn()
        
        # Update price
        price = price * (1 + daily_return)
        
        # Ensure price stays reasonable (prevent going to zero)
        price = max(price, base_price * 0.2)  # Never drop below 20% of base
        price = min(price, base_price * 100)  # Cap at 100x base
        
        # Generate OHLC with realistic intraday movement
        daily_range = abs(volatility * volatility_multiplier * np.random.randn())
        
        high = price * (1 + daily_range * np.random.uniform(0.5, 1.5))
        low = price * (1 - daily_range * np.random.uniform(0.5, 1.5))
        
        # Open and close within high-low range
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        close_price = price  # Close is our tracked price
        
        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume (realistic range)
        base_volume = np.random.uniform(5000000, 50000000)
        volume_multiplier = 1 + (3 * abs(daily_return))  # Higher volume on big moves
        volume = int(base_volume * volume_multiplier * volatility_multiplier)
        
        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)

def main():
    data_dir = Path('sp500_data/daily')
    
    # Get all CSV files
    csv_files = sorted(data_dir.glob('*.csv'))
    total = len(csv_files)
    
    print(f"Found {total} stocks to process")
    print("Generating realistic stock data from 2005 to 2024...")
    print("(No flat price periods - continuous trading simulation)")
    print("-" * 60)
    
    for idx, csv_file in enumerate(csv_files, 1):
        ticker = csv_file.stem
        
        # Generate data
        df = generate_stock_data(ticker)
        
        # Verify no flat periods
        price_range = df['close'].max() - df['close'].min()
        if price_range < 1:
            print(f"WARNING: {ticker} has suspicious flat prices, regenerating...")
            df = generate_stock_data(ticker + "_retry")
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        
        if idx % 50 == 0 or idx == total:
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%) - Latest: {ticker} (Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f})")
    
    print("-" * 60)
    print(f"✓ Successfully generated realistic data for all {total} stocks\!")
    print(f"Date range: 2005-01-03 to 2024-10-03")
    print(f"Approximate records per stock: ~5000 trading days")

if __name__ == "__main__":
    main()
