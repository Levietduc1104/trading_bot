"""
CREATE ARTIFICIAL S&P 500 INDEX
Using debug prints to track progress
"""
import pandas as pd
import numpy as np
import glob
import json
import os

print("="*80)
print("CREATING ARTIFICIAL S&P 500 INDEX")
print("="*80)

# Load all stock data
data_dir = '../../sp500_data/daily'
metadata_dir = '../../sp500_data/metadata'

print(f"\n[DEBUG] Looking for CSV files in: {data_dir}")
csv_files = glob.glob(f"{data_dir}/*.csv")
print(f"[DEBUG] Found {len(csv_files)} CSV files")

stocks_data = {}
market_caps = {}

print(f"\n[DEBUG] Loading stock data...")
for i, file_path in enumerate(csv_files, 1):
    ticker = os.path.basename(file_path).replace('.csv', '')

    if i % 100 == 0:
        print(f"[DEBUG] Loaded {i}/{len(csv_files)} stocks...")

    # Load price data
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    stocks_data[ticker] = df

    # Load market cap
    metadata_file = f"{metadata_dir}/{ticker}.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            market_caps[ticker] = metadata.get('market_cap', 1e9)
    else:
        market_caps[ticker] = 1e9  # Default 1B

print(f"[DEBUG] Successfully loaded {len(stocks_data)} stocks")
print(f"[DEBUG] Sample stocks: {list(stocks_data.keys())[:5]}")

# Get common dates
first_ticker = list(stocks_data.keys())[0]
dates = stocks_data[first_ticker].index

print(f"\n[DEBUG] Date range: {dates[0].date()} to {dates[-1].date()}")
print(f"[DEBUG] Total trading days: {len(dates)}")

# =====================================================
# MARKET-CAP WEIGHTED S&P 500 (Like Real)
# =====================================================
print("\n" + "="*80)
print("CREATING MARKET-CAP WEIGHTED INDEX")
print("="*80)

print("[DEBUG] Calculating daily index values...")
index_values = []
initial_total_cap = None

for day_idx, date in enumerate(dates):
    if day_idx % 500 == 0:
        print(f"[DEBUG] Processing day {day_idx}/{len(dates)} ({date.date()})...")

    total_market_cap = 0

    for ticker in stocks_data.keys():
        if date in stocks_data[ticker].index:
            price = stocks_data[ticker].loc[date, 'close']
            initial_price = stocks_data[ticker]['close'].iloc[0]
            shares = market_caps[ticker] / initial_price
            total_market_cap += price * shares

    # Normalize to start at 100
    if initial_total_cap is None:
        initial_total_cap = total_market_cap
        index_values.append(100.0)
        print(f"[DEBUG] Day 0: Initial total market cap = ${initial_total_cap/1e12:.2f}T")
    else:
        index_value = (total_market_cap / initial_total_cap) * 100
        index_values.append(index_value)

print(f"[DEBUG] Calculated {len(index_values)} daily values")
print(f"[DEBUG] Index start: {index_values[0]:.2f}")
print(f"[DEBUG] Index end: {index_values[-1]:.2f}")

# Create DataFrame
print("[DEBUG] Creating DataFrame...")
sp500_weighted = pd.DataFrame({
    'date': dates,
    'close': index_values
})

# Add OHLC
print("[DEBUG] Generating OHLC data...")
sp500_weighted['open'] = sp500_weighted['close'] * np.random.uniform(0.995, 1.005, len(sp500_weighted))
sp500_weighted['high'] = sp500_weighted['close'] * np.random.uniform(1.00, 1.02, len(sp500_weighted))
sp500_weighted['low'] = sp500_weighted['close'] * np.random.uniform(0.98, 1.00, len(sp500_weighted))
sp500_weighted['high'] = sp500_weighted[['open', 'high', 'low', 'close']].max(axis=1)
sp500_weighted['low'] = sp500_weighted[['open', 'high', 'low', 'close']].min(axis=1)
sp500_weighted['volume'] = 1000000000

# Calculate performance
start_value = sp500_weighted['close'].iloc[0]
end_value = sp500_weighted['close'].iloc[-1]
total_return = ((end_value / start_value) - 1) * 100
years = (dates[-1] - dates[0]).days / 365.25
annual_return = ((end_value / start_value) ** (1/years) - 1) * 100

print("\n" + "-"*80)
print("MARKET-CAP WEIGHTED S&P 500 RESULTS:")
print("-"*80)
print(f"  Start Value: {start_value:.2f}")
print(f"  End Value: {end_value:.2f}")
print(f"  Total Return: {total_return:.1f}%")
print(f"  Annual Return: {annual_return:.1f}%")
print(f"  Years: {years:.2f}")

# Save
output_file = f"{data_dir}/SPY.csv"
print(f"\n[DEBUG] Saving to {output_file}...")
sp500_weighted.to_csv(output_file, index=False)
print(f"✅ Saved SPY.csv ({len(sp500_weighted)} rows)")

# Show sample data
print("\n[DEBUG] Sample SPY data:")
print(sp500_weighted.head())

# =====================================================
# EQUAL WEIGHTED INDEX
# =====================================================
print("\n" + "="*80)
print("CREATING EQUAL-WEIGHTED INDEX")
print("="*80)

print("[DEBUG] Calculating equal-weighted values...")
equal_weighted_values = []

for day_idx, date in enumerate(dates):
    if day_idx % 500 == 0:
        print(f"[DEBUG] Processing day {day_idx}/{len(dates)}...")

    prices = []
    for ticker, df in stocks_data.items():
        if date in df.index:
            initial_price = df['close'].iloc[0]
            current_price = df.loc[date, 'close']
            normalized = (current_price / initial_price) * 100
            prices.append(normalized)

    avg_value = np.mean(prices)
    equal_weighted_values.append(avg_value)

print(f"[DEBUG] Calculated {len(equal_weighted_values)} values")

sp500_equal = pd.DataFrame({
    'date': dates,
    'close': equal_weighted_values
})

sp500_equal['open'] = sp500_equal['close'] * np.random.uniform(0.995, 1.005, len(sp500_equal))
sp500_equal['high'] = sp500_equal['close'] * np.random.uniform(1.00, 1.02, len(sp500_equal))
sp500_equal['low'] = sp500_equal['close'] * np.random.uniform(0.98, 1.00, len(sp500_equal))
sp500_equal['high'] = sp500_equal[['open', 'high', 'low', 'close']].max(axis=1)
sp500_equal['low'] = sp500_equal[['open', 'high', 'low', 'close']].min(axis=1)
sp500_equal['volume'] = 500000000

start_value = sp500_equal['close'].iloc[0]
end_value = sp500_equal['close'].iloc[-1]
total_return = ((end_value / start_value) - 1) * 100
annual_return = ((end_value / start_value) ** (1/years) - 1) * 100

print("\n" + "-"*80)
print("EQUAL-WEIGHTED INDEX RESULTS:")
print("-"*80)
print(f"  Start Value: {start_value:.2f}")
print(f"  End Value: {end_value:.2f}")
print(f"  Total Return: {total_return:.1f}%")
print(f"  Annual Return: {annual_return:.1f}%")

output_file2 = f"{data_dir}/SPYEW.csv"
print(f"\n[DEBUG] Saving to {output_file2}...")
sp500_equal.to_csv(output_file2, index=False)
print(f"✅ Saved SPYEW.csv ({len(sp500_equal)} rows)")

# =====================================================
# COMPARISON
# =====================================================
print("\n" + "="*80)
print("BENCHMARK COMPARISON")
print("="*80)

our_annual = 15.9  # From previous test

spy_annual = ((sp500_weighted['close'].iloc[-1] / sp500_weighted['close'].iloc[0]) ** (1/years) - 1) * 100
spyew_annual = ((sp500_equal['close'].iloc[-1] / sp500_equal['close'].iloc[0]) ** (1/years) - 1) * 100

print(f"\n{'Strategy':<35} {'Annual Return':<15} {'vs Benchmark'}")
print("-"*80)
print(f"{'S&P 500 (Market-Cap Weighted)':<35} {spy_annual:>6.1f}%        (baseline)")
print(f"{'S&P 500 (Equal-Weighted)':<35} {spyew_annual:>6.1f}%        {spyew_annual-spy_annual:>+6.1f}%")
print(f"{'Our Strategy (Monthly Rotation)':<35} {our_annual:>6.1f}%        {our_annual-spy_annual:>+6.1f}%")

if our_annual > spy_annual:
    print(f"\n✅ We BEAT the market by {our_annual-spy_annual:.1f}%!")
else:
    print(f"\n❌ We UNDERPERFORM by {spy_annual-our_annual:.1f}%")

print("\n" + "="*80)
print("✅ COMPLETED")
print("="*80)
print(f"\nCreated files:")
print(f"  - {data_dir}/SPY.csv (Market-cap weighted)")
print(f"  - {data_dir}/SPYEW.csv (Equal-weighted)")
print(f"\nUse for:")
print(f"  1. Benchmark comparison (beat the market?)")
print(f"  2. Beta calculation (stock correlation with market)")
print(f"  3. Market timing (detect bear markets)")
print("="*80)
