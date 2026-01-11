"""
Create separate stock datasets for different 20-year periods.

Each dataset will only contain stocks with actual price data during that period.

Periods:
- stock_data_1963_1983 (20 years)
- stock_data_1983_2003 (20 years)
- stock_data_1990_2024 (34 years - current production dataset)
"""

import os
import pandas as pd
import shutil
from datetime import datetime
import glob

# Define periods (start_date, end_date, folder_name)
PERIODS = [
    ('1963-01-01', '1983-12-31', 'stock_data_1963_1983'),
    ('1983-01-01', '2003-12-31', 'stock_data_1983_2003'),
    ('1990-01-01', '2024-12-31', 'stock_data_1990_2024'),
]

# Source directory with clean S&P 500 stocks
SOURCE_DIR = 'sp500_data/sp500_filtered'
OUTPUT_BASE = 'sp500_data'

def filter_stock_by_date(df, start_date, end_date):
    """Filter dataframe to date range and check if it has valid data"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Filter to date range
    filtered = df[(df.index >= start) & (df.index <= end)]

    # Check if we have valid data (at least 100 trading days with valid prices)
    if len(filtered) < 100:
        return None

    # Check for valid prices (not all zeros, not all NaN)
    if filtered['close'].isna().all() or (filtered['close'] == 0).all():
        return None

    return filtered

def create_period_dataset(start_date, end_date, folder_name):
    """Create a dataset for a specific period"""

    print(f"\n{'='*80}")
    print(f"Creating dataset: {folder_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Get all stock files
    stock_files = glob.glob(f"{SOURCE_DIR}/*.csv")
    stock_files = [f for f in stock_files if not f.endswith('VIX.csv')]

    valid_stocks = []
    invalid_stocks = []

    for stock_file in stock_files:
        ticker = os.path.basename(stock_file).replace('.csv', '')

        try:
            # Read stock data
            df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
            df.columns = [col.lower() for col in df.columns]

            # Filter to period
            filtered_df = filter_stock_by_date(df, start_date, end_date)

            if filtered_df is not None and len(filtered_df) > 0:
                # Save filtered data
                output_file = os.path.join(output_dir, f"{ticker}.csv")
                filtered_df.to_csv(output_file)
                valid_stocks.append(ticker)

                # Print sample info
                if len(valid_stocks) <= 5:
                    print(f"  ✅ {ticker}: {len(filtered_df)} days "
                          f"({filtered_df.index[0].date()} to {filtered_df.index[-1].date()})")
            else:
                invalid_stocks.append(ticker)

        except Exception as e:
            print(f"  ❌ {ticker}: Error - {e}")
            invalid_stocks.append(ticker)

    # Copy VIX data (filter to period)
    vix_source = os.path.join(SOURCE_DIR, 'VIX.csv')
    if os.path.exists(vix_source):
        try:
            vix_df = pd.read_csv(vix_source, index_col=0, parse_dates=True)
            vix_df.columns = [col.lower() for col in vix_df.columns]

            # Filter VIX to period
            filtered_vix = filter_stock_by_date(vix_df, start_date, end_date)

            if filtered_vix is not None:
                vix_dest = os.path.join(output_dir, 'VIX.csv')
                filtered_vix.to_csv(vix_dest)
                print(f"\n  ✅ VIX: {len(filtered_vix)} days")
        except Exception as e:
            print(f"  ⚠️  VIX: Error - {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"DATASET SUMMARY: {folder_name}")
    print(f"{'='*80}")
    print(f"Valid stocks:   {len(valid_stocks)}")
    print(f"Invalid stocks: {len(invalid_stocks)}")
    print(f"Coverage:       {len(valid_stocks)/len(stock_files)*100:.1f}%")
    print(f"Output:         {output_dir}/")

    # Save stock list
    list_file = os.path.join(output_dir, 'stock_list.txt')
    with open(list_file, 'w') as f:
        f.write(f"Dataset: {folder_name}\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Total stocks: {len(valid_stocks)}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Valid stocks:\n")
        for ticker in sorted(valid_stocks):
            f.write(f"  {ticker}\n")

    print(f"Stock list:     {list_file}")
    print(f"{'='*80}\n")

    return len(valid_stocks)

def main():
    print("="*80)
    print("CREATING PERIOD-SPECIFIC STOCK DATASETS")
    print("="*80)
    print(f"Source: {SOURCE_DIR}/")
    print(f"Periods: {len(PERIODS)}")

    results = {}

    for start_date, end_date, folder_name in PERIODS:
        count = create_period_dataset(start_date, end_date, folder_name)
        results[folder_name] = count

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for folder_name, count in results.items():
        print(f"{folder_name:30s} {count:>4d} stocks")
    print("="*80)
    print("\n✅ All datasets created successfully!")
    print("\nYou can now run backtests on different periods:")
    for start_date, end_date, folder_name in PERIODS:
        print(f"  - {folder_name}: {start_date} to {end_date}")

if __name__ == '__main__':
    main()
