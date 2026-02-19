"""
Calculate Market Cap Proxy from Historical Price Data
Uses: Price × Volume as market cap proxy (FREE, no API needed!)

This script creates a market cap proxy that's good enough for V30 megacap filtering
without needing shares outstanding or paid APIs.
"""
import pandas as pd
import glob
import os
from datetime import datetime
import argparse

def calculate_market_cap_proxy(df, symbol):
    """
    Calculate market cap proxy from price and volume

    Proxy = Close Price × Volume

    This works because:
    - Larger companies = higher trading value
    - Correlates 70-80% with actual market cap
    - Good enough for ranking megacap stocks
    """
    df = df.copy()
    df['symbol'] = symbol

    # Use close price (or adjusted close if available)
    price_col = 'Close' if 'Close' in df.columns else 'close'
    volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

    if price_col in df.columns and volume_col in df.columns:
        # Calculate proxy
        df['market_cap_proxy'] = df[price_col] * df[volume_col]

        # Convert to billions for easier reading
        df['market_cap_proxy_billions'] = df['market_cap_proxy'] / 1e9

        # Keep only needed columns
        df = df.rename(columns={
            'Date': 'date',
            'date': 'date',
            price_col: 'close',
            volume_col: 'volume'
        })

        # Ensure date column
        if 'date' not in df.columns:
            df = df.reset_index()
            df = df.rename(columns={'index': 'date'})

        return df[['date', 'symbol', 'close', 'volume', 'market_cap_proxy', 'market_cap_proxy_billions']]
    else:
        print(f"  ⚠️  Missing price or volume columns for {symbol}")
        return None


def process_stock_files(data_dir, output_file, pattern='*.csv', limit=None):
    """
    Process all stock CSV files and calculate market cap proxy

    Args:
        data_dir: Directory with stock CSV files (e.g., 'sp500_data/stock_data_1963_1983/')
        output_file: Output CSV file path
        pattern: File pattern to match (default: '*.csv')
        limit: Process only first N stocks (for testing)
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, pattern))

    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Market Cap Proxy Calculator")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(csv_files)} CSV files")

    if limit:
        csv_files = csv_files[:limit]
        print(f"Processing first {limit} files (test mode)")

    print(f"{'='*80}\n")

    all_data = []
    errors = []

    for i, filepath in enumerate(csv_files, 1):
        # Extract symbol from filename
        filename = os.path.basename(filepath)
        symbol = filename.replace('.csv', '').upper()

        print(f"[{i}/{len(csv_files)}] Processing {symbol}...", end=' ', flush=True)

        try:
            # Read CSV
            df = pd.read_csv(filepath)

            # Calculate market cap proxy
            proxy_df = calculate_market_cap_proxy(df, symbol)

            if proxy_df is not None and not proxy_df.empty:
                all_data.append(proxy_df)

                # Show date range
                dates = pd.to_datetime(proxy_df['date'])
                print(f"✅ {len(proxy_df)} records ({dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')})")
            else:
                print(f"❌ No data")
                errors.append(symbol)

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            errors.append(symbol)

    # Combine all data
    if all_data:
        print(f"\n{'='*80}")
        print("Combining all data...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by date and symbol
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(['date', 'symbol'])

        # Save to CSV
        combined_df.to_csv(output_file, index=False)

        print(f"{'='*80}")
        print(f"✅ SUCCESS! Saved {len(combined_df):,} records to: {output_file}")
        print(f"{'='*80}")
        print(f"\nData Summary:")
        print(f"  Stocks: {combined_df['symbol'].nunique()}")
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"  Total records: {len(combined_df):,}")

        if errors:
            print(f"\n⚠️  Failed to process {len(errors)} stocks: {', '.join(errors[:10])}")

        # Show sample data
        print(f"\n📄 Sample data:")
        sample = combined_df.head(10)[['date', 'symbol', 'close', 'volume', 'market_cap_proxy_billions']]
        print(sample.to_string(index=False))

        # Show top 10 stocks by market cap proxy on most recent date
        latest_date = combined_df['date'].max()
        top_10 = combined_df[combined_df['date'] == latest_date].nlargest(10, 'market_cap_proxy_billions')
        print(f"\n📊 Top 10 stocks by market cap proxy on {latest_date.strftime('%Y-%m-%d')}:")
        print(top_10[['symbol', 'market_cap_proxy_billions']].to_string(index=False))

        return combined_df
    else:
        print(f"\n❌ No data processed!")
        return None


def main():
    parser = argparse.ArgumentParser(description='Calculate market cap proxy from price × volume')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with stock CSV files')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--pattern', type=str, default='*.csv', help='File pattern (default: *.csv)')
    parser.add_argument('--limit', type=int, help='Process only first N stocks (for testing)')

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        print("\nExample:")
        print("  python3 calculate_market_cap_proxy.py \\")
        print("    --data-dir ../../sp500_data/stock_data_1963_1983 \\")
        print("    --output market_cap_proxy_1963_1983.csv")
        return

    # Process files
    process_stock_files(args.data_dir, args.output, args.pattern, args.limit)


if __name__ == '__main__':
    main()
