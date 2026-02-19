"""
IEX Cloud Historical Market Cap Fetcher
Downloads daily market capitalization data from 1990-2024
"""
import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import argparse

# ===== IEX CLOUD API CONFIGURATION =====
IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'  # Get from: https://iexcloud.io/console/
IEX_BASE_URL = 'https://cloud.iexapis.com/stable'

def fetch_market_cap_timeseries(symbol, years=5):
    """
    Fetch historical market cap using IEX time series endpoint
    Returns daily market cap for the specified number of years
    """
    url = f"{IEX_BASE_URL}/time-series/HISTORICAL_MARKET_CAP/{symbol}"

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    params = {
        'token': IEX_API_KEY,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                # Add symbol to each record
                for item in data:
                    item['symbol'] = symbol
                return data
            else:
                return []
        else:
            print(f"  ❌ Error {response.status_code}: {response.text[:200]}")
            return []

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return []


def fetch_market_cap_stats(symbol):
    """
    Fetch current market cap from stock stats endpoint (as fallback)
    """
    url = f"{IEX_BASE_URL}/stock/{symbol}/stats"
    params = {'token': IEX_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'marketcap' in data:
                return {
                    'symbol': symbol,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'marketCap': data['marketcap']
                }
        return None

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None


def fetch_multiple_stocks(symbols, years=5, delay=0.5):
    """Fetch market cap for multiple stocks with rate limiting"""
    all_data = []
    total = len(symbols)

    print(f"\n{'='*80}")
    print(f"Downloading market cap for {total} stocks ({years} years)")
    print(f"IEX Cloud - Rate limit delay: {delay} seconds")
    print(f"{'='*80}\n")

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Fetching {symbol}...", end=' ', flush=True)

        data = fetch_market_cap_timeseries(symbol, years)

        if data:
            all_data.extend(data)
            dates = [item.get('date') for item in data if 'date' in item]
            if dates:
                print(f"✅ {len(data)} records ({min(dates)} to {max(dates)})")
            else:
                print(f"✅ {len(data)} records")
        else:
            print(f"❌ No data")

        # Rate limiting (IEX Cloud: 100K calls/month Starter plan)
        if i < total:
            time.sleep(delay)

    return all_data


def save_to_csv(data, filename):
    """Save market cap data to CSV"""
    if not data:
        print(f"\n❌ No data to save")
        return

    df = pd.DataFrame(data)

    # Standardize column names
    column_mapping = {
        'marketCap': 'market_cap',
        'marketcap': 'market_cap',
        'value': 'market_cap'
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Ensure required columns exist
    if 'date' in df.columns and 'symbol' in df.columns and 'market_cap' in df.columns:
        df = df[['date', 'symbol', 'market_cap']]
        df = df.sort_values(['symbol', 'date'])

    # Convert market cap to billions
    if 'market_cap' in df.columns:
        df['market_cap_billions'] = df['market_cap'] / 1e9

    df.to_csv(filename, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Saved {len(df)} records to: {filename}")
    print(f"{'='*80}")
    print(f"\nData summary:")
    if 'symbol' in df.columns:
        print(f"  Symbols: {df['symbol'].nunique()}")
    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total records: {len(df)}")

    # Show sample
    print(f"\n📄 Sample data:")
    print(df.head(10).to_string(index=False))


def read_symbols_from_file(filepath):
    """Read stock symbols from file (one per line or JSON)"""
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(item) if isinstance(item, str) else item.get('symbol', item.get('ticker')) for item in data]
                elif isinstance(data, dict):
                    for key in ['symbols', 'tickers', 'stocks']:
                        if key in data:
                            return data[key]
        else:
            with open(filepath, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
                return symbols
    except Exception as e:
        print(f"❌ Error reading file {filepath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Download historical market cap from IEX Cloud')
    parser.add_argument('--symbol', type=str, help='Single stock symbol (e.g., AAPL)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--file', type=str, help='File with symbols (txt or json)')
    parser.add_argument('--years', type=int, default=34, help='Number of years of history (max 34 for 1990-2024)')
    parser.add_argument('--output', type=str, help='Output CSV filename')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests (seconds)')

    args = parser.parse_args()

    if IEX_API_KEY == 'YOUR_IEX_API_KEY_HERE':
        print("\n❌ ERROR: IEX API key not set!")
        print("\nSteps to get started:")
        print("1. Sign up at: https://iexcloud.io/")
        print("2. Choose Starter plan ($9/month)")
        print("3. Copy your API token from the console")
        print("4. Edit this file and replace 'YOUR_IEX_API_KEY_HERE' with your token")
        print("\nExample: IEX_API_KEY = 'pk_abc123...'")
        return

    # Determine symbols to fetch
    symbols = []
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.file:
        symbols = read_symbols_from_file(args.file)
        if not symbols:
            print(f"❌ No symbols found in {args.file}")
            return
    else:
        print("❌ Error: Must provide --symbol, --symbols, or --file")
        print("\nExamples:")
        print("  python3 fetch_iex_market_cap.py --symbol AAPL --years 10")
        print("  python3 fetch_iex_market_cap.py --symbols AAPL,MSFT,GOOGL --years 20")
        print("  python3 fetch_iex_market_cap.py --file sp500_symbols.txt --years 34")
        return

    # Set output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"iex_market_cap_{timestamp}.csv"

    # Fetch data
    print(f"\n🚀 IEX Cloud Market Cap Downloader")
    print(f"API Key: {IEX_API_KEY[:20]}...")
    print(f"Symbols: {len(symbols)}")
    print(f"Years: {args.years} (back to {datetime.now().year - args.years})")

    if len(symbols) == 1:
        data = fetch_market_cap_timeseries(symbols[0], args.years)
        all_data = data
    else:
        all_data = fetch_multiple_stocks(symbols, args.years, args.delay)

    # Save to CSV
    if all_data:
        save_to_csv(all_data, output_file)
        print(f"\n✅ SUCCESS! Data saved to: {output_file}")
    else:
        print(f"\n❌ No data downloaded.")
        print("   Check:")
        print("   1. API key is valid")
        print("   2. Symbols are correct")
        print("   3. IEX Cloud subscription is active")


if __name__ == '__main__':
    main()
