"""
Download Historical Market Capitalization from FMP
Fetches daily market cap data for stocks from 2000-2024
"""
import requests
import pandas as pd
import time
import json
from datetime import datetime
import argparse

# ===== FMP API CONFIGURATION =====
FMP_API_KEY = 'yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w'
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3'

def fetch_market_cap(symbol, start_date='2000-01-01', end_date='2024-12-31'):
    """Fetch historical market cap for a single symbol"""
    url = f"{FMP_BASE_URL}/historical-market-capitalization/{symbol}"
    params = {
        'from': start_date,
        'to': end_date,
        'apikey': FMP_API_KEY
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
                print(f"  ⚠️  No data returned for {symbol}")
                return []
        else:
            print(f"  ❌ Error {response.status_code} for {symbol}: {response.text[:200]}")
            return []

    except Exception as e:
        print(f"  ❌ Exception for {symbol}: {e}")
        return []


def fetch_multiple_stocks(symbols, start_date='2000-01-01', end_date='2024-12-31', delay=1.0):
    """Fetch market cap for multiple stocks with rate limiting"""
    all_data = []
    total = len(symbols)

    print(f"\n{'='*80}")
    print(f"Downloading market cap for {total} stocks from {start_date} to {end_date}")
    print(f"Rate limit delay: {delay} seconds between requests")
    print(f"{'='*80}\n")

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Fetching {symbol}...", end=' ')

        data = fetch_market_cap(symbol, start_date, end_date)

        if data:
            all_data.extend(data)
            dates = [item['date'] for item in data if 'date' in item]
            if dates:
                print(f"✅ {len(data)} records ({min(dates)} to {max(dates)})")
            else:
                print(f"✅ {len(data)} records")
        else:
            print(f"❌ No data")

        # Rate limiting (FMP free tier: 250 calls/day, ~5 calls/min safe)
        if i < total:
            time.sleep(delay)

    return all_data


def save_to_csv(data, filename):
    """Save market cap data to CSV"""
    if not data:
        print(f"\n❌ No data to save")
        return

    df = pd.DataFrame(data)

    # Reorder columns
    if 'date' in df.columns and 'symbol' in df.columns and 'marketCap' in df.columns:
        df = df[['date', 'symbol', 'marketCap']]
        df = df.sort_values(['symbol', 'date'])

    df.to_csv(filename, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Saved {len(df)} records to: {filename}")
    print(f"{'='*80}")
    print(f"\nData summary:")
    print(f"  Symbols: {df['symbol'].nunique()}")
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
                # Handle different JSON formats
                if isinstance(data, list):
                    if all(isinstance(item, str) for item in data):
                        return data
                    elif all(isinstance(item, dict) for item in data):
                        # Try common fields
                        for field in ['symbol', 'ticker', 'Symbol', 'Ticker']:
                            if field in data[0]:
                                return [item[field] for item in data]
                elif isinstance(data, dict):
                    # Try common keys
                    for key in ['symbols', 'tickers', 'stocks']:
                        if key in data:
                            return data[key]
        else:
            # Plain text file, one symbol per line
            with open(filepath, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
                return symbols
    except Exception as e:
        print(f"❌ Error reading file {filepath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Download historical market cap from FMP')
    parser.add_argument('--symbol', type=str, help='Single stock symbol (e.g., AAPL)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--file', type=str, help='File with symbols (txt or json)')
    parser.add_argument('--start', type=str, default='2000-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output CSV filename')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')

    args = parser.parse_args()

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
        print("  python3 fetch_fmp_market_cap.py --symbol AAPL")
        print("  python3 fetch_fmp_market_cap.py --symbols AAPL,MSFT,GOOGL")
        print("  python3 fetch_fmp_market_cap.py --file sp500_symbols.txt")
        return

    # Set output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"fmp_market_cap_{timestamp}.csv"

    # Fetch data
    print(f"\n🚀 FMP Market Cap Downloader")
    print(f"API Key: {FMP_API_KEY[:20]}...")
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {args.start} to {args.end}")

    if len(symbols) == 1:
        data = fetch_market_cap(symbols[0], args.start, args.end)
        all_data = data
    else:
        all_data = fetch_multiple_stocks(symbols, args.start, args.end, args.delay)

    # Save to CSV
    if all_data:
        save_to_csv(all_data, output_file)
        print(f"\n✅ SUCCESS! Data saved to: {output_file}")
    else:
        print(f"\n❌ No data downloaded. Check your network connection.")
        print("   Make sure you're running from HOME network (not work/school proxy)")


if __name__ == '__main__':
    main()
