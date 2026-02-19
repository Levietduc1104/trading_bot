"""
Download Historical Fundamental Data from FMP
Fetches quarterly/annual fundamental ratios: P/E, EPS, ROE, ROA, etc.
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

def fetch_financial_ratios(symbol, period='quarter', limit=80):
    """
    Fetch financial ratios for a symbol
    period: 'quarter' or 'annual'
    """
    url = f"{FMP_BASE_URL}/ratios/{symbol}"
    params = {
        'period': period,
        'limit': limit,
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
                return []
        else:
            print(f"  ❌ Error {response.status_code} for {symbol}")
            return []

    except Exception as e:
        print(f"  ❌ Exception for {symbol}: {e}")
        return []


def fetch_key_metrics(symbol, period='quarter', limit=80):
    """
    Fetch key metrics for a symbol
    period: 'quarter' or 'annual'
    """
    url = f"{FMP_BASE_URL}/key-metrics/{symbol}"
    params = {
        'period': period,
        'limit': limit,
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
                return []
        else:
            return []

    except Exception as e:
        print(f"  ❌ Exception for {symbol}: {e}")
        return []


def fetch_income_statement(symbol, period='quarter', limit=80):
    """
    Fetch income statement for a symbol
    period: 'quarter' or 'annual'
    """
    url = f"{FMP_BASE_URL}/income-statement/{symbol}"
    params = {
        'period': period,
        'limit': limit,
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
                return []
        else:
            return []

    except Exception as e:
        print(f"  ❌ Exception for {symbol}: {e}")
        return []


def fetch_comprehensive_fundamentals(symbol, period='quarter', limit=80):
    """
    Fetch comprehensive fundamental data for a symbol
    Combines ratios, key metrics, and income statement
    """
    print(f"  Fetching ratios...", end=' ')
    ratios = fetch_financial_ratios(symbol, period, limit)

    print(f"metrics...", end=' ')
    metrics = fetch_key_metrics(symbol, period, limit)

    print(f"income...", end=' ')
    income = fetch_income_statement(symbol, period, limit)

    # Merge by date
    if ratios and metrics:
        ratios_df = pd.DataFrame(ratios)
        metrics_df = pd.DataFrame(metrics)

        # Merge on date
        merged = pd.merge(ratios_df, metrics_df, on=['symbol', 'date'], how='outer', suffixes=('', '_metrics'))

        # Add income statement if available
        if income:
            income_df = pd.DataFrame(income)
            # Select key income statement fields
            income_fields = ['symbol', 'date', 'revenue', 'netIncome', 'eps', 'epsdiluted']
            income_subset = income_df[[f for f in income_fields if f in income_df.columns]]
            merged = pd.merge(merged, income_subset, on=['symbol', 'date'], how='left')

        return merged.to_dict('records')

    elif ratios:
        return ratios
    elif metrics:
        return metrics
    else:
        return []


def fetch_multiple_stocks(symbols, period='quarter', limit=80, delay=1.0, data_type='ratios'):
    """Fetch fundamental data for multiple stocks with rate limiting"""
    all_data = []
    total = len(symbols)

    print(f"\n{'='*80}")
    print(f"Downloading {data_type} for {total} stocks ({period})")
    print(f"Rate limit delay: {delay} seconds between requests")
    print(f"{'='*80}\n")

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] {symbol}...", end=' ')

        if data_type == 'ratios':
            data = fetch_financial_ratios(symbol, period, limit)
        elif data_type == 'metrics':
            data = fetch_key_metrics(symbol, period, limit)
        elif data_type == 'income':
            data = fetch_income_statement(symbol, period, limit)
        elif data_type == 'comprehensive':
            data = fetch_comprehensive_fundamentals(symbol, period, limit)
        else:
            data = fetch_financial_ratios(symbol, period, limit)

        if data:
            all_data.extend(data)
            if isinstance(data, list) and len(data) > 0 and 'date' in data[0]:
                dates = [item['date'] for item in data if 'date' in item]
                if dates:
                    print(f"✅ {len(data)} periods ({min(dates)} to {max(dates)})")
                else:
                    print(f"✅ {len(data)} periods")
            else:
                print(f"✅ {len(data)} periods")
        else:
            print(f"❌ No data")

        # Rate limiting
        if i < total:
            time.sleep(delay)

    return all_data


def save_to_csv(data, filename):
    """Save fundamental data to CSV"""
    if not data:
        print(f"\n❌ No data to save")
        return

    df = pd.DataFrame(data)

    # Sort by symbol and date
    if 'symbol' in df.columns and 'date' in df.columns:
        df = df.sort_values(['symbol', 'date'])

    df.to_csv(filename, index=False)

    print(f"\n{'='*80}")
    print(f"✅ Saved {len(df)} records to: {filename}")
    print(f"{'='*80}")
    print(f"\nData summary:")
    print(f"  Symbols: {df['symbol'].nunique()}")
    if 'date' in df.columns:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Show key columns if they exist
    key_cols = ['date', 'symbol', 'peRatio', 'eps', 'roeTTM', 'roaTTM', 'debtToEquity', 'currentRatio']
    available_cols = [c for c in key_cols if c in df.columns]

    if available_cols:
        print(f"\n📄 Sample data (key columns):")
        print(df[available_cols].head(10).to_string(index=False))


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
    parser = argparse.ArgumentParser(description='Download historical fundamentals from FMP')
    parser.add_argument('--symbol', type=str, help='Single stock symbol (e.g., AAPL)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--file', type=str, help='File with symbols (txt or json)')
    parser.add_argument('--period', type=str, default='quarter', choices=['quarter', 'annual'], help='Quarterly or annual data')
    parser.add_argument('--limit', type=int, default=80, help='Number of periods to fetch per stock')
    parser.add_argument('--type', type=str, default='comprehensive', choices=['ratios', 'metrics', 'income', 'comprehensive'], help='Data type to fetch')
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
        print("  python3 fetch_fmp_fundamentals.py --symbol AAPL")
        print("  python3 fetch_fmp_fundamentals.py --symbols AAPL,MSFT,GOOGL --period quarter")
        print("  python3 fetch_fmp_fundamentals.py --file sp500_symbols.txt --type comprehensive")
        return

    # Set output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"fmp_fundamentals_{args.type}_{args.period}_{timestamp}.csv"

    # Fetch data
    print(f"\n🚀 FMP Fundamentals Downloader")
    print(f"API Key: {FMP_API_KEY[:20]}...")
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {args.period}")
    print(f"Data type: {args.type}")
    print(f"Limit: {args.limit} periods per stock")

    if len(symbols) == 1:
        if args.type == 'comprehensive':
            data = fetch_comprehensive_fundamentals(symbols[0], args.period, args.limit)
        elif args.type == 'ratios':
            data = fetch_financial_ratios(symbols[0], args.period, args.limit)
        elif args.type == 'metrics':
            data = fetch_key_metrics(symbols[0], args.period, args.limit)
        elif args.type == 'income':
            data = fetch_income_statement(symbols[0], args.period, args.limit)
        all_data = data
    else:
        all_data = fetch_multiple_stocks(symbols, args.period, args.limit, args.delay, args.type)

    # Save to CSV
    if all_data:
        save_to_csv(all_data, output_file)
        print(f"\n✅ SUCCESS! Data saved to: {output_file}")
    else:
        print(f"\n❌ No data downloaded. Check your network connection.")
        print("   Make sure you're running from HOME network (not work/school proxy)")


if __name__ == '__main__':
    main()
