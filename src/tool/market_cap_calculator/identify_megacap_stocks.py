"""
Identify Top Megacap Stocks at Each Date
Uses market cap proxy to get top N stocks for V30 filtering
"""
import pandas as pd
import argparse
from datetime import datetime

def get_megacap_stocks_at_date(df, date, top_n=20):
    """Get top N stocks by market cap proxy on specific date"""
    day_data = df[df['date'] == date]

    if day_data.empty:
        return None

    # Sort by market cap proxy
    top_stocks = day_data.nlargest(top_n, 'market_cap_proxy_billions')

    return top_stocks[['symbol', 'market_cap_proxy_billions']].reset_index(drop=True)


def create_megacap_filter_file(market_cap_file, output_file, top_n=20, sample_dates=None):
    """
    Create a file showing top megacap stocks at key dates

    Args:
        market_cap_file: CSV file with market cap proxy data
        output_file: Output file showing top stocks at each date
        top_n: Number of top stocks to identify
        sample_dates: List of specific dates to check (or None for quarterly)
    """
    print(f"\n{'='*80}")
    print(f"Megacap Stock Identifier")
    print(f"{'='*80}")
    print(f"Input file: {market_cap_file}")
    print(f"Top N: {top_n}")
    print(f"{'='*80}\n")

    # Load market cap proxy data
    df = pd.read_csv(market_cap_file)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stocks: {df['symbol'].nunique()}")

    # If no sample dates provided, use quarterly dates
    if sample_dates is None:
        # Get quarterly dates (every 63 trading days for V30)
        all_dates = sorted(df['date'].unique())

        # Sample every 63 days (quarterly)
        sample_dates = []
        for i in range(0, len(all_dates), 63):
            sample_dates.append(all_dates[i])

        print(f"\nUsing {len(sample_dates)} quarterly dates")
    else:
        sample_dates = [pd.to_datetime(d) for d in sample_dates]
        print(f"\nUsing {len(sample_dates)} custom dates")

    # Get top stocks at each date
    results = []

    print(f"\nIdentifying top {top_n} stocks at each date...")
    print("="*80)

    for i, date in enumerate(sample_dates, 1):
        top_stocks = get_megacap_stocks_at_date(df, date, top_n)

        if top_stocks is not None:
            # Add date column
            top_stocks['date'] = date

            # Add rank
            top_stocks['rank'] = range(1, len(top_stocks) + 1)

            results.append(top_stocks)

            # Show first few and last few
            if i <= 3 or i > len(sample_dates) - 3:
                print(f"\n{date.strftime('%Y-%m-%d')} - Top {top_n} stocks:")
                print(top_stocks[['rank', 'symbol', 'market_cap_proxy_billions']].head(10).to_string(index=False))
        else:
            print(f"  ⚠️  No data for {date}")

    # Combine results
    if results:
        all_results = pd.concat(results, ignore_index=True)

        # Reorder columns
        all_results = all_results[['date', 'rank', 'symbol', 'market_cap_proxy_billions']]

        # Save to CSV
        all_results.to_csv(output_file, index=False)

        print(f"\n{'='*80}")
        print(f"✅ SUCCESS! Saved megacap stock rankings to: {output_file}")
        print(f"{'='*80}")
        print(f"\nSummary:")
        print(f"  Dates analyzed: {len(sample_dates)}")
        print(f"  Top N per date: {top_n}")
        print(f"  Total records: {len(all_results):,}")

        # Show how often each stock appears in top N
        stock_counts = all_results['symbol'].value_counts().head(20)
        print(f"\n📊 Most frequently in top {top_n} (across all dates):")
        print(stock_counts.to_string())

        return all_results
    else:
        print(f"\n❌ No results generated")
        return None


def main():
    parser = argparse.ArgumentParser(description='Identify top megacap stocks for V30 filtering')
    parser.add_argument('--input', type=str, required=True, help='Market cap proxy CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top stocks (default: 20)')
    parser.add_argument('--dates', type=str, help='Comma-separated dates (YYYY-MM-DD) or leave blank for quarterly')

    args = parser.parse_args()

    # Parse dates if provided
    sample_dates = None
    if args.dates:
        sample_dates = [d.strip() for d in args.dates.split(',')]
        print(f"Using custom dates: {sample_dates}")

    # Create megacap filter file
    create_megacap_filter_file(args.input, args.output, args.top_n, sample_dates)


if __name__ == '__main__':
    main()
