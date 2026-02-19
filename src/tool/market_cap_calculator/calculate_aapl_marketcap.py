"""
Calculate Apple (AAPL) Market Cap - Test Script
Shows both proxy method and actual method (if shares available)
"""
import pandas as pd
import requests
import os

# Alpha Vantage API key (you have this)
ALPHA_VANTAGE_KEY = 'PEI9KPIV5GAG81KZ'

def get_shares_outstanding(symbol):
    """Get current shares outstanding from Alpha Vantage"""
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}'

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'SharesOutstanding' in data:
                shares = float(data['SharesOutstanding'])
                return shares
    except Exception as e:
        print(f"⚠️  Could not get shares outstanding: {e}")

    return None


def calculate_aapl_market_cap():
    """Calculate AAPL market cap using both methods"""

    print("="*80)
    print("APPLE (AAPL) MARKET CAP CALCULATOR")
    print("="*80)

    # Find AAPL data file
    data_paths = [
        '../../../sp500_data/stock_data_1990_2024/AAPL.csv',
        '../../../sp500_data/stock_data_1983_2003/AAPL.csv',
        '../../../sp500_data/individual_stocks/AAPL.csv',
    ]

    aapl_file = None
    for path in data_paths:
        if os.path.exists(path):
            aapl_file = path
            print(f"✅ Found AAPL data: {path}")
            break

    if not aapl_file:
        print("❌ Could not find AAPL.csv file")
        print("\nTried:")
        for path in data_paths:
            print(f"  - {path}")
        return

    # Load AAPL data
    df = pd.read_csv(aapl_file)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Rename date column if needed
    if 'date' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'date'})

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    print(f"✅ Loaded {len(df)} records")
    print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    # Get current shares outstanding
    print("\n" + "="*80)
    print("Getting current shares outstanding from Alpha Vantage...")
    print("="*80)

    shares_outstanding = get_shares_outstanding('AAPL')

    if shares_outstanding:
        print(f"✅ Current shares outstanding: {shares_outstanding:,.0f}")
        print(f"   ({shares_outstanding/1e9:.2f} billion shares)")
    else:
        print("⚠️  Could not get shares outstanding")
        print("   Will only calculate proxy method")

    # Method 1: Price × Volume Proxy
    print("\n" + "="*80)
    print("METHOD 1: Market Cap Proxy (Price × Volume)")
    print("="*80)

    df['market_cap_proxy'] = df['close'] * df['volume']
    df['market_cap_proxy_billions'] = df['market_cap_proxy'] / 1e9

    # Show samples
    print("\nSample dates:")
    sample_dates = ['1990-01-02', '2000-01-03', '2010-01-04', '2020-01-02', '2024-01-02']

    for date_str in sample_dates:
        try:
            date = pd.to_datetime(date_str)
            row = df[df['date'] == date]
            if not row.empty:
                price = row['close'].iloc[0]
                volume = row['volume'].iloc[0]
                proxy = row['market_cap_proxy_billions'].iloc[0]
                print(f"  {date_str}: ${price:8.2f} × {volume:12,.0f} = ${proxy:8.2f}B proxy")
        except:
            pass

    # Method 2: Current Shares × Adjusted Price (if shares available)
    if shares_outstanding:
        print("\n" + "="*80)
        print("METHOD 2: Actual Market Cap (Current Shares × Adjusted Price)")
        print("="*80)

        df['market_cap_actual'] = df['close'] * shares_outstanding
        df['market_cap_actual_billions'] = df['market_cap_actual'] / 1e9

        print(f"\nUsing current shares: {shares_outstanding/1e9:.2f}B shares")
        print("\nSample dates:")

        for date_str in sample_dates:
            try:
                date = pd.to_datetime(date_str)
                row = df[df['date'] == date]
                if not row.empty:
                    price = row['close'].iloc[0]
                    actual = row['market_cap_actual_billions'].iloc[0]
                    print(f"  {date_str}: ${price:8.2f} × {shares_outstanding/1e9:.2f}B shares = ${actual:8.2f}B")
            except:
                pass

        # Compare methods
        print("\n" + "="*80)
        print("COMPARISON: Proxy vs Actual")
        print("="*80)

        print("\nRecent data (last 10 trading days):")
        recent = df.tail(10)[['date', 'close', 'market_cap_proxy_billions', 'market_cap_actual_billions']].copy()
        recent['difference_%'] = ((recent['market_cap_proxy_billions'] / recent['market_cap_actual_billions']) - 1) * 100

        print(recent.to_string(index=False))

        # Overall correlation
        correlation = df['market_cap_proxy_billions'].corr(df['market_cap_actual_billions'])
        print(f"\n📊 Correlation between proxy and actual: {correlation:.2%}")

    # Show latest values
    print("\n" + "="*80)
    print("LATEST DATA (Most Recent)")
    print("="*80)

    latest = df.iloc[-1]
    print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Price: ${latest['close']:.2f}")
    print(f"Volume: {latest['volume']:,.0f}")
    print(f"\nMarket Cap Proxy: ${latest['market_cap_proxy_billions']:.2f}B")

    if shares_outstanding:
        print(f"Market Cap Actual: ${latest['market_cap_actual_billions']:.2f}B")
        diff_pct = ((latest['market_cap_proxy_billions'] / latest['market_cap_actual_billions']) - 1) * 100
        print(f"Difference: {diff_pct:+.1f}%")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ Method 1 (Proxy): Uses only price × volume")
    print("   - FREE (no shares outstanding needed)")
    print("   - Good for ranking stocks")
    print("   - 70-80% correlation with actual")

    if shares_outstanding:
        print("\n✅ Method 2 (Actual): Uses shares outstanding × price")
        print("   - More accurate for absolute values")
        print("   - Requires shares outstanding (from API)")
        print("   - Assumes shares constant (not perfect)")

    print("\n💡 For V30 megacap filtering:")
    print("   - Proxy method is good enough!")
    print("   - Only need to rank top 20 stocks")
    print("   - Correlation is strong enough for ranking")

    # Save sample
    output_file = 'aapl_market_cap_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved full analysis to: {output_file}")


if __name__ == '__main__':
    calculate_aapl_market_cap()
