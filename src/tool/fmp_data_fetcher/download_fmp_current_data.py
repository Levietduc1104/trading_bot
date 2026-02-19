"""
Download CURRENT market cap from FMP using /stable/ endpoint
Then calculate historical market cap using your existing price data
"""
import subprocess
import json
import pandas as pd
import time
import os

FMP_API_KEY = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"
FMP_URL = "https://financialmodelingprep.com/stable/profile"

def get_current_data(symbol):
    """Get current market cap and price for a symbol using curl"""
    url = f"{FMP_URL}?symbol={symbol}&apikey={FMP_API_KEY}"

    try:
        # Use curl to bypass proxy issues
        result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                stock = data[0]
                return {
                    'symbol': symbol,
                    'current_price': stock.get('price'),
                    'current_market_cap': stock.get('mktCap'),
                    'company_name': stock.get('companyName'),
                    'sector': stock.get('sector'),
                    'exchange': stock.get('exchange')
                }
    except Exception as e:
        print(f"    ❌ Error: {str(e)[:40]}")

    return None

def main():
    print("="*80)
    print("FMP Current Market Cap Downloader (Premium API)")
    print("="*80)

    # Get stock list
    data_dir = '../../../sp500_data/stock_data_1990_2024'

    if os.path.exists(data_dir):
        all_stocks = sorted([f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')])
        print(f"✅ Found {len(all_stocks)} stocks in data directory")
    else:
        print("❌ Could not find stock data directory")
        return

    # Check what we already have
    existing_symbols = set()
    if os.path.exists('current_market_cap_progress.csv'):
        existing_df = pd.read_csv('current_market_cap_progress.csv')
        existing_symbols = set(existing_df['symbol'].unique())
        print(f"✅ Already downloaded: {len(existing_symbols)} stocks")

    # Find missing stocks
    stocks = sorted([s for s in all_stocks if s not in existing_symbols])

    if not stocks:
        print("\n🎉 All stocks already downloaded!")
        print(f"   Total: {len(existing_symbols)} stocks")
        return

    print(f"⚠️  Missing: {len(stocks)} stocks")
    print(f"\nDownloading {len(stocks)} missing stocks...")
    print("Using curl to bypass proxy (Premium API - no rate limits)")
    print("="*80)

    results = []
    success = 0
    failed = 0

    for i, symbol in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {symbol}...", end=' ', flush=True)

        data = get_current_data(symbol)

        if data and data['current_market_cap']:
            results.append(data)
            success += 1
            print(f"✅ ${data['current_market_cap']/1e9:.1f}B")
        else:
            failed += 1
            print("❌")

        # Rate limiting - Premium allows more but still be respectful
        time.sleep(0.2)

        # Save progress every 50 stocks
        if len(results) > 0 and len(results) % 50 == 0:
            # Append to existing data
            if os.path.exists('current_market_cap_progress.csv'):
                existing_df = pd.read_csv('current_market_cap_progress.csv')
                temp_df = pd.DataFrame(results)
                combined = pd.concat([existing_df, temp_df], ignore_index=True)
                combined.to_csv('current_market_cap_progress.csv', index=False)
            else:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv('current_market_cap_progress.csv', index=False)
            print(f"    💾 Progress saved ({success} new stocks)")

    # Save final results
    if results:
        new_df = pd.DataFrame(results)

        # Merge with existing data
        if os.path.exists('current_market_cap_progress.csv'):
            existing_df = pd.read_csv('current_market_cap_progress.csv')
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=['symbol'], keep='last')
        else:
            final_df = new_df

        final_df.to_csv('current_market_cap_progress.csv', index=False)

        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*80)
        print(f"Successfully downloaded: {success} stocks")
        print(f"Failed: {failed} stocks")
        print(f"Total in database: {len(final_df)} stocks")
        print(f"File: current_market_cap_progress.csv")

        # Check for Mag 7
        mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        have_mag7 = [s for s in mag7 if s in final_df['symbol'].values]
        missing_mag7 = [s for s in mag7 if s not in final_df['symbol'].values]

        print("\n" + "="*80)
        print("MAG 7 STATUS")
        print("="*80)
        print(f"✅ Have: {have_mag7}")
        if missing_mag7:
            print(f"❌ Missing: {missing_mag7}")
        else:
            print(f"🎉 ALL MAG 7 STOCKS AVAILABLE!")

        # Show top 10
        print("\n" + "="*80)
        print("Top 10 by market cap:")
        print("="*80)
        top10 = final_df.nlargest(10, 'current_market_cap')[['symbol', 'company_name', 'current_market_cap']]
        for idx, row in top10.iterrows():
            mc_val = row['current_market_cap'] / 1e9 if row['current_market_cap'] else 0
            print(f"  {row['symbol']:6s}  ${mc_val:8.1f}B  {row['company_name'][:40]}")

        print("\n📊 Next step: Run calculate_historical_market_cap_200.py to process all stocks")
    else:
        print("\n❌ No data downloaded")

if __name__ == '__main__':
    main()
