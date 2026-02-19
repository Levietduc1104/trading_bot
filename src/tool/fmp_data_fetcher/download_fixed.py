"""
Download missing stocks - FIXED VERSION
"""
import requests
import pandas as pd
import time

FMP_API_KEY = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"

def get_current_data(symbol):
    """Get current market cap - WORKING METHOD"""
    url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_API_KEY}"

    try:
        # Use empty proxy dict - this is the method that works\!
        response = requests.get(url, timeout=10, proxies={})

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                stock = data[0]
                return {
                    'symbol': symbol,
                    'current_price': stock.get('price'),
                    'current_market_cap': stock.get('marketCap'),
                    'company_name': stock.get('companyName'),
                    'sector': stock.get('industry'),
                    'exchange': stock.get('exchange')
                }
    except Exception as e:
        pass

    return None

def main():
    print("="*80)
    print("DOWNLOAD MISSING STOCKS - FIXED VERSION")
    print("="*80)

    # Load missing stocks
    with open('missing_stocks.txt', 'r') as f:
        missing = [line.strip() for line in f if line.strip()]

    print(f"Total missing: {len(missing)} stocks")
    print(f"Your API limit: 750 calls/day")
    print("="*80)

    results = []
    success = 0
    failed = []

    for i, symbol in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {symbol}...", end=' ', flush=True)

        data = get_current_data(symbol)

        if data and data['current_market_cap']:
            results.append(data)
            success += 1
            print(f"✅ ${data['current_market_cap']/1e9:.1f}B")
        else:
            failed.append(symbol)
            print("❌")

        time.sleep(0.2)

        if len(results) > 0 and len(results) % 50 == 0:
            existing = pd.read_csv('current_market_cap_progress.csv')
            temp_df = pd.DataFrame(results)
            combined = pd.concat([existing, temp_df], ignore_index=True)
            combined.to_csv('current_market_cap_progress.csv', index=False)
            print(f"    💾 Progress saved ({success} new)")

    if results:
        existing = pd.read_csv('current_market_cap_progress.csv')
        new_df = pd.DataFrame(results)
        final = pd.concat([existing, new_df], ignore_index=True)
        final = final.drop_duplicates(subset=['symbol'], keep='last')
        final.to_csv('current_market_cap_progress.csv', index=False)

        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE\!")
        print("="*80)
        print(f"Successfully downloaded: {success} stocks")
        print(f"Failed: {len(failed)} stocks")
        print(f"Total in database: {len(final)} stocks")

        mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        have = [s for s in mag7 if s in final['symbol'].values]
        missing_mag7 = [s for s in mag7 if s not in final['symbol'].values]

        print("\n" + "="*80)
        print("MAG 7 STATUS")
        print("="*80)
        print(f"✅ Have: {have}")
        if missing_mag7:
            print(f"❌ Still missing: {missing_mag7}")
        else:
            print("🎉 ALL MAG 7 AVAILABLE\!")

        if failed and len(failed) < 30:
            print(f"\n⚠️  Failed stocks: {failed}")

    print("\n📊 Next: python3 calculate_historical_market_cap_200.py")

if __name__ == '__main__':
    main()
