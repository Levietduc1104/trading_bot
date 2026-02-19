"""
Download missing stocks - bypassing proxy
"""
import subprocess
import json
import pandas as pd
import time
import os

FMP_API_KEY = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"

def get_current_data(symbol):
    """Get current market cap using curl with no proxy"""
    url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_API_KEY}"

    try:
        # Bypass proxy by setting environment
        env = os.environ.copy()
        env['http_proxy'] = ''
        env['https_proxy'] = ''
        env['HTTP_PROXY'] = ''
        env['HTTPS_PROXY'] = ''

        result = subprocess.run(['curl', '-s', '--noproxy', '*', url],
                               capture_output=True, text=True, timeout=10, env=env)

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
        pass

    return None

def main():
    print("="*80)
    print("DOWNLOAD MISSING STOCKS (Bypass Proxy)")
    print("="*80)

    # Load missing stocks
    with open('missing_stocks.txt', 'r') as f:
        missing = [line.strip() for line in f if line.strip()]

    print(f"Total missing: {len(missing)} stocks")
    print(f"Your API limit: 750 calls/day (only used 6)")
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

        # Rate limiting
        time.sleep(0.2)

        # Save progress every 50 stocks
        if len(results) > 0 and len(results) % 50 == 0:
            existing = pd.read_csv('current_market_cap_progress.csv')
            temp_df = pd.DataFrame(results)
            combined = pd.concat([existing, temp_df], ignore_index=True)
            combined.to_csv('current_market_cap_progress.csv', index=False)
            print(f"    💾 Progress saved ({success} new)")

    # Final save
    if results:
        existing = pd.read_csv('current_market_cap_progress.csv')
        new_df = pd.DataFrame(results)
        final = pd.concat([existing, new_df], ignore_index=True)
        final = final.drop_duplicates(subset=['symbol'], keep='last')
        final.to_csv('current_market_cap_progress.csv', index=False)

        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*80)
        print(f"Successfully downloaded: {success} stocks")
        print(f"Failed: {len(failed)} stocks")
        print(f"Total in database: {len(final)} stocks")

        # Check Mag 7
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
            print("🎉 ALL MAG 7 AVAILABLE!")

        if failed:
            print(f"\n⚠️  Failed stocks ({len(failed)}): {failed[:30]}")

    print("\n📊 Next: Run calculate_historical_market_cap_200.py")

if __name__ == '__main__':
    main()
