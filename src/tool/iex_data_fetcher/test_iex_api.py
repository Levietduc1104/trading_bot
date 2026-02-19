"""
Test IEX Cloud API Connection and Endpoints
Verifies your API key works and shows what data is available
"""
import requests
import json

# ===== IEX CLOUD API CONFIGURATION =====
IEX_API_KEY = 'YOUR_IEX_API_KEY_HERE'  # Get from: https://iexcloud.io/console/
IEX_BASE_URL = 'https://cloud.iexapis.com/stable'

def test_endpoint(name, url):
    """Test a specific IEX endpoint"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list):
                print(f"✅ SUCCESS - {len(data)} records")
                if len(data) > 0:
                    print(f"\nFirst record:")
                    print(json.dumps(data[0], indent=2))
                    if 'date' in data[0]:
                        dates = [item['date'] for item in data if 'date' in item]
                        if dates:
                            print(f"\n📅 Date range: {min(dates)} to {max(dates)}")

            elif isinstance(data, dict):
                print(f"✅ SUCCESS")
                print(json.dumps(data, indent=2))

            return True

        else:
            print(f"❌ ERROR {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False


def main():
    symbol = 'AAPL'

    print("="*80)
    print("IEX CLOUD API TEST")
    print("="*80)
    print(f"API Key: {IEX_API_KEY[:20] if IEX_API_KEY != 'YOUR_IEX_API_KEY_HERE' else '❌ NOT SET'}...")
    print(f"Test Symbol: {symbol}")
    print("="*80)

    if IEX_API_KEY == 'YOUR_IEX_API_KEY_HERE':
        print("\n❌ ERROR: Please set your IEX API key first!")
        print("\nSteps:")
        print("1. Sign up at: https://iexcloud.io/")
        print("2. Choose Starter plan ($9/month)")
        print("3. Get your API token from console")
        print("4. Edit test_iex_api.py and set IEX_API_KEY")
        return

    working = []
    failed = []

    # Test 1: Quote (current price + market cap)
    test_name = "Quote (current price + market cap)"
    url = f"{IEX_BASE_URL}/stock/{symbol}/quote?token={IEX_API_KEY}"
    if test_endpoint(test_name, url):
        working.append(test_name)
    else:
        failed.append(test_name)

    # Test 2: Stats (company statistics including market cap)
    test_name = "Stats (company statistics)"
    url = f"{IEX_BASE_URL}/stock/{symbol}/stats?token={IEX_API_KEY}"
    if test_endpoint(test_name, url):
        working.append(test_name)
    else:
        failed.append(test_name)

    # Test 3: Historical prices (to check data availability)
    test_name = "Historical Prices (5 years)"
    url = f"{IEX_BASE_URL}/stock/{symbol}/chart/5y?token={IEX_API_KEY}"
    if test_endpoint(test_name, url):
        working.append(test_name)
    else:
        failed.append(test_name)

    # Test 4: Company info
    test_name = "Company Information"
    url = f"{IEX_BASE_URL}/stock/{symbol}/company?token={IEX_API_KEY}"
    if test_endpoint(test_name, url):
        working.append(test_name)
    else:
        failed.append(test_name)

    # Test 5: Time Series - Historical Market Cap
    test_name = "Time Series - Historical Market Cap"
    url = f"{IEX_BASE_URL}/time-series/HISTORICAL_MARKET_CAP/{symbol}?token={IEX_API_KEY}&last=10"
    if test_endpoint(test_name, url):
        working.append(test_name)
    else:
        failed.append(test_name)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Working endpoints: {len(working)}")
    for name in working:
        print(f"   - {name}")

    if failed:
        print(f"\n❌ Failed endpoints: {len(failed)}")
        for name in failed:
            print(f"   - {name}")

    print("\n" + "="*80)

    if len(working) >= 2:
        print("✅ API KEY WORKS! Ready to download data.")
        print("\nNext steps:")
        print("1. Edit fetch_iex_market_cap.py and add your API key")
        print("2. Run: python3 fetch_iex_market_cap.py --symbol AAPL --years 10")
    else:
        print("❌ API key doesn't work or subscription issue")
        print("\nCheck:")
        print("1. API key is correct")
        print("2. IEX Cloud subscription is active")
        print("3. Network connection is working")

    print("="*80)


if __name__ == '__main__':
    main()
