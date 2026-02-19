"""
Test FMP API with New Endpoint Structure
FMP changed from /api/v3/ to /stable/ endpoints
"""
import requests
import json

FMP_API_KEY = 'yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w'
FMP_BASE_URL = 'https://financialmodelingprep.com/stable'
symbol = 'AAPL'

# Test new /stable/ endpoint patterns
endpoints_to_test = [
    # Company information
    ('profile', f'{FMP_BASE_URL}/profile/{symbol}?apikey={FMP_API_KEY}'),
    ('company-profile', f'{FMP_BASE_URL}/company-profile/{symbol}?apikey={FMP_API_KEY}'),
    ('quote', f'{FMP_BASE_URL}/quote/{symbol}?apikey={FMP_API_KEY}'),
    ('search-symbol', f'{FMP_BASE_URL}/search-symbol?query={symbol}&apikey={FMP_API_KEY}'),

    # Financial statements
    ('income-statement', f'{FMP_BASE_URL}/income-statement/{symbol}?apikey={FMP_API_KEY}'),
    ('balance-sheet-statement', f'{FMP_BASE_URL}/balance-sheet-statement/{symbol}?apikey={FMP_API_KEY}'),
    ('cash-flow-statement', f'{FMP_BASE_URL}/cash-flow-statement/{symbol}?apikey={FMP_API_KEY}'),

    # Financial ratios and metrics
    ('ratios', f'{FMP_BASE_URL}/ratios/{symbol}?apikey={FMP_API_KEY}'),
    ('key-metrics', f'{FMP_BASE_URL}/key-metrics/{symbol}?apikey={FMP_API_KEY}'),
    ('financial-ratios', f'{FMP_BASE_URL}/financial-ratios/{symbol}?apikey={FMP_API_KEY}'),

    # Market cap
    ('market-capitalization', f'{FMP_BASE_URL}/market-capitalization/{symbol}?apikey={FMP_API_KEY}'),
    ('historical-market-capitalization', f'{FMP_BASE_URL}/historical-market-capitalization/{symbol}?apikey={FMP_API_KEY}'),

    # Historical price
    ('historical-price-full', f'{FMP_BASE_URL}/historical-price-full/{symbol}?apikey={FMP_API_KEY}'),

    # Enterprise value
    ('enterprise-values', f'{FMP_BASE_URL}/enterprise-values/{symbol}?apikey={FMP_API_KEY}'),

    # Growth metrics
    ('financial-growth', f'{FMP_BASE_URL}/financial-growth/{symbol}?apikey={FMP_API_KEY}'),
]

print("="*80)
print("FMP API Testing - New /stable/ Endpoints")
print("="*80)
print(f"Testing {len(endpoints_to_test)} endpoints...")
print()

working_endpoints = []

for name, url in endpoints_to_test:
    print(f"Testing: {name:40s} ", end='', flush=True)

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            try:
                data = response.json()

                if isinstance(data, dict) and 'Error Message' in data:
                    error_msg = data['Error Message'][:60]
                    print(f"❌ {error_msg}...")

                elif isinstance(data, list):
                    if len(data) > 0:
                        print(f"✅ WORKS! ({len(data)} records)")
                        working_endpoints.append((name, url, data))

                        # Show first record structure
                        if 'date' in data[0]:
                            dates = [item.get('date') for item in data if 'date' in item]
                            if dates:
                                print(f"   📅 Date range: {min(dates)} to {max(dates)}")
                    else:
                        print(f"⚠️  Empty list")

                elif isinstance(data, dict):
                    if len(data) > 0:
                        print(f"✅ WORKS! (dict with {len(data)} keys)")
                        working_endpoints.append((name, url, data))
                    else:
                        print(f"⚠️  Empty dict")

            except json.JSONDecodeError:
                print(f"⚠️  Non-JSON response")

        elif response.status_code == 403:
            try:
                error_data = response.json()
                error_msg = error_data.get('Error Message', 'Forbidden')[:60]
                print(f"❌ 403 - {error_msg}...")
            except:
                print(f"❌ 403 - Forbidden")

        elif response.status_code == 401:
            print(f"❌ 401 - Unauthorized")

        elif response.status_code == 404:
            print(f"❌ 404 - Not found")

        else:
            print(f"❌ {response.status_code}")

    except Exception as e:
        print(f"❌ Exception: {str(e)[:40]}...")

print()
print("="*80)
print(f"SUMMARY: {len(working_endpoints)} working endpoints found")
print("="*80)

if working_endpoints:
    for name, url, sample_data in working_endpoints:
        print(f"\n✅ {name}")
        print(f"   URL: {url}")

        if isinstance(sample_data, list) and len(sample_data) > 0:
            print(f"   Records: {len(sample_data)}")
            print(f"   Sample keys: {list(sample_data[0].keys())[:10]}")
        elif isinstance(sample_data, dict):
            print(f"   Keys: {list(sample_data.keys())[:10]}")

    print()
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("✅ Found working endpoints!")
    print("I'll update the fetch scripts to use these new endpoints.")

else:
    print("\n❌ No working endpoints found!")
    print()
    print("This means:")
    print("1. The API structure has changed significantly")
    print("2. Your API key might need activation/upgrade")
    print("3. Need to check FMP documentation for current endpoints")
    print()
    print("Alternative data sources to consider:")
    print("- IEX Cloud ($9/month) - historical market cap 1990-2024")
    print("- Alpha Vantage (FREE) - current fundamentals only")
    print("- Polygon.io ($29/month) - comprehensive data")

print()
