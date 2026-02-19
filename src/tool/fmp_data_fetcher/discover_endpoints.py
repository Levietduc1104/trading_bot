"""
Discover FMP API Endpoints - Find which endpoints work
Tests various endpoint patterns to find the current working API
"""
import requests
import json

FMP_API_KEY = 'yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w'
symbol = 'AAPL'

# Test different endpoint patterns
endpoints_to_test = [
    # V4 endpoints (possibly new)
    ('v4/profile', f'https://financialmodelingprep.com/api/v4/profile/{symbol}?apikey={FMP_API_KEY}'),
    ('v4/historical-market-cap', f'https://financialmodelingprep.com/api/v4/historical-market-capitalization/{symbol}?apikey={FMP_API_KEY}'),
    ('v4/market-cap', f'https://financialmodelingprep.com/api/v4/market-cap/{symbol}?apikey={FMP_API_KEY}'),

    # Alternative v3 paths
    ('v3/profile-all', f'https://financialmodelingprep.com/api/v3/profile-all/{symbol}?apikey={FMP_API_KEY}'),
    ('v3/company/profile', f'https://financialmodelingprep.com/api/v3/company/profile/{symbol}?apikey={FMP_API_KEY}'),
    ('v3/quote', f'https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}'),

    # Financial statements - alternative paths
    ('v3/financials/income-statement', f'https://financialmodelingprep.com/api/v3/financials/income-statement/{symbol}?apikey={FMP_API_KEY}'),
    ('v4/income-statement', f'https://financialmodelingprep.com/api/v4/income-statement/{symbol}?apikey={FMP_API_KEY}'),

    # Ratios - alternative paths
    ('v3/financial-ratios', f'https://financialmodelingprep.com/api/v3/financial-ratios/{symbol}?apikey={FMP_API_KEY}'),
    ('v4/ratios', f'https://financialmodelingprep.com/api/v4/ratios/{symbol}?apikey={FMP_API_KEY}'),

    # Key metrics - alternative paths
    ('v3/metrics', f'https://financialmodelingprep.com/api/v3/metrics/{symbol}?apikey={FMP_API_KEY}'),
    ('v4/key-metrics', f'https://financialmodelingprep.com/api/v4/key-metrics/{symbol}?apikey={FMP_API_KEY}'),

    # Market cap - alternative paths
    ('v3/market-capitalization', f'https://financialmodelingprep.com/api/v3/market-capitalization/{symbol}?apikey={FMP_API_KEY}'),
    ('v4/market-capitalization', f'https://financialmodelingprep.com/api/v4/market-capitalization/{symbol}?apikey={FMP_API_KEY}'),

    # Historical price (this usually works)
    ('v3/historical-price-full', f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={FMP_API_KEY}'),
]

print("="*80)
print("FMP API Endpoint Discovery")
print("="*80)
print(f"Testing {len(endpoints_to_test)} different endpoint patterns...")
print()

working_endpoints = []

for name, url in endpoints_to_test:
    print(f"Testing: {name:40s} ", end='')

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, dict) and 'Error Message' in data:
                    print(f"❌ {data['Error Message'][:50]}...")
                elif isinstance(data, list) and len(data) > 0:
                    print(f"✅ WORKS! ({len(data)} records)")
                    working_endpoints.append((name, url))
                elif isinstance(data, dict) and len(data) > 0:
                    print(f"✅ WORKS! (dict with {len(data)} keys)")
                    working_endpoints.append((name, url))
                else:
                    print(f"⚠️  Empty response")
            except:
                print(f"⚠️  Non-JSON response")
        elif response.status_code == 403:
            error_msg = response.json().get('Error Message', 'Forbidden')[:50]
            print(f"❌ 403 - {error_msg}...")
        elif response.status_code == 401:
            print(f"❌ 401 - Unauthorized (bad API key?)")
        elif response.status_code == 404:
            print(f"❌ 404 - Not found")
        else:
            print(f"❌ {response.status_code}")

    except Exception as e:
        print(f"❌ Exception: {str(e)[:30]}...")

print()
print("="*80)
print("WORKING ENDPOINTS:")
print("="*80)

if working_endpoints:
    for name, url in working_endpoints:
        print(f"✅ {name}")
        print(f"   {url}")
        print()
else:
    print("❌ No working endpoints found!")
    print()
    print("Possible reasons:")
    print("1. API key is invalid or expired")
    print("2. API version has changed significantly")
    print("3. Need to upgrade to paid plan")
    print("4. FMP has completely restructured their API")
    print()
    print("Next steps:")
    print("1. Check your FMP account at: https://financialmodelingprep.com/developer/docs")
    print("2. Verify your API key is active")
    print("3. Check the documentation for new endpoint structure")
    print("4. Consider alternative data sources (IEX Cloud, Alpha Vantage, etc.)")

print()
print("="*80)
