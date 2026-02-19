"""
Test Financial Modeling Prep (FMP) API - Historical Data Availability
"""
import requests
import json
from datetime import datetime

FMP_API_KEY = 'yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w'
FMP_BASE_URL = 'https://financialmodelingprep.com/api/v3'

def test_endpoint(endpoint_name, url):
    print(f"\n{'='*80}")
    print(f"Testing: {endpoint_name}")
    print(f"{'='*80}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"✅ SUCCESS - {len(data)} records")
                if len(data) > 0:
                    print(f"\nFirst: {json.dumps(data[0], indent=2)}")
                    if len(data) > 1:
                        print(f"\nLast: {json.dumps(data[-1], indent=2)}")
                    if 'date' in data[0]:
                        dates = [item['date'] for item in data if 'date' in item]
                        if dates:
                            print(f"\n📅 {min(dates)} to {max(dates)} ({len(dates)} periods)")
                    elif 'calendarYear' in data[0]:
                        years = [item['calendarYear'] for item in data if 'calendarYear' in item]
                        if years:
                            print(f"\n📅 {min(years)} to {max(years)} ({len(years)} years)")
                return data
            elif isinstance(data, dict):
                if 'historical' in data:
                    hist = data['historical']
                    print(f"✅ SUCCESS - {len(hist)} records")
                    if len(hist) > 0:
                        print(f"\nFirst: {json.dumps(hist[0], indent=2)}")
                        print(f"\nLast: {json.dumps(hist[-1], indent=2)}")
                        dates = [item['date'] for item in hist if 'date' in item]
                        if dates:
                            print(f"\n📅 {min(dates)} to {max(dates)} ({len(dates)} days)")
                else:
                    print(f"✅ SUCCESS\n{json.dumps(data, indent=2)}")
                return data
        else:
            print(f"❌ ERROR {response.status_code}: {response.text[:500]}")
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
    return None

symbol = 'AAPL'
print("="*80)
print("FMP API TEST - AAPL Historical Data Coverage")
print("="*80)

test_endpoint("Company Profile", f"{FMP_BASE_URL}/profile/{symbol}?apikey={FMP_API_KEY}")
test_endpoint("Historical Market Cap (All)", f"{FMP_BASE_URL}/historical-market-capitalization/{symbol}?limit=10000&apikey={FMP_API_KEY}")
test_endpoint("Historical Market Cap (1990-2024)", f"{FMP_BASE_URL}/historical-market-capitalization/{symbol}?from=1990-01-01&to=2024-12-31&apikey={FMP_API_KEY}")
test_endpoint("Income Statement Annual", f"{FMP_BASE_URL}/income-statement/{symbol}?limit=40&apikey={FMP_API_KEY}")
test_endpoint("Income Statement Quarterly", f"{FMP_BASE_URL}/income-statement/{symbol}?period=quarter&limit=80&apikey={FMP_API_KEY}")
test_endpoint("Financial Ratios Annual", f"{FMP_BASE_URL}/ratios/{symbol}?limit=40&apikey={FMP_API_KEY}")
test_endpoint("Financial Ratios Quarterly", f"{FMP_BASE_URL}/ratios/{symbol}?period=quarter&limit=80&apikey={FMP_API_KEY}")
test_endpoint("Key Metrics Annual", f"{FMP_BASE_URL}/key-metrics/{symbol}?limit=40&apikey={FMP_API_KEY}")
test_endpoint("Key Metrics Quarterly", f"{FMP_BASE_URL}/key-metrics/{symbol}?period=quarter&limit=80&apikey={FMP_API_KEY}")
test_endpoint("Historical Prices", f"{FMP_BASE_URL}/historical-price-full/{symbol}?apikey={FMP_API_KEY}")

print("\n" + "="*80)
print("TEST COMPLETE - Review date ranges above")
print("="*80)
