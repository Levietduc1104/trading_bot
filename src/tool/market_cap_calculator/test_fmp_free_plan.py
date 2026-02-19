"""
Test what FMP FREE plan can access
"""
import requests
import json

API_KEY = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"

def test_endpoint(url, description):
    """Test an endpoint and show results"""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print('='*80)
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, dict) and 'Error Message' in data:
                print(f"❌ Error: {data['Error Message']}")
                return False
            elif data:
                print(f"✅ SUCCESS\!")
                print(f"\nData preview:")
                print(json.dumps(data[:2] if isinstance(data, list) else data, indent=2)[:500])
                return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text[:200])
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

def main():
    print("="*80)
    print("FMP FREE Plan Capabilities Test")
    print("="*80)
    
    symbol = "AAPL"
    
    # Test 1: Profile (current market cap and shares)
    test_endpoint(
        f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}",
        "Current Profile (market cap, shares outstanding)"
    )
    
    # Test 2: Historical Market Cap
    test_endpoint(
        f"https://financialmodelingprep.com/api/v3/historical-market-capitalization/{symbol}?apikey={API_KEY}",
        "Historical Market Cap"
    )
    
    # Test 3: Key Metrics (might have historical shares)
    test_endpoint(
        f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?apikey={API_KEY}",
        "Key Metrics"
    )
    
    # Test 4: Historical Price (to see date range)
    test_endpoint(
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={API_KEY}",
        "Historical Price Data"
    )
    
    # Test 5: Company Core Information
    test_endpoint(
        f"https://financialmodelingprep.com/api/v3/company-core-information?symbol={symbol}&apikey={API_KEY}",
        "Company Core Information"
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nNow we know what FREE plan gives you\!")

if __name__ == '__main__':
    main()
