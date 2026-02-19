"""
Download SEC shares outstanding for stocks we have
Uses direct CIK lookup from SEC
"""
import requests
import pandas as pd
import time

# Common CIKs (we'll try to get them)
def try_get_cik_from_sec_search(ticker):
    """Try to find CIK by searching SEC"""
    # SEC search returns companies matching ticker
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&count=10&output=json"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        # This might not work, just try
        return None
    except:
        return None

def download_shares_outstanding_direct(ticker):
    """Download using common CIK patterns"""
    # Try common patterns - most tickers match their CIK padding
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Research/1.0)"}
    
    # For now, just try the Apple example to test
    # We know AAPL = CIK 0000320193
    cik_map = {
        'AAPL': '0000320193',
        'MSFT': '0000789019',
        'GOOGL': '0001652044',
        'AMZN': '0001018724',
        'NVDA': '0001045810',
        # Add more as needed
    }
    
    cik = cik_map.get(ticker)
    if not cik:
        return None
    
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            us_gaap = data.get('facts', {}).get('us-gaap', {})
            shares_data = us_gaap.get('CommonStockSharesOutstanding', {})
            units = shares_data.get('units', {})
            
            if 'shares' in units:
                df = pd.DataFrame(units['shares'])
                df['end'] = pd.to_datetime(df['end'])
                df['symbol'] = ticker
                return df
    except Exception as e:
        print(f" Error: {str(e)[:40]}")
    
    return None

# Test with stocks we know
print("Testing SEC download with known stocks...")
test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

for ticker in test_stocks:
    print(f"{ticker}...", end=' ', flush=True)
    df = download_shares_outstanding_direct(ticker)
    if df is not None:
        print(f"✅ {len(df)} records")
    else:
        print("❌")
    time.sleep(0.2)

print("\n💡 For full download, we need company_tickers.json")
print("Please download it from: https://www.sec.gov/files/company_tickers.json")
