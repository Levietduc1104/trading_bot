"""
Get CURRENT market cap and shares outstanding
Then use with historical prices to calculate historical market cap
"""
import requests
import pandas as pd
import json

FMP_API_KEY = "yJQUTCul5jMlGvqF4ZY6xF7kbWPQ2c7w"

def get_current_info(symbol):
    """Get current market cap and shares from FMP"""
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    params = {'apikey': FMP_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                info = data[0]
                return {
                    'symbol': symbol,
                    'market_cap': info.get('mktCap'),
                    'shares_outstanding': info.get('sharesOutstanding'),
                    'price': info.get('price'),
                    'company_name': info.get('companyName')
                }
    except Exception as e:
        print(f"  ❌ {symbol}: {str(e)[:50]}")
    
    return None

def main():
    import os
    
    print("="*80)
    print("Get Current Market Cap Data")
    print("="*80)
    
    # Get stock list from your data
    data_dir = '../../../sp500_data/stock_data_1990_2024'
    
    if os.path.exists(data_dir):
        stocks = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"✅ Found {len(stocks)} stocks")
    else:
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        print(f"⚠️  Using test list: {len(stocks)} stocks")
    
    # Ask for limit
    print("How many to download? (Enter for all):", end=' ')
    try:
        user_input = input().strip()
        if user_input:
            limit = int(user_input)
            stocks = stocks[:limit]
    except:
        pass
    
    print(f"\nDownloading current data for {len(stocks)} stocks...")
    print("="*80)
    
    results = []
    success = 0
    
    for i, symbol in enumerate(stocks[:50], 1):  # Limit to 50 for testing
        print(f"[{i}/{min(len(stocks), 50)}] {symbol}...", end=' ')
        
        info = get_current_info(symbol)
        
        if info:
            results.append(info)
            success += 1
            print(f"✅ ${info['market_cap']/1e9:.1f}B")
        else:
            print("❌")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv('current_market_cap.csv', index=False)
        
        print("\n" + "="*80)
        print("✅ SUCCESS\!")
        print("="*80)
        print(f"Downloaded: {success} stocks")
        print(f"Saved to: current_market_cap.csv")
        
        print("\nTop 10 by market cap:")
        top10 = df.nlargest(10, 'market_cap')[['symbol', 'company_name', 'market_cap']]
        for _, row in top10.iterrows():
            print(f"  {row['symbol']:6s}  ${row['market_cap']/1e9:8.1f}B  {row['company_name'][:30]}")
        
        print("\n💡 Next step: Use this with historical prices to calculate historical market cap")
    else:
        print("\n❌ No data downloaded - check network connection")

if __name__ == '__main__':
    main()
