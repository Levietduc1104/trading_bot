"""
Download historical shares outstanding from SEC Edgar for all stocks
"""
import requests
import pandas as pd
import json
import time
import os

def get_cik_from_ticker(ticker, ticker_map):
    """Get 10-digit CIK from ticker"""
    ticker = ticker.upper().strip()
    for item in ticker_map.values():
        if item["ticker"].upper() == ticker:
            cik_int = int(item["cik_str"])
            return str(cik_int).zfill(10)
    return None

def download_shares_outstanding(cik10):
    """Download shares outstanding from SEC Edgar"""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Research/1.0)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            
            # Get shares outstanding
            us_gaap = data.get('facts', {}).get('us-gaap', {})
            shares_data = us_gaap.get('CommonStockSharesOutstanding', {})
            units = shares_data.get('units', {})
            
            if 'shares' in units:
                observations = units['shares']
                df = pd.DataFrame(observations)
                df['end'] = pd.to_datetime(df['end'])
                return df
    except:
        pass
    
    return None

def main():
    print("="*80)
    print("SEC Edgar Shares Outstanding Downloader")
    print("="*80)
    
    # Load ticker map
    if not os.path.exists('company_tickers.json'):
        print("Downloading company_tickers.json...")
        response = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with open('company_tickers.json', 'w') as f:
            json.dump(response.json(), f)
    
    with open('company_tickers.json', 'r') as f:
        ticker_map = json.load(f)
    
    print(f"✅ Loaded {len(ticker_map)} company tickers")
    
    # Load current market cap data
    current_df = pd.read_csv('current_market_cap_progress.csv')
    print(f"✅ Processing {len(current_df)} stocks")
    print("="*80)
    
    all_results = []
    success = 0
    failed = 0
    
    for idx, row in current_df.iterrows():
        symbol = row['symbol']
        print(f"[{idx+1}/{len(current_df)}] {symbol}...", end=' ', flush=True)
        
        # Get CIK
        cik10 = get_cik_from_ticker(symbol, ticker_map)
        if not cik10:
            print("❌ No CIK")
            failed += 1
            continue
        
        # Download shares
        df = download_shares_outstanding(cik10)
        
        if df is not None and len(df) > 0:
            df['symbol'] = symbol
            all_results.append(df)
            success += 1
            print(f"✅ {len(df)} records ({df['end'].min().year}-{df['end'].max().year})")
        else:
            print("❌ No data")
            failed += 1
        
        # Rate limiting - be nice to SEC
        time.sleep(0.2)
        
        # Save progress every 50
        if len(all_results) > 0 and len(all_results) % 50 == 0:
            temp_df = pd.concat(all_results, ignore_index=True)
            temp_df.to_csv('sec_shares_progress.csv', index=False)
            print(f"    💾 Progress saved ({len(all_results)} stocks)")
    
    # Save final
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv('sec_shares_outstanding_all.csv', index=False)
        
        print("\n" + "="*80)
        print("✅ SUCCESS\!")
        print("="*80)
        print(f"Success: {success} stocks")
        print(f"Failed: {failed} stocks")
        print(f"Total records: {len(combined):,}")
        print(f"Date range: {combined['end'].min()} to {combined['end'].max()}")
        print(f"\n💾 Saved to: sec_shares_outstanding_all.csv")
    else:
        print("\n❌ No data downloaded")

if __name__ == '__main__':
    main()
