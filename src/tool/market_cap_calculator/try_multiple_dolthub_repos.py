"""
Try multiple DoltHub repositories for market cap data
"""
import requests
import pandas as pd

# List of known DoltHub financial repositories
REPOSITORIES = [
    ("Liquidata", "stock-metrics"),
    ("Liquidata", "sp500-prices"),
    ("Liquidata", "stock-data"),
    ("post-no-preference", "earnings"),
    ("dolthub", "us-stocks-market-cap"),
    ("dolthub", "stock-prices"),
]

# Common table names to try
TABLE_NAMES = [
    "market_cap",
    "marketcap", 
    "stocks",
    "prices",
    "fundamentals",
    "metrics",
    "data"
]

def try_download(owner, repo, table):
    """Try to download from a specific repo/table"""
    url = f"https://www.dolthub.com/csv/{owner}/{repo}/main/{table}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return url, response.content
    except:
        pass
    return None, None

def main():
    print("="*80)
    print("Trying Multiple DoltHub Repositories")
    print("="*80)
    
    found = False
    
    for owner, repo in REPOSITORIES:
        print(f"\n📦 Repository: {owner}/{repo}")
        
        for table in TABLE_NAMES:
            print(f"  Trying table: {table}...", end=' ')
            
            url, content = try_download(owner, repo, table)
            
            if content:
                filename = f"{owner}_{repo}_{table}.csv"
                with open(filename, 'wb') as f:
                    f.write(content)
                
                # Check what we got
                try:
                    df = pd.read_csv(filename, nrows=5)
                    print(f"✅ SUCCESS\!")
                    print(f"    Downloaded: {filename}")
                    print(f"    Rows: {len(content.splitlines())-1}")
                    print(f"    Columns: {list(df.columns)}")
                    print(f"    URL: {url}")
                    print(f"\n    Preview:")
                    print(df.to_string(index=False))
                    found = True
                    print(f"\n{'='*80}")
                    break
                except Exception as e:
                    print(f"❌ Downloaded but couldn't parse")
            else:
                print("❌")
        
        if found:
            print(f"\n✅ Found working repository: {owner}/{repo}")
            print(f"💡 You can now download full data from this repository")
            break
    
    if not found:
        print("\n❌ No working repositories found")
        print("\n💡 Alternative: Visit https://www.dolthub.com/repositories")
        print("   Search for 'market cap' or 'stocks' and download manually")

if __name__ == '__main__':
    main()
