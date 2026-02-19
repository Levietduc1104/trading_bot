"""
Download Financial Data from DoltHub
Repository: post-no-preference/earnings
"""
import requests
import pandas as pd

OWNER = "post-no-preference"
REPO = "earnings"

def download_as_csv(table_name, output_file):
    """Download a table as CSV"""
    csv_url = f"https://www.dolthub.com/csv/{OWNER}/{REPO}/main/{table_name}"
    
    print(f"Downloading from: {csv_url}")
    
    try:
        response = requests.get(csv_url, timeout=120)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded to: {output_file}")
            
            # Show preview
            df = pd.read_csv(output_file, nrows=5)
            print(f"\nPreview:")
            print(df)
            print(f"\nColumns: {list(df.columns)}")
            return True
        else:
            print(f"❌ Download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*80)
    print("DoltHub Financial Data Downloader")
    print("="*80)
    print(f"Repository: {OWNER}/{REPO}")
    print("="*80)
    
    # Try downloading common tables
    tables = ['earnings', 'market_cap', 'fundamentals', 'balance_sheet']
    
    for table_name in tables:
        print(f"\nTrying: {table_name}")
        success = download_as_csv(table_name, f'{table_name}.csv')
        if success:
            print(f"\n✅ Successfully downloaded {table_name}\!")
            break
    
    print(f"\n💡 Visit: https://www.dolthub.com/repositories/{OWNER}/{REPO}")

if __name__ == '__main__':
    main()
