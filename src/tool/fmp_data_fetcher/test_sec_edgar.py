import requests
import pandas as pd

SEC_HEADERS = {
    "User-Agent": "ResearchBot/1.0 (research@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

def get_cik_from_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    for _, item in data.items():
        if item["ticker"].upper() == ticker:
            cik_int = int(item["cik_str"])
            return str(cik_int).zfill(10)
    raise ValueError(f"Ticker not found: {ticker}")

def download_company_facts(cik10: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def facts_to_dataframe(companyfacts: dict, taxonomy="us-gaap") -> pd.DataFrame:
    facts = companyfacts.get("facts", {}).get(taxonomy, {})
    rows = []
    for tag, tag_obj in facts.items():
        units = tag_obj.get("units", {})
        for unit, observations in units.items():
            for ob in observations:
                rows.append({
                    "tag": tag,
                    "unit": unit,
                    "val": ob.get("val"),
                    "start": ob.get("start"),
                    "end": ob.get("end"),
                    "fy": ob.get("fy"),
                    "fp": ob.get("fp"),
                    "form": ob.get("form"),
                    "filed": ob.get("filed"),
                })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Testing SEC Edgar API for {ticker}...")
    print("="*80)
    
    try:
        cik10 = get_cik_from_ticker(ticker)
        print(f"✅ CIK: {cik10}")
        
        companyfacts = download_company_facts(cik10)
        print(f"✅ Downloaded company facts")
        
        df = facts_to_dataframe(companyfacts)
        print(f"✅ Total records: {len(df):,}")
        
        # Check for shares outstanding
        shares_data = df[df['tag'].str.contains('Share', case=False, na=False)]
        print(f"\n📊 Found {len(shares_data)} share-related records")
        print(f"   Tags: {shares_data['tag'].unique()[:10]}")
        
        # Save sample
        df.to_csv('aapl_sec_all.csv', index=False)
        print(f"\n💾 Saved to: aapl_sec_all.csv")
        
        # Show date range
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
        print(f"📅 Date range: {df['end'].min()} to {df['end'].max()}")
        
        print("\n✅ SEC Edgar API works\!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

