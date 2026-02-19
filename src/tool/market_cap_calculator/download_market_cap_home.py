"""
Download Historical Market Cap Data
⚠️ RUN THIS FROM HOME NETWORK (not work/school)

This uses yfinance which requires internet access to Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime

def get_stock_list():
    """Get list of stocks from your existing data"""
    data_dir = '../../../sp500_data/stock_data_1990_2024'
    
    if os.path.exists(data_dir):
        files = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"✅ Found {len(files)} stocks from your data")
        return sorted(files)
    else:
        # Fallback list
        print("⚠️  Using fallback stock list")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'WMT', 'XOM']

def download_market_cap(symbol, start='1990-01-01', end='2024-12-31'):
    """Download market cap for one stock"""
    try:
        print(f"  {symbol}...", end=' ', flush=True)
        
        ticker = yf.Ticker(symbol)
        
        # Get price history
        hist = ticker.history(start=start, end=end)
        
        if hist.empty:
            print("❌ No data")
            return None
        
        # Get current info
        info = ticker.info
        shares = info.get('sharesOutstanding', None)
        
        # Try to get historical shares
        try:
            shares_hist = ticker.get_shares_full(start=start, end=end)
            if shares_hist is not None and not shares_hist.empty:
                hist['shares'] = shares_hist
            elif shares:
                hist['shares'] = shares
        except:
            if shares:
                hist['shares'] = shares
        
        # Calculate market cap
        if 'shares' in hist.columns:
            hist['market_cap'] = hist['Close'] * hist['shares']
            hist['market_cap_B'] = hist['market_cap'] / 1e9
        else:
            print("⚠️  No shares")
            return None
        
        result = pd.DataFrame({
            'date': hist.index.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'close': hist['Close'].values,
            'volume': hist['Volume'].values,
            'shares_outstanding': hist['shares'].values,
            'market_cap_billions': hist['market_cap_B'].values
        })
        
        print(f"✅ {len(result)} records")
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:30]}")
        return None

def main():
    print("="*80)
    print("Market Cap Downloader (yfinance)")
    print("="*80)
    print("⚠️  MUST RUN FROM HOME NETWORK")
    print("⚠️  This will take 2-3 hours for all stocks")
    print("="*80)
    
    # Get stock list
    stocks = get_stock_list()
    
    # Ask for limit
    print(f"\nFound {len(stocks)} stocks")
    print("Enter number to process (or press Enter for all):", end=' ')
    
    try:
        user_input = input().strip()
        if user_input:
            limit = int(user_input)
            stocks = stocks[:limit]
            print(f"Processing first {limit} stocks")
    except:
        pass
    
    print(f"\nDownloading {len(stocks)} stocks...")
    print("="*80)
    
    all_data = []
    success = 0
    failed = 0
    
    for i, symbol in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}]", end=' ')
        
        df = download_market_cap(symbol)
        
        if df is not None:
            all_data.append(df)
            success += 1
            
            # Save progress every 10 stocks
            if len(all_data) % 10 == 0:
                temp_df = pd.concat(all_data, ignore_index=True)
                temp_df.to_csv('market_cap_progress.csv', index=False)
                print(f"  💾 Progress saved ({len(all_data)} stocks)")
        else:
            failed += 1
        
        # Be nice to Yahoo
        time.sleep(0.5)
    
    # Save final result
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df = final_df.sort_values(['date', 'symbol'])
        
        output_file = 'market_cap_1990_2024.csv'
        final_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE\!")
        print("="*80)
        print(f"Success: {success} stocks")
        print(f"Failed: {failed} stocks")
        print(f"Total records: {len(final_df):,}")
        print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        print(f"File size: {os.path.getsize(output_file)/1e6:.1f} MB")
        print(f"\n💾 Saved to: {output_file}")
        
        # Show top 10 on latest date
        latest = final_df[final_df['date'] == final_df['date'].max()]
        top10 = latest.nlargest(10, 'market_cap_billions')[['symbol', 'market_cap_billions']]
        print(f"\nTop 10 stocks on {final_df['date'].max().strftime('%Y-%m-%d')}:")
        for _, row in top10.iterrows():
            print(f"  {row['symbol']:6s}  ${row['market_cap_billions']:8.2f}B")
        
        print("\n✅ Copy this file to your work computer\!")
    else:
        print("\n❌ No data downloaded")
        print("Check your internet connection")

if __name__ == '__main__':
    main()
