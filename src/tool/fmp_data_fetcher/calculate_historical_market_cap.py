"""
Calculate Historical Market Cap from Current Data + Historical Prices

Formula: Historical Market Cap = (Current Market Cap / Current Price) × Historical Price
"""
import pandas as pd
import os
from datetime import datetime

def calculate_historical_market_cap(current_data_file, price_data_dir, output_file):
    """
    Calculate historical market cap for all stocks
    """
    print("="*80)
    print("Historical Market Cap Calculator")
    print("="*80)
    
    # Load current market cap data
    current_df = pd.read_csv(current_data_file)
    print(f"✅ Loaded current data: {len(current_df)} stocks")
    
    # Check price data directory
    if not os.path.exists(price_data_dir):
        print(f"❌ Price data directory not found: {price_data_dir}")
        return
    
    print(f"✅ Price data directory: {price_data_dir}")
    print("="*80)
    
    all_data = []
    success = 0
    failed = 0
    
    for idx, row in current_df.iterrows():
        symbol = row['symbol']
        current_price = row['current_price']
        current_market_cap = row['current_market_cap']
        
        print(f"[{idx+1}/{len(current_df)}] {symbol}...", end=' ', flush=True)
        
        # Skip if no current data
        if pd.isna(current_price) or pd.isna(current_market_cap):
            print("❌ No current data")
            failed += 1
            continue
        
        # Load historical prices
        price_file = os.path.join(price_data_dir, f"{symbol}.csv")
        
        if not os.path.exists(price_file):
            print("❌ No price file")
            failed += 1
            continue
        
        try:
            # Read historical prices
            prices_df = pd.read_csv(price_file)
            
            # Standardize column names
            prices_df.columns = prices_df.columns.str.lower()
            
            # Rename date column if needed
            if 'date' not in prices_df.columns:
                prices_df = prices_df.rename(columns={prices_df.columns[0]: 'date'})
            
            prices_df['date'] = pd.to_datetime(prices_df['date'])
            
            # Get close price column
            price_col = 'close' if 'close' in prices_df.columns else 'adj close'
            
            if price_col not in prices_df.columns:
                print("❌ No price column")
                failed += 1
                continue
            
            # Calculate shares outstanding (constant)
            shares_outstanding = current_market_cap / current_price
            
            # Calculate historical market cap
            prices_df['market_cap'] = prices_df[price_col] * shares_outstanding
            prices_df['market_cap_billions'] = prices_df['market_cap'] / 1e9
            
            # Create result dataframe
            result = pd.DataFrame({
                'date': prices_df['date'],
                'symbol': symbol,
                'close': prices_df[price_col],
                'volume': prices_df.get('volume', None),
                'shares_outstanding': shares_outstanding,
                'market_cap': prices_df['market_cap'],
                'market_cap_billions': prices_df['market_cap_billions']
            })
            
            all_data.append(result)
            success += 1
            print(f"✅ {len(result)} records")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}")
            failed += 1
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['date', 'symbol'])
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ SUCCESS\!")
        print("="*80)
        print(f"Success: {success} stocks")
        print(f"Failed: {failed} stocks")
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"Output file: {output_file} ({os.path.getsize(output_file)/1e6:.1f} MB)")
        
        # Show sample - top 10 on latest date
        latest_date = combined_df['date'].max()
        latest = combined_df[combined_df['date'] == latest_date]
        top10 = latest.nlargest(10, 'market_cap_billions')[['symbol', 'market_cap_billions']]
        
        print(f"\nTop 10 stocks on {latest_date.strftime('%Y-%m-%d')}:")
        for idx, row in top10.iterrows():
            print(f"  {row['symbol']:6s}  ${row['market_cap_billions']:8.1f}B")
        
        print(f"\n💾 Saved to: {output_file}")
        print("✅ Ready for V30 megacap filtering\!")
        
        return combined_df
    else:
        print("\n❌ No data generated")
        return None

if __name__ == '__main__':
    calculate_historical_market_cap(
        current_data_file='current_market_cap_progress.csv',
        price_data_dir='../../../sp500_data/stock_data_1990_2024',
        output_file='historical_market_cap_1990_2024.csv'
    )
