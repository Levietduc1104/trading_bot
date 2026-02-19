"""
Calculate Historical Market Cap for 200 FMP stocks
Save to data/market_cap folder
"""
import pandas as pd
import os

def calculate_historical_market_cap():
    print("="*80)
    print("Historical Market Cap Calculator - 200 FMP Stocks")
    print("="*80)
    
    # Input files
    current_data_file = 'current_market_cap_progress.csv'
    price_data_dir = '../../../sp500_data/stock_data_1990_2024'
    output_dir = '../../../../data/market_cap'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load current market cap data
    current_df = pd.read_csv(current_data_file)
    print(f"✅ Loaded current data: {len(current_df)} stocks")
    print(f"✅ Price data directory: {price_data_dir}")
    print(f"✅ Output directory: {output_dir}")
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
        if pd.isna(current_price) or pd.isna(current_market_cap) or current_price == 0:
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
            
            # Handle date column
            date_col = 'date' if 'date' in prices_df.columns else prices_df.columns[0]
            prices_df = prices_df.rename(columns={date_col: 'date'})
            prices_df['date'] = pd.to_datetime(prices_df['date'])
            
            # Get close price column
            if 'close' in prices_df.columns:
                price_col = 'close'
            elif 'adj close' in prices_df.columns:
                price_col = 'adj close'
            else:
                print("❌ No price column")
                failed += 1
                continue
            
            # Calculate shares outstanding (assumed constant)
            shares_outstanding = current_market_cap / current_price
            
            # Calculate historical market cap
            prices_df['shares_outstanding'] = shares_outstanding
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
            
            date_min = result['date'].min().year
            date_max = result['date'].max().year
            print(f"✅ {len(result):,} records ({date_min}-{date_max})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}")
            failed += 1
    
    # Save results
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['date', 'market_cap_billions'], ascending=[True, False])
        
        # Save to data folder
        output_file = os.path.join(output_dir, 'historical_market_cap_1990_2024.csv')
        combined_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ SUCCESS\!")
        print("="*80)
        print(f"Success: {success} stocks")
        print(f"Failed: {failed} stocks")
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        file_size_mb = os.path.getsize(output_file) / 1e6
        print(f"Output file: {output_file} ({file_size_mb:.1f} MB)")
        
        # Show top 10 on latest date
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
    calculate_historical_market_cap()
