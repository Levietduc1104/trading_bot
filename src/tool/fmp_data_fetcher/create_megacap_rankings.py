"""
Create Megacap Rankings Over Time
Find top N stocks at quarterly intervals (for V30 strategy)
"""
import pandas as pd
import os

def create_megacap_rankings(top_n=20):
    print("="*80)
    print(f"Creating Top {top_n} Megacap Rankings Over Time")
    print("="*80)
    
    # Load historical market cap data
    data_file = '../../../../data/market_cap/historical_market_cap_1990_2024.csv'
    output_file = f'../../../../data/market_cap/megacap_rankings_top{top_n}_quarterly.csv'
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"✅ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✅ Stocks: {df['symbol'].nunique()}")
    print("="*80)
    
    # Get unique dates and sample quarterly (every 63 trading days)
    all_dates = sorted(df['date'].unique())
    quarterly_dates = all_dates[::63]  # Every ~3 months
    
    print(f"Creating rankings for {len(quarterly_dates)} quarterly dates...")
    print("="*80)
    
    results = []
    
    for i, date in enumerate(quarterly_dates, 1):
        # Get all stocks on this date
        day_data = df[df['date'] == date].copy()
        
        if len(day_data) == 0:
            continue
        
        # Sort by market cap
        day_data = day_data.sort_values('market_cap_billions', ascending=False)
        
        # Get top N
        top_stocks = day_data.head(top_n).copy()
        top_stocks['rank'] = range(1, len(top_stocks) + 1)
        
        # Keep only needed columns
        top_stocks = top_stocks[['date', 'rank', 'symbol', 'market_cap_billions', 'close']]
        
        results.append(top_stocks)
        
        # Show progress
        if i % 10 == 0 or i <= 5 or i > len(quarterly_dates) - 5:
            top3 = top_stocks.head(3)['symbol'].tolist()
            print(f"[{i}/{len(quarterly_dates)}] {date.strftime('%Y-%m-%d')}: {', '.join(top3)}...")
    
    # Combine results
    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ SUCCESS\!")
        print("="*80)
        print(f"Total dates: {len(quarterly_dates)}")
        print(f"Total records: {len(combined):,}")
        print(f"Output: {output_file}")
        
        # Show most frequent megacaps
        print(f"\n{'='*80}")
        print(f"MOST FREQUENT IN TOP {top_n} (All Time)")
        print(f"{'='*80}")
        
        stock_counts = combined['symbol'].value_counts().head(20)
        appearances = len(quarterly_dates)
        
        for symbol, count in stock_counts.items():
            pct = (count / appearances) * 100
            print(f"  {symbol:6s}  {count:4d} / {appearances} dates ({pct:5.1f}%)")
        
        # Show "Mag 7" evolution
        print(f"\n{'='*80}")
        print("MAG 7 EVOLUTION (Top 7 at Key Dates)")
        print(f"{'='*80}")
        
        key_dates = ['1990-01-02', '2000-01-03', '2010-01-04', '2020-01-02', '2024-01-02']
        
        for date_str in key_dates:
            date = pd.to_datetime(date_str)
            date_data = combined[combined['date'] == date]
            if not date_data.empty:
                top7 = date_data.head(7)
                print(f"\n{date_str}:")
                for _, row in top7.iterrows():
                    print(f"  {row['rank']}) {row['symbol']:6s}  ${row['market_cap_billions']:8.1f}B")
        
        print(f"\n💾 Saved to: {output_file}")
        print("✅ Ready for V30 strategy\!")
        
        return combined
    else:
        print("\n❌ No rankings created")
        return None

if __name__ == '__main__':
    # Create top 20 rankings (V30 typically uses top 10-20)
    create_megacap_rankings(top_n=20)
    
    # Also create top 7 for "Mag 7" analysis
    print("\n" + "="*80)
    create_megacap_rankings(top_n=7)
