"""
Flexible Megacap Stock Selector
Different methods to select megacap stocks for V30/V31
"""
import pandas as pd
import os

class MegacapSelector:
    def __init__(self, market_cap_file=None):
        """
        Initialize megacap selector
        
        Args:
            market_cap_file: Path to historical market cap CSV
        """
        if market_cap_file is None:
            market_cap_file = '../data/market_cap/historical_market_cap_1990_2024.csv'
        
        self.df = pd.read_csv(market_cap_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✅ Loaded market cap data: {len(self.df):,} records")
        print(f"   Stocks: {self.df['symbol'].nunique()}")
        print(f"   Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
    
    def get_megacaps_simple(self, date, top_n=20):
        """
        Method 1: Simple top N by market cap
        
        Args:
            date: Selection date
            top_n: Number of stocks to select
        
        Returns:
            List of symbols
        """
        day_data = self.df[self.df['date'] == pd.Timestamp(date)]
        top_stocks = day_data.nlargest(top_n, 'market_cap_billions')['symbol'].tolist()
        return top_stocks
    
    def get_megacaps_min_market_cap(self, date, min_market_cap_billions=100):
        """
        Method 2: All stocks above a minimum market cap threshold
        
        Args:
            date: Selection date
            min_market_cap_billions: Minimum market cap in billions (default: $100B)
        
        Returns:
            List of symbols
        """
        day_data = self.df[self.df['date'] == pd.Timestamp(date)]
        large_stocks = day_data[day_data['market_cap_billions'] >= min_market_cap_billions]
        return large_stocks.sort_values('market_cap_billions', ascending=False)['symbol'].tolist()
    
    def get_megacaps_percentile(self, date, percentile=90):
        """
        Method 3: Top percentile by market cap
        
        Args:
            date: Selection date
            percentile: Percentile threshold (default: 90 = top 10%)
        
        Returns:
            List of symbols
        """
        day_data = self.df[self.df['date'] == pd.Timestamp(date)]
        threshold = day_data['market_cap_billions'].quantile(percentile/100)
        large_stocks = day_data[day_data['market_cap_billions'] >= threshold]
        return large_stocks.sort_values('market_cap_billions', ascending=False)['symbol'].tolist()
    
    def get_megacaps_consistent(self, start_date, end_date, min_appearances=0.8, top_n=50):
        """
        Method 4: Stocks that consistently appear in top N over a period
        
        Args:
            start_date: Start of period
            end_date: End of period
            min_appearances: Minimum % of time in top N (default: 0.8 = 80%)
            top_n: Top N to check (default: 50)
        
        Returns:
            List of symbols (sorted by frequency)
        """
        period_data = self.df[(self.df['date'] >= pd.Timestamp(start_date)) & 
                              (self.df['date'] <= pd.Timestamp(end_date))]
        
        # Count appearances in top N
        dates = period_data['date'].unique()
        total_dates = len(dates)
        
        stock_counts = {}
        for date in dates:
            day_data = period_data[period_data['date'] == date]
            top_stocks = day_data.nlargest(top_n, 'market_cap_billions')['symbol'].tolist()
            for symbol in top_stocks:
                stock_counts[symbol] = stock_counts.get(symbol, 0) + 1
        
        # Filter by minimum appearances
        threshold = int(total_dates * min_appearances)
        consistent_stocks = [(symbol, count) for symbol, count in stock_counts.items() if count >= threshold]
        consistent_stocks.sort(key=lambda x: x[1], reverse=True)
        
        return [symbol for symbol, count in consistent_stocks]
    
    def get_megacaps_sector_diversified(self, date, sectors, stocks_per_sector=3):
        """
        Method 5: Sector-diversified megacaps
        Requires sector data
        
        Args:
            date: Selection date
            sectors: List of sectors to include
            stocks_per_sector: Number of stocks per sector
        
        Returns:
            List of symbols
        """
        # Note: This requires sector information which we don't have yet
        # Placeholder for future implementation
        raise NotImplementedError("Sector diversification requires sector data from FMP")
    
    def get_megacaps_growth_weighted(self, date, lookback_days=252, top_n=20):
        """
        Method 6: Top N weighted by both market cap AND recent growth
        
        Args:
            date: Selection date
            lookback_days: Days to look back for growth calc (default: 252 = 1 year)
            top_n: Number of stocks
        
        Returns:
            List of symbols
        """
        current_date = pd.Timestamp(date)
        lookback_date = current_date - pd.Timedelta(days=lookback_days)
        
        # Get current market caps
        current = self.df[self.df['date'] == current_date]
        
        # Get past market caps
        past = self.df[self.df['date'] >= lookback_date].groupby('symbol').first().reset_index()
        
        # Merge and calculate growth
        merged = current.merge(past[['symbol', 'market_cap_billions']], 
                              on='symbol', suffixes=('_current', '_past'))
        
        merged['growth'] = (merged['market_cap_billions_current'] / merged['market_cap_billions_past']) - 1
        
        # Score: 70% market cap, 30% growth
        merged['score'] = (0.7 * merged['market_cap_billions_current'] + 
                          0.3 * merged['growth'] * merged['market_cap_billions_current'])
        
        top_stocks = merged.nlargest(top_n, 'score')['symbol'].tolist()
        return top_stocks


# Helper function for V30 integration
def load_megacap_selector(market_cap_file=None):
    """
    Load megacap selector for use in V30/V31 strategies
    
    Returns:
        MegacapSelector instance
    """
    return MegacapSelector(market_cap_file)


if __name__ == '__main__':
    # Test different selection methods
    selector = MegacapSelector('../data/market_cap/historical_market_cap_1990_2024.csv')
    
    test_date = '2024-01-01'
    
    print("\n" + "="*80)
    print("MEGACAP SELECTION METHODS COMPARISON")
    print("="*80)
    
    # Method 1: Simple top 20
    print(f"\n1️⃣  Top 20 by market cap:")
    top20 = selector.get_megacaps_simple(test_date, top_n=20)
    print(f"   {', '.join(top20[:10])}...")
    
    # Method 2: Minimum $100B market cap
    print(f"\n2️⃣  All stocks >= $100B market cap:")
    min100b = selector.get_megacaps_min_market_cap(test_date, min_market_cap_billions=100)
    print(f"   Count: {len(min100b)}")
    print(f"   {', '.join(min100b[:10])}...")
    
    # Method 3: Top 10% (90th percentile)
    print(f"\n3️⃣  Top 10% by market cap:")
    top10pct = selector.get_megacaps_percentile(test_date, percentile=90)
    print(f"   Count: {len(top10pct)}")
    print(f"   {', '.join(top10pct[:10])}...")
    
    # Method 4: Consistently in top 50 over past 5 years
    print(f"\n4️⃣  Consistently in top 50 (past 5 years, 80% of time):")
    consistent = selector.get_megacaps_consistent('2019-01-01', '2024-01-01', 
                                                   min_appearances=0.8, top_n=50)
    print(f"   Count: {len(consistent)}")
    print(f"   {', '.join(consistent[:10])}...")
    
    # Method 6: Growth-weighted
    print(f"\n6️⃣  Top 20 growth-weighted (70% size, 30% 1yr growth):")
    growth_weighted = selector.get_megacaps_growth_weighted(test_date, lookback_days=252, top_n=20)
    print(f"   {', '.join(growth_weighted[:10])}...")
    
    print("\n" + "="*80)
    print("✅ Choose the method that fits your V30 strategy best\!")
    print("="*80)
