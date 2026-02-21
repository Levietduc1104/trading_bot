import pandas as pd
import numpy as np
from datetime import datetime

def calculate_drawdowns(df):
    """Calculate drawdown series"""
    df = df.copy()
    
    # Calculate running maximum
    df['cummax'] = df['Close'].cummax()
    
    # Calculate drawdown
    df['drawdown'] = (df['Close'] - df['cummax']) / df['cummax'] * 100
    df['drawdown_dollars'] = df['Close'] - df['cummax']
    
    return df

def find_drawdown_periods(df):
    """Find distinct drawdown periods"""
    df = df.copy()
    
    # Mark new peaks
    df['is_peak'] = df['Close'] == df['cummax']
    
    drawdown_periods = []
    in_drawdown = False
    current_period = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Start of drawdown
        if not in_drawdown and row['drawdown'] < 0:
            in_drawdown = True
            current_period = {
                'peak_date': df.iloc[i-1]['Date'] if i > 0 else row['Date'],
                'peak_price': df.iloc[i-1]['Close'] if i > 0 else row['Close'],
                'trough_date': None,
                'trough_price': None,
                'recovery_date': None,
                'recovery_price': None,
                'max_drawdown': 0,
                'drawdown_days': 0,
                'recovery_days': 0
            }
        
        # Track maximum drawdown in this period
        if in_drawdown:
            if row['drawdown'] < current_period['max_drawdown']:
                current_period['max_drawdown'] = row['drawdown']
                current_period['trough_date'] = row['Date']
                current_period['trough_price'] = row['Close']
        
        # End of drawdown (recovery)
        if in_drawdown and row['drawdown'] == 0:
            in_drawdown = False
            current_period['recovery_date'] = row['Date']
            current_period['recovery_price'] = row['Close']
            
            # Calculate durations
            current_period['drawdown_days'] = (current_period['trough_date'] - current_period['peak_date']).days
            current_period['recovery_days'] = (current_period['recovery_date'] - current_period['trough_date']).days
            current_period['total_days'] = (current_period['recovery_date'] - current_period['peak_date']).days
            
            drawdown_periods.append(current_period)
    
    # If still in drawdown at end
    if in_drawdown and current_period:
        current_period['recovery_date'] = df.iloc[-1]['Date']
        current_period['recovery_price'] = df.iloc[-1]['Close']
        current_period['drawdown_days'] = (current_period['trough_date'] - current_period['peak_date']).days
        current_period['total_days'] = (current_period['recovery_date'] - current_period['peak_date']).days
        current_period['recovery_days'] = current_period['total_days'] - current_period['drawdown_days']
        drawdown_periods.append(current_period)
    
    return drawdown_periods

def main():
    print("=" * 120)
    print("SPY MAXIMUM DRAWDOWN ANALYSIS")
    print("=" * 120)
    print()
    
    # Load data
    df = pd.read_csv('trading_bot/sp500_data/individual_stocks/SPY.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= '2005-01-01'].copy()
    df = df.reset_index(drop=True)
    
    print(f"Analysis Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total Days: {len(df):,}")
    print()
    
    # Calculate drawdowns
    df = calculate_drawdowns(df)
    
    # Overall statistics
    max_dd = df['drawdown'].min()
    max_dd_date = df.loc[df['drawdown'].idxmin(), 'Date']
    max_dd_price = df.loc[df['drawdown'].idxmin(), 'Close']
    max_dd_peak_price = df.loc[df['drawdown'].idxmin(), 'cummax']
    
    print("=" * 120)
    print("OVERALL MAXIMUM DRAWDOWN")
    print("=" * 120)
    print()
    print(f"Maximum Drawdown: {max_dd:.2f}%")
    print(f"Date of Maximum DD: {max_dd_date.date()}")
    print(f"Price at Bottom: ${max_dd_price:.2f}")
    print(f"Peak Price Before: ${max_dd_peak_price:.2f}")
    print(f"Dollar Loss from Peak: ${max_dd_peak_price - max_dd_price:.2f}")
    print()
    
    # Find all drawdown periods
    drawdown_periods = find_drawdown_periods(df)
    
    # Sort by severity
    drawdown_periods_sorted = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])
    
    print("=" * 120)
    print("TOP 10 WORST DRAWDOWN PERIODS")
    print("=" * 120)
    print()
    
    print(f"{'Rank':<6} {'Peak Date':<12} {'Trough Date':<12} {'Recovery':<12} {'Max DD':<10} {'DD Days':<10} {'Recovery Days':<15} {'Total Days':<12}")
    print("-" * 120)
    
    for i, period in enumerate(drawdown_periods_sorted[:10], 1):
        recovery_status = period['recovery_date'].date() if period['recovery_date'] else "Still Down"
        print(f"{i:<6} {period['peak_date'].date()\!s:<12} {period['trough_date'].date()\!s:<12} "
              f"{str(recovery_status):<12} {period['max_drawdown']:>8.2f}% {period['drawdown_days']:>9} "
              f"{period['recovery_days']:>14} {period['total_days']:>11}")
    
    print()
    
    # Identify the periods
    print("=" * 120)
    print("DETAILED ANALYSIS OF WORST DRAWDOWNS")
    print("=" * 120)
    print()
    
    for i, period in enumerate(drawdown_periods_sorted[:5], 1):
        print(f"{i}. Drawdown: {period['max_drawdown']:.2f}%")
        print("-" * 120)
        print(f"   Peak Date:        {period['peak_date'].date()} (Price: ${period['peak_price']:.2f})")
        print(f"   Trough Date:      {period['trough_date'].date()} (Price: ${period['trough_price']:.2f})")
        print(f"   Recovery Date:    {period['recovery_date'].date() if period['recovery_date'] else 'Not Recovered'}")
        print(f"   ")
        print(f"   Time to Bottom:   {period['drawdown_days']} days ({period['drawdown_days']/30:.1f} months)")
        print(f"   Time to Recover:  {period['recovery_days']} days ({period['recovery_days']/30:.1f} months)")
        print(f"   Total Duration:   {period['total_days']} days ({period['total_days']/365:.1f} years)")
        print(f"   ")
        print(f"   Dollar Loss:      ${period['peak_price'] - period['trough_price']:.2f}")
        print(f"   Recovery Gain:    ${period['recovery_price'] - period['trough_price']:.2f}" if period['recovery_price'] else "   Still recovering...")
        
        # Identify the crisis
        year = period['trough_date'].year
        if year == 2008 or year == 2009:
            crisis = "2008 Financial Crisis / Great Recession"
        elif year == 2020:
            crisis = "COVID-19 Pandemic Crash"
        elif year == 2022:
            crisis = "2022 Bear Market (Inflation/Rate Hikes)"
        elif year == 2011:
            crisis = "2011 US Debt Ceiling Crisis"
        elif year == 2018:
            crisis = "2018 Q4 Correction"
        elif year == 2015 or year == 2016:
            crisis = "2015-2016 Market Correction"
        else:
            crisis = "Market Correction"
        
        print(f"   Event:            {crisis}")
        print()
    
    # Year-by-year maximum drawdown
    print("=" * 120)
    print("MAXIMUM DRAWDOWN BY YEAR")
    print("=" * 120)
    print()
    
    df['year'] = df['Date'].dt.year
    
    yearly_stats = []
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        max_dd_year = year_data['drawdown'].min()
        
        # Get start and end prices
        start_price = year_data.iloc[0]['Close']
        end_price = year_data.iloc[-1]['Close']
        year_return = ((end_price - start_price) / start_price) * 100
        
        yearly_stats.append({
            'year': year,
            'max_drawdown': max_dd_year,
            'year_return': year_return,
            'start_price': start_price,
            'end_price': end_price
        })
    
    yearly_df = pd.DataFrame(yearly_stats)
    yearly_df = yearly_df.sort_values('max_drawdown')
    
    print(f"{'Year':<8} {'Max Drawdown':<15} {'Annual Return':<15} {'Start Price':<15} {'End Price':<15} {'Severity':<20}")
    print("-" * 120)
    
    for _, row in yearly_df.iterrows():
        if row['max_drawdown'] < -30:
            severity = "🔴 SEVERE CRASH"
        elif row['max_drawdown'] < -20:
            severity = "🟠 MAJOR CORRECTION"
        elif row['max_drawdown'] < -10:
            severity = "🟡 CORRECTION"
        elif row['max_drawdown'] < -5:
            severity = "🟢 MINOR PULLBACK"
        else:
            severity = "✅ STABLE"
        
        print(f"{int(row['year']):<8} {row['max_drawdown']:>13.2f}% {row['year_return']:>13.2f}% "
              f"${row['start_price']:>13.2f} ${row['end_price']:>13.2f} {severity:<20}")
    
    print()
    
    # Summary statistics
    print("=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print()
    
    # Count drawdowns by severity
    severe_years = len(yearly_df[yearly_df['max_drawdown'] < -30])
    major_years = len(yearly_df[(yearly_df['max_drawdown'] >= -30) & (yearly_df['max_drawdown'] < -20)])
    correction_years = len(yearly_df[(yearly_df['max_drawdown'] >= -20) & (yearly_df['max_drawdown'] < -10)])
    minor_years = len(yearly_df[(yearly_df['max_drawdown'] >= -10) & (yearly_df['max_drawdown'] < -5)])
    stable_years = len(yearly_df[yearly_df['max_drawdown'] >= -5])
    
    total_years = len(yearly_df)
    
    print(f"Total Years Analyzed: {total_years}")
    print()
    print("Frequency by Severity:")
    print(f"  🔴 Severe Crash (>30% DD):      {severe_years} years ({severe_years/total_years*100:.0f}%)")
    print(f"  🟠 Major Correction (20-30% DD): {major_years} years ({major_years/total_years*100:.0f}%)")
    print(f"  🟡 Correction (10-20% DD):       {correction_years} years ({correction_years/total_years*100:.0f}%)")
    print(f"  🟢 Minor Pullback (5-10% DD):    {minor_years} years ({minor_years/total_years*100:.0f}%)")
    print(f"  ✅ Stable (<5% DD):              {stable_years} years ({stable_years/total_years*100:.0f}%)")
    print()
    
    # Average recovery time
    completed_drawdowns = [p for p in drawdown_periods if p['recovery_days'] > 0 and p['max_drawdown'] < -10]
    if completed_drawdowns:
        avg_recovery = np.mean([p['recovery_days'] for p in completed_drawdowns])
        avg_total = np.mean([p['total_days'] for p in completed_drawdowns])
        
        print(f"Average Recovery Time (for >10% drawdowns):")
        print(f"  Time to Bottom:   {np.mean([p['drawdown_days'] for p in completed_drawdowns]):.0f} days")
        print(f"  Time to Recover:  {avg_recovery:.0f} days ({avg_recovery/30:.1f} months)")
        print(f"  Total Duration:   {avg_total:.0f} days ({avg_total/365:.1f} years)")
    
    print()
    print("=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)
    print()
    
    print("1. WORST CRASHES:")
    worst_3 = yearly_df.head(3)
    for _, row in worst_3.iterrows():
        print(f"   - {int(row['year'])}: {row['max_drawdown']:.1f}% drawdown")
    
    print()
    print("2. RISK PROFILE:")
    print(f"   - You will experience 10%+ corrections in ~{(severe_years + major_years + correction_years)/total_years*100:.0f}% of years")
    print(f"   - Severe crashes (>30%) happen roughly every {total_years/max(severe_years, 1):.0f} years")
    print(f"   - Recovery can take 6 months to 2+ years")
    
    print()
    print("3. INVESTMENT IMPLICATIONS:")
    print("   - Emergency fund essential (6-12 months)")
    print("   - Don't invest money needed within 5 years")
    print("   - Buy during crashes for best long-term returns")
    print("   - Stay invested - markets always recover")
    
    print()
    print("=" * 120)
    
    # Save detailed results
    yearly_df.to_csv('spy_drawdown_by_year.csv', index=False)
    
    # Save drawdown periods
    dd_df = pd.DataFrame(drawdown_periods_sorted)
    dd_df.to_csv('spy_drawdown_periods.csv', index=False)
    
    print()
    print("Results saved:")
    print("  - spy_drawdown_by_year.csv")
    print("  - spy_drawdown_periods.csv")

if __name__ == "__main__":
    main()
