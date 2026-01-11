"""
Filter each period dataset to top 500 best companies based on available data.

Since we don't have historical market cap, we use:
1. Dollar volume (price * volume) - proxy for company size and liquidity
2. Average trading volume - liquidity indicator
3. Average price - avoid penny stocks
4. Data completeness - companies that survived the full period

This simulates S&P 500 selection for historical periods.
"""

import os
import pandas as pd
import numpy as np
import glob
import shutil
from datetime import datetime

# Periods to process
PERIODS = [
    ('1963-01-01', '1983-12-31', 'stock_data_1963_1983'),
    ('1983-01-01', '2003-12-31', 'stock_data_1983_2003'),
    ('1990-01-01', '2024-12-31', 'stock_data_1990_2024'),
]

TOP_N = 500  # Select top 500 companies per period

def calculate_company_score(df, ticker, start_date, end_date):
    """
    Calculate company quality score based on available data:
    - Dollar volume (price * volume) - proxy for size/liquidity
    - Trading volume - liquidity
    - Price level - avoid penny stocks
    - Data completeness - survival through period
    """

    # Filter to period
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    period_df = df[(df.index >= start) & (df.index <= end)]

    if len(period_df) < 100:
        return None

    # Calculate metrics
    try:
        # 1. Dollar volume (price * volume) - MAIN METRIC
        # This is our proxy for company size when market cap unavailable
        dollar_volume = (period_df['close'] * period_df['volume']).mean()

        # 2. Average daily volume (liquidity)
        avg_volume = period_df['volume'].mean()

        # 3. Average price (avoid extreme penny stocks)
        avg_price = period_df['close'].mean()

        # Filter out extreme penny stocks (< $0.10 average) - more lenient
        if avg_price < 0.10:
            return None

        # 4. Data completeness (percentage of period with data)
        expected_days = (end - start).days
        actual_days = len(period_df)
        completeness = actual_days / expected_days if expected_days > 0 else 0

        # 5. Data quality (no extreme zeros or NaNs)
        valid_data_ratio = (~period_df['close'].isna()).sum() / len(period_df)
        non_zero_ratio = (period_df['close'] > 0).sum() / len(period_df)

        # Require at least 90% valid data - more lenient to get more stocks
        if valid_data_ratio < 0.90 or non_zero_ratio < 0.90:
            return None

        # 6. Volatility (standard deviation of returns)
        returns = period_df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        score = {
            'ticker': ticker,
            'dollar_volume': dollar_volume,
            'avg_volume': avg_volume,
            'avg_price': avg_price,
            'completeness': completeness,
            'valid_data_ratio': valid_data_ratio,
            'volatility': volatility,
            'trading_days': len(period_df),
            'start_date': period_df.index[0],
            'end_date': period_df.index[-1],
        }

        # Composite score: weighted combination
        # Dollar volume is most important (represents size + liquidity)
        composite_score = (
            np.log1p(dollar_volume) * 0.60 +   # Dollar volume (60%) - main metric
            np.log1p(avg_volume) * 0.20 +      # Volume (20%)
            completeness * 100 * 0.15 +        # Data completeness (15%)
            np.log1p(avg_price) * 0.05         # Price level (5%)
        )

        score['composite_score'] = composite_score

        return score

    except Exception as e:
        print(f"  ⚠️  {ticker}: Error calculating score - {e}")
        return None

def filter_top_companies(source_dir, output_dir, start_date, end_date, top_n=500):
    """Filter to top N companies based on composite score"""

    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(source_dir)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Target: Top {min(top_n, 9999)} companies (or all if fewer)")
    print(f"{'='*80}")

    # Get all stock files
    stock_files = glob.glob(f"{source_dir}/*.csv")
    stock_files = [f for f in stock_files if not f.endswith('VIX.csv') and not f.endswith('stock_list.txt')]

    print(f"\nAnalyzing {len(stock_files)} stocks...")

    # Calculate scores for all stocks
    scores = []
    filtered_out = []

    for i, stock_file in enumerate(stock_files):
        ticker = os.path.basename(stock_file).replace('.csv', '')

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(stock_files)} stocks...")

        try:
            df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
            df.columns = [col.lower() for col in df.columns]

            score = calculate_company_score(df, ticker, start_date, end_date)
            if score is not None:
                scores.append(score)
            else:
                filtered_out.append(ticker)

        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
            filtered_out.append(ticker)

    print(f"\n  Total analyzed: {len(stock_files)}")
    print(f"  Valid stocks:   {len(scores)}")
    print(f"  Filtered out:   {len(filtered_out)} (penny stocks, bad data)")

    # Convert to DataFrame and sort by composite score
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('composite_score', ascending=False)

    # Select top N (or all if fewer than N)
    actual_top_n = min(top_n, len(scores_df))
    top_stocks = scores_df.head(actual_top_n)

    print(f"\n{'='*80}")
    print(f"TOP {len(top_stocks)} COMPANIES SELECTED")
    print(f"{'='*80}")
    print(f"\nTop 10 by composite score:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Ticker':<8} {'Score':<10} {'Avg Price':<12} {'Dollar Vol':<18} {'Avg Volume':<15}")
    print("-" * 80)
    for rank, (i, row) in enumerate(top_stocks.head(10).iterrows(), 1):
        print(f"{rank:<6} {row['ticker']:<8} {row['composite_score']:<10.1f} "
              f"${row['avg_price']:<11.2f} ${row['dollar_volume']:>16,.0f}  "
              f"{row['avg_volume']:>14,.0f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Copy selected stocks
    copied = 0
    for ticker in top_stocks['ticker'].values:
        source_file = f"{source_dir}/{ticker}.csv"
        dest_file = f"{output_dir}/{ticker}.csv"

        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            copied += 1

    # Copy VIX data
    vix_source = f"{source_dir}/VIX.csv"
    if os.path.exists(vix_source):
        shutil.copy2(vix_source, f"{output_dir}/VIX.csv")
        print(f"\n✅ VIX data copied")

    # Save selection report
    report_file = f"{output_dir}/top{actual_top_n}_selection.csv"
    top_stocks.to_csv(report_file, index=False)

    # Save summary
    summary_file = f"{output_dir}/selection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Top {actual_top_n} Companies Selection Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Source: {source_dir}\n")
        f.write(f"Total candidates: {len(stock_files)}\n")
        f.write(f"Valid stocks: {len(scores_df)}\n")
        f.write(f"Filtered out: {len(filtered_out)} (penny stocks < $1, bad data)\n")
        f.write(f"Selected: {len(top_stocks)}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Selection Criteria (weighted):\n")
        f.write(f"  - Dollar volume (price * volume): 60% [proxy for size]\n")
        f.write(f"  - Trading volume (liquidity): 20%\n")
        f.write(f"  - Data completeness: 15%\n")
        f.write(f"  - Price level (> $1 required): 5%\n\n")

        f.write(f"Filters Applied:\n")
        f.write(f"  - Minimum price: $0.10 (exclude extreme penny stocks)\n")
        f.write(f"  - Minimum data quality: 90% valid data\n")
        f.write(f"  - Minimum trading days: 100 days\n\n")

        f.write(f"Top 50 Companies:\n")
        f.write(f"{'-'*100}\n")
        f.write(f"{'Rank':<6} {'Ticker':<8} {'Score':<10} {'Avg Price':<12} {'Dollar Volume':<20} {'Avg Volume':<15} {'Days':<8}\n")
        f.write(f"{'-'*100}\n")

        for rank, (i, row) in enumerate(top_stocks.head(50).iterrows(), 1):
            f.write(f"{rank:<6} {row['ticker']:<8} {row['composite_score']:<10.1f} "
                   f"${row['avg_price']:<11.2f} ${row['dollar_volume']:<19,.0f} "
                   f"{row['avg_volume']:<15,.0f} {row['trading_days']:<8}\n")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Stocks copied:     {copied}")
    print(f"Output directory:  {output_dir}")
    print(f"Selection report:  {report_file}")
    print(f"Summary:           {summary_file}")
    print(f"{'='*80}\n")

    return len(top_stocks)

def main():
    """Process all periods"""

    print("="*80)
    print("FILTERING TO TOP 500 COMPANIES PER PERIOD")
    print("="*80)
    print(f"\nSimulates S&P 500 selection for historical periods.")
    print(f"Selection based on: dollar volume (price*vol), volume, liquidity, quality")
    print(f"\nNote: Dollar volume (price * volume) is used as proxy for company size")
    print(f"      since historical market cap data is not available.\n")

    results = {}

    for start_date, end_date, folder_name in PERIODS:
        source_dir = f"sp500_data/{folder_name}"
        output_dir = f"sp500_data/{folder_name}_top500"

        if not os.path.exists(source_dir):
            print(f"\n⚠️  Skipping {folder_name} - source not found")
            continue

        count = filter_top_companies(source_dir, output_dir, start_date, end_date, TOP_N)
        results[folder_name] = count

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - TOP 500 DATASETS CREATED")
    print("="*80)
    print(f"{'Dataset':<40} {'Original':<12} {'Top 500':<12} {'Change':<12}")
    print("-"*80)

    for start_date, end_date, folder_name in PERIODS:
        if folder_name in results:
            source_dir = f"sp500_data/{folder_name}"
            original_count = len(glob.glob(f"{source_dir}/*.csv")) - 1  # Exclude VIX
            top500_count = results[folder_name]
            change = "All included" if top500_count == original_count else f"-{original_count - top500_count} stocks"

            output_folder = f"{folder_name}_top500"
            print(f"{output_folder:<40} {original_count:<12} {top500_count:<12} {change:<12}")

    print("="*80)
    print("\n✅ Top 500 filtering complete!")
    print("\nNew datasets created:")
    for start_date, end_date, folder_name in PERIODS:
        if folder_name in results:
            print(f"  - sp500_data/{folder_name}_top500/  ({results[folder_name]} stocks)")

    print("\nThese datasets contain the best companies for each period,")
    print("selected using S&P 500-like criteria (size, volume, quality).")
    print("\nDollar volume (price * volume) is the main selection metric,")
    print("representing both company size and trading liquidity.")

if __name__ == '__main__':
    main()
