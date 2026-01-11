"""
Split all_stock_data.csv into individual stock files

This script reads the large CSV file and creates separate files for each ticker
in the sp500_data/individual_stocks folder.
"""

import pandas as pd
import os
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def split_stock_data():
    """Split the combined CSV into individual stock files"""

    input_file = 'sp500_data/all_stock_data.csv'
    output_dir = 'sp500_data/individual_stocks'

    logger.info(f"Reading {input_file}...")
    logger.info("This may take a few minutes for a 3.5GB file...")

    # Track progress
    ticker_data = defaultdict(list)
    chunk_size = 100000  # Process 100k rows at a time
    total_rows = 0

    # Read in chunks to manage memory
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size), 1):
        total_rows += len(chunk)

        # Group by ticker in this chunk
        for ticker, group in chunk.groupby('Ticker'):
            ticker_data[ticker].append(group)

        if chunk_num % 10 == 0:
            logger.info(f"Processed {total_rows:,} rows, found {len(ticker_data)} unique tickers so far...")

    logger.info(f"Finished reading {total_rows:,} rows")
    logger.info(f"Found {len(ticker_data)} unique tickers")

    # Write each ticker to its own file
    logger.info(f"Writing individual stock files to {output_dir}...")

    for ticker_num, (ticker, df_list) in enumerate(ticker_data.items(), 1):
        # Combine all chunks for this ticker
        ticker_df = pd.concat(df_list, ignore_index=True)

        # Sort by date
        ticker_df = ticker_df.sort_values('Date')

        # Set Date as index
        ticker_df = ticker_df.set_index('Date')

        # Remove the Ticker column (redundant in individual files)
        ticker_df = ticker_df.drop('Ticker', axis=1)

        # Save to CSV
        output_file = os.path.join(output_dir, f'{ticker}.csv')
        ticker_df.to_csv(output_file)

        if ticker_num % 100 == 0:
            logger.info(f"Written {ticker_num}/{len(ticker_data)} files...")

    logger.info(f"✓ Successfully split into {len(ticker_data)} individual stock files")
    logger.info(f"✓ Files saved to: {output_dir}")

    # Print sample of tickers
    sample_tickers = list(ticker_data.keys())[:10]
    logger.info(f"Sample tickers: {', '.join(sample_tickers)}")

if __name__ == '__main__':
    split_stock_data()
