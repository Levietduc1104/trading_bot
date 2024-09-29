import pandas as pd

def convert_to_readable_datetime(trades_df):
    trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime'], unit='s', origin='unix').dt.strftime('%Y-%m-%d %H:%M:%S')
    trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'], unit='s', origin='unix').dt.strftime('%Y-%m-%d %H:%M:%S')
    return trades_df
