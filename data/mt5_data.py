# The `MT5Data` class is designed to connect to MetaTrader 5, retrieve historical data for a specified
# symbol and timeframes, and resample the confirmation data to match the primary data index.
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime


class MT5Data:
    def __init__(self, login, password, server, symbol, primary_timeframe, confirm_timeframe, start_date, end_date):
        self.login = login
        self.password = password
        self.server = server
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.confirm_timeframe = confirm_timeframe
        if not self.connecting():
            raise ConnectionError("Failed to connect to MetaTrader 5.")
        
        self.primary_data = self.request_historical_data(self.primary_timeframe, start_date=start_date, end_date=end_date)
        self.confirm_data = self.request_historical_data(self.confirm_timeframe, start_date=start_date, end_date=end_date)
        self.resample_confirmation_data()

    def connecting(self):
        if not mt5.initialize():
            print("Failed to initialize connection to MetaTrader 5.")
            return False
        if not mt5.login(self.login, self.password, self.server):
            print("Failed to log in to MetaTrader account.")
            return False
        print("Connected to MetaTrader 5 successfully.")
        return True

    def request_historical_data(self, timeframe, start_date=None, end_date=None):
        utc_from = pd.to_datetime(start_date, utc=True)
        utc_now = pd.to_datetime(end_date, utc=True)
        
        rates = mt5.copy_rates_range(self.symbol, timeframe, utc_from, utc_now)
        if rates is None or len(rates) == 0:
            print("Failed to retrieve historical data or no data available.")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.index = df.index.tz_localize(None)  # Remove timezone information
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
        
        return df
    
    def resample_confirmation_data(self):
        self.confirm_data = self.confirm_data.reindex(self.primary_data.index, method='ffill')

