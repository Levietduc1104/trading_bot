# from strategies.trend_detector import TrendDetector
import pandas as pd
from backtesting import Strategy, Backtest
import numpy as np
from datetime import time
 

class MultiTimeFrameStrategy(Strategy):
    fast_ema_period: int = 13
    medium_ema_period: int = 34
    slow_ema_period: int = 89
    atr_len: int = 5
    adx_len: int = 5
    risk_percent: int = 0.01  # Risk 1% of account balance per trade
    pip_value: int = 10  # Pip value for XAU/USD (standard lot size)
    trailing_stop_distance: int = 2  # Distance for trailing stop in pips
    last_trade_type = None

    def __init__(self, broker, primary_data, confirm_data, *args, **kwargs):
        super().__init__(broker, *args, **kwargs)
        self.primary_data = primary_data
        self.confirm_data = confirm_data
        self.stop_loss_level = None  # To track stop-loss manually
        self.trend_detector = TrendDetector()
        self.last_trade_type = None

    def init(self):
        self.fast_ema_primary = self.I(self.ema, self.primary_data['Close'], self.fast_ema_period)
        self.medium_ema_primary = self.I(self.ema, self.primary_data['Close'], self.medium_ema_period)
        self.slow_ema_primary = self.I(self.ema, self.primary_data['Close'], self.slow_ema_period)
        self.rsi_primary = self.I(self.rsi, self.primary_data['Close'], period=14)

        # Registering ATR and ADX calculation for efficiency
        self.atr = self.I(self.trend_detector.atr,
                          self.primary_data['High'],
                          self.primary_data['Low'],
                          self.primary_data['Close'])

        self.di_plus, self.di_minus, self.adx = self.I(self.trend_detector.adx,
                                                       self.primary_data['High'],
                                                       self.primary_data['Low'],
                                                       self.primary_data['Close'])
        # Initialize the last trade type flag to None
        self.last_trade_type = None

    def rsi(self, data, period=14):
        """
        Calculates the Relative Strength Index (RSI).
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss  # Relative Strength
        rsi = 100 - (100 / (1 + rs))  # RSI calculation

        return rsi.to_numpy()

    def ema(self, data, period):
        return pd.Series(data).ewm(span=period, adjust=False).mean().to_numpy()

    def calculate_lot_size(self, sl_pips):
        account_balance = self._broker.equity  # Access the current equity balance
        risk_amount = account_balance * self.risk_percent
        lot_size = risk_amount / (sl_pips * self.pip_value)
        return lot_size

    def update_trailing_stop(self, is_long):
        """
        Update trailing stop dynamically based on the current price.
        For a long position, move the stop loss up as the price increases.
        For a short position, move the stop loss down as the price decreases.
        """
        current_price = self.data.Close[-1]

        if is_long:
            # For long positions, move stop loss up as the price moves up
            if self.stop_loss_level is None:
                self.stop_loss_level = current_price - self.trailing_stop_distance
            else:
                # Move the stop loss up but never lower it
                self.stop_loss_level = max(self.stop_loss_level, current_price - self.trailing_stop_distance)
        else:
            # For short positions, move stop loss down as the price moves down
            if self.stop_loss_level is None:
                self.stop_loss_level = current_price + self.trailing_stop_distance
            else:
                # Move the stop loss down but never raise it
                self.stop_loss_level = min(self.stop_loss_level, current_price + self.trailing_stop_distance)

    def check_stop_loss(self):
        if self.position and self.stop_loss_level is not None:
            if self.position.is_long and self.data.Close[-1] <= self.stop_loss_level:
                self.position.close()
                print("Position closed by trailing stop-loss.")
            elif self.position.is_short and self.data.Close[-1] >= self.stop_loss_level:
                self.position.close()
                print("Position closed by trailing stop-loss.")
    def next(self):
        current_time = self.data.index[-1]

        # Close all positions at 11 PM on Friday
        if current_time.weekday() == 4 and current_time.time() >= time(17, 0):
            if self.position:
                self.position.close()
                print("Closed all positions on Friday at 11 PM.")
            return

        # Fetch the current EMAs
        current_fast_ema_primary = self.fast_ema_primary[-1]
        current_medium_ema_primary = self.medium_ema_primary[-1]
        current_slow_ema_primary = self.slow_ema_primary[-1]
        current_price = self.data.Close[-1]

        # Fetch current ATR and ADX values
        current_atr = self.atr[-1]
        current_adx = self.adx[-1]

        # Calculate ATR moving average
        atr_ma = pd.Series(self.atr).rolling(window=self.atr_len).mean().iloc[-1]

        # Detect a sideways market
        sideways = (current_atr <= atr_ma) or (current_adx <= 20)
        if sideways:
            print(f"{current_time} Sideways market detected: ATR={current_atr}, ADX={current_adx}")
            return

        sl_pips = 20  # Initial stop-loss in pips
        lot_size = self.calculate_lot_size(sl_pips)

        # Ensure we close positions based on EMA crossovers
        if self.position.is_long and (current_fast_ema_primary < current_medium_ema_primary):
            self.position.close()
            self.last_trade_type = "long"  # Set the last trade type as 'long'
            print(f"{current_time} Exited long position due to EMA crossover.")

        elif self.position.is_short and current_fast_ema_primary >= current_medium_ema_primary:
            self.position.close()
            self.last_trade_type = "short"  # Set the last trade type as 'short'
            print(f"{current_time} Exited short position due to EMA crossover.")

        # Update trailing stop-loss
        if self.position:
            self.check_stop_loss()

        # BUY Condition: Only allow buy if the last trade was not a buy
        if self.last_trade_type != "long" and 25 < current_adx < 70 and current_medium_ema_primary > current_slow_ema_primary and \
                current_fast_ema_primary > current_medium_ema_primary:
            if not (0.5 <= abs(current_fast_ema_primary - current_medium_ema_primary) and
                    0.5 <= abs(current_medium_ema_primary - current_slow_ema_primary)):
                print(f"{current_time} - EMA distance condition not met. No trade.")
                return

            if self.position and self.position.is_short:
                self.position.close()
                print("Exited short position to enter long position.")

            if not self.position or self.position.is_short:
                self.buy(sl=self.data.Close[-1] - 4, tp=self.data.Close[-1] + 5, size=lot_size)
                self.last_trade_type = "long"
                print(f"{current_time} Opened long position with lot size: {lot_size:.2f}")

        # SELL Condition: Only allow sell if the last trade was not a sell
        elif self.last_trade_type != "short" and current_medium_ema_primary < current_slow_ema_primary and \
                current_fast_ema_primary < current_medium_ema_primary:
            if not (0.35 <= abs(current_fast_ema_primary - current_medium_ema_primary) <= 2 and
                    0.35 <= abs(current_medium_ema_primary - current_slow_ema_primary) <= 2):
                print(f"{current_time} - EMA distance condition not met. No trade.")
                return

            if self.position and self.position.is_long:
                self.position.close()
                print("Exited long position to enter short position.")

            if not self.position or self.position.is_long:
                self.sell(sl=self.data.Close[-1] + 4, tp=self.data.Close[-1] - 5, size=lot_size)
                self.last_trade_type = "short"
                print(f"{current_time} Opened short position with lot size: {lot_size:.2f}")


class TrendDetector:
    def __init__(self, atr_len=10, adx_len=14):
        self.atr_len = atr_len
        self.adx_len = adx_len

    # Calculate Average True Range (ATR)
    def atr(self, high, low, close):
        tr1 = high - low  # True range 1
        tr2 = abs(high - close.shift(1))  # True range 2
        tr3 = abs(low - close.shift(1))  # True range 3
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_len).mean()  # Rolling mean of True Range
        return atr

    # Calculate Average Directional Index (ADX)
    def adx(self, high, low, close):
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)

        tr = pd.Series(tr, index=high.index)
        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=high.index)

        atr = tr.rolling(window=self.adx_len).mean()
        dm_plus_smoothed = dm_plus.ewm(span=self.adx_len, min_periods=self.adx_len).mean()
        dm_minus_smoothed = dm_minus.ewm(span=self.adx_len, min_periods=self.adx_len).mean()

        di_plus = 100 * dm_plus_smoothed / atr
        di_minus = 100 * dm_minus_smoothed / atr
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=self.adx_len).mean()

        return di_plus, di_minus, adx
    
        # AlphaTrend Calculation (based on ATR and RSI/MFI logic)
    def alpha_trend(self, high, low, close, rsi_period=14):
        # Calculate ATR
        atr = self.atr(high, low, close)
        
        # Calculate upper and lower bands
        upT = low - atr * self.alpha_multiplier
        downT = high + atr * self.alpha_multiplier
        
        # Calculate trend based on RSI or other indicator
        rsi = ta.rsi(close, rsi_period)  # Using RSI, can replace with MFI
        AlphaTrend = pd.Series(np.zeros(len(close)), index=close.index)  # Initialize AlphaTrend as zeros
        
        # Apply logic based on the AlphaTrend script
        for i in range(1, len(close)):
            if rsi[i] >= 50:
                AlphaTrend[i] = max(AlphaTrend[i-1], upT[i])
            else:
                AlphaTrend[i] = min(AlphaTrend[i-1], downT[i])
        return AlphaTrend

    # Detect crossovers for buy and sell signals
    def detect_signals(self, alpha_trend):
        buy_signal = (alpha_trend > alpha_trend.shift(2)) & (alpha_trend.shift(1) <= alpha_trend.shift(3))
        sell_signal = (alpha_trend < alpha_trend.shift(2)) & (alpha_trend.shift(1) >= alpha_trend.shift(3))
        return buy_signal, sell_signal