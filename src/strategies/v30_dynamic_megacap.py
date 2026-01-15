"""
V30: DYNAMIC MEGA-CAP SPLIT STRATEGY
=====================================
Evolution of V29 that works across ALL time periods by dynamically
identifying mega-cap stocks using trading value as proxy for market cap.

Strategy:
- 70% allocation to Top 3 Dynamic Mega-Caps (by momentum)
- 30% allocation to Top 2 Momentum stocks (non mega-caps)
- VIX-based cash reserves (up to 70% in crisis)
- 15% trailing stop losses
"""
import numpy as np
import pandas as pd

class V30Strategy:
    def __init__(self, bot, config=None):
        self.bot = bot
        self.config = config or {
            'megacap_allocation': 0.70,
            'num_megacap': 3,
            'num_momentum': 2,
            'trailing_stop': 0.15,
            'max_portfolio_dd': 0.25,
            'vix_crisis': 35,
            'num_top_megacaps': 7,
            'lookback_trading_value': 20,
        }
        self.initial_capital = bot.initial_capital

    def identify_megacaps(self, date, top_n=7):
        """Dynamically identify top N mega-cap stocks using trading value proxy"""
        # ETFs to exclude (SPY, QQQ, etc.)
        ETF_EXCLUSIONS = {'SPY', 'SPY 2', 'QQQ', 'IVV', 'VOO', 'VTI', 'DIA', 'IWM', 'EFA', 'EEM'}
        
        lookback = self.config['lookback_trading_value']
        trading_values = {}
        for ticker, df in self.bot.stocks_data.items():
            # Skip ETFs
            if ticker in ETF_EXCLUSIONS:
                continue
            
            df_at_date = df[df.index <= date]
            if len(df_at_date) >= lookback:
                recent = df_at_date.tail(lookback)
                avg_trading_value = (recent['close'] * recent['volume']).mean()
                trading_values[ticker] = avg_trading_value
        sorted_stocks = sorted(trading_values.items(), key=lambda x: x[1], reverse=True)
        return [ticker for ticker, _ in sorted_stocks[:top_n]]

    def get_vix_cash_reserve(self, vix):
        if vix < 15: return 0.05
        elif vix < 20: return 0.10
        elif vix < 25: return 0.20
        elif vix < 30: return 0.35
        elif vix < self.config['vix_crisis']: return 0.50
        else: return 0.70

    def get_portfolio_dd_multiplier(self, portfolio_df):
        if portfolio_df is None or len(portfolio_df) < 2:
            return 1.0
        peak = portfolio_df['value'].cummax().iloc[-1]
        current = portfolio_df['value'].iloc[-1]
        dd = (current - peak) / peak
        if dd > -0.05: return 1.0
        elif dd > -0.10: return 0.90
        elif dd > -0.15: return 0.75
        elif dd > -0.20: return 0.50
        else: return 0.25

    def check_trailing_stop(self, ticker, current_price, holdings):
        if ticker not in holdings:
            return False
        peak_price = holdings[ticker].get('peak_price', current_price)
        stop_price = peak_price * (1 - self.config['trailing_stop'])
        return current_price < stop_price

    def run_backtest(self, start_year=1963, end_year=2024):
        first_ticker = list(self.bot.stocks_data.keys())[0]
        all_dates = self.bot.stocks_data[first_ticker].index
        all_dates = all_dates[(all_dates >= f'{start_year}-01-01') & (all_dates <= f'{end_year}-12-31')]

        portfolio_values = []
        holdings = {}
        cash = self.initial_capital
        last_rebalance = None

        for date in all_dates:
            # Update peak prices and check trailing stops
            for ticker in list(holdings.keys()):
                df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    holdings[ticker]['peak_price'] = max(holdings[ticker].get('peak_price', current_price), current_price)
                    if self.check_trailing_stop(ticker, current_price, holdings):
                        cash += holdings[ticker]['shares'] * current_price
                        del holdings[ticker]

            # Monthly rebalancing (day 7-15)
            is_rebalance = (last_rebalance is None or 
                ((date.year, date.month) != (last_rebalance.year, last_rebalance.month) and 7 <= date.day <= 15))

            if is_rebalance:
                last_rebalance = date
                # Liquidate
                for ticker in list(holdings.keys()):
                    df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        cash += holdings[ticker]['shares'] * df_at_date.iloc[-1]['close']
                holdings = {}

                # VIX cash reserve
                vix = 20
                if self.bot.vix_data is not None:
                    vix_at_date = self.bot.vix_data[self.bot.vix_data.index <= date]
                    if len(vix_at_date) > 0:
                        vix = vix_at_date.iloc[-1]['close']

                cash_reserve = self.get_vix_cash_reserve(vix)
                portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
                dd_multiplier = self.get_portfolio_dd_multiplier(portfolio_df)
                invest_amount = cash * (1 - cash_reserve) * dd_multiplier

                # DYNAMIC MEGA-CAP IDENTIFICATION
                megacaps = self.identify_megacaps(date, self.config['num_top_megacaps'])

                # Score mega-caps by momentum
                megacap_scores = {}
                for ticker in megacaps:
                    if ticker in self.bot.stocks_data:
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) >= 20:
                            mom20 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1) * 100
                            megacap_scores[ticker] = mom20

                top_megacaps = sorted(megacap_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['num_megacap']]

                # Score non-mega-cap stocks
                momentum_scores = {}
                for ticker, df in self.bot.stocks_data.items():
                    if ticker in megacaps:
                        continue
                    df_at_date = df[df.index <= date]
                    if len(df_at_date) >= 100:
                        try:
                            momentum_scores[ticker] = self.bot.score_stock(ticker, df_at_date)
                        except:
                            pass

                top_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['num_momentum']]

                # Allocate
                megacap_amount = invest_amount * self.config['megacap_allocation']
                momentum_amount = invest_amount * (1 - self.config['megacap_allocation'])

                # Buy mega-caps
                if top_megacaps:
                    per_megacap = megacap_amount / len(top_megacaps)
                    for ticker, score in top_megacaps:
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) > 0:
                            price = df_at_date.iloc[-1]['close']
                            shares = per_megacap / price
                            holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                            cash -= per_megacap * 1.001

                # Buy momentum stocks
                if top_momentum:
                    per_mom = momentum_amount / len(top_momentum)
                    for ticker, score in top_momentum:
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) > 0:
                            price = df_at_date.iloc[-1]['close']
                            shares = per_mom / price
                            holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                            cash -= per_mom * 1.001

            # Calculate portfolio value
            stocks_value = sum(h['shares'] * self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date].iloc[-1]['close'] 
                              for t, h in holdings.items() if len(self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date]) > 0)
            portfolio_values.append({'date': date, 'value': cash + stocks_value})

        return pd.DataFrame(portfolio_values).set_index('date')

def calculate_metrics(portfolio_df, initial_capital):
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    returns = portfolio_df['value'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    return {'final_value': final_value, 'total_return': total_return, 'annual_return': annual_return,
            'max_drawdown': max_drawdown, 'sharpe': sharpe, 'years': years}
