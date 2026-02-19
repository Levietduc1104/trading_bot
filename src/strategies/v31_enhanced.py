"""
V31 Enhanced: V31 with Multi-Factor Position Sizing
Builds on proven V31 strategy by improving position sizing algorithm

Changes from standard V31:
- Position sizing: Inverse volatility only → Multi-factor (vol + momentum + mean reversion + correlation)
- Min position: 10% → 8% (less fragmentation)
- Max position: 25% → 22% (less concentration risk)

Expected improvement: +1-2% annual return
"""

import numpy as np
import pandas as pd
import os
import logging
from src.backtest.transaction_costs import TransactionCostModel
from src.core.covered_calls_manager import CoveredCallsManager
from src.strategies.enhanced_position_sizing import EnhancedPositionSizer

logger = logging.getLogger(__name__)

class V31EnhancedStrategy:
    def __init__(self, bot, config=None, use_transaction_costs=False, broker='interactive_brokers', enable_covered_calls=False):
        self.bot = bot
        self.config = config or {
            'megacap_allocation': 0.70,
            'num_megacap': 3,
            'num_momentum': 2,
            'trailing_stop': 0.15,
            'max_portfolio_dd': 0.25,
            'vix_crisis': 35,
            'num_top_megacaps': 7,
            'rebalance_frequency': 'quarterly',
        }
        self.initial_capital = bot.initial_capital

        # Enhanced position sizer
        self.position_sizer = EnhancedPositionSizer(config={
            'volatility_weight': 0.40,
            'momentum_weight': 0.30,
            'mean_reversion_weight': 0.20,
            'correlation_weight': 0.10,
            'vol_lookback': 20,
            'momentum_lookback': 60,
            'mean_reversion_lookback': 20,
            'min_position_size': 0.08,
            'max_position_size': 0.22,
        })

        # Trade history tracking
        self.trade_history = []

        # Load real market cap data
        market_cap_path = '../data/market_cap/historical_market_cap_1990_2024.csv'
        if not os.path.exists(market_cap_path):
            market_cap_path = 'data/market_cap/historical_market_cap_1990_2024.csv'

        print(f"Loading real market cap data from {market_cap_path}...")
        self.market_cap_df = pd.read_csv(market_cap_path)
        self.market_cap_df['date'] = pd.to_datetime(self.market_cap_df['date'])

        # Optimize: Create dictionary for fast lookups
        self.market_cap_by_date = {}
        for date, group in self.market_cap_df.groupby('date'):
            self.market_cap_by_date[date] = group

        self.market_cap_symbols = set(self.market_cap_df['symbol'].unique())
        print(f"✅ Loaded {len(self.market_cap_df):,} market cap records ({len(self.market_cap_by_date)} dates, {len(self.market_cap_symbols)} symbols)")

        self.use_transaction_costs = use_transaction_costs
        self.broker = broker
        self.cost_model = TransactionCostModel(broker=broker) if use_transaction_costs else None
        self.total_costs = 0.0
        self.num_rebalances = 0

        # V31: Covered calls integration
        self.enable_covered_calls = enable_covered_calls
        self.covered_calls_manager = CoveredCallsManager(
            quarterly_premium_rate=0.01,
            enabled=enable_covered_calls
        ) if enable_covered_calls else None

        if enable_covered_calls:
            logger.info("✅ V31 Enhanced Covered Calls ENABLED - Target: 4% annual premium")

    def identify_megacaps(self, date, top_n=7):
        """Dynamically identify top N mega-cap stocks using REAL historical market cap"""
        target_date = pd.Timestamp(date)
        available_dates = sorted(self.market_cap_by_date.keys())
        closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))

        date_data = self.market_cap_by_date[closest_date]
        top_stocks = date_data.nlargest(top_n, 'market_cap_billions')
        return top_stocks['symbol'].tolist()

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

    def calculate_trade_cost(self, ticker, shares, price, date):
        """Calculate realistic transaction costs for a trade"""
        if not self.use_transaction_costs or self.cost_model is None:
            return shares * price * 0.001

        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
        if len(df_at_date) < 20:
            return shares * price * 0.001

        recent_volume = df_at_date.tail(20)['volume'].mean()
        market_cap_category = 'large'
        recent_returns = df_at_date.tail(20)['close'].pct_change().dropna()
        daily_vol = recent_returns.std() if len(recent_returns) > 0 else 0.02

        cost_info = self.cost_model.total_execution_cost(
            ticker=ticker,
            shares=abs(shares),
            price=price,
            avg_daily_volume=recent_volume,
            market_cap=market_cap_category,
            volatility=daily_vol,
            order_type='market'
        )

        return cost_info['total_cost']

    def run_backtest(self, start_year=1963, end_year=2024):
        # Use SPY or find stock with longest data range
        if 'SPY' in self.bot.stocks_data:
            reference_ticker = 'SPY'
        else:
            longest_ticker = None
            max_length = 0
            for ticker, df in self.bot.stocks_data.items():
                if len(df) > max_length:
                    max_length = len(df)
                    longest_ticker = ticker
            reference_ticker = longest_ticker

        all_dates = self.bot.stocks_data[reference_ticker].index
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
                        shares = holdings[ticker]['shares']
                        proceeds = shares * current_price
                        cost = self.calculate_trade_cost(ticker, shares, current_price, date)
                        cash += proceeds - cost
                        self.total_costs += cost
                        # Record trade
                        self.trade_history.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'SELL',
                            'reason': 'trailing_stop',
                            'shares': shares,
                            'price': current_price,
                            'value': proceeds,
                            'cost': cost
                        })
                        del holdings[ticker]

            # Quarterly rebalancing
            is_rebalance = (
                last_rebalance is None or
                (date.month in [1, 4, 7, 10] and
                 7 <= date.day <= 15 and
                 (last_rebalance.year != date.year or last_rebalance.month != date.month))
            )

            if is_rebalance:
                last_rebalance = date
                self.num_rebalances += 1

                # Collect covered calls premium
                if self.enable_covered_calls and self.covered_calls_manager:
                    stocks_value = sum(
                        h['shares'] * self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date].iloc[-1]['close']
                        for t, h in holdings.items()
                        if len(self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date]) > 0
                    )
                    portfolio_value = cash + stocks_value

                    premium = self.covered_calls_manager.calculate_premium_backtest(
                        portfolio_value=portfolio_value,
                        current_date=date
                    )
                    cash += premium

                # Liquidate
                for ticker in list(holdings.keys()):
                    df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        shares = holdings[ticker]['shares']
                        price = df_at_date.iloc[-1]['close']
                        proceeds = shares * price
                        cost = self.calculate_trade_cost(ticker, shares, price, date)
                        cash += proceeds - cost
                        self.total_costs += cost
                        # Record trade
                        self.trade_history.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'SELL',
                            'reason': 'rebalance',
                            'shares': shares,
                            'price': price,
                            'value': proceeds,
                            'cost': cost
                        })
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

                # Identify mega-caps
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

                # ENHANCED POSITION SIZING (NEW!)
                megacap_amount = invest_amount * self.config['megacap_allocation']
                momentum_amount = invest_amount * (1 - self.config['megacap_allocation'])

                # Use enhanced position sizing for mega-caps
                if top_megacaps:
                    megacap_allocations = self.position_sizer.calculate_enhanced_positions(
                        top_megacaps, date, self.bot, megacap_amount
                    )

                    for ticker, allocation in megacap_allocations.items():
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) > 0:
                            price = df_at_date.iloc[-1]['close']
                            shares = allocation / price
                            cost = self.calculate_trade_cost(ticker, shares, price, date)
                            holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                            cash -= allocation + cost
                            self.total_costs += cost
                            # Record trade
                            self.trade_history.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'reason': 'rebalance_megacap',
                                'shares': shares,
                                'price': price,
                                'value': allocation,
                                'cost': cost
                            })

                # Use enhanced position sizing for momentum stocks
                if top_momentum:
                    momentum_allocations = self.position_sizer.calculate_enhanced_positions(
                        top_momentum, date, self.bot, momentum_amount
                    )

                    for ticker, allocation in momentum_allocations.items():
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) > 0:
                            price = df_at_date.iloc[-1]['close']
                            shares = allocation / price
                            cost = self.calculate_trade_cost(ticker, shares, price, date)
                            holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                            cash -= allocation + cost
                            self.total_costs += cost
                            # Record trade
                            self.trade_history.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'reason': 'rebalance_momentum',
                                'shares': shares,
                                'price': price,
                                'value': allocation,
                                'cost': cost
                            })

            # Calculate portfolio value
            stocks_value = sum(h['shares'] * self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date].iloc[-1]['close']
                              for t, h in holdings.items() if len(self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date]) > 0)
            portfolio_values.append({'date': date, 'value': cash + stocks_value})

        # Export trade history to CSV
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            os.makedirs('output', exist_ok=True)
            trade_df.to_csv('output/tier2_trades.csv', index=False)
            logger.info(f"✅ Saved {len(trade_df)} trades to: output/tier2_trades.csv")

        return pd.DataFrame(portfolio_values).set_index('date')

    def get_premium_stats(self):
        """Get covered calls premium statistics"""
        if self.covered_calls_manager:
            return self.covered_calls_manager.get_premium_stats()
        return {'total_premium': 0, 'num_collections': 0, 'avg_premium': 0, 'annual_rate': 0}

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
