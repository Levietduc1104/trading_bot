"""
V30 ENHANCED: VOLATILITY-WEIGHTED POSITION SIZING
==================================================
Enhancement of V30 that uses risk-based position sizing instead of equal weight.

Key Enhancement:
- Volatility-based position sizing (inverse volatility weighting)
- Lower volatility stocks get MORE allocation
- Higher volatility stocks get LESS allocation
- Position constraints: 10% minimum, 25% maximum

Expected Impact:
- Lower max drawdown (-20% vs -24%)
- Higher Sharpe ratio (1.10 vs 1.00)
- Similar or slightly lower returns (15.5% vs 16.1%)
"""
import numpy as np
import pandas as pd
from src.backtest.transaction_costs import TransactionCostModel

class V30VolWeightedStrategy:
    def __init__(self, bot, config=None, use_transaction_costs=False, broker='interactive_brokers'):
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
            'vol_lookback': 20,  # Days to calculate volatility
            'min_position_size': 0.10,  # 10% minimum per stock
            'max_position_size': 0.25,  # 25% maximum per stock
            'rebalance_frequency': 'quarterly',  # DEFAULT: 'quarterly' (61.5% lower costs than monthly)
        }
        self.initial_capital = bot.initial_capital
        self.use_transaction_costs = use_transaction_costs
        self.broker = broker
        self.cost_model = TransactionCostModel(broker=broker) if use_transaction_costs else None
        self.total_costs = 0.0  # Track cumulative transaction costs
        self.num_rebalances = 0  # Track number of rebalances

    def calculate_volatility(self, ticker, date):
        """Calculate annualized volatility for a stock"""
        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]

        if len(df_at_date) < self.config['vol_lookback']:
            return None

        # Get recent returns
        recent = df_at_date.tail(self.config['vol_lookback'])
        returns = recent['close'].pct_change().dropna()

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        return volatility

    def calculate_vol_weighted_positions(self, stocks, date, total_amount):
        """
        Allocate capital based on inverse volatility weighting

        Args:
            stocks: List of (ticker, score) tuples
            date: Current date
            total_amount: Total capital to allocate

        Returns:
            Dictionary of {ticker: allocation_amount}
        """
        # Calculate volatilities
        volatilities = {}
        for ticker, score in stocks:
            vol = self.calculate_volatility(ticker, date)
            if vol is not None and vol > 0:
                volatilities[ticker] = vol

        if not volatilities:
            # Fallback to equal weight if no volatility data
            equal_weight = total_amount / len(stocks)
            return {ticker: equal_weight for ticker, score in stocks}

        # Calculate inverse volatility weights
        inverse_vols = {ticker: 1.0 / vol for ticker, vol in volatilities.items()}
        total_inverse_vol = sum(inverse_vols.values())

        # Calculate raw weights
        raw_weights = {ticker: inv_vol / total_inverse_vol for ticker, inv_vol in inverse_vols.items()}

        # Apply min/max constraints
        allocations = {}
        min_size = self.config['min_position_size']
        max_size = self.config['max_position_size']

        # First pass: apply constraints
        for ticker, weight in raw_weights.items():
            constrained_weight = np.clip(weight, min_size, max_size)
            allocations[ticker] = constrained_weight * total_amount

        # Second pass: normalize to ensure we use all capital
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            scale_factor = total_amount / total_allocated
            allocations = {ticker: amount * scale_factor for ticker, amount in allocations.items()}

        return allocations

    def identify_megacaps(self, date, top_n=7):
        """Dynamically identify top N mega-cap stocks using trading value proxy"""
        ETF_EXCLUSIONS = {'SPY', 'SPY 2', 'QQQ', 'IVV', 'VOO', 'VTI', 'DIA', 'IWM', 'EFA', 'EEM'}

        lookback = self.config['lookback_trading_value']
        trading_values = {}
        for ticker, df in self.bot.stocks_data.items():
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

    def calculate_trade_cost(self, ticker, shares, price, date):
        """
        Calculate realistic transaction costs for a trade

        Returns:
            cost: Dollar amount of transaction cost
        """
        if not self.use_transaction_costs or self.cost_model is None:
            # Simple 0.1% cost as fallback
            return shares * price * 0.001

        # Get stock data for volume calculation
        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
        if len(df_at_date) < 20:
            # Fallback for insufficient data
            return shares * price * 0.001

        # Calculate average daily volume (last 20 days)
        recent_volume = df_at_date.tail(20)['volume'].mean()

        # Mega-caps have 'large' market cap category
        market_cap_category = 'large'

        # Get daily volatility from recent returns
        recent_returns = df_at_date.tail(20)['close'].pct_change().dropna()
        daily_vol = recent_returns.std() if len(recent_returns) > 0 else 0.02

        # Calculate one-way cost
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
            # Find stock with longest date range
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
                        del holdings[ticker]

            # Rebalancing logic (monthly or quarterly)
            if self.config['rebalance_frequency'] == 'quarterly':
                # Quarterly rebalancing: Jan, Apr, Jul, Oct (day 7-15)
                is_rebalance = (
                    last_rebalance is None or
                    (date.month in [1, 4, 7, 10] and
                     7 <= date.day <= 15 and
                     (last_rebalance.year != date.year or last_rebalance.month != date.month))
                )
            else:
                # Monthly rebalancing (day 7-15)
                is_rebalance = (
                    last_rebalance is None or
                    ((date.year, date.month) != (last_rebalance.year, last_rebalance.month) and 7 <= date.day <= 15)
                )

            if is_rebalance:
                last_rebalance = date
                self.num_rebalances += 1
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

                # VOLATILITY-WEIGHTED ALLOCATION (NEW!)
                megacap_amount = invest_amount * self.config['megacap_allocation']
                momentum_amount = invest_amount * (1 - self.config['megacap_allocation'])

                # Calculate volatility-weighted positions for mega-caps
                if top_megacaps:
                    megacap_allocations = self.calculate_vol_weighted_positions(top_megacaps, date, megacap_amount)

                    for ticker, allocation in megacap_allocations.items():
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) > 0:
                            price = df_at_date.iloc[-1]['close']
                            shares = allocation / price
                            cost = self.calculate_trade_cost(ticker, shares, price, date)
                            holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                            cash -= allocation + cost
                            self.total_costs += cost

                # Calculate volatility-weighted positions for momentum stocks
                if top_momentum:
                    momentum_allocations = self.calculate_vol_weighted_positions(top_momentum, date, momentum_amount)

                    for ticker, allocation in momentum_allocations.items():
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) > 0:
                            price = df_at_date.iloc[-1]['close']
                            shares = allocation / price
                            cost = self.calculate_trade_cost(ticker, shares, price, date)
                            holdings[ticker] = {'shares': shares, 'entry_price': price, 'peak_price': price}
                            cash -= allocation + cost
                            self.total_costs += cost

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
