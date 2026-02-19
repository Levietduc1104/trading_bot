"""
V31 Tier 2: FA-Enhanced Growth Scoring Strategy (Simplified)

Overrides V31's stock scoring to combine momentum with growth potential.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from .v31_enhanced import V31EnhancedStrategy
from ..data.historical_fa_data_loader import HistoricalFADataLoader

logger = logging.getLogger(__name__)


class V31Tier2GrowthScoringStrategy(V31EnhancedStrategy):
    """
    V31 + FA-Enhanced Growth Scoring

    Modification: Instead of pure momentum scoring, use 70% momentum + 30% growth potential
    """

    def __init__(self, bot, use_transaction_costs=True, broker='interactive_brokers',
                 enable_covered_calls=True, enable_growth_scoring=True,
                 momentum_weight=0.70, growth_weight=0.30):
        super().__init__(bot=bot,
                        use_transaction_costs=use_transaction_costs,
                        broker=broker,
                        enable_covered_calls=enable_covered_calls)

        self.enable_growth_scoring = enable_growth_scoring
        self.momentum_weight = momentum_weight
        self.growth_weight = growth_weight

        # Load historical FA data
        if enable_growth_scoring:
            self.fa_loader = HistoricalFADataLoader()
            self.fa_loader.load_all()
            logger.info("✅ Historical FA Growth Scoring ENABLED (Tier 2)")
            logger.info(f"   Momentum Weight: {momentum_weight*100:.0f}%")
            logger.info(f"   Growth Score Weight: {growth_weight*100:.0f}%")

    def calculate_growth_potential_score(self, fa_data):
        """
        Calculate growth potential score (0-100) based on fundamental metrics
        """
        if not fa_data:
            return 50  # Neutral score

        score = 0

        # 1. Profitability Efficiency (25 points)
        roe = fa_data.get('returnOnEquity', 0)
        if roe > 0.25:
            score += 25
        elif roe > 0.20:
            score += 20
        elif roe > 0.15:
            score += 15
        elif roe > 0.10:
            score += 10
        elif roe > 0.05:
            score += 5

        # 2. Margin Quality (20 points)
        operating_margin = fa_data.get('operatingProfitMargin', 0)
        net_margin = fa_data.get('netProfitMargin', 0)

        if operating_margin > 0.20 and net_margin > 0.15:
            score += 20
        elif operating_margin > 0.15 and net_margin > 0.10:
            score += 15
        elif operating_margin > 0.10 and net_margin > 0.05:
            score += 10
        elif operating_margin > 0.05 and net_margin > 0:
            score += 5

        # 3. Cash Generation (20 points)
        fcf_yield = fa_data.get('freeCashFlowYield', 0)
        income_quality = fa_data.get('incomeQuality', 0)

        if fcf_yield > 0.10:
            score += 12
        elif fcf_yield > 0.07:
            score += 9
        elif fcf_yield > 0.05:
            score += 6
        elif fcf_yield > 0.03:
            score += 3

        if income_quality > 1.0:
            score += 8
        elif income_quality > 0.8:
            score += 5
        elif income_quality > 0.5:
            score += 3

        # 4. Capital Efficiency (20 points)
        roic = fa_data.get('returnOnInvestedCapital', 0)
        roce = fa_data.get('returnOnCapitalEmployed', 0)

        if roic > 0.20 or roce > 0.20:
            score += 20
        elif roic > 0.15 or roce > 0.15:
            score += 15
        elif roic > 0.10 or roce > 0.10:
            score += 10
        elif roic > 0.05 or roce > 0.05:
            score += 5

        # 5. Financial Health (15 points)
        current_ratio = fa_data.get('currentRatio', 0)
        debt_to_equity = fa_data.get('debtToEquityRatio', 999)

        if current_ratio > 2.0:
            score += 8
        elif current_ratio > 1.5:
            score += 6
        elif current_ratio > 1.0:
            score += 4
        elif current_ratio > 0.8:
            score += 2

        if debt_to_equity < 0.5:
            score += 7
        elif debt_to_equity < 1.0:
            score += 5
        elif debt_to_equity < 2.0:
            score += 3
        elif debt_to_equity < 3.0:
            score += 1

        return min(score, 100)

    def score_megacap_with_growth(self, ticker, date, momentum_score):
        """Score a megacap stock with growth potential"""
        if not self.enable_growth_scoring:
            return momentum_score

        fa_data = self.fa_loader.get_fa_data_at_date(ticker, date)
        growth_score = self.calculate_growth_potential_score(fa_data)

        # Combined: 70% momentum + 30% growth
        combined = (momentum_score * self.momentum_weight +
                   growth_score * self.growth_weight)
        return combined

    def score_stock_with_growth(self, ticker, df_at_date, date):
        """Score a non-megacap stock with growth potential"""
        if not self.enable_growth_scoring:
            return self.bot.score_stock(ticker, df_at_date)

        # Get momentum score from bot
        try:
            momentum_score = self.bot.score_stock(ticker, df_at_date)
        except:
            momentum_score = 0

        # Get growth potential score
        fa_data = self.fa_loader.get_fa_data_at_date(ticker, date)
        growth_score = self.calculate_growth_potential_score(fa_data)

        # Combined: 70% momentum + 30% growth
        combined = (momentum_score * self.momentum_weight +
                   growth_score * self.growth_weight)
        return combined

    def run_backtest(self, start_year=1963, end_year=2024):
        """
        Override run_backtest to inject growth scoring

        This is almost identical to parent V31, but replaces:
        - Line 244: megacap_scores[ticker] = mom20
        - Line 256: momentum_scores[ticker] = self.bot.score_stock(...)

        With growth-enhanced versions.
        """
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

                # Score mega-caps - GROWTH ENHANCED
                megacap_scores = {}
                for ticker in megacaps:
                    if ticker in self.bot.stocks_data:
                        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                        if len(df_at_date) >= 20:
                            mom20 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1) * 100
                            # TIER 2 MODIFICATION: Add growth scoring
                            enhanced_score = self.score_megacap_with_growth(ticker, date, mom20)
                            megacap_scores[ticker] = enhanced_score

                top_megacaps = sorted(megacap_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['num_megacap']]

                # Score non-mega-cap stocks - GROWTH ENHANCED
                momentum_scores = {}
                for ticker, df in self.bot.stocks_data.items():
                    if ticker in megacaps:
                        continue
                    df_at_date = df[df.index <= date]
                    if len(df_at_date) >= 100:
                        try:
                            # TIER 2 MODIFICATION: Add growth scoring
                            enhanced_score = self.score_stock_with_growth(ticker, df_at_date, date)
                            momentum_scores[ticker] = enhanced_score
                        except:
                            pass

                top_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['num_momentum']]

                # ENHANCED POSITION SIZING
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
            stocks_value = sum(
                h['shares'] * self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date].iloc[-1]['close']
                for t, h in holdings.items()
                if len(self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date]) > 0
            )
            portfolio_value = cash + stocks_value

            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': cash,
                'stocks_value': stocks_value
            })

        # Export trade history to CSV
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            import os
            os.makedirs('output', exist_ok=True)
            trade_df.to_csv('output/tier2_trades.csv', index=False)
            logger.info(f"✅ Saved {len(trade_df)} trades to: output/tier2_trades.csv")

        return pd.DataFrame(portfolio_values).set_index('date')
