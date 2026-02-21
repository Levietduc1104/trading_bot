"""
Phase 2: V31 + ML Stock Ranking Strategy

Replaces simple momentum scoring with ML predictions
Uses ALL 131 features (20 technical + 111 fundamental)
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional
from .v31_enhanced import V31EnhancedStrategy
from ..ml.feature_extraction import MLFeatureExtractor
from ..ml.stock_ranker import MLStockRanker
from ..data.historical_fa_data_loader import HistoricalFADataLoader

logger = logging.getLogger(__name__)


class V31MLStrategy(V31EnhancedStrategy):
    """
    V31 + ML Stock Ranking (Phase 2)

    Enhancements over V31:
    - ML-based stock scoring instead of simple momentum
    - Uses 131 features (20 technical + 111 fundamental)
    - LightGBM with strong overfitting prevention
    - Walk-forward validation during training
    """

    def __init__(self, bot, use_transaction_costs=True, broker='interactive_brokers',
                 enable_covered_calls=True, monthly_contribution=0,
                 ml_model=None, n_features_to_select=50):
        super().__init__(bot=bot,
                        use_transaction_costs=use_transaction_costs,
                        broker=broker,
                        enable_covered_calls=enable_covered_calls,
                        monthly_contribution=monthly_contribution)

        # ML components
        self.ml_model = ml_model  # Pre-trained model (optional)
        self.feature_extractor = MLFeatureExtractor()
        self.fa_loader = HistoricalFADataLoader()
        self.fa_loader.load_all()

        # Feature selection
        self.n_features_to_select = n_features_to_select

        logger.info("✅ V31 ML Strategy initialized (Phase 2)")
        logger.info(f"   ML Model: {'Pre-trained' if ml_model else 'Not loaded'}")
        logger.info(f"   Features: 131 total (20 technical + 111 fundamental)")
        logger.info(f"   Feature Selection: Top {n_features_to_select} by importance")

    def score_stock_ml(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """
        Score a stock using ML model

        Returns:
            Predicted forward return (higher = better)
        """
        if self.ml_model is None:
            logger.warning("ML model not trained! Using fallback momentum scoring")
            return self.score_stock_fallback(ticker, date)

        # Extract all features
        fa_data = self.fa_loader.get_fa_data_at_date(ticker, date)
        features = self.feature_extractor.extract_features(ticker, date, self.bot, fa_data)

        if features is None:
            return None

        # Get ML prediction
        try:
            prediction = self.ml_model.predict(features)
            return prediction
        except Exception as e:
            logger.warning(f"ML prediction failed for {ticker}: {e}")
            return None

    def score_stock_fallback(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Fallback to simple momentum if ML fails"""
        df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
        if len(df_at_date) >= 60:
            return (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1) * 100
        return None

    def get_regime_multiplier(self, date: pd.Timestamp) -> float:
        """
        Returns an equity exposure multiplier based on SPY vs its 200-day MA.

        Regimes:
          Strong Bull  (SPY > MA200 * 1.05):  1.00  -- full exposure
          Bull         (SPY > MA200):          0.90  -- slight reduction
          Caution      (SPY < MA200 by <5%):  0.65  -- meaningful reduction
          Bear         (SPY < MA200 by 5-15%): 0.40  -- significant reduction
          Deep Bear    (SPY < MA200 by >15%): 0.20  -- near-cash
        """
        if 'SPY' not in self.bot.stocks_data:
            return 1.0
        spy = self.bot.stocks_data['SPY']
        spy_hist = spy[spy.index <= date]
        if len(spy_hist) < 200:
            return 1.0
        current = spy_hist['close'].iloc[-1]
        ma200   = spy_hist['close'].tail(200).mean()
        ratio   = current / ma200
        if   ratio >= 1.05: return 1.00   # strong bull
        elif ratio >= 1.00: return 0.90   # bull
        elif ratio >= 0.95: return 0.65   # caution
        elif ratio >= 0.85: return 0.40   # bear
        else:               return 0.20   # deep bear

    def _apply_ml_confidence_weights(self, tickers_scores, base_allocations, total_amount, blend=0.40):
        """
        Blend base_allocations (from EnhancedPositionSizer) with ML-score-proportional weights.
        blend=0.40 means 40% ML score weight, 60% base weight.

        Steps:
        1. Normalize ML scores to [0, 1] within the group (min-max)
        2. Convert to weights (sum to 1.0)
        3. Blend: final_weight = 0.60 * base_weight + 0.40 * ml_weight
        4. Rescale to total_amount
        """
        if not tickers_scores or not base_allocations:
            return base_allocations
        scores = {t: max(s, 0) for t, s in tickers_scores}  # floor at 0
        score_sum = sum(scores.values())
        if score_sum <= 0:
            return base_allocations
        # ML weights (proportional to score)
        ml_weights = {t: scores[t] / score_sum for t in base_allocations if t in scores}
        # Fill any missing tickers with 0 ml weight
        for t in base_allocations:
            if t not in ml_weights:
                ml_weights[t] = 0.0
        # Base weights (from EnhancedPositionSizer)
        base_sum = sum(base_allocations.values())
        if base_sum <= 0:
            return base_allocations
        base_weights = {t: v / base_sum for t, v in base_allocations.items()}
        # Blend
        blended = {t: (1 - blend) * base_weights.get(t, 0) + blend * ml_weights.get(t, 0)
                   for t in base_allocations}
        # Normalize and apply constraints (8%-22% of group)
        n = len(blended)
        min_w = 0.08
        max_w = min(0.22, 1.0 / max(n, 1))
        total_w = sum(blended.values())
        if total_w <= 0:
            return base_allocations
        final = {t: max(min_w, min(max_w, w / total_w)) for t, w in blended.items()}
        # Rescale to total_amount
        final_sum = sum(final.values())
        return {t: w / final_sum * total_amount for t, w in final.items()}

    def run_backtest(self, start_year=1963, end_year=2024):
        """
        Run backtest with ML scoring

        Same structure as V31, but uses ML to score stocks
        """
        if 'SPY' in self.bot.stocks_data:
            reference_ticker = 'SPY'
        else:
            longest_ticker = max(self.bot.stocks_data.items(), key=lambda x: len(x[1]))[0]
            reference_ticker = longest_ticker

        all_dates = self.bot.stocks_data[reference_ticker].index
        all_dates = all_dates[(all_dates >= f'{start_year}-01-01') & (all_dates <= f'{end_year}-12-31')]

        portfolio_values = []
        holdings = {}
        cash = self.initial_capital
        last_rebalance = None
        last_contribution_month = None

        # Log ML status
        if self.ml_model:
            logger.info(f"✅ Running backtest with ML model")
            n_features = self.ml_model.train_metrics.get('n_features_used', 'N/A')
            val_corr = self.ml_model.train_metrics.get('val_corr', None)
            logger.info(f"   Features: {n_features}")
            if val_corr is not None:
                logger.info(f"   Validation Correlation: {val_corr:.3f}")
        else:
            logger.warning("⚠️  No ML model loaded - using fallback momentum scoring")

        # Add monthly contribution tracking
        if self.monthly_contribution > 0:
            logger.info(f"💰 Monthly Contribution: ${self.monthly_contribution:,.0f}/month")

        for date in all_dates:
            # Monthly contributions
            if self.monthly_contribution > 0:
                current_month = (date.year, date.month)
                if last_contribution_month is None or current_month != last_contribution_month:
                    cash += self.monthly_contribution
                    self.total_contributions += self.monthly_contribution
                    last_contribution_month = current_month

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

                # Covered calls premium
                if self.enable_covered_calls and self.covered_calls_manager:
                    stocks_value = sum(
                        h['shares'] * self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date].iloc[-1]['close']
                        for t, h in holdings.items()
                        if len(self.bot.stocks_data[t][self.bot.stocks_data[t].index <= date]) > 0
                    )
                    portfolio_value = cash + stocks_value
                    premium = self.covered_calls_manager.calculate_premium_backtest(portfolio_value, date)
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

                cash_reserve  = self.get_vix_cash_reserve(vix)
                portfolio_df  = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
                dd_multiplier = self.get_portfolio_dd_multiplier(portfolio_df)
                regime_mult   = self.get_regime_multiplier(date)
                invest_amount = cash * (1 - cash_reserve) * dd_multiplier * regime_mult
                logger.debug(f"  {date.date()} | vix={vix:.1f} cash_reserve={cash_reserve:.2f} dd_mult={dd_multiplier:.2f} regime_mult={regime_mult:.2f}")

                # Identify mega-caps
                megacaps = self.identify_megacaps(date, self.config['num_top_megacaps'])

                # Score mega-caps with ML
                megacap_scores = {}
                for ticker in megacaps:
                    if ticker in self.bot.stocks_data:
                        score = self.score_stock_ml(ticker, date)
                        if score is not None:
                            megacap_scores[ticker] = score

                top_megacaps = sorted(megacap_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['num_megacap']]

                # Score non-megacap stocks with ML
                momentum_scores = {}
                for ticker in self.bot.stocks_data.keys():
                    if ticker in megacaps:
                        continue
                    df_at_date = self.bot.stocks_data[ticker][self.bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) >= 100:
                        try:
                            score = self.score_stock_ml(ticker, date)
                            if score is not None:
                                momentum_scores[ticker] = score
                        except:
                            pass

                top_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['num_momentum']]

                # Position sizing
                megacap_amount = invest_amount * self.config['megacap_allocation']
                momentum_amount = invest_amount * (1 - self.config['megacap_allocation'])

                # Allocate mega-caps
                if top_megacaps:
                    megacap_base = self.position_sizer.calculate_enhanced_positions(
                        top_megacaps, date, self.bot, megacap_amount
                    )
                    megacap_allocations = self._apply_ml_confidence_weights(
                        top_megacaps, megacap_base, megacap_amount
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

                # Allocate momentum stocks
                if top_momentum:
                    momentum_base = self.position_sizer.calculate_enhanced_positions(
                        top_momentum, date, self.bot, momentum_amount
                    )
                    momentum_allocations = self._apply_ml_confidence_weights(
                        top_momentum, momentum_base, momentum_amount
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

        # Export trade history
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            import os
            os.makedirs('output', exist_ok=True)
            trade_df.to_csv('output/ml_trades.csv', index=False)
            logger.info(f"✅ Saved {len(trade_df)} trades to: output/ml_trades.csv")

        return pd.DataFrame(portfolio_values).set_index('date')
