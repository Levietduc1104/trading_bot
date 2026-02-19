"""
Enhanced Position Sizing for V31
Improves on inverse volatility weighting by adding momentum and diversification factors

Base V31: Inverse volatility weighting only
Enhanced: Volatility (40%) + Momentum (30%) + Mean Reversion (20%) + Correlation (10%)

Expected improvement: +1-2% annual return
"""

import numpy as np
import pandas as pd
from datetime import timedelta

class EnhancedPositionSizer:
    def __init__(self, config=None):
        self.config = config or {
            'volatility_weight': 0.40,      # 40% weight on inverse volatility
            'momentum_weight': 0.30,        # 30% weight on momentum
            'mean_reversion_weight': 0.20,  # 20% weight on mean reversion
            'correlation_weight': 0.10,     # 10% weight on diversification

            'vol_lookback': 20,             # Days for volatility calculation
            'momentum_lookback': 60,        # Days for momentum calculation
            'mean_reversion_lookback': 20,  # Days for mean reversion

            'min_position_size': 0.08,      # Minimum 8% per stock (up from 5%)
            'max_position_size': 0.22,      # Maximum 22% per stock (down from 25%)
        }

    def calculate_volatility_score(self, ticker, date, bot):
        """Calculate inverse volatility score (lower vol = higher score)"""
        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]

        if len(df_at_date) < self.config['vol_lookback']:
            return None

        recent = df_at_date.tail(self.config['vol_lookback'])
        returns = recent['close'].pct_change().dropna()

        if len(returns) == 0:
            return None

        # Annualized volatility
        volatility = returns.std() * np.sqrt(252)

        if volatility == 0:
            return None

        # Inverse volatility score (normalize to 0-1 range)
        # Typical stock volatility: 15-50%
        # Lower vol (15%) → higher score (1.0)
        # Higher vol (50%) → lower score (0.3)
        inv_vol_score = 1.0 / volatility
        normalized_score = np.clip(inv_vol_score / 0.067, 0.3, 1.0)  # 1/15 ≈ 0.067

        return normalized_score

    def calculate_momentum_score(self, ticker, date, bot):
        """Calculate momentum score (higher momentum = higher score)"""
        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]

        if len(df_at_date) < self.config['momentum_lookback']:
            return None

        # Calculate 60-day momentum
        current_price = df_at_date.iloc[-1]['close']
        past_price = df_at_date.iloc[-self.config['momentum_lookback']]['close']
        momentum_pct = (current_price / past_price - 1) * 100

        # Normalize to 0-1 range
        # Strong momentum (+30%) → 1.0
        # Neutral (0%) → 0.5
        # Weak (-30%) → 0.0
        normalized_score = np.clip((momentum_pct + 30) / 60, 0.0, 1.0)

        return normalized_score

    def calculate_mean_reversion_score(self, ticker, date, bot):
        """Calculate mean reversion score (oversold = higher score)"""
        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]

        if len(df_at_date) < self.config['mean_reversion_lookback']:
            return None

        recent = df_at_date.tail(self.config['mean_reversion_lookback'])
        current_price = recent.iloc[-1]['close']
        sma = recent['close'].mean()

        # How far below SMA (oversold)?
        deviation = (current_price - sma) / sma * 100

        # Normalize to 0-1 range
        # 10% below SMA → 1.0 (max oversold bonus)
        # At SMA → 0.5 (neutral)
        # 10% above SMA → 0.0 (no bonus)
        normalized_score = np.clip(0.5 - (deviation / 20), 0.0, 1.0)

        return normalized_score

    def calculate_correlation_score(self, ticker, date, bot, other_tickers):
        """Calculate diversification score (lower correlation = higher score)"""
        if not other_tickers:
            return 1.0  # No other stocks, full score

        df_ticker = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]

        if len(df_ticker) < 60:
            return 0.5  # Insufficient data, neutral score

        # Calculate this stock's returns
        ticker_returns = df_ticker.tail(60)['close'].pct_change().dropna()

        if len(ticker_returns) < 30:
            return 0.5

        # Calculate average correlation with other stocks
        correlations = []
        for other_ticker in other_tickers:
            if other_ticker == ticker:
                continue

            df_other = bot.stocks_data[other_ticker][bot.stocks_data[other_ticker].index <= date]
            if len(df_other) < 60:
                continue

            other_returns = df_other.tail(60)['close'].pct_change().dropna()

            if len(other_returns) < 30:
                continue

            # Align dates
            common_dates = ticker_returns.index.intersection(other_returns.index)
            if len(common_dates) < 30:
                continue

            ticker_aligned = ticker_returns.loc[common_dates]
            other_aligned = other_returns.loc[common_dates]

            corr = ticker_aligned.corr(other_aligned)
            if not np.isnan(corr):
                correlations.append(corr)

        if not correlations:
            return 0.5  # No correlations calculated, neutral

        avg_correlation = np.mean(correlations)

        # Convert correlation to score
        # Low correlation (0.3) → 1.0 (great diversification)
        # High correlation (0.9) → 0.0 (poor diversification)
        # Mega-caps typically have 0.6-0.8 correlation
        normalized_score = np.clip(1.0 - (avg_correlation - 0.3) / 0.6, 0.0, 1.0)

        return normalized_score

    def calculate_combined_score(self, ticker, date, bot, other_tickers):
        """Calculate weighted combination of all factors"""
        vol_score = self.calculate_volatility_score(ticker, date, bot)
        mom_score = self.calculate_momentum_score(ticker, date, bot)
        mr_score = self.calculate_mean_reversion_score(ticker, date, bot)
        corr_score = self.calculate_correlation_score(ticker, date, bot, other_tickers)

        # If any critical score is None, return None
        if vol_score is None or mom_score is None:
            return None

        # Mean reversion and correlation are optional (default to 0.5 if None)
        mr_score = mr_score if mr_score is not None else 0.5
        corr_score = corr_score if corr_score is not None else 0.5

        # Weighted combination
        combined_score = (
            vol_score * self.config['volatility_weight'] +
            mom_score * self.config['momentum_weight'] +
            mr_score * self.config['mean_reversion_weight'] +
            corr_score * self.config['correlation_weight']
        )

        return combined_score

    def calculate_enhanced_positions(self, stocks, date, bot, total_amount):
        """
        Allocate capital using enhanced multi-factor position sizing

        Args:
            stocks: List of (ticker, score) tuples
            date: Current date
            bot: Portfolio bot instance
            total_amount: Total capital to allocate

        Returns:
            Dictionary of {ticker: allocation_amount}
        """
        tickers = [ticker for ticker, _ in stocks]

        # Calculate combined scores for all stocks
        combined_scores = {}
        for ticker, _ in stocks:
            score = self.calculate_combined_score(ticker, date, bot, tickers)
            if score is not None and score > 0:
                combined_scores[ticker] = score

        if not combined_scores:
            # Fallback to equal weight
            equal_weight = total_amount / len(stocks)
            return {ticker: equal_weight for ticker, _ in stocks}

        # Calculate raw weights from scores
        total_score = sum(combined_scores.values())
        raw_weights = {ticker: score / total_score for ticker, score in combined_scores.items()}

        # Apply min/max constraints
        min_size = self.config['min_position_size']
        max_size = self.config['max_position_size']

        allocations = {}
        for ticker, weight in raw_weights.items():
            constrained_weight = np.clip(weight, min_size, max_size)
            allocations[ticker] = constrained_weight * total_amount

        # Normalize to ensure we use all capital
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            scale_factor = total_amount / total_allocated
            allocations = {ticker: amount * scale_factor for ticker, amount in allocations.items()}

        return allocations

    def get_allocation_details(self, ticker, date, bot, other_tickers):
        """Get detailed breakdown of allocation factors for a stock"""
        vol_score = self.calculate_volatility_score(ticker, date, bot)
        mom_score = self.calculate_momentum_score(ticker, date, bot)
        mr_score = self.calculate_mean_reversion_score(ticker, date, bot)
        corr_score = self.calculate_correlation_score(ticker, date, bot, other_tickers)

        return {
            'volatility_score': vol_score,
            'momentum_score': mom_score,
            'mean_reversion_score': mr_score,
            'correlation_score': corr_score,
            'combined_score': self.calculate_combined_score(ticker, date, bot, other_tickers),
        }
