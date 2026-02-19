"""
Position Sizing Module - Basic Implementation

Provides simple volatility-weighted allocation for position sizing.
Start with basic approach before moving to complex models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PositionSizer:
    """
    Position sizing calculator using basic volatility-weighting.

    Philosophy: Allocate inversely to volatility - safer stocks get more capital.
    This provides better risk control than equal weighting.
    """

    def __init__(self, lookback_days: int = 20, min_position_pct: float = 0.05,
                 max_position_pct: float = 0.30):
        """
        Initialize position sizer.

        Args:
            lookback_days: Days to calculate volatility (default 20)
            min_position_pct: Minimum position size as % of capital (default 5%)
            max_position_pct: Maximum position size as % of capital (default 30%)
        """
        self.lookback_days = lookback_days
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct

    def volatility_weighted_allocation(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        selected_stocks: List[Tuple[str, float]],
        invest_amount: float,
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Allocate capital inversely proportional to stock volatility.
        Lower volatility = Higher allocation

        Args:
            stocks_data: Dict of ticker -> price DataFrame
            selected_stocks: List of (ticker, score) tuples
            invest_amount: Total amount to allocate
            date: Current date for calculation

        Returns:
            Dict of ticker -> allocation amount
        """
        volatilities = {}

        # Calculate annualized volatility for each stock
        for ticker, score in selected_stocks:
            if ticker not in stocks_data:
                continue

            df = stocks_data[ticker][stocks_data[ticker].index <= date]

            if len(df) >= self.lookback_days:
                # Calculate returns and volatility
                returns = df['close'].pct_change().tail(self.lookback_days)
                daily_vol = returns.std()

                # Annualize volatility
                annual_vol = daily_vol * np.sqrt(252)

                # Store (use max to avoid division by zero)
                volatilities[ticker] = max(annual_vol, 0.01)

        # If no valid volatilities, fall back to equal weight
        if not volatilities:
            equal_weight = invest_amount / len(selected_stocks)
            return {ticker: equal_weight for ticker, _ in selected_stocks}

        # Calculate inverse volatility weights
        inv_vols = {ticker: 1.0 / vol for ticker, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        # Allocate capital with position size limits
        allocations = {}
        for ticker in inv_vols:
            # Calculate raw weight
            weight = inv_vols[ticker] / total_inv_vol

            # Apply min/max constraints
            weight = max(self.min_position_pct, min(weight, self.max_position_pct))

            allocations[ticker] = invest_amount * weight

        # Normalize to ensure total equals invest_amount
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            for ticker in allocations:
                allocations[ticker] = allocations[ticker] / total_allocated * invest_amount

        return allocations

    def equal_weight_allocation(
        self,
        selected_stocks: List[Tuple[str, float]],
        invest_amount: float
    ) -> Dict[str, float]:
        """
        Simple equal weight allocation (baseline for comparison).

        Args:
            selected_stocks: List of (ticker, score) tuples
            invest_amount: Total amount to allocate

        Returns:
            Dict of ticker -> allocation amount
        """
        per_stock = invest_amount / len(selected_stocks)
        return {ticker: per_stock for ticker, _ in selected_stocks}
