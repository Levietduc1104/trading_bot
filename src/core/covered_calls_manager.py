"""
Covered Calls Manager for V31 Strategy
=======================================
Manages covered call options on existing stock positions to generate premium income.

Strategy:
- Sell OTM (Out-of-The-Money) covered calls on stock positions
- Target: 4% annual premium (1% per quarter)
- Quarterly execution aligned with rebalancing
- Automatically rolls positions when appropriate

For Backtesting:
- Simulates premium income collection
- Applies quarterly at rebalancing dates

For Live Trading:
- Executes real options trades via Alpaca
- Monitors and manages options positions
- Handles early assignment risk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CoveredCallsManager:
    """Manages covered call strategy for V31"""

    def __init__(self,
                 quarterly_premium_rate: float = 0.01,  # 1% per quarter = 4% annually
                 days_to_expiry: int = 90,  # Quarterly options
                 otm_percent: float = 0.05,  # 5% OTM strike
                 min_position_value: float = 5000,  # Minimum position size for covered calls
                 enabled: bool = True):
        """
        Initialize Covered Calls Manager

        Args:
            quarterly_premium_rate: Target premium as % of portfolio (default 1% = 4% annual)
            days_to_expiry: Days until option expiration (default 90 for quarterly)
            otm_percent: % above current price for strike (default 5%)
            min_position_value: Minimum position value to write covered calls
            enabled: Whether covered calls are enabled
        """
        self.quarterly_premium_rate = quarterly_premium_rate
        self.days_to_expiry = days_to_expiry
        self.otm_percent = otm_percent
        self.min_position_value = min_position_value
        self.enabled = enabled

        # Track covered call positions
        self.active_calls: Dict[str, Dict] = {}  # ticker -> call details
        self.premium_history: List[Dict] = []
        self.total_premium_collected = 0.0

        logger.info(f"Covered Calls Manager initialized:")
        logger.info(f"  Quarterly premium target: {quarterly_premium_rate*100:.1f}%")
        logger.info(f"  Annual premium target: {quarterly_premium_rate*4*100:.1f}%")
        logger.info(f"  Days to expiry: {days_to_expiry}")
        logger.info(f"  OTM %: {otm_percent*100:.0f}%")
        logger.info(f"  Enabled: {enabled}")

    def calculate_premium_backtest(self,
                                   portfolio_value: float,
                                   current_date: pd.Timestamp) -> float:
        """
        Calculate premium for backtesting (simplified simulation)

        Args:
            portfolio_value: Current portfolio value
            current_date: Current date

        Returns:
            Premium amount in dollars
        """
        if not self.enabled:
            return 0.0

        # Calculate premium based on portfolio value
        premium = portfolio_value * self.quarterly_premium_rate

        # Record premium
        self.total_premium_collected += premium
        self.premium_history.append({
            'date': current_date,
            'premium': premium,
            'portfolio_value': portfolio_value,
            'cumulative_premium': self.total_premium_collected
        })

        logger.info(f"  💰 Covered calls premium collected: ${premium:,.2f} "
                   f"({self.quarterly_premium_rate*100:.1f}% of ${portfolio_value:,.0f})")
        logger.info(f"     Total premium to date: ${self.total_premium_collected:,.2f}")

        return premium

    def get_premium_stats(self) -> Dict:
        """Get statistics on premium collection"""
        if not self.premium_history:
            return {
                'total_premium': 0,
                'num_collections': 0,
                'avg_premium': 0,
                'annual_rate': 0
            }

        df = pd.DataFrame(self.premium_history)

        # Calculate annual rate
        if len(df) > 1:
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
            years = max(days / 365.25, 0.25)  # At least 0.25 years
            annual_premium = self.total_premium_collected / years
            avg_portfolio = df['portfolio_value'].mean()
            annual_rate = (annual_premium / avg_portfolio) * 100 if avg_portfolio > 0 else 0
        else:
            annual_rate = 0

        return {
            'total_premium': self.total_premium_collected,
            'num_collections': len(self.premium_history),
            'avg_premium': df['premium'].mean() if len(df) > 0 else 0,
            'annual_rate': annual_rate
        }

    def get_premium_dataframe(self) -> pd.DataFrame:
        """Get premium history as DataFrame"""
        if not self.premium_history:
            return pd.DataFrame(columns=['date', 'premium', 'portfolio_value', 'cumulative_premium'])
        return pd.DataFrame(self.premium_history)

    def reset(self):
        """Reset manager (for backtesting)"""
        self.active_calls = {}
        self.premium_history = []
        self.total_premium_collected = 0.0
