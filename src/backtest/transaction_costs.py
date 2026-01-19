"""
Transaction Cost Model
Based on "Machine Learning for Trading" Chapter 2: Market Microstructure

References:
- Bid-ask spread modeling
- Market impact (Kyle's Lambda, Almgren-Chriss)
- Slippage estimation
- Commission structures
"""
import numpy as np
import pandas as pd


class TransactionCostModel:
    """
    Realistic transaction cost modeling for backtesting

    Components:
    1. Bid-ask spread (varies by stock liquidity)
    2. Market impact (price moves against you on large orders)
    3. Slippage (execution price != decision price)
    4. Commission (broker fees)
    """

    def __init__(self, broker='interactive_brokers'):
        self.broker = broker

        # Commission structure (Interactive Brokers IBKR Pro)
        self.commissions = {
            'interactive_brokers': {
                'per_share': 0.0035,  # $0.0035 per share
                'minimum': 0.35,       # $0.35 minimum
                'maximum': 0.01        # 1% of trade value maximum
            },
            'robinhood': {
                'per_share': 0.0,
                'minimum': 0.0,
                'maximum': 0.0
            }
        }

    def estimate_spread(self, ticker, price, market_cap_category='mid'):
        """
        Estimate bid-ask spread based on stock characteristics

        Research shows spread is function of:
        - Price level (higher price = lower % spread)
        - Market cap (larger = lower spread)
        - Volatility (higher vol = higher spread)

        Args:
            ticker: Stock ticker
            price: Current stock price
            market_cap_category: 'large', 'mid', 'small', 'micro'

        Returns:
            spread_pct: Half-spread as percentage (you pay this going in AND out)
        """
        # Empirical spreads from market data
        base_spreads = {
            'large': 0.0005,   # 0.05% - S&P 100 (AAPL, MSFT, GOOGL)
            'mid': 0.0015,     # 0.15% - S&P 500 ex-100
            'small': 0.0035,   # 0.35% - Russell 2000
            'micro': 0.0100    # 1.00% - Very small cap
        }

        spread = base_spreads.get(market_cap_category, 0.0020)

        # Price adjustment: very low-price stocks have higher % spreads
        if price < 5:
            spread *= 2.0
        elif price < 10:
            spread *= 1.5
        elif price < 20:
            spread *= 1.2

        return spread

    def estimate_market_impact(self, shares, avg_daily_volume, volatility=0.02):
        """
        Market impact using simplified Kyle's Lambda model

        When you buy/sell a large amount relative to daily volume,
        you move the price against yourself.

        Kyle's Lambda: ΔP = λ × Q
        where λ (lambda) depends on volatility and liquidity

        Args:
            shares: Number of shares to trade
            avg_daily_volume: Average daily trading volume
            volatility: Daily volatility (default 2%)

        Returns:
            impact_pct: Expected price impact as percentage
        """
        if avg_daily_volume == 0:
            return 0.01  # 1% penalty for illiquid stocks

        # Participation rate: what % of daily volume are you trading?
        participation_rate = shares / avg_daily_volume

        # Kyle's lambda (simplified)
        # Impact increases non-linearly with participation rate
        if participation_rate < 0.01:  # < 1% of daily volume
            impact = participation_rate * volatility * 0.1
        elif participation_rate < 0.05:  # 1-5%
            impact = participation_rate * volatility * 0.3
        elif participation_rate < 0.10:  # 5-10%
            impact = participation_rate * volatility * 0.5
        else:  # > 10% - very large impact
            impact = participation_rate * volatility * 1.0

        return impact

    def estimate_slippage(self, order_type='market', volatility=0.02):
        """
        Slippage: difference between decision price and execution price

        Causes:
        - Time delay between signal and execution
        - Market orders walk the book
        - Price movement during order execution

        Args:
            order_type: 'market' or 'limit'
            volatility: Daily volatility

        Returns:
            slippage_pct: Expected slippage as percentage
        """
        if order_type == 'market':
            # Market orders: 0.05% - 0.20% slippage
            base_slippage = 0.001  # 0.1%
            # Higher volatility = more slippage
            vol_adjustment = volatility * 2.0
            return base_slippage + vol_adjustment
        else:
            # Limit orders: lower slippage but risk of non-execution
            return 0.0005  # 0.05%

    def calculate_commission(self, shares, price):
        """
        Calculate broker commission

        Args:
            shares: Number of shares
            price: Price per share

        Returns:
            commission: Total commission in dollars
        """
        config = self.commissions[self.broker]

        # Per-share commission
        comm = shares * config['per_share']

        # Apply minimum
        comm = max(comm, config['minimum'])

        # Apply maximum (% of trade value)
        trade_value = shares * price
        max_comm = trade_value * config['maximum']
        comm = min(comm, max_comm)

        return comm

    def total_execution_cost(self, ticker, shares, price,
                            avg_daily_volume, market_cap='mid',
                            volatility=0.02, order_type='market'):
        """
        Calculate total cost to execute a trade

        Args:
            ticker: Stock ticker
            shares: Number of shares to trade
            price: Price per share
            avg_daily_volume: Average daily volume
            market_cap: 'large', 'mid', 'small', 'micro'
            volatility: Daily volatility (default 2%)
            order_type: 'market' or 'limit'

        Returns:
            dict with cost breakdown
        """
        # 1. Bid-ask spread (one-way, you pay this cost)
        spread = self.estimate_spread(ticker, price, market_cap)
        spread_cost = shares * price * spread

        # 2. Market impact
        impact = self.estimate_market_impact(shares, avg_daily_volume, volatility)
        impact_cost = shares * price * impact

        # 3. Slippage
        slippage = self.estimate_slippage(order_type, volatility)
        slippage_cost = shares * price * slippage

        # 4. Commission
        commission = self.calculate_commission(shares, price)

        # Total
        total_cost = spread_cost + impact_cost + slippage_cost + commission
        trade_value = shares * price
        total_cost_pct = (total_cost / trade_value) * 100 if trade_value > 0 else 0

        return {
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'slippage_cost': slippage_cost,
            'commission': commission,
            'total_cost': total_cost,
            'total_cost_pct': total_cost_pct,
            'effective_price': price * (1 + spread + impact + slippage) + (commission / shares)
        }

    def round_trip_cost(self, ticker, shares, price, avg_daily_volume,
                       market_cap='mid', volatility=0.02):
        """
        Calculate cost for buying AND selling (round trip)
        This is the true cost of a position.

        Returns:
            total round-trip cost as percentage of trade value
        """
        buy_cost = self.total_execution_cost(
            ticker, shares, price, avg_daily_volume,
            market_cap, volatility, 'market'
        )

        sell_cost = self.total_execution_cost(
            ticker, shares, price, avg_daily_volume,
            market_cap, volatility, 'market'
        )

        total_pct = buy_cost['total_cost_pct'] + sell_cost['total_cost_pct']

        return {
            'buy_cost_pct': buy_cost['total_cost_pct'],
            'sell_cost_pct': sell_cost['total_cost_pct'],
            'round_trip_pct': total_pct,
            'buy_cost': buy_cost['total_cost'],
            'sell_cost': sell_cost['total_cost'],
            'total_cost': buy_cost['total_cost'] + sell_cost['total_cost']
        }


# Example usage
if __name__ == '__main__':
    model = TransactionCostModel(broker='interactive_brokers')

    # Example 1: Large cap stock (AAPL)
    print("=" * 60)
    print("Example 1: AAPL (Large Cap)")
    print("=" * 60)
    cost = model.total_execution_cost(
        ticker='AAPL',
        shares=100,
        price=180.0,
        avg_daily_volume=50_000_000,  # 50M shares/day
        market_cap='large',
        volatility=0.015  # 1.5% daily vol
    )
    print(f"Trade: 100 shares @ $180")
    print(f"Spread cost: ${cost['spread_cost']:.2f}")
    print(f"Impact cost: ${cost['impact_cost']:.2f}")
    print(f"Slippage cost: ${cost['slippage_cost']:.2f}")
    print(f"Commission: ${cost['commission']:.2f}")
    print(f"Total cost: ${cost['total_cost']:.2f} ({cost['total_cost_pct']:.3f}%)")
    print(f"Effective price: ${cost['effective_price']:.2f}")

    # Example 2: Small cap stock
    print("\n" + "=" * 60)
    print("Example 2: Small Cap Stock")
    print("=" * 60)
    cost2 = model.total_execution_cost(
        ticker='SMALL',
        shares=500,
        price=25.0,
        avg_daily_volume=100_000,  # Only 100K shares/day
        market_cap='small',
        volatility=0.035  # 3.5% daily vol (more volatile)
    )
    print(f"Trade: 500 shares @ $25")
    print(f"Total cost: ${cost2['total_cost']:.2f} ({cost2['total_cost_pct']:.3f}%)")

    # Example 3: Round trip
    print("\n" + "=" * 60)
    print("Example 3: Round Trip Cost (Buy + Sell)")
    print("=" * 60)
    rt_cost = model.round_trip_cost(
        ticker='SPY',
        shares=100,
        price=450.0,
        avg_daily_volume=80_000_000,
        market_cap='large',
        volatility=0.01
    )
    print(f"Round trip on $45,000 position:")
    print(f"Buy cost: {rt_cost['buy_cost_pct']:.3f}%")
    print(f"Sell cost: {rt_cost['sell_cost_pct']:.3f}%")
    print(f"Total round trip: {rt_cost['round_trip_pct']:.3f}%")
    print(f"Total cost: ${rt_cost['total_cost']:.2f}")
    print(f"\nTo break even, stock must move +{rt_cost['round_trip_pct']:.2f}%")
