"""
Fundamental Data Fetcher using yfinance
========================================
Fetch market cap, P/E ratio, and other fundamentals for V30 strategy

This module provides fundamental data that Alpaca doesn't offer:
- Market Cap
- P/E Ratio
- EPS
- Revenue
- Profit Margin
- Dividend Yield
- Beta
- 52-week high/low
- And more!

Install: pip install yfinance
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class FundamentalDataFetcher:
    """
    Fetch fundamental data using yfinance (Yahoo Finance)
    Free and unlimited!
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.cache = {}  # Cache to avoid repeated API calls

    def get_fundamentals(self, symbol: str) -> Dict:
        """
        Get comprehensive fundamental data for a symbol

        Args:
            symbol: Stock ticker (e.g., 'AAPL')

        Returns:
            Dict with fundamental data
        """
        # Check cache first
        if symbol in self.cache:
            if self.debug:
                print(f"[DEBUG] Using cached data for {symbol}")
            return self.cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamentals = {
                # Valuation
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),

                # Profitability
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),

                # Growth
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),

                # Financial Health
                'current_ratio': info.get('currentRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'quick_ratio': info.get('quickRatio', None),

                # Dividend
                'dividend_yield': info.get('dividendYield', None),
                'payout_ratio': info.get('payoutRatio', None),

                # Risk
                'beta': info.get('beta', None),

                # Size & Sector
                'sector': info.get('sector', None),
                'industry': info.get('industry', None),

                # Price Info
                'current_price': info.get('currentPrice', None),
                'target_price': info.get('targetMeanPrice', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),

                # Earnings
                'eps': info.get('trailingEps', None),
                'forward_eps': info.get('forwardEps', None),
            }

            # Cache the result
            self.cache[symbol] = fundamentals

            if self.debug:
                print(f"[DEBUG] Fetched fundamentals for {symbol}")

            return fundamentals

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error fetching {symbol}: {e}")
            return {}

    def get_fundamentals_batch(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get fundamentals for multiple symbols at once

        Args:
            symbols: List of stock tickers

        Returns:
            DataFrame with fundamental data for all symbols
        """
        if self.debug:
            print(f"[DEBUG] Fetching fundamentals for {len(symbols)} symbols...")

        data = []
        for symbol in symbols:
            fundamentals = self.get_fundamentals(symbol)
            if fundamentals:
                fundamentals['symbol'] = symbol
                data.append(fundamentals)

        return pd.DataFrame(data)

    def get_quality_score(self, symbol: str) -> float:
        """
        Calculate a quality score (0-100) based on fundamentals

        Higher score = Better quality stock

        Criteria:
        - High profit margin
        - Low P/E ratio (value)
        - Strong revenue growth
        - Low debt to equity
        - High ROE
        """
        fundamentals = self.get_fundamentals(symbol)

        if not fundamentals:
            return 0.0

        score = 0.0
        max_score = 100.0

        # Profitability (30 points)
        profit_margin = fundamentals.get('profit_margin')
        if profit_margin:
            score += min(profit_margin * 100, 30)  # Max 30 points

        # Valuation (20 points)
        pe_ratio = fundamentals.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            # Lower P/E is better (inverse relationship)
            # P/E < 15 = 20 points, P/E 15-25 = 10-20 points, P/E > 25 = 0-10 points
            if pe_ratio < 15:
                score += 20
            elif pe_ratio < 25:
                score += 20 - ((pe_ratio - 15) * 1.0)
            else:
                score += max(0, 10 - ((pe_ratio - 25) * 0.2))

        # Growth (20 points)
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth:
            score += min(revenue_growth * 100, 20)  # Max 20 points

        # Financial Health (15 points)
        debt_to_equity = fundamentals.get('debt_to_equity')
        if debt_to_equity is not None:
            # Lower debt is better
            if debt_to_equity < 50:
                score += 15
            elif debt_to_equity < 100:
                score += 15 - ((debt_to_equity - 50) * 0.3)
            else:
                score += max(0, 5 - ((debt_to_equity - 100) * 0.05))

        # ROE (15 points)
        roe = fundamentals.get('roe')
        if roe:
            score += min(roe * 100, 15)  # Max 15 points

        return min(score, max_score)

    def filter_quality_stocks(self, symbols: List[str], min_score: float = 50.0) -> List[str]:
        """
        Filter stocks by quality score

        Args:
            symbols: List of stock tickers
            min_score: Minimum quality score (0-100)

        Returns:
            List of symbols that meet quality criteria
        """
        quality_stocks = []

        for symbol in symbols:
            score = self.get_quality_score(symbol)
            if score >= min_score:
                quality_stocks.append(symbol)
                if self.debug:
                    print(f"[DEBUG] {symbol}: Quality Score = {score:.1f} ✓")
            elif self.debug:
                print(f"[DEBUG] {symbol}: Quality Score = {score:.1f} ✗")

        return quality_stocks


def demo():
    """Demo the fundamental data fetcher"""
    print("\n" + "="*60)
    print("FUNDAMENTAL DATA FETCHER DEMO")
    print("="*60)

    fetcher = FundamentalDataFetcher(debug=True)

    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'META']

    print("\n[1] Single Stock Example - AAPL")
    print("-" * 60)
    aapl_data = fetcher.get_fundamentals('AAPL')
    if aapl_data:
        print(f"\nAAPL Fundamentals:")
        print(f"  Market Cap: ${aapl_data.get('market_cap', 0):,.0f}")
        print(f"  P/E Ratio: {aapl_data.get('pe_ratio', 0):.2f}")
        print(f"  EPS: ${aapl_data.get('eps', 0):.2f}")
        print(f"  Profit Margin: {aapl_data.get('profit_margin', 0)*100:.2f}%")
        print(f"  Revenue Growth: {aapl_data.get('revenue_growth', 0)*100:.2f}%")
        print(f"  Dividend Yield: {aapl_data.get('dividend_yield', 0)*100:.2f}%")
        print(f"  Beta: {aapl_data.get('beta', 0):.2f}")
        print(f"  Sector: {aapl_data.get('sector', 'N/A')}")

    print("\n[2] Batch Fetch - Multiple Stocks")
    print("-" * 60)
    df = fetcher.get_fundamentals_batch(test_symbols)
    if not df.empty:
        print(f"\nFetched data for {len(df)} stocks")
        print(df[['symbol', 'market_cap', 'pe_ratio', 'profit_margin', 'beta']].to_string())

    print("\n[3] Quality Scoring")
    print("-" * 60)
    for symbol in test_symbols:
        score = fetcher.get_quality_score(symbol)
        print(f"  {symbol}: Quality Score = {score:.1f}/100")

    print("\n[4] Quality Filtering")
    print("-" * 60)
    quality_stocks = fetcher.filter_quality_stocks(test_symbols, min_score=60.0)
    print(f"\nStocks with quality score >= 60:")
    print(f"  {', '.join(quality_stocks)}")

    print("\n" + "="*60)
    print("✓ Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    demo()
