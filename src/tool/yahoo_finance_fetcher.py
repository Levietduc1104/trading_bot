"""
Yahoo Finance Financial Data Fetcher
=====================================

Comprehensive tool to fetch fundamental data from Yahoo Finance
Includes: profitability, balance sheet, valuation, growth metrics

Usage:
    from src.tool.yahoo_finance_fetcher import YahooFinanceFetcher

    fetcher = YahooFinanceFetcher()

    # Fetch single stock
    data = fetcher.fetch_stock_fundamentals('AAPL')

    # Fetch multiple stocks
    data_dict = fetcher.fetch_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'])

    # Save to CSV
    fetcher.save_to_csv(data_dict, 'output/fundamentals.csv')
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """Fetches comprehensive fundamental data from Yahoo Finance"""

    def __init__(self, cache_file: Optional[str] = None):
        """
        Args:
            cache_file: Optional CSV file to cache results
        """
        self.cache_file = cache_file
        self.cache = {}

        if cache_file:
            try:
                df = pd.read_csv(cache_file, index_col=0)
                self.cache = df.to_dict('index')
                logger.info(f"Loaded {len(self.cache)} cached stocks from {cache_file}")
            except FileNotFoundError:
                logger.info("No cache file found, starting fresh")

    def fetch_stock_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch comprehensive fundamental data for a single stock

        Returns dict with:
        - Profitability metrics
        - Balance sheet health
        - Valuation metrics
        - Growth metrics
        - Risk metrics
        """
        # Check cache first
        if ticker in self.cache:
            logger.debug(f"Using cached data for {ticker}")
            return self.cache[ticker]

        logger.info(f"Fetching fundamentals for {ticker}")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Initialize result dictionary
            data = {
                'ticker': ticker,
                'fetch_date': datetime.now().strftime('%Y-%m-%d'),
            }

            # ====== PROFITABILITY METRICS ======
            data['net_profit_margin'] = self._safe_get(info, 'profitMargins', 0) * 100  # Convert to %
            data['operating_margin'] = self._safe_get(info, 'operatingMargins', 0) * 100
            data['gross_margin'] = self._safe_get(info, 'grossMargins', 0) * 100
            data['ebitda_margin'] = self._safe_get(info, 'ebitdaMargins', 0) * 100
            data['return_on_equity'] = self._safe_get(info, 'returnOnEquity', 0) * 100
            data['return_on_assets'] = self._safe_get(info, 'returnOnAssets', 0) * 100

            # Free cash flow
            data['free_cash_flow'] = self._safe_get(info, 'freeCashflow', 0)
            data['operating_cash_flow'] = self._safe_get(info, 'operatingCashflow', 0)

            # Profitability flags
            data['is_profitable'] = 1 if data['net_profit_margin'] > 0 else 0
            data['has_positive_fcf'] = 1 if data['free_cash_flow'] > 0 else 0

            # ====== BALANCE SHEET HEALTH ======
            data['total_cash'] = self._safe_get(info, 'totalCash', 0)
            data['total_debt'] = self._safe_get(info, 'totalDebt', 0)
            data['total_assets'] = self._safe_get(info, 'totalAssets', 0)
            data['current_assets'] = self._safe_get(info, 'totalCurrentAssets', 0)
            data['current_liabilities'] = self._safe_get(info, 'totalCurrentLiabilities', 0)

            # Ratios
            data['debt_to_equity'] = self._safe_get(info, 'debtToEquity', 0) / 100  # Convert from % to ratio
            data['current_ratio'] = self._safe_get(info, 'currentRatio', 0)
            data['quick_ratio'] = self._safe_get(info, 'quickRatio', 0)

            # Net debt
            data['net_debt'] = data['total_debt'] - data['total_cash']

            # Balance sheet health score
            data['balance_sheet_healthy'] = self._calculate_balance_sheet_health(data)

            # ====== VALUATION METRICS ======
            data['pe_ratio'] = self._safe_get(info, 'trailingPE', 0)
            data['forward_pe'] = self._safe_get(info, 'forwardPE', 0)
            data['price_to_sales'] = self._safe_get(info, 'priceToSalesTrailing12Months', 0)
            data['price_to_book'] = self._safe_get(info, 'priceToBook', 0)
            data['peg_ratio'] = self._safe_get(info, 'pegRatio', 0)
            data['enterprise_value'] = self._safe_get(info, 'enterpriseValue', 0)
            data['ev_to_ebitda'] = self._safe_get(info, 'enterpriseToEbitda', 0)
            data['ev_to_revenue'] = self._safe_get(info, 'enterpriseToRevenue', 0)

            # Valuation flags
            data['has_reasonable_valuation'] = self._check_reasonable_valuation(data)

            # ====== GROWTH METRICS ======
            data['revenue_growth'] = self._safe_get(info, 'revenueGrowth', 0) * 100
            data['earnings_growth'] = self._safe_get(info, 'earningsGrowth', 0) * 100
            data['revenue'] = self._safe_get(info, 'totalRevenue', 0)
            data['earnings'] = self._safe_get(info, 'netIncomeToCommon', 0)

            # Growth quality (earnings growth should match revenue growth for quality companies)
            if data['revenue_growth'] != 0:
                data['growth_quality_ratio'] = data['earnings_growth'] / data['revenue_growth']
            else:
                data['growth_quality_ratio'] = 0

            # Quality growth flag (earnings growing faster than revenue)
            data['has_quality_growth'] = 1 if data['growth_quality_ratio'] > 0.8 else 0

            # ====== RISK METRICS ======
            data['beta'] = self._safe_get(info, 'beta', 1.0)
            data['annual_volatility'] = self._safe_get(info, 'fiftyTwoWeekChange', 0)  # Proxy

            # ====== SIZE & LIQUIDITY ======
            data['market_cap'] = self._safe_get(info, 'marketCap', 0)
            data['is_megacap'] = 1 if data['market_cap'] > 200e9 else 0
            data['is_large_cap'] = 1 if data['market_cap'] > 50e9 else 0

            data['avg_volume'] = self._safe_get(info, 'averageVolume', 0)
            data['avg_volume_10day'] = self._safe_get(info, 'averageVolume10days', 0)

            # Calculate dollar volume (proxy using current price)
            current_price = self._safe_get(info, 'currentPrice', 0)
            data['dollar_volume'] = current_price * data['avg_volume']

            # ====== DIVIDEND & INCOME ======
            data['dividend_yield'] = self._safe_get(info, 'dividendYield', 0) * 100 if self._safe_get(info, 'dividendYield') else 0
            data['payout_ratio'] = self._safe_get(info, 'payoutRatio', 0) * 100
            data['pays_dividend'] = 1 if data['dividend_yield'] > 0 else 0

            # ====== INTEREST RATE SENSITIVITY ======
            # Duration proxy: Market Cap / Free Cash Flow
            if data['free_cash_flow'] > 0:
                data['stock_duration'] = data['market_cap'] / data['free_cash_flow']
            else:
                data['stock_duration'] = 999  # Very high duration = very rate sensitive

            data['is_rate_sensitive'] = 1 if data['stock_duration'] > 30 else 0

            # ====== ANALYST ESTIMATES ======
            data['target_price'] = self._safe_get(info, 'targetMeanPrice', 0)
            data['num_analyst_opinions'] = self._safe_get(info, 'numberOfAnalystOpinions', 0)

            if current_price > 0 and data['target_price'] > 0:
                data['upside_to_target'] = ((data['target_price'] / current_price) - 1) * 100
            else:
                data['upside_to_target'] = 0

            # ====== OWNERSHIP ======
            data['insider_ownership'] = self._safe_get(info, 'heldPercentInsiders', 0) * 100
            data['institutional_ownership'] = self._safe_get(info, 'heldPercentInstitutions', 0) * 100

            # ====== SHORT INTEREST ======
            data['short_percent'] = self._safe_get(info, 'shortPercentOfFloat', 0) * 100
            data['short_ratio'] = self._safe_get(info, 'shortRatio', 0)
            data['is_heavily_shorted'] = 1 if data['short_percent'] > 10 else 0

            # ====== QUALITY SCORE (COMPOSITE) ======
            data['quality_score'] = self._calculate_quality_score(data)

            # ====== ADDITIONAL INFO ======
            data['sector'] = self._safe_get(info, 'sector', 'Unknown')
            data['industry'] = self._safe_get(info, 'industry', 'Unknown')
            data['current_price'] = current_price

            # Cache result
            self.cache[ticker] = data

            return data

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'fetch_date': datetime.now().strftime('%Y-%m-%d')
            }

    def _safe_get(self, info: Dict, key: str, default=None):
        """Safely get value from info dict, handling None/NaN/Inf"""
        value = info.get(key, default)

        if value is None:
            return default

        # Handle numeric values
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return default

        return value

    def _calculate_balance_sheet_health(self, data: Dict) -> int:
        """Calculate balance sheet health score (0 or 1)"""
        score = 0

        # Good debt level
        if data['debt_to_equity'] < 2.0:
            score += 1

        # Good liquidity
        if data['current_ratio'] > 1.0:
            score += 1

        # Has cash
        if data['total_cash'] > 0:
            score += 1

        # More cash than debt
        if data['total_cash'] > data['total_debt']:
            score += 1

        # Return 1 if at least 3 out of 4 conditions met
        return 1 if score >= 3 else 0

    def _check_reasonable_valuation(self, data: Dict) -> int:
        """Check if valuation is reasonable (not extreme)"""
        # PE < 80 (avoid extreme growth valuations)
        if data['pe_ratio'] > 0 and data['pe_ratio'] < 80:
            return 1

        # If no PE, check P/S < 15
        if data['price_to_sales'] > 0 and data['price_to_sales'] < 15:
            return 1

        return 0

    def _calculate_quality_score(self, data: Dict) -> float:
        """Calculate composite quality score (0-10)"""
        score = 0.0

        # Profitability (3 points max)
        if data['is_profitable']:
            score += 1.0
        if data['net_profit_margin'] > 10:
            score += 1.0
        if data['has_positive_fcf']:
            score += 1.0

        # Balance sheet (2 points max)
        if data['balance_sheet_healthy']:
            score += 2.0

        # Size & liquidity (2 points max)
        if data['is_large_cap']:
            score += 1.0
        if data['dollar_volume'] > 100e6:
            score += 1.0

        # Valuation (1 point max)
        if data['has_reasonable_valuation']:
            score += 1.0

        # Growth quality (1 point max)
        if data['has_quality_growth']:
            score += 1.0

        # Institutional ownership (1 point max)
        if data['institutional_ownership'] > 60:
            score += 1.0

        return score

    def fetch_multiple_stocks(self, tickers: List[str], delay: float = 0.5) -> Dict[str, Dict]:
        """
        Fetch fundamentals for multiple stocks with rate limiting

        Args:
            tickers: List of ticker symbols
            delay: Delay between requests in seconds (to avoid rate limiting)

        Returns:
            Dict mapping ticker to fundamental data
        """
        results = {}
        total = len(tickers)

        logger.info(f"Fetching fundamentals for {total} stocks...")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{total}] Fetching {ticker}")
            results[ticker] = self.fetch_stock_fundamentals(ticker)

            # Rate limiting
            if i < total:
                time.sleep(delay)

        logger.info(f"Completed fetching {total} stocks")
        return results

    def save_to_csv(self, data: Dict[str, Dict], filename: str):
        """Save fetched data to CSV"""
        df = pd.DataFrame.from_dict(data, orient='index')
        df.to_csv(filename)
        logger.info(f"Saved {len(df)} stocks to {filename}")

    def load_from_csv(self, filename: str) -> Dict[str, Dict]:
        """Load previously saved data from CSV"""
        df = pd.read_csv(filename, index_col=0)
        return df.to_dict('index')

    def get_quality_stocks(self, data: Dict[str, Dict], min_quality_score: float = 7.0) -> List[str]:
        """
        Filter stocks by quality score

        Args:
            data: Dict of ticker -> fundamental data
            min_quality_score: Minimum quality score (0-10)

        Returns:
            List of tickers that meet quality threshold
        """
        quality_stocks = []

        for ticker, info in data.items():
            if 'quality_score' in info and info['quality_score'] >= min_quality_score:
                quality_stocks.append(ticker)

        logger.info(f"Found {len(quality_stocks)} quality stocks (score >= {min_quality_score})")
        return quality_stocks

    def apply_quality_filters(self, data: Dict[str, Dict]) -> List[str]:
        """
        Apply hard quality filters (strict requirements)

        Filters:
        - Profitable (net margin > 5%)
        - Healthy balance sheet (debt/equity < 2.0)
        - Good liquidity (current ratio > 1.0)
        - Large cap (market cap > $50B)
        - Reasonable valuation (P/E < 80 or P/S < 15)
        - Positive FCF

        Returns:
            List of tickers that pass all filters
        """
        filtered_stocks = []

        for ticker, info in data.items():
            # Skip if error
            if 'error' in info:
                continue

            # Apply filters
            passes = True

            # Profitability
            if info.get('net_profit_margin', 0) < 5:
                passes = False

            # FCF
            if info.get('free_cash_flow', 0) <= 0:
                passes = False

            # Balance sheet
            if info.get('debt_to_equity', 999) > 2.0:
                passes = False

            if info.get('current_ratio', 0) < 1.0:
                passes = False

            # Size
            if info.get('market_cap', 0) < 50e9:
                passes = False

            # Valuation
            if not info.get('has_reasonable_valuation', 0):
                passes = False

            if passes:
                filtered_stocks.append(ticker)

        logger.info(f"Filtered to {len(filtered_stocks)} quality stocks (passed all filters)")
        return filtered_stocks

    def create_summary_report(self, data: Dict[str, Dict], output_file: str = None):
        """Create summary report of fetched fundamentals"""
        df = pd.DataFrame.from_dict(data, orient='index')

        # Remove error rows
        df = df[~df.index.str.contains('error', na=False)]

        summary = f"""
        FUNDAMENTAL DATA SUMMARY
        ========================
        Total stocks: {len(df)}

        PROFITABILITY:
        - Profitable companies: {df['is_profitable'].sum()} ({df['is_profitable'].mean()*100:.1f}%)
        - Positive FCF: {df['has_positive_fcf'].sum()} ({df['has_positive_fcf'].mean()*100:.1f}%)
        - Avg net margin: {df['net_profit_margin'].mean():.1f}%
        - Avg operating margin: {df['operating_margin'].mean():.1f}%

        BALANCE SHEET:
        - Healthy balance sheets: {df['balance_sheet_healthy'].sum()} ({df['balance_sheet_healthy'].mean()*100:.1f}%)
        - Avg debt/equity: {df['debt_to_equity'].mean():.2f}
        - Avg current ratio: {df['current_ratio'].mean():.2f}

        SIZE:
        - Mega-caps (>$200B): {df['is_megacap'].sum()}
        - Large-caps (>$50B): {df['is_large_cap'].sum()}
        - Avg market cap: ${df['market_cap'].mean()/1e9:.1f}B

        VALUATION:
        - Avg P/E: {df['pe_ratio'].mean():.1f}
        - Avg P/S: {df['price_to_sales'].mean():.1f}
        - Reasonable valuations: {df['has_reasonable_valuation'].sum()} ({df['has_reasonable_valuation'].mean()*100:.1f}%)

        QUALITY:
        - Avg quality score: {df['quality_score'].mean():.1f}/10
        - High quality (>=7): {(df['quality_score'] >= 7).sum()} stocks
        - Medium quality (5-7): {((df['quality_score'] >= 5) & (df['quality_score'] < 7)).sum()} stocks
        - Low quality (<5): {(df['quality_score'] < 5).sum()} stocks
        """

        print(summary)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(summary)
            logger.info(f"Saved summary to {output_file}")

        return summary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test with a few stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'SNAP', 'RBLX', 'TSLA', 'META']

    fetcher = YahooFinanceFetcher()

    print("Fetching fundamental data...")
    data = fetcher.fetch_multiple_stocks(test_tickers)

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)

    for ticker, info in data.items():
        if 'error' in info:
            print(f"\n{ticker}: ERROR - {info['error']}")
        else:
            print(f"\n{ticker}:")
            print(f"  Quality Score: {info['quality_score']:.1f}/10")
            print(f"  Profitable: {'✓' if info['is_profitable'] else '✗'}")
            print(f"  Net Margin: {info['net_profit_margin']:.1f}%")
            print(f"  Debt/Equity: {info['debt_to_equity']:.2f}")
            print(f"  Current Ratio: {info['current_ratio']:.2f}")
            print(f"  P/E Ratio: {info['pe_ratio']:.1f}")
            print(f"  Market Cap: ${info['market_cap']/1e9:.1f}B")

    # Save to CSV
    fetcher.save_to_csv(data, 'output/test_fundamentals.csv')

    # Create summary
    print("\n" + "="*60)
    fetcher.create_summary_report(data)

    # Apply quality filters
    print("\n" + "="*60)
    print("QUALITY FILTERED STOCKS:")
    print("="*60)
    quality_stocks = fetcher.apply_quality_filters(data)
    print(f"Passed filters: {quality_stocks}")
