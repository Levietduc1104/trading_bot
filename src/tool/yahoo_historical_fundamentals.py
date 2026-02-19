"""
Yahoo Finance Historical Fundamentals Fetcher
==============================================

Fetches HISTORICAL financial statements from Yahoo Finance
Works for backtesting 2015-2024 (gets 5+ years of data)

Key difference from yahoo_finance_fetcher.py:
- That file: Gets CURRENT info dict (2024 only)
- This file: Gets HISTORICAL financial statements (2015-2024)

Usage:
    from src.tool.yahoo_historical_fundamentals import YahooHistoricalFetcher

    fetcher = YahooHistoricalFetcher()

    # Get historical fundamentals
    data = fetcher.fetch_historical_fundamentals('AAPL')

    # Get profitability at specific date
    is_profitable = fetcher.is_profitable_at_date('AAPL', '2022-01-01')
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pickle
import os

logger = logging.getLogger(__name__)


class YahooHistoricalFetcher:
    """Fetches historical financial statements from Yahoo Finance"""

    def __init__(self, cache_dir: str = 'output/fundamentals_cache'):
        """
        Args:
            cache_dir: Directory to cache financial statements
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_historical_fundamentals(self, ticker: str, force_refresh: bool = False) -> Dict:
        """
        Fetch historical financial statements for a stock

        Returns dict with:
        - income_statement: DataFrame with multiple years
        - balance_sheet: DataFrame with multiple years
        - cash_flow: DataFrame with multiple years
        - metadata: Parsed metrics by year
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_hist.pkl")

        # Check cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.debug(f"Loaded cached data for {ticker}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")

        logger.info(f"Fetching historical fundamentals for {ticker}")

        try:
            stock = yf.Ticker(ticker)

            # Fetch financial statements
            income_stmt = stock.financials  # Annual income statement
            balance_sheet = stock.balance_sheet  # Annual balance sheet
            cash_flow = stock.cashflow  # Annual cash flow statement

            # Also fetch quarterly for more granularity
            income_stmt_q = stock.quarterly_financials
            balance_sheet_q = stock.quarterly_balance_sheet
            cash_flow_q = stock.quarterly_cashflow

            # Parse into usable format
            metrics_annual = self._parse_financials_by_year(
                ticker, income_stmt, balance_sheet, cash_flow, 'annual'
            )

            metrics_quarterly = self._parse_financials_by_year(
                ticker, income_stmt_q, balance_sheet_q, cash_flow_q, 'quarterly'
            )

            result = {
                'ticker': ticker,
                'fetch_date': datetime.now().strftime('%Y-%m-%d'),
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'income_statement_quarterly': income_stmt_q,
                'balance_sheet_quarterly': balance_sheet_q,
                'cash_flow_quarterly': cash_flow_q,
                'metrics_annual': metrics_annual,
                'metrics_quarterly': metrics_quarterly,
            }

            # Cache result
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            logger.info(f"Fetched {ticker}: {len(metrics_annual)} years of annual data")

            return result

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'fetch_date': datetime.now().strftime('%Y-%m-%d')
            }

    def _parse_financials_by_year(
        self,
        ticker: str,
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        period_type: str
    ) -> Dict[str, Dict]:
        """
        Parse financial statements into yearly metrics

        Returns: Dict mapping year -> metrics dict
        """
        metrics_by_year = {}

        # Get all available dates (columns in DataFrames)
        if income_stmt is None or income_stmt.empty:
            logger.warning(f"No income statement for {ticker}")
            return {}

        dates = income_stmt.columns

        for date in dates:
            year = date.year
            year_str = f"{year}"

            metrics = {
                'ticker': ticker,
                'year': year,
                'date': date,
                'period_type': period_type,
            }

            # ====== INCOME STATEMENT ======
            try:
                # Revenue
                metrics['total_revenue'] = self._safe_extract(income_stmt, 'Total Revenue', date)

                # Gross profit
                metrics['gross_profit'] = self._safe_extract(income_stmt, 'Gross Profit', date)

                # Operating income
                metrics['operating_income'] = self._safe_extract(income_stmt, 'Operating Income', date)

                # Net income
                metrics['net_income'] = self._safe_extract(income_stmt, 'Net Income', date)

                # EBITDA
                metrics['ebitda'] = self._safe_extract(income_stmt, 'EBITDA', date)

                # Calculate margins
                if metrics['total_revenue'] and metrics['total_revenue'] != 0:
                    if metrics['gross_profit']:
                        metrics['gross_margin'] = (metrics['gross_profit'] / metrics['total_revenue']) * 100
                    else:
                        metrics['gross_margin'] = 0

                    if metrics['operating_income']:
                        metrics['operating_margin'] = (metrics['operating_income'] / metrics['total_revenue']) * 100
                    else:
                        metrics['operating_margin'] = 0

                    if metrics['net_income']:
                        metrics['net_profit_margin'] = (metrics['net_income'] / metrics['total_revenue']) * 100
                    else:
                        metrics['net_profit_margin'] = 0

                    if metrics['ebitda']:
                        metrics['ebitda_margin'] = (metrics['ebitda'] / metrics['total_revenue']) * 100
                    else:
                        metrics['ebitda_margin'] = 0
                else:
                    metrics['gross_margin'] = 0
                    metrics['operating_margin'] = 0
                    metrics['net_profit_margin'] = 0
                    metrics['ebitda_margin'] = 0

                # Profitability flag
                metrics['is_profitable'] = 1 if metrics.get('net_profit_margin', 0) > 0 else 0

            except Exception as e:
                logger.debug(f"Error parsing income statement for {ticker} {year}: {e}")

            # ====== BALANCE SHEET ======
            try:
                # Assets
                metrics['total_assets'] = self._safe_extract(balance_sheet, 'Total Assets', date)
                metrics['current_assets'] = self._safe_extract(balance_sheet, 'Current Assets', date)
                metrics['cash'] = self._safe_extract(balance_sheet, 'Cash', date,
                                                      alt_keys=['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'])

                # Liabilities
                metrics['total_liabilities'] = self._safe_extract(balance_sheet, 'Total Liabilities Net Minority Interest', date,
                                                                   alt_keys=['Total Liabilities'])
                metrics['current_liabilities'] = self._safe_extract(balance_sheet, 'Current Liabilities', date)
                metrics['total_debt'] = self._safe_extract(balance_sheet, 'Total Debt', date)

                # Equity
                metrics['stockholders_equity'] = self._safe_extract(balance_sheet, 'Stockholders Equity', date,
                                                                      alt_keys=['Total Equity Gross Minority Interest'])

                # Calculate ratios
                if metrics['current_liabilities'] and metrics['current_liabilities'] != 0:
                    metrics['current_ratio'] = metrics['current_assets'] / metrics['current_liabilities']
                else:
                    metrics['current_ratio'] = 0

                if metrics['stockholders_equity'] and metrics['stockholders_equity'] != 0:
                    metrics['debt_to_equity'] = (metrics['total_debt'] or 0) / metrics['stockholders_equity']
                else:
                    metrics['debt_to_equity'] = 0

                # Net debt
                metrics['net_debt'] = (metrics['total_debt'] or 0) - (metrics['cash'] or 0)

                # Balance sheet health
                balance_sheet_score = 0
                if metrics['debt_to_equity'] < 2.0:
                    balance_sheet_score += 1
                if metrics['current_ratio'] > 1.0:
                    balance_sheet_score += 1
                if metrics.get('cash', 0) > 0:
                    balance_sheet_score += 1
                if metrics.get('cash', 0) > metrics.get('total_debt', 0):
                    balance_sheet_score += 1

                metrics['balance_sheet_healthy'] = 1 if balance_sheet_score >= 3 else 0

            except Exception as e:
                logger.debug(f"Error parsing balance sheet for {ticker} {year}: {e}")

            # ====== CASH FLOW ======
            try:
                # Operating cash flow
                metrics['operating_cash_flow'] = self._safe_extract(cash_flow, 'Operating Cash Flow', date)

                # Free cash flow
                metrics['free_cash_flow'] = self._safe_extract(cash_flow, 'Free Cash Flow', date)

                # Capital expenditure
                metrics['capex'] = self._safe_extract(cash_flow, 'Capital Expenditure', date)

                # FCF flag
                metrics['has_positive_fcf'] = 1 if metrics.get('free_cash_flow', 0) > 0 else 0

            except Exception as e:
                logger.debug(f"Error parsing cash flow for {ticker} {year}: {e}")

            # ====== QUALITY SCORE ======
            quality_score = 0

            # Profitability (3 points)
            if metrics.get('is_profitable', 0):
                quality_score += 1
            if metrics.get('net_profit_margin', 0) > 10:
                quality_score += 1
            if metrics.get('has_positive_fcf', 0):
                quality_score += 1

            # Balance sheet (2 points)
            if metrics.get('balance_sheet_healthy', 0):
                quality_score += 2

            # Size (estimate as 2 points if has data)
            if metrics.get('total_revenue', 0) > 10e9:  # >$10B revenue
                quality_score += 2

            metrics['quality_score'] = quality_score

            metrics_by_year[year_str] = metrics

        return metrics_by_year

    def _safe_extract(self, df: pd.DataFrame, key: str, date, alt_keys: List[str] = None) -> float:
        """Safely extract value from financial DataFrame"""
        if df is None or df.empty:
            return 0

        # Try main key
        if key in df.index:
            value = df.loc[key, date]
            if pd.notna(value):
                return float(value)

        # Try alternative keys
        if alt_keys:
            for alt_key in alt_keys:
                if alt_key in df.index:
                    value = df.loc[alt_key, date]
                    if pd.notna(value):
                        return float(value)

        return 0

    def is_profitable_at_date(self, ticker: str, target_date: str) -> bool:
        """
        Check if stock was profitable at a specific date

        Args:
            ticker: Stock symbol
            target_date: Date string like '2022-01-01'

        Returns:
            True if profitable, False otherwise
        """
        data = self.fetch_historical_fundamentals(ticker)

        if 'error' in data:
            return False

        target_year = pd.to_datetime(target_date).year

        # Try annual data first
        metrics_annual = data.get('metrics_annual', {})
        for year_str, metrics in metrics_annual.items():
            if int(year_str) == target_year:
                return metrics.get('is_profitable', 0) == 1

        # If no exact match, use closest previous year
        available_years = sorted([int(y) for y in metrics_annual.keys()])
        closest_year = None
        for year in available_years:
            if year <= target_year:
                closest_year = year

        if closest_year:
            return metrics_annual[str(closest_year)].get('is_profitable', 0) == 1

        return False

    def get_metrics_at_date(self, ticker: str, target_date: str) -> Dict:
        """
        Get all metrics closest to target date

        Returns dict with profitability, balance sheet, cash flow metrics
        """
        data = self.fetch_historical_fundamentals(ticker)

        if 'error' in data:
            return {}

        target_year = pd.to_datetime(target_date).year

        # Use annual data
        metrics_annual = data.get('metrics_annual', {})

        # Find closest previous year
        available_years = sorted([int(y) for y in metrics_annual.keys()])
        closest_year = None
        for year in available_years:
            if year <= target_year:
                closest_year = year

        if closest_year:
            return metrics_annual[str(closest_year)]

        return {}

    def fetch_multiple_stocks(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch historical fundamentals for multiple stocks"""
        results = {}
        total = len(tickers)

        logger.info(f"Fetching historical fundamentals for {total} stocks...")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{total}] Fetching {ticker}")
            results[ticker] = self.fetch_historical_fundamentals(ticker)

        logger.info(f"Completed fetching {total} stocks")
        return results

    def create_quality_filters_dict(self, tickers: List[str], reference_date: str = '2022-01-01') -> Dict[str, bool]:
        """
        Create dict of ticker -> passes_quality_filters for backtesting

        Args:
            tickers: List of tickers
            reference_date: Date to check quality at

        Returns:
            Dict mapping ticker -> True/False (passes filters)
        """
        quality_dict = {}

        for ticker in tickers:
            metrics = self.get_metrics_at_date(ticker, reference_date)

            if not metrics:
                quality_dict[ticker] = False
                continue

            # Apply filters
            passes = True

            # Profitability
            if metrics.get('net_profit_margin', 0) < 5:
                passes = False

            # FCF
            if not metrics.get('has_positive_fcf', 0):
                passes = False

            # Balance sheet
            if metrics.get('debt_to_equity', 999) > 2.0:
                passes = False

            if metrics.get('current_ratio', 0) < 1.0:
                passes = False

            quality_dict[ticker] = passes

        passed = sum(quality_dict.values())
        logger.info(f"Quality filters: {passed}/{len(tickers)} stocks passed")

        return quality_dict


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test with stocks that crashed in 2022
    test_tickers = ['AAPL', 'MSFT', 'SNAP', 'RBLX', 'ROKU', 'DOCU']

    fetcher = YahooHistoricalFetcher()

    print("Fetching historical fundamentals...\n")

    for ticker in test_tickers:
        data = fetcher.fetch_historical_fundamentals(ticker)

        if 'error' in data:
            print(f"{ticker}: ERROR - {data['error']}\n")
            continue

        print(f"\n{'='*60}")
        print(f"{ticker} - Historical Fundamentals")
        print(f"{'='*60}")

        # Show available years
        metrics = data.get('metrics_annual', {})
        print(f"Years available: {sorted(metrics.keys())}")

        # Show 2021 and 2022 metrics (before/during crash)
        for year in ['2021', '2022']:
            if year in metrics:
                m = metrics[year]
                print(f"\n{year}:")
                print(f"  Profitable: {'✓' if m.get('is_profitable') else '✗'}")
                print(f"  Net Margin: {m.get('net_profit_margin', 0):.1f}%")
                print(f"  Operating Margin: {m.get('operating_margin', 0):.1f}%")
                print(f"  Debt/Equity: {m.get('debt_to_equity', 0):.2f}")
                print(f"  Current Ratio: {m.get('current_ratio', 0):.2f}")
                print(f"  Has Positive FCF: {'✓' if m.get('has_positive_fcf') else '✗'}")
                print(f"  Quality Score: {m.get('quality_score', 0)}/7")

    # Test quality filters at specific date
    print(f"\n\n{'='*60}")
    print("QUALITY FILTERS TEST (January 2022)")
    print(f"{'='*60}")

    quality_dict = fetcher.create_quality_filters_dict(test_tickers, '2022-01-01')

    for ticker, passes in quality_dict.items():
        status = "✓ PASS" if passes else "✗ FAIL"
        print(f"{ticker}: {status}")
