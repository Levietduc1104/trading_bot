"""
Historical FA Data Loader - Fast access to historical fundamental data with date-based queries

Loads quarterly historical data from FMP (2005-2025)
Supports date-based lookups to avoid look-ahead bias in backtesting
"""
import json
import os
import logging
from datetime import datetime, timedelta
from bisect import bisect_left

logger = logging.getLogger(__name__)


class HistoricalFADataLoader:
    """
    Loads and provides fast access to historical fundamental data

    Data sources:
    - financial_ratios_historical.json (64 fields, quarterly, 2005-2025)
    - key_metrics_historical.json (43 fields, quarterly, 2005-2025)
    - earnings_surprises.json (actual vs estimated EPS, 2006-2025)
    - analyst_estimates.json (forward consensus EPS/revenue, 2006-2025)
    - income_statements.json (quarterly P&L, 2006-2025)
    - balance_sheets.json (quarterly balance sheet, 2006-2025)
    - cashflow_statements.json (quarterly cash flows, 2006-2025)

    Key feature: Date-based queries to avoid look-ahead bias
    """

    def __init__(self, data_dir='sp500_data/fmp_fundamentals_historical',
                 premium_dir='sp500_data/fmp_fundamentals_premium'):
        self.data_dir = data_dir
        self.premium_dir = premium_dir
        self.ratios = {}
        self.metrics = {}
        # Premium data
        self.earnings = {}       # earnings surprises
        self.analyst = {}        # analyst estimates
        self.income = {}         # income statements
        self.balance = {}        # balance sheets
        self.cashflow = {}       # cash flow statements
        self.loaded = False

        # Date index for fast lookups
        self.date_index = {}          # {ticker: [sorted_dates]} for ratios
        self.earnings_date_idx = {}   # {ticker: [sorted_dates]}
        self.analyst_date_idx = {}
        self.income_date_idx = {}
        self.balance_date_idx = {}
        self.cashflow_date_idx = {}

    def load_all(self):
        """Load all historical FA data into memory (called once at startup)"""
        if self.loaded:
            return

        logger.info("Loading historical FMP fundamental data...")

        # Load historical ratios
        ratios_path = os.path.join(self.data_dir, 'financial_ratios_historical.json')
        if os.path.exists(ratios_path):
            with open(ratios_path, 'r') as f:
                self.ratios = json.load(f)
            logger.info(f"✓ Loaded historical ratios for {len(self.ratios)} stocks")
        else:
            logger.warning(f"Historical ratios not found: {ratios_path}")

        # Load historical metrics
        metrics_path = os.path.join(self.data_dir, 'key_metrics_historical.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"✓ Loaded historical metrics for {len(self.metrics)} stocks")
        else:
            logger.warning(f"Historical metrics not found: {metrics_path}")

        # Load premium data
        for attr, fname, label in [
            ('earnings', 'earnings_surprises.json',   'earnings surprises'),
            ('analyst',  'analyst_estimates.json',    'analyst estimates'),
            ('income',   'income_statements.json',    'income statements'),
            ('balance',  'balance_sheets.json',       'balance sheets'),
            ('cashflow', 'cashflow_statements.json',  'cashflow statements'),
        ]:
            path = os.path.join(self.premium_dir, fname)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    setattr(self, attr, json.load(f))
                logger.info(f"✓ Loaded {label} for {len(getattr(self, attr))} stocks")
            else:
                logger.warning(f"Premium data not found: {path}")

        self.loaded = True

        # Build date indices for fast queries
        self._build_date_index()

        common_stocks = set(self.ratios.keys()) & set(self.metrics.keys())
        logger.info(f"✓ Complete historical FA data for {len(common_stocks)} stocks")

    def _build_date_index(self):
        """Build date index for fast binary search"""
        for ticker in self.ratios.keys():
            dates = [r['date'] for r in self.ratios[ticker]]
            self.date_index[ticker] = sorted(dates)

        for attr, idx_attr in [
            ('earnings', 'earnings_date_idx'),
            ('analyst',  'analyst_date_idx'),
            ('income',   'income_date_idx'),
            ('balance',  'balance_date_idx'),
            ('cashflow', 'cashflow_date_idx'),
        ]:
            src = getattr(self, attr)
            idx = {}
            for ticker, records in src.items():
                dates = sorted([r['date'] for r in records if r.get('date')])
                if dates:
                    idx[ticker] = dates
            setattr(self, idx_attr, idx)

    def _get_record_at_date(self, records, date_index, ticker, target_date_str):
        """
        Generic: return most recent record for ticker with date <= target_date_str.
        Returns the raw record dict or None.
        """
        if ticker not in date_index:
            return None
        dates = date_index[ticker]
        idx = bisect_left(dates, target_date_str)
        if idx >= len(dates):
            idx = len(dates) - 1
        elif dates[idx] > target_date_str:
            if idx == 0:
                return None
            idx -= 1
        target = dates[idx]
        src = records.get(ticker, [])
        for r in src:
            if r.get('date') == target:
                return r
        return None

    def get_earnings_at_date(self, ticker, target_date):
        """Return most recent earnings surprise record before target_date."""
        if isinstance(target_date, str):
            d = target_date
        else:
            d = target_date.strftime('%Y-%m-%d')
        return self._get_record_at_date(self.earnings, self.earnings_date_idx, ticker, d)

    def get_analyst_at_date(self, ticker, target_date):
        """Return most recent analyst estimate record before target_date."""
        if isinstance(target_date, str):
            d = target_date
        else:
            d = target_date.strftime('%Y-%m-%d')
        return self._get_record_at_date(self.analyst, self.analyst_date_idx, ticker, d)

    def get_income_at_date(self, ticker, target_date):
        """Return most recent income statement before target_date."""
        if isinstance(target_date, str):
            d = target_date
        else:
            d = target_date.strftime('%Y-%m-%d')
        return self._get_record_at_date(self.income, self.income_date_idx, ticker, d)

    def get_balance_at_date(self, ticker, target_date):
        """Return most recent balance sheet before target_date."""
        if isinstance(target_date, str):
            d = target_date
        else:
            d = target_date.strftime('%Y-%m-%d')
        return self._get_record_at_date(self.balance, self.balance_date_idx, ticker, d)

    def get_cashflow_at_date(self, ticker, target_date):
        """Return most recent cash flow statement before target_date."""
        if isinstance(target_date, str):
            d = target_date
        else:
            d = target_date.strftime('%Y-%m-%d')
        return self._get_record_at_date(self.cashflow, self.cashflow_date_idx, ticker, d)

    def get_income_history(self, ticker, target_date, n_quarters=4):
        """Return last n_quarters income statements before target_date (most recent first)."""
        if isinstance(target_date, str):
            d = target_date
        else:
            d = target_date.strftime('%Y-%m-%d')
        if ticker not in self.income_date_idx:
            return []
        dates = self.income_date_idx[ticker]
        idx = bisect_left(dates, d)
        if idx >= len(dates):
            idx = len(dates) - 1
        elif dates[idx] > d:
            idx -= 1
        # Collect up to n_quarters records going backwards
        result = []
        src = {r['date']: r for r in self.income.get(ticker, [])}
        for i in range(idx, max(-1, idx - n_quarters), -1):
            if i >= 0 and dates[i] in src:
                result.append(src[dates[i]])
        return result

    def get_fa_data_at_date(self, ticker, target_date):
        """
        Get fundamental data as of a specific date (most recent quarter before target_date)

        Args:
            ticker: Stock symbol
            target_date: Target date (datetime or string 'YYYY-MM-DD')

        Returns:
            dict: Combined FA data (ratios + metrics) or None if not found
        """
        if not self.loaded:
            self.load_all()

        if ticker not in self.ratios or ticker not in self.metrics:
            return None

        # Convert target_date to string format
        if isinstance(target_date, str):
            target_date_str = target_date
        else:
            target_date_str = target_date.strftime('%Y-%m-%d')

        # Find most recent quarter before target_date using binary search
        if ticker not in self.date_index:
            return None

        dates = self.date_index[ticker]
        idx = bisect_left(dates, target_date_str)

        # If exact match or after target, use previous quarter
        if idx >= len(dates):
            idx = len(dates) - 1
        elif dates[idx] > target_date_str and idx > 0:
            idx -= 1

        # Get data for that quarter
        ratio_data = None
        metric_data = None

        for r in self.ratios[ticker]:
            if r['date'] == dates[idx]:
                ratio_data = r
                break

        for m in self.metrics[ticker]:
            if m['date'] == dates[idx]:
                metric_data = m
                break

        if not ratio_data or not metric_data:
            return None

        # Combine ratio and metric data
        fa_data = {}
        fa_data.update(ratio_data)
        fa_data.update(metric_data)

        return fa_data

    def safe_get(self, ticker, target_date, field, default=0):
        """
        Safely get a specific field for a ticker at a specific date

        Args:
            ticker: Stock symbol
            target_date: Target date (datetime or string)
            field: Field name (e.g., 'returnOnEquity', 'netProfitMargin')
            default: Default value if field is missing or None

        Returns:
            float: Field value or default
        """
        fa_data = self.get_fa_data_at_date(ticker, target_date)
        if fa_data is None:
            return default

        value = fa_data.get(field, default)
        return value if value is not None else default

    def has_data(self, ticker):
        """Check if we have historical FA data for this ticker"""
        if not self.loaded:
            self.load_all()
        return (ticker in self.ratios and ticker in self.metrics)

    def get_quality_stats(self):
        """Get summary statistics about data quality"""
        if not self.loaded:
            self.load_all()

        common_stocks = set(self.ratios.keys()) & set(self.metrics.keys())

        stats = {
            'total_stocks': len(common_stocks),
            'ratios_count': len(self.ratios),
            'metrics_count': len(self.metrics),
        }

        if len(common_stocks) > 0:
            # Sample quality metrics (using latest quarter)
            high_current_ratio = 0
            high_roe = 0
            profitable = 0
            low_debt = 0

            for ticker in common_stocks:
                if len(self.ratios[ticker]) > 0 and len(self.metrics[ticker]) > 0:
                    latest_ratio = self.ratios[ticker][0]
                    latest_metric = self.metrics[ticker][0]

                    if latest_ratio.get('currentRatio', 0) > 1.2:
                        high_current_ratio += 1
                    if latest_metric.get('returnOnEquity', 0) > 0.15:
                        high_roe += 1
                    if latest_ratio.get('netProfitMargin', 0) > 0.05:
                        profitable += 1
                    if latest_ratio.get('debtToEquityRatio', 999) < 1.0:
                        low_debt += 1

            stats.update({
                'high_current_ratio_count': high_current_ratio,
                'high_current_ratio_pct': high_current_ratio / len(common_stocks) * 100,
                'high_roe_count': high_roe,
                'high_roe_pct': high_roe / len(common_stocks) * 100,
                'profitable_count': profitable,
                'profitable_pct': profitable / len(common_stocks) * 100,
                'low_debt_count': low_debt,
                'low_debt_pct': low_debt / len(common_stocks) * 100,
            })

        return stats


# Singleton instance for reuse
_historical_fa_loader_instance = None

def get_historical_fa_loader():
    """Get singleton historical FA data loader instance"""
    global _historical_fa_loader_instance
    if _historical_fa_loader_instance is None:
        _historical_fa_loader_instance = HistoricalFADataLoader()
        _historical_fa_loader_instance.load_all()
    return _historical_fa_loader_instance
