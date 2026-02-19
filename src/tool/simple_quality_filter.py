"""
Quality Filter Using Existing Metadata
========================================

Uses existing metadata from sp500_data/metadata/*.json
Adds simple quality filters to avoid speculative stocks

Available data:
- P/E ratio, EPS
- Market cap
- Beta, volatility
- Sector

Quality filters:
1. Market cap > $50B (large cap only)
2. P/E ratio reasonable (5 < P/E < 80)
3. Has positive EPS (profitable)
4. Moderate volatility (< 60%)
5. Reasonable valuation (P/E < 80)
"""

import json
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SimpleQualityFilter:
    """Quality filter using existing metadata"""

    def __init__(self, metadata_dir: str = 'sp500_data/metadata'):
        self.metadata_dir = metadata_dir
        self.metadata_cache = {}

    def load_metadata(self, ticker: str) -> Dict:
        """Load metadata from JSON file"""
        if ticker in self.metadata_cache:
            return self.metadata_cache[ticker]

        json_file = os.path.join(self.metadata_dir, f"{ticker}.json")

        if not os.path.exists(json_file):
            logger.warning(f"No metadata for {ticker}")
            return {}

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.metadata_cache[ticker] = data
                return data
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            return {}

    def load_all_metadata(self) -> Dict[str, Dict]:
        """Load all metadata files"""
        metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]

        logger.info(f"Loading {len(metadata_files)} metadata files...")

        all_metadata = {}
        for file in metadata_files:
            ticker = file.replace('.json', '')
            all_metadata[ticker] = self.load_metadata(ticker)

        return all_metadata

    def passes_quality_filters(self, ticker: str) -> bool:
        """
        Check if stock passes quality filters

        Filters:
        1. Market cap > $50B (large cap)
        2. Positive EPS (profitable)
        3. Reasonable P/E (5 < P/E < 80)
        4. Moderate volatility (< 60%)
        5. Not extreme beta (< 2.5)

        Returns:
            True if passes all filters, False otherwise
        """
        metadata = self.load_metadata(ticker)

        if not metadata:
            return False

        # Filter 1: Market cap > $50B
        market_cap = metadata.get('market_cap', 0)
        if market_cap < 50e9:
            logger.debug(f"{ticker}: FAIL - Market cap ${market_cap/1e9:.1f}B < $50B")
            return False

        # Filter 2: Positive EPS (profitable)
        eps = metadata.get('eps', 0)
        if eps <= 0:
            logger.debug(f"{ticker}: FAIL - EPS ${eps:.2f} <= 0 (unprofitable)")
            return False

        # Filter 3: Reasonable P/E ratio
        pe_ratio = metadata.get('pe_ratio', 0)
        if pe_ratio <= 0 or pe_ratio > 80:
            logger.debug(f"{ticker}: FAIL - P/E {pe_ratio:.1f} (extreme valuation)")
            return False

        # Filter 4: Moderate volatility
        volatility = metadata.get('annual_volatility_pct', 0)
        if volatility > 60:
            logger.debug(f"{ticker}: FAIL - Volatility {volatility:.1f}% > 60%")
            return False

        # Filter 5: Reasonable beta
        beta = metadata.get('beta', 1.0)
        if beta > 2.5:
            logger.debug(f"{ticker}: FAIL - Beta {beta:.2f} > 2.5 (too volatile)")
            return False

        logger.debug(f"{ticker}: PASS all filters")
        return True

    def get_quality_stocks(self, tickers: List[str]) -> List[str]:
        """
        Filter list of tickers to only quality stocks

        Returns:
            List of tickers that pass quality filters
        """
        quality_stocks = []

        for ticker in tickers:
            if self.passes_quality_filters(ticker):
                quality_stocks.append(ticker)

        logger.info(f"Quality filter: {len(quality_stocks)}/{len(tickers)} stocks passed")
        return quality_stocks

    def get_quality_score(self, ticker: str) -> float:
        """
        Calculate quality score (0-10) based on available data

        Scoring:
        - Profitability (EPS > 0): 2 points
        - Low P/E (<25): 2 points
        - Large market cap (>$200B): 2 points
        - Low volatility (<40%): 2 points
        - Low beta (<1.2): 2 points

        Returns:
            Quality score 0-10
        """
        metadata = self.load_metadata(ticker)

        if not metadata:
            return 0

        score = 0.0

        # Profitability (2 points)
        eps = metadata.get('eps', 0)
        if eps > 0:
            score += 2.0

        # Valuation (2 points)
        pe_ratio = metadata.get('pe_ratio', 0)
        if 5 < pe_ratio < 25:
            score += 2.0
        elif 25 <= pe_ratio < 40:
            score += 1.0

        # Size (2 points)
        market_cap = metadata.get('market_cap', 0)
        if market_cap > 200e9:
            score += 2.0
        elif market_cap > 100e9:
            score += 1.0

        # Volatility (2 points)
        volatility = metadata.get('annual_volatility_pct', 0)
        if volatility < 30:
            score += 2.0
        elif volatility < 40:
            score += 1.0

        # Beta (2 points)
        beta = metadata.get('beta', 1.0)
        if beta < 1.0:
            score += 2.0
        elif beta < 1.5:
            score += 1.0

        return score

    def create_quality_report(self, tickers: List[str]):
        """Create summary report of quality filtering"""
        total = len(tickers)
        passed = 0
        failed_reasons = {
            'market_cap': 0,
            'eps': 0,
            'pe_ratio': 0,
            'volatility': 0,
            'beta': 0
        }

        quality_stocks = []
        speculative_stocks = []

        for ticker in tickers:
            metadata = self.load_metadata(ticker)

            if not metadata:
                continue

            # Check each filter
            passes = True

            # Market cap
            if metadata.get('market_cap', 0) < 50e9:
                failed_reasons['market_cap'] += 1
                passes = False

            # EPS
            if metadata.get('eps', 0) <= 0:
                failed_reasons['eps'] += 1
                passes = False

            # P/E ratio
            pe = metadata.get('pe_ratio', 0)
            if pe <= 0 or pe > 80:
                failed_reasons['pe_ratio'] += 1
                passes = False

            # Volatility
            if metadata.get('annual_volatility_pct', 0) > 60:
                failed_reasons['volatility'] += 1
                passes = False

            # Beta
            if metadata.get('beta', 1.0) > 2.5:
                failed_reasons['beta'] += 1
                passes = False

            if passes:
                passed += 1
                quality_stocks.append(ticker)
            else:
                speculative_stocks.append(ticker)

        report = f"""
        QUALITY FILTER REPORT
        =====================
        Total stocks: {total}
        Passed filters: {passed} ({passed/total*100:.1f}%)
        Failed filters: {total - passed} ({(total-passed)/total*100:.1f}%)

        FAILURE REASONS:
        - Small market cap (<$50B): {failed_reasons['market_cap']} stocks
        - Negative EPS (unprofitable): {failed_reasons['eps']} stocks
        - Extreme P/E ratio: {failed_reasons['pe_ratio']} stocks
        - High volatility (>60%): {failed_reasons['volatility']} stocks
        - High beta (>2.5): {failed_reasons['beta']} stocks

        QUALITY STOCKS (sample): {', '.join(quality_stocks[:20])}

        SPECULATIVE STOCKS (sample): {', '.join(speculative_stocks[:20])}
        """

        print(report)
        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'failure_reasons': failed_reasons,
            'quality_stocks': quality_stocks,
            'speculative_stocks': speculative_stocks
        }


if __name__ == '__main__':
    # Test the quality filter
    logging.basicConfig(level=logging.INFO)

    filter_obj = SimpleQualityFilter()

    # Test with stocks from 2022 crash
    test_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA',  # Quality mega-caps
        'SNAP', 'RBLX', 'ROKU', 'DOCU', 'PARA', 'PINS', 'ETSY',   # Speculative stocks that crashed
    ]

    print("\n" + "="*70)
    print("TESTING QUALITY FILTERS ON 2022 CRASH STOCKS")
    print("="*70)

    print("\nQUALITY MEGA-CAPS:")
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']:
        metadata = filter_obj.load_metadata(ticker)
        passes = filter_obj.passes_quality_filters(ticker)
        score = filter_obj.get_quality_score(ticker)

        if metadata:
            status = "✓ PASS" if passes else "✗ FAIL"
            print(f"  {ticker}: {status} (Score: {score:.1f}/10)")
            print(f"    Market Cap: ${metadata.get('market_cap', 0)/1e9:.1f}B, "
                  f"EPS: ${metadata.get('eps', 0):.2f}, "
                  f"P/E: {metadata.get('pe_ratio', 0):.1f}, "
                  f"Vol: {metadata.get('annual_volatility_pct', 0):.1f}%")

    print("\nSPECULATIVE STOCKS (that crashed in 2022):")
    for ticker in ['SNAP', 'RBLX', 'ROKU', 'DOCU']:
        metadata = filter_obj.load_metadata(ticker)
        passes = filter_obj.passes_quality_filters(ticker)
        score = filter_obj.get_quality_score(ticker)

        if metadata:
            status = "✓ PASS" if passes else "✗ FAIL"
            print(f"  {ticker}: {status} (Score: {score:.1f}/10)")
            print(f"    Market Cap: ${metadata.get('market_cap', 0)/1e9:.1f}B, "
                  f"EPS: ${metadata.get('eps', 0):.2f}, "
                  f"P/E: {metadata.get('pe_ratio', 0):.1f}, "
                  f"Vol: {metadata.get('annual_volatility_pct', 0):.1f}%")

    # Generate full report
    print("\n" + "="*70)
    all_tickers = [f.replace('.json', '') for f in os.listdir('sp500_data/metadata') if f.endswith('.json')]
    filter_obj.create_quality_report(all_tickers)
