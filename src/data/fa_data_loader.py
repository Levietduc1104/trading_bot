"""
FA Data Loader - Fast access to FMP fundamental data
Loads and caches financial ratios, key metrics, and financial scores
"""
import json
import os
import logging

logger = logging.getLogger(__name__)


class FADataLoader:
    """
    Loads and provides fast access to FMP fundamental data

    Data sources:
    - financial_ratios_ttm.json (60 fields: profitability, leverage, liquidity)
    - key_metrics_ttm.json (43 fields: returns, valuation, efficiency)
    - financial_scores.json (11 fields: Altman Z-Score, Piotroski F-Score)
    """

    def __init__(self, data_dir='sp500_data/fmp_fundamentals'):
        self.data_dir = data_dir
        self.ratios = {}
        self.metrics = {}
        self.scores = {}
        self.loaded = False

    def load_all(self):
        """Load all FA data into memory (called once at startup)"""
        if self.loaded:
            return

        logger.info("Loading FMP fundamental data...")

        # Load financial ratios
        ratios_path = os.path.join(self.data_dir, 'financial_ratios_ttm.json')
        if os.path.exists(ratios_path):
            with open(ratios_path, 'r') as f:
                self.ratios = json.load(f)
            logger.info(f"✓ Loaded ratios for {len(self.ratios)} stocks")
        else:
            logger.warning(f"Financial ratios not found: {ratios_path}")

        # Load key metrics
        metrics_path = os.path.join(self.data_dir, 'key_metrics_ttm.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"✓ Loaded metrics for {len(self.metrics)} stocks")
        else:
            logger.warning(f"Key metrics not found: {metrics_path}")

        # Load financial scores
        scores_path = os.path.join(self.data_dir, 'financial_scores.json')
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                self.scores = json.load(f)
            logger.info(f"✓ Loaded scores for {len(self.scores)} stocks")
        else:
            logger.warning(f"Financial scores not found: {scores_path}")

        self.loaded = True

        # Calculate coverage
        common_stocks = set(self.ratios.keys()) & set(self.metrics.keys()) & set(self.scores.keys())
        logger.info(f"✓ Complete FA data for {len(common_stocks)} stocks")

    def get_fa_data(self, ticker):
        """
        Get all fundamental data for a ticker

        Returns:
            dict: Combined FA data, or None if ticker not found
        """
        if not self.loaded:
            self.load_all()

        if ticker not in self.ratios or ticker not in self.metrics or ticker not in self.scores:
            return None

        # Combine all data sources
        fa_data = {}

        # Add ratios
        if ticker in self.ratios:
            fa_data.update(self.ratios[ticker])

        # Add metrics
        if ticker in self.metrics:
            fa_data.update(self.metrics[ticker])

        # Add scores
        if ticker in self.scores:
            fa_data.update(self.scores[ticker])

        return fa_data

    def safe_get(self, ticker, field, default=0):
        """
        Safely get a specific field for a ticker

        Args:
            ticker: Stock symbol
            field: Field name (e.g., 'altmanZScore', 'returnOnEquityTTM')
            default: Default value if field is missing or None

        Returns:
            float: Field value or default
        """
        fa_data = self.get_fa_data(ticker)
        if fa_data is None:
            return default

        value = fa_data.get(field, default)
        return value if value is not None else default

    def has_data(self, ticker):
        """Check if we have FA data for this ticker"""
        if not self.loaded:
            self.load_all()
        return (ticker in self.ratios and
                ticker in self.metrics and
                ticker in self.scores)

    def get_quality_stats(self):
        """Get summary statistics about data quality"""
        if not self.loaded:
            self.load_all()

        common_stocks = set(self.ratios.keys()) & set(self.metrics.keys()) & set(self.scores.keys())

        stats = {
            'total_stocks': len(common_stocks),
            'ratios_count': len(self.ratios),
            'metrics_count': len(self.metrics),
            'scores_count': len(self.scores),
        }

        if len(common_stocks) > 0:
            # Quality distribution
            safe_zone = sum(1 for s in common_stocks
                          if self.safe_get(s, 'altmanZScore', 0) > 3.0)
            strong_piotroski = sum(1 for s in common_stocks
                                  if self.safe_get(s, 'piotroskiScore', 0) >= 7)
            high_roe = sum(1 for s in common_stocks
                          if self.safe_get(s, 'returnOnEquityTTM', 0) > 0.15)
            low_debt = sum(1 for s in common_stocks
                          if self.safe_get(s, 'debtToEquityRatioTTM', 999) < 1.0)

            stats.update({
                'safe_zone_count': safe_zone,
                'safe_zone_pct': safe_zone / len(common_stocks) * 100,
                'strong_piotroski_count': strong_piotroski,
                'strong_piotroski_pct': strong_piotroski / len(common_stocks) * 100,
                'high_roe_count': high_roe,
                'high_roe_pct': high_roe / len(common_stocks) * 100,
                'low_debt_count': low_debt,
                'low_debt_pct': low_debt / len(common_stocks) * 100,
            })

        return stats


# Singleton instance for reuse
_fa_loader_instance = None

def get_fa_loader():
    """Get singleton FA data loader instance"""
    global _fa_loader_instance
    if _fa_loader_instance is None:
        _fa_loader_instance = FADataLoader()
        _fa_loader_instance.load_all()
    return _fa_loader_instance
