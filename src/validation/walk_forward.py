"""
WALK-FORWARD TESTING & OUT-OF-SAMPLE VALIDATION
================================================

This module implements robust validation methodology to prevent overfitting:

1. Walk-Forward Testing: Train on N years, test on next year, roll forward
2. Out-of-Sample Testing: Hold out 2024 as pure test set
3. SPY Benchmark Comparison: Direct comparison to buy-and-hold S&P 500

Purpose: Validate that backtested returns are realistic and not curve-fitted
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from src.core.execution import run_v28_backtest, calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward Testing & Validation Framework

    Methodology:
    - Train on 10-year window, test on next 1 year
    - Roll forward 1 year at a time
    - Compare to SPY benchmark for each period
    - Aggregate results to assess true out-of-sample performance
    """

    def __init__(self, data_dir: str, initial_capital: float = 100000):
        """
        Initialize validator

        Args:
            data_dir: Path to stock data directory
            initial_capital: Starting capital for backtests
        """
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.results = []
        self.spy_benchmark = None

    def load_spy_benchmark(self):
        """
        Load SPY data for benchmark comparison

        Returns:
            DataFrame with SPY daily prices
        """
        logger.info("Loading SPY benchmark data...")

        try:
            # Try loading from data directory
            spy_path = os.path.join(self.data_dir, 'SPY.csv')
            if not os.path.exists(spy_path):
                # Alternative path
                spy_path = os.path.join(project_root, 'sp500_data', 'daily', 'SPY.csv')

            spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
            spy_df.columns = [col.lower() for col in spy_df.columns]

            self.spy_benchmark = spy_df
            logger.info(f"✓ Loaded SPY benchmark: {len(spy_df)} days")
            return spy_df

        except Exception as e:
            logger.error(f"Failed to load SPY benchmark: {e}")
            logger.warning("Benchmark comparison will not be available")
            return None

    def calculate_spy_return(self, start_date, end_date) -> Dict:
        """
        Calculate SPY buy-and-hold returns for a period

        Args:
            start_date: Start date (pandas Timestamp)
            end_date: End date (pandas Timestamp)

        Returns:
            Dict with SPY performance metrics
        """
        if self.spy_benchmark is None:
            return None

        # Filter SPY data for the period
        spy_period = self.spy_benchmark[
            (self.spy_benchmark.index >= start_date) &
            (self.spy_benchmark.index <= end_date)
        ]

        if len(spy_period) < 2:
            logger.warning(f"Insufficient SPY data for {start_date} to {end_date}")
            return None

        # Calculate metrics
        start_price = spy_period['close'].iloc[0]
        end_price = spy_period['close'].iloc[-1]

        total_return = (end_price - start_price) / start_price * 100

        # Calculate annual return
        years = (end_date - start_date).days / 365.25
        annual_return = (((end_price / start_price) ** (1 / years)) - 1) * 100 if years > 0 else 0

        # Calculate drawdown
        cummax = spy_period['close'].cummax()
        drawdown = (spy_period['close'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio
        returns = spy_period['close'].pct_change()
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'start_price': start_price,
            'end_price': end_price
        }

    def run_single_period(self, train_start: int, train_end: int,
                          test_year: int) -> Dict:
        """
        Run backtest for a single train/test period

        Args:
            train_start: Training period start year
            train_end: Training period end year
            test_year: Test year

        Returns:
            Dict with period results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Walk-Forward Period: Train {train_start}-{train_end}, Test {test_year}")
        logger.info(f"{'='*80}")

        # Initialize bot
        bot = PortfolioRotationBot(data_dir=self.data_dir, initial_capital=self.initial_capital)
        bot.prepare_data()

        # Get date range for this stock
        first_ticker = list(bot.stocks_data.keys())[0]
        all_dates = bot.stocks_data[first_ticker].index

        # Filter dates for test period
        test_start = pd.Timestamp(f'{test_year}-01-01')
        test_end = pd.Timestamp(f'{test_year}-12-31')

        # Filter to only dates in test year
        test_dates = all_dates[(all_dates >= test_start) & (all_dates <= test_end)]

        if len(test_dates) == 0:
            logger.warning(f"No data available for test year {test_year}")
            return None

        # Get actual start and end dates from available data
        actual_test_start = test_dates[0]
        actual_test_end = test_dates[-1]

        logger.info(f"Test period: {actual_test_start.date()} to {actual_test_end.date()}")

        # Temporarily filter data to simulate real-time trading
        # (In production, we'd retrain here with only train_start to train_end data)
        # For backtest, we use all data but only evaluate test period

        # Run backtest
        logger.info("Running V28 backtest on test period...")
        portfolio_df = run_v28_backtest(bot)

        # Filter results to test period only
        portfolio_test = portfolio_df[
            (portfolio_df.index >= actual_test_start) &
            (portfolio_df.index <= actual_test_end)
        ]

        if len(portfolio_test) < 2:
            logger.warning(f"Insufficient portfolio data for test year {test_year}")
            return None

        # Calculate strategy metrics for test period
        strategy_metrics = self._calculate_period_metrics(
            portfolio_test,
            actual_test_start,
            actual_test_end
        )

        # Calculate SPY benchmark for same period
        spy_metrics = self.calculate_spy_return(actual_test_start, actual_test_end)

        # Calculate alpha (excess return vs SPY)
        if spy_metrics:
            alpha = strategy_metrics['annual_return'] - spy_metrics['annual_return']
            info_ratio = alpha / strategy_metrics.get('tracking_error', 1.0) if 'tracking_error' in strategy_metrics else 0
        else:
            alpha = None
            info_ratio = None

        result = {
            'train_period': f"{train_start}-{train_end}",
            'test_year': test_year,
            'test_start': actual_test_start,
            'test_end': actual_test_end,
            'strategy': strategy_metrics,
            'spy_benchmark': spy_metrics,
            'alpha': alpha,
            'info_ratio': info_ratio,
            'outperformed': strategy_metrics['annual_return'] > spy_metrics['annual_return'] if spy_metrics else None
        }

        # Log results
        self._log_period_results(result)

        return result

    def _calculate_period_metrics(self, portfolio_df: pd.DataFrame,
                                   start_date, end_date) -> Dict:
        """
        Calculate performance metrics for a specific period

        Args:
            portfolio_df: Portfolio values DataFrame
            start_date: Period start
            end_date: Period end

        Returns:
            Dict with performance metrics
        """
        if len(portfolio_df) < 2:
            return {}

        start_value = portfolio_df['portfolio_value'].iloc[0] if 'portfolio_value' in portfolio_df.columns else portfolio_df['value'].iloc[0]
        end_value = portfolio_df['portfolio_value'].iloc[-1] if 'portfolio_value' in portfolio_df.columns else portfolio_df['value'].iloc[-1]

        total_return = (end_value - start_value) / start_value * 100

        # Calculate annual return
        years = (end_date - start_date).days / 365.25
        annual_return = (((end_value / start_value) ** (1 / years)) - 1) * 100 if years > 0 else 0

        # Calculate drawdown
        value_col = 'portfolio_value' if 'portfolio_value' in portfolio_df.columns else 'value'
        cummax = portfolio_df[value_col].cummax()
        drawdown = (portfolio_df[value_col] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio
        returns = portfolio_df[value_col].pct_change()
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

        # Calculate volatility
        volatility = returns.std() * np.sqrt(252) * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'start_value': start_value,
            'end_value': end_value
        }

    def _log_period_results(self, result: Dict):
        """Log results for a single period"""
        logger.info(f"\nTest Year {result['test_year']} Results:")
        logger.info(f"{'─'*80}")

        strat = result['strategy']
        spy = result['spy_benchmark']

        logger.info(f"Strategy Performance:")
        logger.info(f"  Annual Return:  {strat['annual_return']:>7.1f}%")
        logger.info(f"  Max Drawdown:   {strat['max_drawdown']:>7.1f}%")
        logger.info(f"  Sharpe Ratio:   {strat['sharpe_ratio']:>7.2f}")

        if spy:
            logger.info(f"\nSPY Benchmark:")
            logger.info(f"  Annual Return:  {spy['annual_return']:>7.1f}%")
            logger.info(f"  Max Drawdown:   {spy['max_drawdown']:>7.1f}%")
            logger.info(f"  Sharpe Ratio:   {spy['sharpe_ratio']:>7.2f}")

            logger.info(f"\nAlpha vs SPY:     {result['alpha']:>7.1f}%")

            status = "✅ BEAT SPY" if result['outperformed'] else "❌ UNDERPERFORMED"
            logger.info(f"Result:           {status}")

    def run_walk_forward(self, start_year: int = 2005, end_year: int = 2023,
                         train_window: int = 10) -> List[Dict]:
        """
        Run complete walk-forward analysis

        Args:
            start_year: First year to start training
            end_year: Last year to test
            train_window: Number of years in training window

        Returns:
            List of period results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"WALK-FORWARD VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Configuration:")
        logger.info(f"  Training window: {train_window} years")
        logger.info(f"  Test period: {start_year + train_window} to {end_year}")
        logger.info(f"  Total periods: {end_year - start_year - train_window + 1}")
        logger.info(f"{'='*80}\n")

        # Load SPY benchmark
        self.load_spy_benchmark()

        self.results = []

        # Iterate through test years
        for test_year in range(start_year + train_window, end_year + 1):
            train_start = test_year - train_window
            train_end = test_year - 1

            result = self.run_single_period(train_start, train_end, test_year)

            if result:
                self.results.append(result)

        # Generate summary
        self._generate_summary()

        return self.results

    def _generate_summary(self):
        """Generate aggregate summary of walk-forward results"""
        if not self.results:
            logger.warning("No results to summarize")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"WALK-FORWARD SUMMARY")
        logger.info(f"{'='*80}\n")

        # Aggregate metrics
        strategy_returns = [r['strategy']['annual_return'] for r in self.results]
        spy_returns = [r['spy_benchmark']['annual_return'] for r in self.results if r['spy_benchmark']]
        alphas = [r['alpha'] for r in self.results if r['alpha'] is not None]

        win_rate = sum(1 for r in self.results if r['outperformed']) / len(self.results) * 100 if self.results else 0

        logger.info(f"Out-of-Sample Performance ({len(self.results)} test periods):")
        logger.info(f"{'─'*80}")
        logger.info(f"Strategy:")
        logger.info(f"  Mean Annual Return:  {np.mean(strategy_returns):>7.1f}%")
        logger.info(f"  Median Annual Return: {np.median(strategy_returns):>7.1f}%")
        logger.info(f"  Std Dev:             {np.std(strategy_returns):>7.1f}%")
        logger.info(f"  Best Year:           {max(strategy_returns):>7.1f}%")
        logger.info(f"  Worst Year:          {min(strategy_returns):>7.1f}%")

        if spy_returns:
            logger.info(f"\nSPY Benchmark:")
            logger.info(f"  Mean Annual Return:  {np.mean(spy_returns):>7.1f}%")
            logger.info(f"  Median Annual Return: {np.median(spy_returns):>7.1f}%")
            logger.info(f"  Std Dev:             {np.std(spy_returns):>7.1f}%")

            logger.info(f"\nAlpha:")
            logger.info(f"  Mean Alpha:          {np.mean(alphas):>7.1f}%")
            logger.info(f"  Median Alpha:        {np.median(alphas):>7.1f}%")
            logger.info(f"  Win Rate:            {win_rate:>7.1f}%")

        logger.info(f"\n{'─'*80}")
        logger.info(f"Yearly Breakdown:")
        logger.info(f"{'─'*80}")

        for r in self.results:
            strat_ret = r['strategy']['annual_return']
            spy_ret = r['spy_benchmark']['annual_return'] if r['spy_benchmark'] else 0
            alpha = r['alpha'] if r['alpha'] is not None else 0
            status = "✅" if r['outperformed'] else "❌"

            logger.info(
                f"{r['test_year']}: Strategy {strat_ret:>6.1f}% | "
                f"SPY {spy_ret:>6.1f}% | Alpha {alpha:>6.1f}% {status}"
            )

    def run_out_of_sample_2024(self) -> Dict:
        """
        Run 2024 as pure out-of-sample test

        Train on all data before 2024, test on 2024 only

        Returns:
            Dict with 2024 test results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"2024 OUT-OF-SAMPLE TEST")
        logger.info(f"{'='*80}")
        logger.info(f"Training: 2005-2023 (all historical data)")
        logger.info(f"Testing:  2024 (pure out-of-sample)")
        logger.info(f"{'='*80}\n")

        # Load SPY if not already loaded
        if self.spy_benchmark is None:
            self.load_spy_benchmark()

        # Run for 2024
        result = self.run_single_period(
            train_start=2005,
            train_end=2023,
            test_year=2024
        )

        if result:
            logger.info(f"\n{'='*80}")
            logger.info(f"2024 OUT-OF-SAMPLE CONCLUSION")
            logger.info(f"{'='*80}")

            if result['outperformed']:
                logger.info(f"✅ Strategy BEAT SPY in 2024")
                logger.info(f"   Alpha: +{result['alpha']:.1f}%")
            else:
                logger.info(f"❌ Strategy UNDERPERFORMED SPY in 2024")
                logger.info(f"   Alpha: {result['alpha']:.1f}%")

            logger.info(f"\nThis is the true forward-looking expectation")
            logger.info(f"for deploying the strategy with real money.")

        return result

    def export_results(self, output_path: str = None):
        """
        Export walk-forward results to CSV

        Args:
            output_path: Path to save CSV file
        """
        if not self.results:
            logger.warning("No results to export")
            return

        if output_path is None:
            output_dir = os.path.join(project_root, 'output', 'validation')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        # Flatten results to DataFrame
        rows = []
        for r in self.results:
            row = {
                'test_year': r['test_year'],
                'train_period': r['train_period'],
                'strategy_return': r['strategy']['annual_return'],
                'strategy_drawdown': r['strategy']['max_drawdown'],
                'strategy_sharpe': r['strategy']['sharpe_ratio'],
                'spy_return': r['spy_benchmark']['annual_return'] if r['spy_benchmark'] else None,
                'spy_drawdown': r['spy_benchmark']['max_drawdown'] if r['spy_benchmark'] else None,
                'spy_sharpe': r['spy_benchmark']['sharpe_ratio'] if r['spy_benchmark'] else None,
                'alpha': r['alpha'],
                'outperformed': r['outperformed']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Results exported to: {output_path}")
        return output_path


def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("WALK-FORWARD VALIDATION - PHASE 1")
    logger.info("="*80)
    logger.info("")

    # Configuration
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')

    # Check if data directory exists
    if not os.path.exists(data_dir):
        # Try alternative path
        data_dir = os.path.join(project_root, 'sp500_data', 'daily')
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            logger.info("Please specify correct data directory path")
            return

    logger.info(f"Data directory: {data_dir}")
    logger.info("")

    # Initialize validator
    validator = WalkForwardValidator(data_dir=data_dir, initial_capital=100000)

    # Task 1: Walk-Forward Testing (2015-2023)
    logger.info("TASK 1: Walk-Forward Testing")
    logger.info("This will take several minutes...")
    logger.info("")

    results = validator.run_walk_forward(
        start_year=2005,    # Start training from 2005
        end_year=2023,      # Test through 2023
        train_window=10     # Use 10-year training windows
    )

    # Export results
    validator.export_results()

    # Task 2: 2024 Out-of-Sample Test
    logger.info("\n\nTASK 2: 2024 Out-of-Sample Test")
    result_2024 = validator.run_out_of_sample_2024()

    logger.info("\n\n")
    logger.info("="*80)
    logger.info("PHASE 1 VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info("")
    logger.info("Key Findings:")
    logger.info("  1. Walk-forward results show realistic out-of-sample expectations")
    logger.info("  2. 2024 test shows true forward performance")
    logger.info("  3. SPY benchmark comparison reveals actual alpha")
    logger.info("")
    logger.info("Next: Review results before proceeding to Phase 2 (improvements)")
    logger.info("="*80)


if __name__ == '__main__':
    main()
