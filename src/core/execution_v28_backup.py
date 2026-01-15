"""
END-TO-END PORTFOLIO TRADING SYSTEM EXECUTION
==============================================

V28: MOMENTUM LEADERS (Production Strategy)

This script runs the complete V28 trading system:
1. Loads stock data
2. Runs backtest with momentum leader selection + regime-based sizing
3. Saves results to database
4. Generates visualizations
5. Creates performance reports

Strategy: V28 Momentum Leaders
- 52-week breakout bonus (0-20 pts) - prioritize stocks near all-time highs
- Relative strength vs SPY (0-15 pts) - only buy market leaders
- Dynamic portfolio size (3-10 stocks) based on market regime (V27)
- Kelly position sizing (weight ∝ √score) (V22)
- VIX-based regime detection
- Portfolio-level drawdown control
- Expected: 9.4% annual, -18.0% max DD, 1.00 Sharpe, 85% win rate

Output:
- Database: output/data/trading_results.db
- Plots: output/plots/trading_analysis.html
- Reports: output/reports/performance_report.txt
- Logs: output/logs/execution.log
"""
import sys
import os
import time
from datetime import datetime
import logging

# Setup logging
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_root, 'output', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'execution.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
import pandas as pd
import numpy as np


def calculate_kelly_weights_sqrt(scored_stocks):
    """
    Calculate position weights using Square Root Kelly method

    This is the CORE INNOVATION of V22.

    Args:
        scored_stocks: List of (ticker, score) tuples

    Returns:
        dict: {ticker: weight} where weights sum to 1.0

    Example:
        Scores: [(AAPL, 120), (MSFT, 100), (GOOGL, 80), (NVDA, 70), (META, 60)]
        Sqrt:   [10.95, 10.0, 8.94, 8.37, 7.75]
        Weights: [23.9%, 21.9%, 19.5%, 18.3%, 16.9%]
    """
    tickers = [t for t, s in scored_stocks]
    scores = [s for t, s in scored_stocks]

    # Square root of each score
    sqrt_scores = [np.sqrt(max(0, score)) for score in scores]
    total_sqrt = sum(sqrt_scores)

    # Normalize to sum to 1.0
    if total_sqrt > 0:
        weights = {
            ticker: np.sqrt(max(0, score)) / total_sqrt
            for ticker, score in scored_stocks
        }
    else:
        # Fallback to equal weight if scores are invalid
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    return weights


def run_v28_backtest(bot):
    """
    Run V28 backtest with Momentum Leaders + Regime-Based Portfolio Size

    V28 NEW Features:
    - 52-week breakout bonus (0-20 pts) - stocks near highs
    - Relative strength vs SPY (0-15 pts) - market leaders only

    V27 Features (maintained):
    - Regime-based portfolio sizing (3-10 stocks)
    - Kelly position sizing (Square Root)
    - VIX-based cash reserves
    - Drawdown control
    """
    start_backtest = time.time()
    logger.info("Running V28 backtest (Momentum Leaders)...")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

    for date in dates[100:]:
        # Monthly rebalancing (day 7-10)
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 10
            )
        )

        if is_rebalance_day:
            # Liquidate holdings
            for ticker in list(holdings.keys()):
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * current_price
            holdings = {}
            last_rebalance_date = date

            # Get current VIX (optimized)
            if bot.vix_data is not None:
                vix_at_date = bot.get_data_up_to_date('VIX', date) if 'VIX' in bot.date_index_cache else bot.vix_data[bot.vix_data.index <= date]
            else:
                vix_at_date = None

            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

            # Score stocks (V13 scoring) - OPTIMIZED
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            # V27: Determine portfolio size based on regime
            top_n = bot.determine_portfolio_size(date)

            # Get top N stocks (dynamic based on regime)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # VIX-based cash reserve (V8)
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            invest_amount = cash * (1 - cash_reserve)

            # 🆕 KELLY POSITION SIZING (Square Root method)
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)

            # Calculate allocations with Kelly weights
            allocations = {
                ticker: invest_amount * weight
                for ticker, weight in kelly_weights.items()
            }

            # Apply V12 drawdown control
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

            # Buy stocks (optimized)
            for ticker, _ in top_stocks:
                df_at_date = bot.get_data_up_to_date(ticker, date)
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    allocation_amount = allocations.get(ticker, 0)
                    shares = allocation_amount / current_price
                    holdings[ticker] = shares
                    fee = allocation_amount * 0.001
                    cash -= (allocation_amount + fee)

        # Calculate daily portfolio value (optimized - runs every day)
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.get_data_up_to_date(ticker, date)
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

    backtest_time = time.time() - start_backtest
    num_days = len(dates[100:])
    logger.info(f"✓ Backtest execution: {backtest_time:.1f}s ({num_days} trading days)")

    return portfolio_df


def log_header(title):
    """Log formatted section header"""
    logger.info("=" * 80)
    logger.info(title.center(80))
    logger.info("=" * 80)


def run_backtest(data_dir='sp500_data/stock_data_1983_2003_top500', initial_capital=100000):
    """
    Run backtest with adaptive regime protection

    Uses top 500 filtered dataset (163 stocks) from 1963-1983 with adjusted prices + dividends

    Returns:
        tuple: (portfolio_df, bot, metrics)
    """
    log_header("STEP 1: LOADING DATA & RUNNING BACKTEST")

    logger.info(f"Initializing bot with:")
    logger.info(f"  - Data directory: {data_dir}")
    logger.info(f"  - Initial capital: ${initial_capital:,.0f}")

    # Set full path for data directory
    full_data_dir = os.path.join(project_root, data_dir)
    bot = PortfolioRotationBot(data_dir=full_data_dir, initial_capital=initial_capital)

    logger.info("Loading stock data...")
    start_load = time.time()
    bot.prepare_data()
    load_time = time.time() - start_load
    logger.info(f"✓ Data loading: {load_time:.1f}s")

    logger.info("Scoring stocks...")
    bot.score_all_stocks()

    logger.info("Running backtest with V28 MOMENTUM LEADERS...")
    logger.info("Configuration:")
    logger.info("  V28 Production: Momentum Leaders + Regime-Based Portfolio ⭐")
    logger.info("  - NEW V28 Momentum Factors:")
    logger.info("    • 52-week breakout bonus (0-20 pts) - prioritize stocks near highs")
    logger.info("    • Relative strength vs SPY (0-15 pts) - only buy market leaders")
    logger.info("  - V27 Portfolio Sizing:")
    logger.info("    • Strong Bull (VIX<15, SPY>>MA200): 3 stocks (concentrate)")
    logger.info("    • Bull (VIX<20, SPY>MA200): 4 stocks")
    logger.info("    • Normal (VIX 20-30): 5 stocks")
    logger.info("    • Volatile (VIX 30-40): 7 stocks (diversify)")
    logger.info("    • Crisis (VIX>40): 10 stocks (maximum diversification)")
    logger.info("  - Position sizing: Kelly-weighted (weight ∝ √score)")
    logger.info("  - Drawdown control: Progressive exposure reduction (0.25x to 1.0x)")
    logger.info("  - Monthly rebalancing (day 7-10)")
    logger.info("  - Dynamic cash reserve (5% to 70% based on VIX)")
    logger.info("  - Trading fee: 0.1% per trade (10 basis points)")
    logger.info("")
    logger.info("  Expected performance:")
    logger.info("    • Annual return: 9.4% (+0.9% vs V27)")
    logger.info("    • Max drawdown: -18.0% (slightly worse)")
    logger.info("    • Sharpe ratio: 1.00 (lower but acceptable)")
    logger.info("    • Win rate: 85% (17/20 positive years) +10% vs V27")

    # Run V28 backtest with momentum leaders
    portfolio_df = run_v28_backtest(bot)

    # Rename 'value' column to 'portfolio_value' for consistency
    if 'value' in portfolio_df.columns:
        portfolio_df = portfolio_df.rename(columns={'value': 'portfolio_value'})

    # Load SPY benchmark for comparison
    logger.info("\nLoading SPY benchmark for performance comparison...")
    spy_benchmark = load_spy_benchmark(data_dir)

    # Calculate metrics with SPY comparison
    metrics = calculate_metrics(portfolio_df, initial_capital, spy_benchmark)

    log_header("BACKTEST COMPLETE")
    log_metrics(metrics)

    return portfolio_df, bot, metrics


def load_spy_benchmark(data_dir='sp500_data/daily'):
    """
    Load SPY benchmark data for comparison

    Returns:
        DataFrame with SPY prices or None if not found
    """
    try:
        full_data_dir = os.path.join(project_root, data_dir)
        spy_path = os.path.join(full_data_dir, 'SPY.csv')

        if not os.path.exists(spy_path):
            logger.warning(f"SPY.csv not found at {spy_path}")
            return None

        spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
        spy_df.columns = [col.lower() for col in spy_df.columns]
        logger.info(f"✓ Loaded SPY benchmark: {len(spy_df)} days")
        return spy_df
    except Exception as e:
        logger.warning(f"Could not load SPY benchmark: {e}")
        return None


def calculate_spy_metrics(spy_df, start_date, end_date, initial_capital):
    """
    Calculate SPY buy-and-hold returns for comparison

    Args:
        spy_df: SPY DataFrame
        start_date: Start date (Timestamp)
        end_date: End date (Timestamp)
        initial_capital: Initial investment

    Returns:
        Dict with SPY metrics or None
    """
    if spy_df is None:
        return None

    # Filter to date range
    spy_period = spy_df[(spy_df.index >= start_date) & (spy_df.index <= end_date)]

    if len(spy_period) < 2:
        logger.warning("Insufficient SPY data for comparison")
        return None

    start_price = spy_period['close'].iloc[0]
    end_price = spy_period['close'].iloc[-1]

    # Calculate returns
    total_return = (end_price - start_price) / start_price * 100
    final_value = initial_capital * (1 + total_return / 100)

    years = (end_date - start_date).days / 365.25
    annual_return = (((end_price / start_price) ** (1 / years)) - 1) * 100 if years > 0 else 0

    # Calculate drawdown
    cummax = spy_period['close'].cummax()
    drawdown = (spy_period['close'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Calculate Sharpe ratio
    returns = spy_period['close'].pct_change()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

    # Yearly returns
    spy_yearly = {}
    spy_period_copy = spy_period.copy()
    spy_period_copy['year'] = spy_period_copy.index.year
    for year in spy_period_copy['year'].unique():
        year_data = spy_period_copy[spy_period_copy['year'] == year]
        if len(year_data) > 0:
            year_start = year_data['close'].iloc[0]
            year_end = year_data['close'].iloc[-1]
            year_return = (year_end - year_start) / year_start * 100
            spy_yearly[year] = year_return

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'yearly_returns': spy_yearly
    }


def calculate_metrics(portfolio_df, initial_capital, spy_benchmark=None):
    """
    Calculate performance metrics from portfolio DataFrame

    Args:
        portfolio_df: Portfolio values DataFrame
        initial_capital: Starting capital
        spy_benchmark: Optional SPY DataFrame for benchmark comparison

    Returns:
        Dict with strategy metrics and SPY comparison
    """
    logger.debug("Calculating performance metrics...")

    final_value = portfolio_df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Calculate annual return
    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]
    years = (end_date - start_date).days / 365.25
    annual_return = (((final_value / initial_capital) ** (1 / years)) - 1) * 100

    # Calculate drawdown
    cummax = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # Calculate Sharpe ratio
    returns = portfolio_df['portfolio_value'].pct_change()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

    # Yearly returns
    portfolio_df['year'] = portfolio_df.index.year
    yearly_returns = {}
    for year in portfolio_df['year'].unique():
        year_data = portfolio_df[portfolio_df['year'] == year]
        if len(year_data) > 0:
            year_start = year_data['portfolio_value'].iloc[0]
            year_end = year_data['portfolio_value'].iloc[-1]
            year_return = (year_end - year_start) / year_start * 100
            yearly_returns[year] = year_return

    logger.debug(f"Metrics calculated: {annual_return:.1f}% annual return")

    # Calculate SPY benchmark metrics
    spy_metrics = calculate_spy_metrics(spy_benchmark, start_date, end_date, initial_capital)

    # Calculate alpha vs SPY
    alpha = None
    if spy_metrics:
        alpha = annual_return - spy_metrics['annual_return']
        logger.info(f"Alpha vs SPY: {alpha:+.1f}%")

    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': 0,
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'years': years,
        'yearly_returns': yearly_returns,
        'spy_benchmark': spy_metrics,
        'alpha': alpha
    }


def log_metrics(metrics):
    """Log performance metrics with SPY comparison"""
    logger.info("")
    logger.info("="*80)
    logger.info("STRATEGY PERFORMANCE")
    logger.info("="*80)
    logger.info(f"Initial Capital: ${metrics['initial_capital']:,.0f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Total Return:    {metrics['total_return']:.1f}%")
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Period:          {metrics['start_date']} to {metrics['end_date']}")
    logger.info(f"Duration:        {metrics['years']:.1f} years")

    # SPY Benchmark Comparison
    if metrics.get('spy_benchmark'):
        spy = metrics['spy_benchmark']
        alpha = metrics.get('alpha', 0)

        logger.info("")
        logger.info("="*80)
        logger.info("SPY BENCHMARK COMPARISON")
        logger.info("="*80)
        logger.info(f"SPY Final Value: ${spy['final_value']:,.0f}")
        logger.info(f"SPY Total Return:{spy['total_return']:>7.1f}%")
        logger.info(f"SPY Annual Return:{spy['annual_return']:>6.1f}%")
        logger.info(f"SPY Max Drawdown:{spy['max_drawdown']:>7.1f}%")
        logger.info(f"SPY Sharpe Ratio:{spy['sharpe_ratio']:>7.2f}")
        logger.info("")
        logger.info("-"*80)
        logger.info(f"ALPHA (Strategy - SPY): {alpha:+.1f}%")
        logger.info("-"*80)

        if alpha > 0:
            logger.info(f"✅ Strategy OUTPERFORMED SPY by {alpha:.1f}% annually")
        else:
            logger.info(f"❌ Strategy UNDERPERFORMED SPY by {abs(alpha):.1f}% annually")

    logger.info("")
    logger.info("="*80)
    logger.info("YEARLY RETURNS COMPARISON")
    logger.info("="*80)

    positive_years = sum(1 for r in metrics['yearly_returns'].values() if r > 0)
    total_years = len(metrics['yearly_returns'])
    logger.info(f"Strategy Win Rate: {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)")

    if metrics.get('spy_benchmark'):
        spy_yearly = metrics['spy_benchmark']['yearly_returns']
        logger.info("")
        logger.info(f"{'Year':<6} {'Strategy':>10} {'SPY':>10} {'Alpha':>10} {'Result':<10}")
        logger.info("-"*80)

        for year, strat_ret in sorted(metrics['yearly_returns'].items()):
            spy_ret = spy_yearly.get(year, 0)
            year_alpha = strat_ret - spy_ret
            status = "✅ Beat SPY" if year_alpha > 0 else "❌ Lost to SPY"
            logger.info(f"{year:<6} {strat_ret:>9.1f}% {spy_ret:>9.1f}% {year_alpha:>+9.1f}% {status}")
    else:
        logger.info("")
        for year, ret in sorted(metrics['yearly_returns'].items()):
            status = "✅" if ret > 0 else "❌"
            logger.info(f"  {year}: {ret:6.1f}% {status}")


def save_to_database(portfolio_df, metrics, strategy_type='V28_MOMENTUM_LEADERS'):
    """Save results to database"""
    log_header("STEP 2: SAVING TO DATABASE")

    import sqlite3

    db_path = os.path.join(project_root, 'output', 'data', 'trading_results.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logger.debug(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{strategy_type}_{timestamp}"

    logger.info(f"Saving backtest run: {run_name}")

    try:
        # Save run summary
        cursor.execute('''
            INSERT INTO backtest_runs (
                run_name, strategy_type, initial_capital, final_value,
                total_return, annual_return, max_drawdown, sharpe_ratio,
                start_date, end_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_name,
            strategy_type,
            metrics['initial_capital'],
            metrics['final_value'],
            metrics['total_return'],
            metrics['annual_return'],
            metrics['max_drawdown'],
            metrics['sharpe_ratio'],
            metrics['start_date'],
            metrics['end_date']
        ))

        run_id = cursor.lastrowid
        logger.info(f"Run ID: {run_id}")

        # Save portfolio values
        logger.info("Saving portfolio value history...")
        logger.debug(f"Saving {len(portfolio_df)} portfolio records...")

        for idx, row in portfolio_df.iterrows():
            cursor.execute('''
                INSERT INTO portfolio_values (
                    run_id, date, portfolio_value, daily_return, cumulative_return, drawdown
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                str(idx),
                row.get('portfolio_value'),
                row.get('daily_return', 0),
                row.get('cumulative_return', 0),
                row.get('drawdown', 0)
            ))

        # Save yearly returns
        logger.info("Saving yearly returns...")
        logger.debug(f"Saving {len(metrics['yearly_returns'])} yearly return records...")

        for year, annual_return in metrics['yearly_returns'].items():
            cursor.execute('''
                INSERT INTO yearly_returns (run_id, year, annual_return)
                VALUES (?, ?, ?)
            ''', (run_id, year, annual_return))

        conn.commit()
        logger.info(f"✅ Results saved to database: {db_path}")

    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed")

    return run_id


def create_text_report(portfolio_df, metrics, output_dir='output/reports'):
    """Create a text performance report"""
    log_header("STEP 3: CREATING PERFORMANCE REPORT")

    full_output_dir = os.path.join(project_root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = os.path.join(full_output_dir, f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    logger.debug(f"Creating report: {report_file}")

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PORTFOLIO TRADING SYSTEM - PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Strategy: V28 Production (Momentum Leaders)\n")
        f.write("=" * 80 + "\n\n")

        # Summary metrics
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Initial Capital:       ${metrics['initial_capital']:>15,.0f}\n")
        f.write(f"Final Value:           ${metrics['final_value']:>15,.0f}\n")
        f.write(f"Total Return:          {metrics['total_return']:>15.1f}%\n")
        f.write(f"Annual Return:         {metrics['annual_return']:>15.1f}%\n")
        f.write(f"Max Drawdown:          {metrics['max_drawdown']:>15.1f}%\n")
        f.write(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>15.2f}\n")
        f.write(f"Period:                {metrics['start_date']} to {metrics['end_date']}\n")
        f.write(f"Duration:              {metrics['years']:>15.1f} years\n\n")

        # SPY Benchmark Comparison
        if metrics.get('spy_benchmark'):
            spy = metrics['spy_benchmark']
            alpha = metrics.get('alpha', 0)

            f.write("=" * 80 + "\n")
            f.write("SPY BENCHMARK COMPARISON\n")
            f.write("=" * 80 + "\n")
            f.write(f"SPY Initial Capital:   ${spy['initial_capital']:>15,.0f}\n")
            f.write(f"SPY Final Value:       ${spy['final_value']:>15,.0f}\n")
            f.write(f"SPY Total Return:      {spy['total_return']:>15.1f}%\n")
            f.write(f"SPY Annual Return:     {spy['annual_return']:>15.1f}%\n")
            f.write(f"SPY Max Drawdown:      {spy['max_drawdown']:>15.1f}%\n")
            f.write(f"SPY Sharpe Ratio:      {spy['sharpe_ratio']:>15.2f}\n\n")

            f.write("-" * 80 + "\n")
            f.write(f"ALPHA (Strategy - SPY): {alpha:>14.1f}%\n")
            f.write("-" * 80 + "\n")

            if alpha > 0:
                f.write(f"✅ Strategy OUTPERFORMED SPY by {alpha:.1f}% annually\n\n")
            else:
                f.write(f"❌ Strategy UNDERPERFORMED SPY by {abs(alpha):.1f}% annually\n\n")

        # Yearly returns
        f.write("=" * 80 + "\n")
        f.write("YEARLY RETURNS\n")
        f.write("=" * 80 + "\n")
        positive_years = sum(1 for r in metrics['yearly_returns'].values() if r > 0)
        total_years = len(metrics['yearly_returns'])
        f.write(f"Strategy Win Rate: {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)\n\n")

        if metrics.get('spy_benchmark'):
            spy_yearly = metrics['spy_benchmark']['yearly_returns']
            f.write(f"{'Year':<6} {'Strategy':>10} {'SPY':>10} {'Alpha':>10} {'Result':<15}\n")
            f.write("-" * 80 + "\n")

            for year, strat_ret in sorted(metrics['yearly_returns'].items()):
                spy_ret = spy_yearly.get(year, 0)
                year_alpha = strat_ret - spy_ret
                status = "✅ Beat SPY" if year_alpha > 0 else "❌ Lost to SPY"
                f.write(f"{year:<6} {strat_ret:>9.1f}% {spy_ret:>9.1f}% {year_alpha:>+9.1f}% {status:<15}\n")
        else:
            for year, ret in sorted(metrics['yearly_returns'].items()):
                status = "✅" if ret > 0 else "❌"
                f.write(f"  {year}:  {ret:>7.1f}% {status}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("STRATEGY CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write("  V28: MOMENTUM LEADERS + REGIME-BASED PORTFOLIO SIZE\n")
        f.write("  NEW V28 Momentum Factors:\n")
        f.write("    - 52-week breakout bonus (0-20 pts) - stocks near all-time highs\n")
        f.write("    - Relative strength vs SPY (0-15 pts) - only buy market leaders\n")
        f.write("  V27 Portfolio Sizing (maintained):\n")
        f.write("    * Strong Bull (VIX<15, SPY>>MA200): 3 stocks (concentrate)\n")
        f.write("    * Bull (VIX<20, SPY>MA200): 4 stocks\n")
        f.write("    * Normal (VIX 20-30): 5 stocks\n")
        f.write("    * Volatile (VIX 30-40, SPY<MA200): 7 stocks (diversify)\n")
        f.write("    * Crisis (VIX>40): 10 stocks (maximum diversification)\n")
        f.write("  Other Features:\n")
        f.write("    - Position Weighting: Kelly-weighted (weight ∝ √score)\n")
        f.write("    - Rebalancing: Monthly (day 7-10 of each month)\n")
        f.write("    - Cash Reserve: Dynamic (5% to 70% based on VIX)\n")
        f.write("    - Drawdown Control:\n")
        f.write("      * DD < 10%:  100% invested\n")
        f.write("      * DD 10-15%: 75% invested\n")
        f.write("      * DD 15-20%: 50% invested\n")
        f.write("      * DD ≥ 20%:  25% invested (maximum defense)\n")
        f.write("    - Regime Detection: VIX-based (forward-looking)\n")
        f.write("    - Trading Fee: 0.1% per trade\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    logger.info(f"✅ Report saved to: {report_file}")
    return report_file


def generate_visualizations():
    """Generate interactive visualizations"""
    log_header("STEP 4: GENERATING VISUALIZATIONS")

    logger.info("Running visualization script...")
    logger.info("This may take a minute to generate interactive charts...")

    # Run the visualize_trades.py script
    import subprocess
    import shutil

    visualize_script = os.path.join(project_root, 'src', 'visualize', 'visualize_trades.py')

    try:
        result = subprocess.run(
            [sys.executable, visualize_script],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ Visualizations generated successfully")

            # Copy to output folder
            source_html = os.path.join(project_root, 'src', 'visualize', 'trading_analysis.html')
            dest_html = os.path.join(project_root, 'output', 'plots', 'trading_analysis.html')

            if os.path.exists(source_html):
                shutil.copy2(source_html, dest_html)
                logger.info(f"✅ Visualization saved to: {dest_html}")
                return dest_html
            else:
                logger.warning("Visualization file not found at expected location")
                return None
        else:
            logger.error(f"Visualization failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("Visualization timed out after 5 minutes")
        return None
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return None

def main():
    """Main execution function"""
    log_header("PORTFOLIO TRADING SYSTEM - EXECUTION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Project Root: {project_root}")

    try:
        # Step 1: Run backtest
        portfolio_df, bot, metrics = run_backtest()

        # Step 2: Save to database
        run_id = save_to_database(portfolio_df, metrics)

        # Step 3: Create text report
        report_file = create_text_report(portfolio_df, metrics)

        # Step 4: Generate visualizations
        viz_file = generate_visualizations()

        # Summary
        log_header("EXECUTION COMPLETE")
        logger.info(f"✅ Backtest completed successfully\!")
        logger.info("")
        logger.info(f"Outputs:")
        logger.info(f"  📊 Database:      output/data/trading_results.db")
        logger.info(f"  📈 Report:        {report_file}")
        logger.info(f"  📋 Logs:          output/logs/execution.log")
        if viz_file:
            logger.info(f"  🎨 Visualization: {viz_file}")
        else:
            logger.warning("  ⚠️  Visualization: Failed to generate")
        logger.info("")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Strategy: V28 Production (Momentum Leaders)")
        logger.info(f"Portfolio: Dynamic 3-10 stocks + Momentum filters")
        logger.info(f"Annual Return: {metrics['annual_return']:.1f}%")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info("")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
