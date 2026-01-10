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
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * current_price
            holdings = {}
            last_rebalance_date = date

            # Get current VIX
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

            # Score stocks (V13 scoring)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
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

            # Buy stocks
            for ticker, _ in top_stocks:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    allocation_amount = allocations.get(ticker, 0)
                    shares = allocation_amount / current_price
                    holdings[ticker] = shares
                    fee = allocation_amount * 0.001
                    cash -= (allocation_amount + fee)

        # Calculate daily portfolio value
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    return portfolio_df


def log_header(title):
    """Log formatted section header"""
    logger.info("=" * 80)
    logger.info(title.center(80))
    logger.info("=" * 80)


def run_backtest(data_dir='sp500_data/sp500_filtered', initial_capital=100000):
    """
    Run backtest with adaptive regime protection

    Uses filtered S&P 500 dataset (464 stocks) with adjusted prices + dividends

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
    bot.prepare_data()

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

    # Calculate metrics
    metrics = calculate_metrics(portfolio_df, initial_capital)

    log_header("BACKTEST COMPLETE")
    log_metrics(metrics)

    return portfolio_df, bot, metrics


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics from portfolio DataFrame"""
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
        'yearly_returns': yearly_returns
    }


def log_metrics(metrics):
    """Log performance metrics"""
    logger.info("")
    logger.info(f"Initial Capital: ${metrics['initial_capital']:,.0f}")
    logger.info(f"Final Value:     ${metrics['final_value']:,.0f}")
    logger.info(f"Total Return:    {metrics['total_return']:.1f}%")
    logger.info(f"Annual Return:   {metrics['annual_return']:.1f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Period:          {metrics['start_date']} to {metrics['end_date']}")
    logger.info(f"Duration:        {metrics['years']:.1f} years")
    logger.info("")
    logger.info("Yearly Returns:")
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

        # Yearly returns
        f.write("YEARLY RETURNS\n")
        f.write("-" * 80 + "\n")
        positive_years = sum(1 for r in metrics['yearly_returns'].values() if r > 0)
        total_years = len(metrics['yearly_returns'])
        f.write(f"Win Rate: {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)\n\n")

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
