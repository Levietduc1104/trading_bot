"""
LightGBM ML Strategy Execution
Test if LightGBM performs better than RandomForest
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.core.execution import (
    log_header, calculate_portfolio_metrics, calculate_spy_benchmark,
    save_to_database, generate_visualization, create_report, logger
)
from src.core.execution import run_ml_strategy_backtest, score_stocks_at_date
from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from datetime import datetime

# Monkey-patch to use LightGBM instead of RandomForest
import src.strategies.ml_stock_ranker_lgbm as lgbm_module
sys.modules['src.strategies.ml_stock_ranker_simple'] = type(sys)('ml_stock_ranker_simple')
sys.modules['src.strategies.ml_stock_ranker_simple'].MLStockRanker = lgbm_module.LGBMStockRanker


def run_lgbm_backtest(start_year=2015, end_year=2024):
    """Run LightGBM ML strategy backtest"""
    log_header("STEP 1: LOADING DATA")

    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    logger.info(f"Data directory: {data_dir}")

    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")

    log_header("STEP 2: RUNNING LIGHTGBM ML BACKTEST")

    logger.info("Strategy: ML with LIGHTGBM (instead of RandomForest)")
    logger.info("Configuration:")
    logger.info("  - ML Model: LightGBM Regressor")
    logger.info("  - Parameters: 100 trees, max_depth=5, learning_rate=0.05")
    logger.info("  - Regularization: L1=0.1, L2=0.1, early stopping")
    logger.info("  - Features: 13 technical indicators")
    logger.info("  - Retraining: Every 6 months")
    logger.info("  - Top stocks: 5 highest ML scores")
    logger.info("  - VIX-based cash reserves (5-70%)")
    logger.info("")

    # Use the standard ML backtest function (now using LightGBM via monkey-patch)
    portfolio_df = run_ml_strategy_backtest(bot, start_year, end_year, enhanced=False)

    metrics = calculate_portfolio_metrics(portfolio_df, 100000)
    spy_metrics = calculate_spy_benchmark(bot, start_year, end_year)

    alpha = metrics['annual_return'] - spy_metrics['annual_return']

    logger.info(f"Backtest Period: {start_year}-{end_year}")
    logger.info("")
    logger.info("Results:")
    logger.info(f"  LGBM Annual Return:  {metrics['annual_return']:.1f}%")
    logger.info(f"  LGBM Total Return:   {metrics['total_return']:.1f}%")
    logger.info(f"  LGBM Max Drawdown:   {metrics['max_drawdown']:.1f}%")
    logger.info(f"  LGBM Sharpe Ratio:   {metrics['sharpe']:.2f}")
    logger.info(f"  LGBM Final Value:    ${metrics['final_value']:,.0f}")
    logger.info("")
    logger.info(f"  SPY Annual Return:   {spy_metrics['annual_return']:.1f}%")
    logger.info(f"  SPY Max Drawdown:    {spy_metrics['max_drawdown']:.1f}%")
    logger.info("")
    logger.info(f"  Alpha vs SPY:        {alpha:+.1f}%")
    logger.info(f"  DD Improvement:      {metrics['max_drawdown'] - spy_metrics['max_drawdown']:+.1f}%")
    logger.info("")
    logger.info("COMPARISON TO OTHER STRATEGIES:")
    logger.info(f"  RandomForest ML:  29.3% annual, -59.0% DD")
    logger.info(f"  V30:              16.1% annual, -23.8% DD")
    logger.info(f"  LightGBM ML:      {metrics['annual_return']:.1f}% annual, {metrics['max_drawdown']:.1f}% DD")

    strategy_name = 'ML_LIGHTGBM'
    return portfolio_df, bot, metrics, spy_metrics, strategy_name


if __name__ == '__main__':
    log_header("LIGHTGBM ML STRATEGY - EXECUTION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run backtest
    portfolio_df, bot, metrics, spy_metrics, strategy_name = run_lgbm_backtest(2015, 2024)

    # Save results
    run_id = save_to_database(portfolio_df, metrics, spy_metrics, strategy_name)
    viz_path = generate_visualization(portfolio_df, bot, metrics, spy_metrics, 2015, 2024, strategy_name)
    report_path = create_report(portfolio_df, metrics, spy_metrics, 2015, 2024, strategy_name)

    log_header("EXECUTION COMPLETE")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
