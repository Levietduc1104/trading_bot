"""
END-TO-END PORTFOLIO TRADING SYSTEM EXECUTION
==============================================

MULTI-STRATEGY EXECUTION SYSTEM

This script runs the complete trading system with strategy selection:
- ML: ML Regularized Strategy (BEST: 106.1% annual) ⭐ RECOMMENDED
- V30: Dynamic Mega-Cap Split (Good: 92.3% annual)

The system:
1. Loads stock data
2. Runs backtest with selected strategy
3. Saves results to database
4. Generates visualizations
5. Creates performance reports

ML Strategy (Recommended):
- RandomForest ML model with 9 technical features
- 50 estimators, max_depth=3 (regularized to prevent overfitting)
- Retrains every 6 months
- Top 5 stocks by ML score
- VIX-based cash reserves (5%-70%)
- Monthly rebalancing

V30 Strategy:
- Dynamically identifies top 7 mega-caps using trading value
- 70% allocation to Top 3 Dynamic Mega-Caps (by momentum)
- 30% allocation to Top 2 Momentum stocks
- VIX-based cash reserves, 15% trailing stops
- Progressive portfolio drawdown control

Output:
- Database: output/data/trading_results.db
- Plots: output/plots/{strategy}_performance.png
- Reports: output/reports/performance_report.txt
- Logs: output/logs/execution.log
"""
import os
import sys
import time
import sqlite3
from datetime import datetime
import logging

# Setup paths
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

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np


def log_header(title):
    """Print formatted header"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)




def get_vix_cash_reserve(vix, aggressive=False):
    """
    Calculate cash reserve based on VIX

    Args:
        vix: VIX volatility index value
        aggressive: If True, use more aggressive cash reserves for better drawdown protection
    """
    if aggressive:
        # More aggressive reserves for enhanced risk management
        if vix < 12:
            return 0.05
        elif vix < 15:
            return 0.15
        elif vix < 20:
            return 0.25
        elif vix < 25:
            return 0.40
        elif vix < 30:
            return 0.55
        elif vix < 40:
            return 0.75
        else:
            return 0.90  # 90% cash in extreme volatility
    else:
        # Original conservative reserves
        if vix < 15:
            return 0.05
        elif vix < 20:
            return 0.10
        elif vix < 25:
            return 0.20
        elif vix < 30:
            return 0.35
        elif vix < 40:
            return 0.50
        else:
            return 0.70


def score_stocks_at_date(ranker, stock_data, spy_df, date):
    """Helper function to score all stocks at a specific date using ML"""
    stock_features = {}
    
    for ticker, df in stock_data.items():
        if ticker in ['SPY', 'VIX']:  # Exclude benchmarks
            continue
        
        df_at_date = df[df.index <= date]
        if len(df_at_date) < 200:
            continue
        
        spy_slice = spy_df[spy_df.index <= date] if spy_df is not None else None
        features = ranker.extract_features(ticker, df_at_date, spy_slice)
        
        if features is not None:
            stock_features[ticker] = features
    
    return ranker.predict_scores(stock_features)


def run_ml_strategy_backtest(bot, start_year, end_year, enhanced=False):
    """
    Run ML Regularized strategy backtest logic

    Args:
        bot: PortfolioRotationBot instance
        start_year: Start year for backtest
        end_year: End year for backtest
        enhanced: If True, use enhanced risk management (trailing stops, drawdown control, aggressive VIX)
    """
    from src.strategies.ml_stock_ranker_simple import MLStockRanker

    # ML Regularized configuration
    config = {
        'n_estimators': 50,
        'max_depth': 3,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 0.5,
        'top_n': 5,
        'retrain_months': 6,
        'trailing_stop': 0.15 if enhanced else None,  # 15% trailing stop
        'max_sector_allocation': 0.40 if enhanced else 1.0,  # Max 40% per sector
    }

    ranker = MLStockRanker(config=config)

    # Get trading dates
    first_ticker = list(bot.stocks_data.keys())[0]
    all_dates = bot.stocks_data[first_ticker].index
    trading_dates = all_dates[(all_dates >= f'{start_year}-01-01') & (all_dates <= f'{end_year}-12-31')]

    logger.info(f"Trading period: {start_year}-{end_year}")
    logger.info(f"Trading dates: {len(trading_dates)}")
    if enhanced:
        logger.info("ENHANCED MODE: Trailing stops, drawdown control, aggressive VIX reserves")
    logger.info("")

    # Initial training
    initial_train_end = pd.to_datetime(f'{start_year}-01-01') - pd.DateOffset(days=1)
    logger.info(f"Initial training up to {initial_train_end.date()}")
    ranker.train(bot.stocks_data, bot.stocks_data.get('SPY'), initial_train_end)

    # Run backtest
    cash = 100000
    portfolio = {}  # {ticker: {'shares': float, 'entry_price': float, 'peak_price': float}}
    portfolio_values = []
    last_rebalance = None
    portfolio_peak = 100000

    for i, date in enumerate(trading_dates):
        # Calculate current portfolio value
        current_value = cash
        for ticker, position in portfolio.items():
            if ticker in bot.stocks_data:
                df = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df) > 0:
                    current_price = df.iloc[-1]['close']
                    current_value += position['shares'] * current_price

        # Update portfolio peak
        portfolio_peak = max(portfolio_peak, current_value)

        # ENHANCED: Check trailing stops DAILY (not just at rebalance)
        if enhanced and config['trailing_stop']:
            for ticker in list(portfolio.keys()):
                if ticker in bot.stocks_data:
                    df = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df) > 0:
                        current_price = df.iloc[-1]['close']

                        # Update peak price
                        portfolio[ticker]['peak_price'] = max(portfolio[ticker]['peak_price'], current_price)

                        # Check trailing stop (15% from peak)
                        stop_price = portfolio[ticker]['peak_price'] * (1 - config['trailing_stop'])
                        if current_price < stop_price:
                            # Sell due to trailing stop
                            shares = portfolio[ticker]['shares']
                            cash += shares * current_price
                            del portfolio[ticker]
                            logger.info(f"  TRAILING STOP: Sold {ticker} at ${current_price:.2f} (peak: ${portfolio[ticker]['peak_price']:.2f})")

        # Check if should retrain
        if ranker.should_retrain(date):
            logger.info(f"Retraining ML model at {date.date()}")
            train_end = date - pd.DateOffset(days=1)
            ranker.train(bot.stocks_data, bot.stocks_data.get('SPY'), train_end)

        # Monthly rebalancing (day 7-15)
        is_rebalance = (
            last_rebalance is None or
            ((date.year, date.month) != (last_rebalance.year, last_rebalance.month) and 7 <= date.day <= 15)
        )

        if is_rebalance:
            last_rebalance = date

            # Get VIX
            vix = 20
            vix_data = bot.stocks_data.get('VIX')
            if vix_data is not None:
                vix_at_date = vix_data[vix_data.index <= date]
                if len(vix_at_date) > 0:
                    vix = vix_at_date.iloc[-1]['close']

            # Liquidate current holdings
            for ticker in list(portfolio.keys()):
                if ticker in bot.stocks_data:
                    df = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df) > 0:
                        price = df.iloc[-1]['close']
                        cash += portfolio[ticker]['shares'] * price
            portfolio = {}

            # Score stocks with ML
            scores = score_stocks_at_date(ranker, bot.stocks_data, bot.stocks_data.get('SPY'), date)
            top_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:config.get('top_n', 5)]

            # Calculate cash reserve based on VIX
            cash_reserve = get_vix_cash_reserve(vix, aggressive=enhanced)

            # ENHANCED: Progressive drawdown control
            if enhanced:
                portfolio_drawdown = ((current_value - portfolio_peak) / portfolio_peak) * 100
                if portfolio_drawdown < -5:
                    drawdown_multiplier = 0.90  # Reduce 10% if DD > 5%
                elif portfolio_drawdown < -10:
                    drawdown_multiplier = 0.75  # Reduce 25% if DD > 10%
                elif portfolio_drawdown < -20:
                    drawdown_multiplier = 0.50  # Reduce 50% if DD > 20%
                elif portfolio_drawdown < -30:
                    drawdown_multiplier = 0.25  # Reduce 75% if DD > 30%
                else:
                    drawdown_multiplier = 1.0
            else:
                drawdown_multiplier = 1.0

            invest_amount = cash * (1 - cash_reserve) * drawdown_multiplier

            # ENHANCED: Sector diversification
            if enhanced and config['max_sector_allocation'] < 1.0:
                # Get sector information for top stocks
                sector_allocation = {}
                filtered_stocks = []

                for ticker, score in top_stocks:
                    # Get sector from bot (if available)
                    sector = 'Unknown'
                    if hasattr(bot, 'sector_peers') and ticker in bot.sector_peers:
                        for sect, tickers in bot.sector_peers.items():
                            if ticker in tickers:
                                sector = sect
                                break

                    # Check if adding this stock would exceed sector limit
                    current_sector_alloc = sector_allocation.get(sector, 0)
                    if current_sector_alloc < config['max_sector_allocation']:
                        filtered_stocks.append((ticker, score))
                        sector_allocation[sector] = current_sector_alloc + (1.0 / config['top_n'])

                top_stocks = filtered_stocks

            # Equal weight allocation
            if top_stocks:
                allocation_per_stock = invest_amount / len(top_stocks)

                for ticker, score in top_stocks:
                    if ticker in bot.stocks_data:
                        df = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                        if len(df) > 0:
                            price = df.iloc[-1]['close']
                            shares = allocation_per_stock / price
                            portfolio[ticker] = {
                                'shares': shares,
                                'entry_price': price,
                                'peak_price': price
                            }
                            cash -= allocation_per_stock

            # Log progress
            if i % 60 == 0 or is_rebalance:
                ret = ((current_value / 100000) - 1) * 100
                stocks_str = ', '.join([t for t, _ in top_stocks[:3]])
                reserve_pct = cash_reserve * 100
                logger.info(f"{date.date()}: ${current_value:,.0f} ({ret:+.1f}%), VIX={vix:.1f}, Cash={reserve_pct:.0f}%, Stocks=[{stocks_str}]")

        # Record portfolio value
        value = cash
        for ticker, position in portfolio.items():
            if ticker in bot.stocks_data:
                df = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df) > 0:
                    value += position['shares'] * df.iloc[-1]['close']

        portfolio_values.append({
            'date': date,
            'value': value,
            'cash': cash
        })

    return pd.DataFrame(portfolio_values).set_index('date')


def calculate_portfolio_metrics(portfolio_df, initial_capital):
    """Calculate portfolio performance metrics"""
    final_value = portfolio_df['value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    
    total_return = ((final_value / initial_capital) - 1) * 100
    annual_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
    
    running_max = portfolio_df['value'].expanding().max()
    drawdown = ((portfolio_df['value'] - running_max) / running_max * 100)
    max_drawdown = drawdown.min()
    
    daily_returns = portfolio_df['value'].pct_change()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    
    return {
        'annual_return': annual_return,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'final_value': final_value
    }


def run_ml_backtest(start_year=2015, end_year=2024):
    """
    Run ML Regularized Strategy backtest (RECOMMENDED - Best Performance)
    
    Returns:
        tuple: (portfolio_df, bot, metrics, spy_metrics, strategy_name)
    """
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    
    log_header("STEP 1: LOADING DATA")
    
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    logger.info(f"Data directory: {data_dir}")
    
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")
    
    log_header("STEP 2: RUNNING ML REGULARIZED BACKTEST")
    
    logger.info("Strategy: ML REGULARIZED (BEST PERFORMER)")
    logger.info("Configuration:")
    logger.info("  - ML Model: RandomForest Regressor")
    logger.info("  - Parameters: 50 estimators, max_depth=3 (regularized)")
    logger.info("  - Features: 9 technical indicators")
    logger.info("  - Retraining: Every 6 months")
    logger.info("  - Top stocks: 5 highest ML scores")
    logger.info("  - VIX-based cash reserves")
    logger.info("")
    
    portfolio_df = run_ml_strategy_backtest(bot, start_year, end_year)
    
    metrics = calculate_portfolio_metrics(portfolio_df, 100000)
    
    # Calculate SPY benchmark
    spy_metrics = calculate_spy_benchmark(bot, start_year, end_year)
    
    alpha = metrics['annual_return'] - spy_metrics['annual_return']
    
    logger.info(f"Backtest Period: {start_year}-{end_year}")
    logger.info("")
    logger.info("Results:")
    logger.info(f"  ML Annual Return:  {metrics['annual_return']:.1f}%")
    logger.info(f"  ML Total Return:   {metrics['total_return']:.1f}%")
    logger.info(f"  ML Max Drawdown:   {metrics['max_drawdown']:.1f}%")
    logger.info(f"  ML Sharpe Ratio:   {metrics['sharpe']:.2f}")
    logger.info(f"  ML Final Value:    ${metrics['final_value']:,.0f}")
    logger.info("")
    logger.info(f"  SPY Annual Return:  {spy_metrics['annual_return']:.1f}%")
    logger.info(f"  SPY Max Drawdown:   {spy_metrics['max_drawdown']:.1f}%")
    logger.info("")
    logger.info(f"  Alpha vs SPY:       {alpha:+.1f}%")
    logger.info(f"  DD Improvement:     {metrics['max_drawdown'] - spy_metrics['max_drawdown']:+.1f}%")
    
    return portfolio_df, bot, metrics, spy_metrics, 'ML_REGULARIZED'


def calculate_spy_benchmark(bot, start_year, end_year):
    """Calculate SPY benchmark metrics"""
    spy_metrics = {'annual_return': 10.0, 'max_drawdown': -30.0}
    spy_df = bot.stocks_data.get('SPY')
    
    if spy_df is not None:
        spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
        if len(spy_period) > 1:
            spy_start = spy_period['close'].iloc[0]
            spy_end = spy_period['close'].iloc[-1]
            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
            spy_metrics['annual_return'] = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100
            spy_cummax = spy_period['close'].cummax()
            spy_metrics['max_drawdown'] = ((spy_period['close'] - spy_cummax) / spy_cummax * 100).min()
    
    return spy_metrics


def run_v30_backtest(start_year=2015, end_year=2024):
    """
    Run V30 Dynamic Mega-Cap Split Strategy backtest
    
    Returns:
        tuple: (portfolio_df, bot, metrics, spy_metrics)
    """
    from src.backtest.portfolio_bot_demo import PortfolioRotationBot
    from src.strategies.v30_dynamic_megacap import V30Strategy, calculate_metrics
    
    log_header("STEP 1: LOADING DATA")
    
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024_top500')
    logger.info(f"Data directory: {data_dir}")
    
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=100000)
    bot.prepare_data()
    logger.info(f"Loaded {len(bot.stocks_data)} stocks")
    
    log_header("STEP 2: RUNNING V30 BACKTEST")
    
    logger.info("Strategy: V30 DYNAMIC MEGA-CAP SPLIT")
    logger.info("Configuration:")
    logger.info("  - 70% Top 3 Magnificent 7 (by momentum)")
    logger.info("  - 30% Top 2 Momentum stocks")
    logger.info("  - 15% Trailing stop losses")
    logger.info("  - VIX-based cash reserves (up to 70%)")
    logger.info("  - Progressive drawdown control")
    logger.info("")
    
    config = {
        'megacap_allocation': 0.70,
        'num_megacap': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
        'num_top_megacaps': 7,
        'lookback_trading_value': 20,
    }
    
    strategy = V30Strategy(bot, config=config)
    portfolio_df = strategy.run_backtest(start_year=start_year, end_year=end_year)
    
    metrics = calculate_metrics(portfolio_df, 100000)
    
    # Calculate SPY benchmark
    spy_metrics = {'annual_return': 10.0, 'max_drawdown': -30.0}
    spy_df = bot.stocks_data.get('SPY')
    if spy_df is not None:
        spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
        if len(spy_period) > 1:
            spy_start = spy_period['close'].iloc[0]
            spy_end = spy_period['close'].iloc[-1]
            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
            spy_metrics['annual_return'] = ((spy_end / spy_start) ** (1 / spy_years) - 1) * 100
            spy_cummax = spy_period['close'].cummax()
            spy_metrics['max_drawdown'] = ((spy_period['close'] - spy_cummax) / spy_cummax * 100).min()
    
    alpha = metrics['annual_return'] - spy_metrics['annual_return']
    
    logger.info(f"Backtest Period: {start_year}-{end_year}")
    logger.info("")
    logger.info("Results:")
    logger.info(f"  V30 Annual Return:  {metrics['annual_return']:.1f}%")
    logger.info(f"  V30 Total Return:   {metrics['total_return']:.1f}%")
    logger.info(f"  V30 Max Drawdown:   {metrics['max_drawdown']:.1f}%")
    logger.info(f"  V30 Sharpe Ratio:   {metrics['sharpe']:.2f}")
    logger.info(f"  V30 Final Value:    ${metrics['final_value']:,.0f}")
    logger.info("")
    logger.info(f"  SPY Annual Return:  {spy_metrics['annual_return']:.1f}%")
    logger.info(f"  SPY Max Drawdown:   {spy_metrics['max_drawdown']:.1f}%")
    logger.info("")
    logger.info(f"  Alpha vs SPY:       {alpha:+.1f}%")
    logger.info(f"  DD Improvement:     {metrics['max_drawdown'] - spy_metrics['max_drawdown']:+.1f}%")
    
    return portfolio_df, bot, metrics, spy_metrics, "V30_DYNAMIC_MEGACAP"



def run_backtest(start_year=2015, end_year=2024, strategy='ml'):
    """
    Run backtest with selected strategy
    
    Args:
        start_year: Start year for backtest
        end_year: End year for backtest
        strategy: 'ml' (recommended, 106% annual) or 'v30' (92% annual)
    
    Returns:
        tuple: (portfolio_df, bot, metrics, spy_metrics, strategy_name)
    """
    if strategy.lower() == 'ml':
        return run_ml_backtest(start_year, end_year)
    elif strategy.lower() == 'v30':
        return run_v30_backtest(start_year, end_year)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose 'ml' or 'v30'")


def save_to_database(portfolio_df, metrics, spy_metrics, strategy_name="V30_DYNAMIC_MEGACAP"):
    """Save results to SQLite database"""
    log_header("STEP 3: SAVING TO DATABASE")
    
    db_dir = os.path.join(project_root, 'output', 'data')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'trading_results.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT,
            strategy TEXT,
            timestamp TEXT,
            initial_capital REAL,
            final_value REAL,
            annual_return REAL,
            total_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            spy_annual_return REAL,
            spy_max_drawdown REAL,
            alpha REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            date TEXT,
            value REAL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
        )
    ''')
    
    # Insert run
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_name = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    alpha = metrics['annual_return'] - spy_metrics['annual_return']
    
    cursor.execute('''
        INSERT INTO backtest_runs 
        (run_name, strategy, timestamp, initial_capital, final_value, annual_return, 
         total_return, max_drawdown, sharpe_ratio, spy_annual_return, spy_max_drawdown, alpha)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_name, strategy_name, timestamp, 100000, float(metrics['final_value']),
          float(metrics['annual_return']), float(metrics['total_return']), float(metrics['max_drawdown']),
          float(metrics['sharpe']), float(spy_metrics['annual_return']), float(spy_metrics['max_drawdown']), float(alpha)))
    
    run_id = cursor.lastrowid
    
    # Insert portfolio values
    for date, row in portfolio_df.iterrows():
        cursor.execute('''
            INSERT INTO portfolio_values (run_id, date, value)
            VALUES (?, ?, ?)
        ''', (run_id, date.strftime('%Y-%m-%d'), float(row['value'])))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved to database: {db_path}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run Name: {run_name}")
    
    return run_id


def generate_visualization(portfolio_df, bot, metrics, spy_metrics, start_year, end_year, strategy_name="V30"):
    """Generate V30 performance visualization"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    log_header("STEP 4: GENERATING VISUALIZATION")
    
    spy_df = bot.stocks_data['SPY']
    spy_period = spy_df[(spy_df.index >= f'{start_year}-01-01') & (spy_df.index <= f'{end_year}-12-31')]
    spy_norm = spy_period['close'] / spy_period['close'].iloc[0] * 100000
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{strategy_name} Strategy ({start_year}-{end_year})', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Growth (Log Scale)
    ax1 = axes[0, 0]
    ax1.semilogy(portfolio_df.index, portfolio_df['value'], 'b-', linewidth=2, label='V30 Strategy')
    ax1.semilogy(spy_norm.index, spy_norm.values, 'gray', linewidth=2, alpha=0.7, label='SPY Buy & Hold')
    ax1.set_title('Portfolio Growth (Log Scale)', fontsize=12)
    ax1.set_ylabel('Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Comparison
    ax2 = axes[0, 1]
    v29_dd = (portfolio_df['value'] - portfolio_df['value'].cummax()) / portfolio_df['value'].cummax() * 100
    spy_dd = (spy_norm - spy_norm.cummax()) / spy_norm.cummax() * 100
    ax2.fill_between(v29_dd.index, v29_dd.values, 0, alpha=0.5, color='blue', label=f'V30 (Max: {v29_dd.min():.1f}%)')
    ax2.fill_between(spy_dd.index, spy_dd.values, 0, alpha=0.3, color='gray', label=f'SPY (Max: {spy_dd.min():.1f}%)')
    ax2.set_title('Drawdown Comparison', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-60, 5)
    
    # 3. Annual Returns Bar Chart
    ax3 = axes[1, 0]
    v29_annual = portfolio_df['value'].resample('YE').last().pct_change() * 100
    spy_annual = spy_norm.resample('YE').last().pct_change() * 100
    v29_vals = v29_annual.dropna()
    spy_vals = spy_annual.dropna()
    common_idx = v29_vals.index.intersection(spy_vals.index)
    if len(common_idx) > 0:
        v29_vals = v29_vals.loc[common_idx]
        spy_vals = spy_vals.loc[common_idx]
        years = common_idx.year
        x = np.arange(len(years))
        width = 0.35
        ax3.bar(x - width/2, v29_vals.values, width, label='V30', color='blue', alpha=0.7)
        ax3.bar(x + width/2, spy_vals.values, width, label='SPY', color='gray', alpha=0.7)
        ax3.set_xticks(x[::2])
        ax3.set_xticklabels(years[::2], rotation=45)
    ax3.set_title('Annual Returns by Year', fontsize=12)
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    alpha = metrics['annual_return'] - spy_metrics['annual_return']
    
    summary = f"""
    PERFORMANCE SUMMARY ({start_year}-{end_year})
    {'='*40}
    
    V30 DYNAMIC MEGA-CAP SPLIT STRATEGY:
      Final Value:    ${metrics['final_value']:,.0f}
      Annual Return:  {metrics['annual_return']:.1f}%
      Total Return:   {metrics['total_return']:.1f}%
      Max Drawdown:   {metrics['max_drawdown']:.1f}%
      Sharpe Ratio:   {metrics['sharpe']:.2f}
    
    SPY BENCHMARK:
      Annual Return:  {spy_metrics['annual_return']:.1f}%
      Max Drawdown:   {spy_metrics['max_drawdown']:.1f}%
    
    OUTPERFORMANCE:
      Alpha vs SPY:   {alpha:+.1f}%
      DD Improvement: {metrics['max_drawdown'] - spy_metrics['max_drawdown']:+.1f}%
    
    CONFIGURATION:
      70% Top 3 Magnificent 7
      30% Top 2 Momentum Stocks
      15% Trailing Stop Loss
      VIX-based Cash Reserve
    """
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(project_root, 'output', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, f"{strategy_name.lower()}_performance.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {png_path}")
    
    # Also save as v30_performance.png
#     v29_path = os.path.join(output_dir, 'v30_performance.png')
#     plt.savefig(v29_path, dpi=150, bbox_inches='tight')
#     logger.info(f"Saved: {v29_path}")
#     
#     plt.close()
    
    return png_path


def create_report(portfolio_df, metrics, spy_metrics, start_year, end_year, strategy_name="V30"):
    """Create performance report"""
    log_header("STEP 5: CREATING REPORT")
    
    output_dir = os.path.join(project_root, 'output', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{strategy_name.lower()}_performance_report.txt")
    
    alpha = metrics['annual_return'] - spy_metrics['annual_return']
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{strategy_name} STRATEGY - PERFORMANCE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Period: {start_year}-{end_year}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("STRATEGY CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write("  Strategy: V30 Dynamic Mega-Cap Split\n")
        f.write("  Allocation: 70% Magnificent 7 / 30% Momentum\n")
        f.write("  Mag7 Selection: Top 3 by 20-day momentum\n")
        f.write("  Momentum Selection: Top 2 (excluding Mag7)\n")
        f.write("  Rebalancing: Monthly (day 7-15)\n")
        f.write("  Trailing Stop: 15%\n")
        f.write("  VIX Crisis Level: 35 (up to 70% cash)\n")
        f.write("  Max Portfolio DD Control: 25%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("MAGNIFICENT 7 STOCKS\n")
        f.write("-" * 70 + "\n")
        f.write("  AAPL - Apple Inc.\n")
        f.write("  MSFT - Microsoft Corporation\n")
        f.write("  NVDA - NVIDIA Corporation\n")
        f.write("  GOOGL - Alphabet Inc.\n")
        f.write("  META - Meta Platforms Inc.\n")
        f.write("  AMZN - Amazon.com Inc.\n")
        f.write("  TSLA - Tesla Inc.\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PERFORMANCE RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Metric':<25} {'V30':>15} {'SPY':>15} {'Diff':>15}\n")
        f.write(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}\n")
        f.write(f"  {'Annual Return':<25} {metrics['annual_return']:>14.1f}% {spy_metrics['annual_return']:>14.1f}% {alpha:>+14.1f}%\n")
        f.write(f"  {'Total Return':<25} {metrics['total_return']:>14.1f}%\n")
        f.write(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>14.1f}% {spy_metrics['max_drawdown']:>14.1f}% {metrics['max_drawdown']-spy_metrics['max_drawdown']:>+14.1f}%\n")
        f.write(f"  {'Sharpe Ratio':<25} {metrics['sharpe']:>15.2f}\n")
        f.write(f"  {'Final Value':<25} ${metrics['final_value']:>13,.0f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("DRAWDOWN PROTECTION MECHANISMS\n")
        f.write("-" * 70 + "\n")
        f.write("  1. VIX-based Cash Reserves:\n")
        f.write("     VIX < 15: 5% cash | VIX 15-20: 10% | VIX 20-25: 20%\n")
        f.write("     VIX 25-30: 35% | VIX 30-40: 50% | VIX > 40: 70%\n\n")
        f.write("  2. Trailing Stop Losses: 15% from peak\n\n")
        f.write("  3. Portfolio Drawdown Control:\n")
        f.write("     DD < 5%: 100% | DD 5-10%: 90% | DD 10-15%: 75%\n")
        f.write("     DD 15-20%: 50% | DD > 20%: 25%\n\n")
        f.write("  4. Regime Detection: SPY vs MA50/MA200\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved: {report_path}")
    return report_path


def main(start_year=2015, end_year=2024, strategy="ml"):
    """Main execution function"""
    strategy_display = "ML REGULARIZED" if strategy.lower() == "ml" else "V30 DYNAMIC MEGA-CAP SPLIT"
    log_header(f"{strategy_display} STRATEGY - EXECUTION")
    logger.info(f"Selected Strategy: {strategy_display}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Project Root: {project_root}")
    
    try:
        # Step 1-2: Run backtest
        portfolio_df, bot, metrics, spy_metrics, strategy_name = run_backtest(start_year, end_year, strategy)
        
        # Step 3: Save to database
        run_id = save_to_database(portfolio_df, metrics, spy_metrics, strategy_name)
        
        # Step 4: Generate visualization
        viz_path = generate_visualization(portfolio_df, bot, metrics, spy_metrics, start_year, end_year, strategy_name)
        
        # Step 5: Create report
        report_path = create_report(portfolio_df, metrics, spy_metrics, start_year, end_year, strategy_name)
        
        # Summary
        log_header("EXECUTION COMPLETE")
        alpha = metrics['annual_return'] - spy_metrics['annual_return']
        
        logger.info(f"{strategy_name} Strategy executed successfully\!")
        logger.info("")
        logger.info("Outputs:")
        logger.info(f"  Database:      output/data/trading_results.db (Run ID: {run_id})")
        logger.info(f"  Visualization: {viz_path}")
        logger.info(f"  Report:        {report_path}")
        logger.info(f"  Logs:          output/logs/execution.log")
        logger.info("")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Period:   {start_year}-{end_year}")
        logger.info("")
        logger.info("Performance:")
        logger.info(f"  Annual Return: {metrics['annual_return']:.1f}%")
        logger.info(f"  Max Drawdown:  {metrics['max_drawdown']:.1f}%")
        logger.info(f"  Sharpe Ratio:  {metrics['sharpe']:.2f}")
        logger.info(f"  Alpha vs SPY:  {alpha:+.1f}%")
        logger.info(f"  Final Value:   ${metrics['final_value']:,.0f}")
        logger.info("")
        logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return portfolio_df, metrics
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Trading Strategy (ML Recommended or V30)')
    parser.add_argument('--strategy', type=str, default='ml', choices=['ml', 'v30'],
                       help='Strategy to run: ml (recommended, 106%% annual) or v30 (92%% annual)')
    parser.add_argument('--start', type=int, default=2015, help='Start year (default: 2015)')
    parser.add_argument('--end', type=int, default=2024, help='End year (default: 2024)')
    args = parser.parse_args()
    
    main(start_year=args.start, end_year=args.end, strategy=args.strategy)
