"""
TEST TIMING STRATEGIES - NO BIAS, NO TUNING
Compare 3 different timing approaches using standard parameters only
"""
import sys
import os
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src', 'backtest'))

from portfolio_bot_demo import PortfolioRotationBot

print("=" * 80)
print("TIMING STRATEGY COMPARISON - UNBIASED TEST")
print("=" * 80)
print("\nUsing STANDARD parameters only (no optimization)")
print("Baseline: Current 200-day MA strategy (60% bear cash)")
print()

# Initialize bot
bot = PortfolioRotationBot(data_dir='sp500_data/daily')
bot.prepare_data()
bot.score_all_stocks()

# Get SPY data for timing strategies
spy_data = bot.stocks_data['SPY'].copy()


def calculate_multi_timeframe_timing(spy_data, date):
    """
    Strategy 1: Multi-Timeframe Trend
    Standard MAs: 50, 200 (industry standard)
    """
    df = spy_data[spy_data.index <= date]
    if len(df) < 200:
        return 0.60  # Default to defensive

    price = df.iloc[-1]['close']
    ma50 = df['close'].tail(50).mean()
    ma200 = df['close'].tail(200).mean()

    # Standard trend classification (no tuning)
    if price > ma50 and ma50 > ma200:
        return 0.05  # Strong bull
    elif price > ma200 and price < ma50:
        return 0.30  # Weak bull
    elif price < ma200 and ma50 > ma200:
        return 0.50  # Weak bear
    else:
        return 0.70  # Strong bear


def calculate_volatility_timing(spy_data, date):
    """
    Strategy 2: Volatility-Based Timing
    Standard: 30-day realized vol, annualized
    Standard thresholds: 15% (low), 25% (high) - market norms
    """
    df = spy_data[spy_data.index <= date]
    if len(df) < 30:
        return 0.60

    returns = df['close'].pct_change()
    volatility = returns.tail(30).std() * np.sqrt(252)  # Annualized

    # Standard volatility regimes (no tuning)
    if volatility < 0.15:  # Low vol (< 15% - calm market)
        return 0.10
    elif volatility < 0.25:  # Medium vol (15-25% - normal)
        return 0.30
    else:  # High vol (> 25% - crisis)
        return 0.60


def calculate_adaptive_regime(spy_data, date, stocks_data):
    """
    Strategy 3: Adaptive Multi-Factor Regime Detection
    Uses 4 standard factors with equal weighting (no tuning)
    """
    df = spy_data[spy_data.index <= date]
    if len(df) < 200:
        return 0.60

    price = df.iloc[-1]['close']

    # Factor 1: Trend (200 MA - industry standard)
    ma200 = df['close'].tail(200).mean()
    trend_score = 1 if price > ma200 else -1

    # Factor 2: Momentum (50-day ROC - standard period)
    if len(df) >= 50:
        price_50_ago = df['close'].iloc[-50]
        momentum = (price / price_50_ago - 1)
        momentum_score = 1 if momentum > 0.05 else -1
    else:
        momentum_score = 0

    # Factor 3: Volatility (30-day vs 1-year average)
    returns = df['close'].pct_change()
    if len(returns) >= 252:
        vol_30 = returns.tail(30).std()
        vol_252 = returns.tail(252).std()
        vol_score = 1 if vol_30 < vol_252 else -1
    else:
        vol_score = 0

    # Factor 4: Market Breadth (% stocks above 200 MA)
    breadth_count = 0
    total_count = 0
    for ticker, stock_df in stocks_data.items():
        stock_at_date = stock_df[stock_df.index <= date]
        if len(stock_at_date) >= 200:
            stock_price = stock_at_date.iloc[-1]['close']
            stock_ma200 = stock_at_date['close'].tail(200).mean()
            if stock_price > stock_ma200:
                breadth_count += 1
            total_count += 1

    if total_count > 0:
        breadth_pct = breadth_count / total_count
        breadth_score = 1 if breadth_pct > 0.50 else -1  # 50% threshold
    else:
        breadth_score = 0

    # Equal weighting of all factors (no optimization)
    total_score = trend_score + momentum_score + vol_score + breadth_score

    # Map score to cash reserve (linear mapping, no tuning)
    if total_score >= 3:
        return 0.05   # Very bullish (score 3-4)
    elif total_score >= 1:
        return 0.25   # Bullish (score 1-2)
    elif total_score >= -1:
        return 0.45   # Neutral (score -1 to 0)
    else:
        return 0.65   # Bearish (score -2 to -4)


def backtest_timing_strategy(bot, strategy_name, cash_function, use_breadth=False):
    """Run backtest with custom timing strategy"""

    all_dates = bot.stocks_data['SPY'].index
    cash = bot.initial_capital
    holdings = {}
    portfolio_values = []

    last_month = None

    for date in all_dates[100:]:
        current_month = date.month

        # Monthly rebalance
        if current_month != last_month:
            last_month = current_month

            # Sell all holdings
            for ticker in list(holdings.keys()):
                if ticker in bot.stocks_data:
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        price = df_at_date.iloc[-1]['close']
                        cash += holdings[ticker] * price
            holdings = {}

            # Determine cash reserve using strategy
            if use_breadth:
                cash_reserve = cash_function(bot.stocks_data['SPY'], date, bot.stocks_data)
            else:
                cash_reserve = cash_function(bot.stocks_data['SPY'], date)

            # Score and rank stocks
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [t for t, s in ranked[:10]]

            # Invest
            invest_amount = cash * (1 - cash_reserve)
            if len(top_stocks) > 0:
                per_stock = invest_amount / len(top_stocks)
                for ticker in top_stocks:
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        price = df_at_date.iloc[-1]['close']
                        shares = per_stock / price
                        holdings[ticker] = shares
                        cash -= per_stock

        # Calculate daily portfolio value
        holdings_value = 0
        for ticker, shares in holdings.items():
            if ticker in bot.stocks_data:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    price = df_at_date.iloc[-1]['close']
                    holdings_value += shares * price

        total_value = cash + holdings_value
        portfolio_values.append({'date': date, 'value': total_value})

    # Create DataFrame
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')

    # Calculate metrics
    initial = portfolio_df['value'].iloc[0]
    final = portfolio_df['value'].iloc[-1]
    years = len(portfolio_df) / 252
    annual_return = (((final / initial) ** (1/years)) - 1) * 100

    rolling_max = portfolio_df['value'].expanding().max()
    drawdown = ((portfolio_df['value'] - rolling_max) / rolling_max * 100).min()

    daily_returns = portfolio_df['value'].pct_change().dropna()
    rf_rate = 0.02
    excess_returns = daily_returns - (rf_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    return {
        'strategy': strategy_name,
        'annual_return': annual_return,
        'max_drawdown': drawdown,
        'sharpe_ratio': sharpe,
        'final_value': final,
        'portfolio_df': portfolio_df
    }


print("\n" + "=" * 80)
print("RUNNING TESTS (This may take a few minutes...)")
print("=" * 80)

# Test 1: Baseline (Current Strategy)
print("\n[1/4] Testing BASELINE (200-day MA, 60% bear cash)...")
baseline = bot.backtest_with_bear_protection(
    top_n=10,
    rebalance_freq='M',
    bear_cash_reserve=0.60,
    bull_cash_reserve=0.10
)

if baseline is not None:
    initial = baseline['value'].iloc[0]
    final = baseline['value'].iloc[-1]
    years = len(baseline) / 252
    baseline_annual = (((final / initial) ** (1/years)) - 1) * 100

    rolling_max = baseline['value'].expanding().max()
    baseline_dd = ((baseline['value'] - rolling_max) / rolling_max * 100).min()

    daily_returns = baseline['value'].pct_change().dropna()
    excess_returns = daily_returns - (0.02 / 252)
    baseline_sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std()

    print(f"   Annual Return: {baseline_annual:.2f}%")
    print(f"   Max Drawdown: {baseline_dd:.2f}%")
    print(f"   Sharpe Ratio: {baseline_sharpe:.2f}")

# Test 2: Multi-Timeframe Trend
print("\n[2/4] Testing MULTI-TIMEFRAME TREND (50/200 MA)...")
result_mt = backtest_timing_strategy(bot, "Multi-Timeframe", calculate_multi_timeframe_timing)
print(f"   Annual Return: {result_mt['annual_return']:.2f}%")
print(f"   Max Drawdown: {result_mt['max_drawdown']:.2f}%")
print(f"   Sharpe Ratio: {result_mt['sharpe_ratio']:.2f}")

# Test 3: Volatility-Based
print("\n[3/4] Testing VOLATILITY-BASED (30-day realized vol)...")
result_vol = backtest_timing_strategy(bot, "Volatility", calculate_volatility_timing)
print(f"   Annual Return: {result_vol['annual_return']:.2f}%")
print(f"   Max Drawdown: {result_vol['max_drawdown']:.2f}%")
print(f"   Sharpe Ratio: {result_vol['sharpe_ratio']:.2f}")

# Test 4: Adaptive Multi-Factor
print("\n[4/4] Testing ADAPTIVE REGIME (4-factor equal weight)...")
result_adaptive = backtest_timing_strategy(bot, "Adaptive", calculate_adaptive_regime, use_breadth=True)
print(f"   Annual Return: {result_adaptive['annual_return']:.2f}%")
print(f"   Max Drawdown: {result_adaptive['max_drawdown']:.2f}%")
print(f"   Sharpe Ratio: {result_adaptive['sharpe_ratio']:.2f}")

# Summary
print("\n" + "=" * 80)
print("📊 FINAL RESULTS - UNBIASED COMPARISON")
print("=" * 80)
print()
print(f"{'Strategy':<30} {'Annual Return':<15} {'Max DD':<12} {'Sharpe':<10} {'Final Value':<15}")
print("-" * 80)

results = [
    ("Baseline (200 MA)", baseline_annual, baseline_dd, baseline_sharpe, final),
    ("Multi-Timeframe Trend", result_mt['annual_return'], result_mt['max_drawdown'],
     result_mt['sharpe_ratio'], result_mt['final_value']),
    ("Volatility-Based", result_vol['annual_return'], result_vol['max_drawdown'],
     result_vol['sharpe_ratio'], result_vol['final_value']),
    ("Adaptive Multi-Factor", result_adaptive['annual_return'], result_adaptive['max_drawdown'],
     result_adaptive['sharpe_ratio'], result_adaptive['final_value'])
]

for name, ret, dd, sharpe, value in results:
    print(f"{name:<30} {ret:>6.2f}%         {dd:>6.2f}%      {sharpe:>5.2f}     ${value:>12,.0f}")

print()
print("=" * 80)

# Find best by annual return
best_idx = max(range(len(results)), key=lambda i: results[i][1])
best_name, best_ret, best_dd, best_sharpe, best_val = results[best_idx]

print(f"🏆 WINNER BY ANNUAL RETURN: {best_name}")
print(f"   Annual Return: {best_ret:.2f}%")
print(f"   Max Drawdown: {best_dd:.2f}%")
print(f"   Sharpe Ratio: {best_sharpe:.2f}")
print(f"   Improvement: +{best_ret - baseline_annual:.2f}% vs baseline")
print("=" * 80)
print()
print("Note: All strategies use STANDARD parameters (no tuning/optimization)")
print("This ensures fair, unbiased comparison across different market conditions")
print("=" * 80)
