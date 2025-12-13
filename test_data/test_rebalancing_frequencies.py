"""
TEST REBALANCING STRATEGIES - UNBIASED, NO OVERFITTING
Compare standard rebalancing frequencies used in industry
NO parameter tuning, NO optimization - just test what's commonly used
"""
import sys
import os
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src', 'backtest'))

from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("REBALANCING STRATEGY COMPARISON - UNBIASED TEST")
print("="*80)
print("\nTesting 4 STANDARD rebalancing frequencies (no tuning):")
print("1. Weekly (52x/year) - Active approach")
print("2. Monthly (12x/year) - Current baseline")
print("3. Quarterly (4x/year) - Passive approach")
print("4. Semi-Annual (2x/year) - Very passive")
print()
print("Note: Using standard adaptive regime for all tests")
print()

# Initialize bot
bot = PortfolioRotationBot(data_dir='sp500_data/daily')
bot.prepare_data()
bot.score_all_stocks()

results = []


# ============================================================================
# 1. WEEKLY REBALANCING (52x per year)
# ============================================================================
print("\n[1/4] Testing WEEKLY rebalancing (52x per year)...")

portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,
    rebalance_freq='W',
    use_adaptive_regime=True
)

initial = portfolio_df['value'].iloc[0]
final = portfolio_df['value'].iloc[-1]
years = len(portfolio_df) / 252
annual_return = (((final / initial) ** (1/years)) - 1) * 100

rolling_max = portfolio_df['value'].expanding().max()
drawdown = ((portfolio_df['value'] - rolling_max) / rolling_max * 100).min()

daily_returns = portfolio_df['value'].pct_change().dropna()
excess_returns = daily_returns - (0.02 / 252)
sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

print(f"   Annual Return: {annual_return:.2f}%")
print(f"   Max Drawdown: {drawdown:.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")

results.append({
    'strategy': 'Weekly',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final,
    'rebalances_per_year': 52
})


# ============================================================================
# 2. MONTHLY REBALANCING (12x per year) - CURRENT BASELINE
# ============================================================================
print("\n[2/4] Testing MONTHLY rebalancing (12x per year) - BASELINE...")

portfolio_df = bot.backtest_with_bear_protection(
    top_n=10,
    rebalance_freq='M',
    use_adaptive_regime=True
)

initial = portfolio_df['value'].iloc[0]
final = portfolio_df['value'].iloc[-1]
years = len(portfolio_df) / 252
annual_return = (((final / initial) ** (1/years)) - 1) * 100

rolling_max = portfolio_df['value'].expanding().max()
drawdown = ((portfolio_df['value'] - rolling_max) / rolling_max * 100).min()

daily_returns = portfolio_df['value'].pct_change().dropna()
excess_returns = daily_returns - (0.02 / 252)
sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

print(f"   Annual Return: {annual_return:.2f}%")
print(f"   Max Drawdown: {drawdown:.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")

results.append({
    'strategy': 'Monthly (Current)',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final,
    'rebalances_per_year': 12
})


# ============================================================================
# 3. QUARTERLY REBALANCING (4x per year)
# ============================================================================
print("\n[3/4] Testing QUARTERLY rebalancing (4x per year)...")

all_dates = bot.stocks_data['SPY'].index
cash = bot.initial_capital
holdings = {}
portfolio_values = []
quarters_passed = 0
last_quarter = None

for date in all_dates[100:]:
    current_quarter = (date.month - 1) // 3  # 0, 1, 2, 3 for Q1, Q2, Q3, Q4

    if current_quarter != last_quarter:
        last_quarter = current_quarter

        # Liquidate
        for ticker in list(holdings.keys()):
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                cash += holdings[ticker] * price
        holdings = {}

        # Use adaptive regime
        cash_reserve = bot.calculate_adaptive_regime(date)

        # Score stocks
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

    # Calculate daily value
    holdings_value = 0
    for ticker, shares in holdings.items():
        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
        if len(df_at_date) > 0:
            price = df_at_date.iloc[-1]['close']
            holdings_value += shares * price

    total_value = cash + holdings_value
    portfolio_values.append({'date': date, 'value': total_value})

portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
initial = portfolio_df['value'].iloc[0]
final = portfolio_df['value'].iloc[-1]
years = len(portfolio_df) / 252
annual_return = (((final / initial) ** (1/years)) - 1) * 100

rolling_max = portfolio_df['value'].expanding().max()
drawdown = ((portfolio_df['value'] - rolling_max) / rolling_max * 100).min()

daily_returns = portfolio_df['value'].pct_change().dropna()
excess_returns = daily_returns - (0.02 / 252)
sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

print(f"   Annual Return: {annual_return:.2f}%")
print(f"   Max Drawdown: {drawdown:.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")

results.append({
    'strategy': 'Quarterly',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final,
    'rebalances_per_year': 4
})


# ============================================================================
# 4. SEMI-ANNUAL REBALANCING (2x per year)
# ============================================================================
print("\n[4/4] Testing SEMI-ANNUAL rebalancing (2x per year)...")

all_dates = bot.stocks_data['SPY'].index
cash = bot.initial_capital
holdings = {}
portfolio_values = []
last_half = None

for date in all_dates[100:]:
    current_half = 0 if date.month <= 6 else 1  # H1 or H2

    if current_half != last_half:
        last_half = current_half

        # Liquidate
        for ticker in list(holdings.keys()):
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                cash += holdings[ticker] * price
        holdings = {}

        # Use adaptive regime
        cash_reserve = bot.calculate_adaptive_regime(date)

        # Score stocks
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

    # Calculate daily value
    holdings_value = 0
    for ticker, shares in holdings.items():
        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
        if len(df_at_date) > 0:
            price = df_at_date.iloc[-1]['close']
            holdings_value += shares * price

    total_value = cash + holdings_value
    portfolio_values.append({'date': date, 'value': total_value})

portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
initial = portfolio_df['value'].iloc[0]
final = portfolio_df['value'].iloc[-1]
years = len(portfolio_df) / 252
annual_return = (((final / initial) ** (1/years)) - 1) * 100

rolling_max = portfolio_df['value'].expanding().max()
drawdown = ((portfolio_df['value'] - rolling_max) / rolling_max * 100).min()

daily_returns = portfolio_df['value'].pct_change().dropna()
excess_returns = daily_returns - (0.02 / 252)
sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

print(f"   Annual Return: {annual_return:.2f}%")
print(f"   Max Drawdown: {drawdown:.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")

results.append({
    'strategy': 'Semi-Annual',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final,
    'rebalances_per_year': 2
})


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL RESULTS - REBALANCING FREQUENCY COMPARISON")
print("="*80)
print()
print(f"{'Strategy':<20} {'Rebal/Year':<12} {'Annual Return':<15} {'Max DD':<12} {'Sharpe':<10} {'Final Value':<15}")
print("-"*95)

for r in results:
    print(f"{r['strategy']:<20} {r['rebalances_per_year']:>4}         {r['annual_return']:>6.2f}%         {r['max_drawdown']:>6.2f}%      {r['sharpe_ratio']:>5.2f}     ${r['final_value']:>12,.0f}")

print()
print("="*80)

# Find best by annual return
best_idx = max(range(len(results)), key=lambda i: results[i]['annual_return'])
best = results[best_idx]

# Find best by Sharpe
best_sharpe_idx = max(range(len(results)), key=lambda i: results[i]['sharpe_ratio'])
best_sharpe = results[best_sharpe_idx]

print(f"🏆 WINNER BY ANNUAL RETURN: {best['strategy']}")
print(f"   Annual Return: {best['annual_return']:.2f}%")
print(f"   Max Drawdown: {best['max_drawdown']:.2f}%")
print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
print(f"   Rebalances: {best['rebalances_per_year']}x per year")

baseline = next(r for r in results if 'Current' in r['strategy'])
print(f"\n   Improvement vs Monthly: +{best['annual_return'] - baseline['annual_return']:.2f}% annual return")

print(f"\n🎯 WINNER BY SHARPE RATIO: {best_sharpe['strategy']}")
print(f"   Annual Return: {best_sharpe['annual_return']:.2f}%")
print(f"   Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("-"*80)
print("• More frequent rebalancing = higher transaction costs (not modeled)")
print("• Less frequent rebalancing = more drift from optimal portfolio")
print("• Quarterly/Semi-Annual may have higher tax efficiency")
print("• All strategies use same adaptive regime detection (fair comparison)")
print("="*80)
