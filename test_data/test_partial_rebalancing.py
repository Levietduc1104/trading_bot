"""
TEST PARTIAL REBALANCING STRATEGIES - UNBIASED COMPARISON
Compare full liquidation vs smart partial rebalancing approaches
NO parameter tuning - using standard/logical thresholds only
"""
import sys
import os
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src', 'backtest'))

from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("PARTIAL REBALANCING STRATEGY COMPARISON - UNBIASED TEST")
print("="*80)
print("\nTesting 4 rebalancing approaches:")
print("1. Full Liquidation (current) - Sell ALL, buy ALL")
print("2. Partial Rebalancing - Keep overlap, only trade differences")
print("3. Threshold Partial - Keep if rank < 15 (standard buffer)")
print("4. Gradual Replacement - Replace worst 3 each month")
print()
print("All use Monthly rebalancing + Adaptive Regime")
print()

# Initialize bot
bot = PortfolioRotationBot(data_dir='sp500_data/daily')
bot.prepare_data()
bot.score_all_stocks()

results = []


# ============================================================================
# 1. FULL LIQUIDATION (CURRENT BASELINE)
# ============================================================================
print("\n[1/4] Testing FULL LIQUIDATION (current baseline)...")
print("   Sell ALL stocks every month, buy new top 10")

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
    'strategy': 'Full Liquidation (Current)',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final
})


# ============================================================================
# 2. PARTIAL REBALANCING (OVERLAP-BASED)
# ============================================================================
print("\n[2/4] Testing PARTIAL REBALANCING (keep overlap)...")
print("   Keep stocks still in top 10, only trade differences")

all_dates = bot.stocks_data['SPY'].index
cash = bot.initial_capital
holdings = {}
portfolio_values = []
last_month = None
total_trades = 0

for date in all_dates[100:]:
    current_month = date.month

    if current_month != last_month:
        last_month = current_month

        # Use adaptive regime
        cash_reserve = bot.calculate_adaptive_regime(date)

        # Score all stocks
        current_scores = {}
        for ticker, df in bot.stocks_data.items():
            df_at_date = df[df.index <= date]
            if len(df_at_date) >= 100:
                try:
                    current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                except:
                    pass

        ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
        new_top_10 = set([t for t, s in ranked[:10]])
        current_holdings = set(holdings.keys())

        # PARTIAL REBALANCING LOGIC
        # 1. Keep stocks that are still in top 10
        keep_stocks = current_holdings & new_top_10

        # 2. Sell stocks no longer in top 10
        sell_stocks = current_holdings - new_top_10
        for ticker in sell_stocks:
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                cash += holdings[ticker] * price
                del holdings[ticker]
                total_trades += 1

        # 3. Buy new stocks that entered top 10
        buy_stocks = new_top_10 - current_holdings

        # Calculate total portfolio value
        holdings_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                holdings_value += shares * price

        total_value = cash + holdings_value

        # Allocate capital: target = (1 - cash_reserve) split among top 10
        target_total_invested = total_value * (1 - cash_reserve)
        target_per_stock = target_total_invested / len(new_top_10) if len(new_top_10) > 0 else 0

        # Buy new stocks
        for ticker in buy_stocks:
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                shares = target_per_stock / price
                holdings[ticker] = shares
                cash -= target_per_stock
                total_trades += 1

        # Rebalance existing holdings to target weights
        for ticker in keep_stocks:
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                current_value = holdings[ticker] * price
                diff = target_per_stock - current_value

                if abs(diff) > 0.01:  # Only trade if difference > $0.01
                    shares_to_trade = diff / price
                    holdings[ticker] += shares_to_trade
                    cash -= diff

    # Calculate daily portfolio value
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

avg_trades_per_month = total_trades / (len(portfolio_df) / 21)

print(f"   Annual Return: {annual_return:.2f}%")
print(f"   Max Drawdown: {drawdown:.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")
print(f"   Avg Trades/Month: {avg_trades_per_month:.1f}")

results.append({
    'strategy': 'Partial Rebalancing',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final
})


# ============================================================================
# 3. THRESHOLD PARTIAL (KEEP IF RANK < 15)
# ============================================================================
print("\n[3/4] Testing THRESHOLD PARTIAL (keep if rank < 15)...")
print("   Keep stocks ranked in top 15, only sell if drops below #15")

all_dates = bot.stocks_data['SPY'].index
cash = bot.initial_capital
holdings = {}
portfolio_values = []
last_month = None

for date in all_dates[100:]:
    current_month = date.month

    if current_month != last_month:
        last_month = current_month

        cash_reserve = bot.calculate_adaptive_regime(date)

        # Score all stocks
        current_scores = {}
        for ticker, df in bot.stocks_data.items():
            df_at_date = df[df.index <= date]
            if len(df_at_date) >= 100:
                try:
                    current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                except:
                    pass

        ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top 15 for keeping threshold
        top_15 = set([t for t, s in ranked[:15]])
        top_10 = set([t for t, s in ranked[:10]])

        # Keep holdings that are still in top 15 (buffer zone)
        keep_stocks = set(holdings.keys()) & top_15

        # Sell stocks that dropped below rank 15
        sell_stocks = set(holdings.keys()) - top_15
        for ticker in sell_stocks:
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                cash += holdings[ticker] * price
                del holdings[ticker]

        # Need to add stocks to reach 10 holdings
        current_count = len(holdings)
        need_to_buy = 10 - current_count

        # Buy top-ranked stocks not yet held
        buy_candidates = [t for t, s in ranked if t not in holdings][:need_to_buy]

        # Calculate target allocation
        holdings_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                holdings_value += shares * price

        total_value = cash + holdings_value
        target_total_invested = total_value * (1 - cash_reserve)
        target_per_stock = target_total_invested / 10

        # Buy new stocks
        for ticker in buy_candidates:
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                shares = target_per_stock / price
                holdings[ticker] = shares
                cash -= target_per_stock

        # Rebalance existing to target
        for ticker in list(holdings.keys()):
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                price = df_at_date.iloc[-1]['close']
                current_value = holdings[ticker] * price
                diff = target_per_stock - current_value

                if abs(diff) > 0.01:
                    shares_to_trade = diff / price
                    holdings[ticker] += shares_to_trade
                    cash -= diff

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
    'strategy': 'Threshold (Top 15)',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final
})


# ============================================================================
# 4. GRADUAL REPLACEMENT (REPLACE WORST 3 EACH MONTH)
# ============================================================================
print("\n[4/4] Testing GRADUAL REPLACEMENT (replace worst 3/month)...")
print("   Each month, replace 3 worst-performing stocks")

all_dates = bot.stocks_data['SPY'].index
cash = bot.initial_capital
holdings = {}
portfolio_values = []
last_month = None

for date in all_dates[100:]:
    current_month = date.month

    if current_month != last_month:
        last_month = current_month

        cash_reserve = bot.calculate_adaptive_regime(date)

        # Score all stocks
        current_scores = {}
        for ticker, df in bot.stocks_data.items():
            df_at_date = df[df.index <= date]
            if len(df_at_date) >= 100:
                try:
                    current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                except:
                    pass

        ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)

        if len(holdings) < 10:
            # Initial fill: buy top 10
            top_10 = [t for t, s in ranked[:10]]
            holdings_value = 0
            for ticker, shares in holdings.items():
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    price = df_at_date.iloc[-1]['close']
                    holdings_value += shares * price

            total_value = cash + holdings_value
            target_total = total_value * (1 - cash_reserve)
            target_per_stock = target_total / 10

            for ticker in top_10:
                if ticker not in holdings:
                    df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                    if len(df_at_date) > 0:
                        price = df_at_date.iloc[-1]['close']
                        shares = target_per_stock / price
                        holdings[ticker] = shares
                        cash -= target_per_stock
        else:
            # Rank current holdings by score
            holdings_scores = [(ticker, current_scores.get(ticker, 0)) for ticker in holdings.keys()]
            holdings_sorted = sorted(holdings_scores, key=lambda x: x[1])

            # Replace worst 3 stocks
            replace_count = min(3, len(holdings_sorted))
            worst_stocks = [t for t, s in holdings_sorted[:replace_count]]

            # Sell worst 3
            for ticker in worst_stocks:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * price
                    del holdings[ticker]

            # Buy top 3 not currently held
            buy_candidates = [t for t, s in ranked if t not in holdings][:replace_count]

            holdings_value = 0
            for ticker, shares in holdings.items():
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    price = df_at_date.iloc[-1]['close']
                    holdings_value += shares * price

            total_value = cash + holdings_value
            target_per_stock = (total_value * (1 - cash_reserve)) / 10

            for ticker in buy_candidates:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    price = df_at_date.iloc[-1]['close']
                    shares = target_per_stock / price
                    holdings[ticker] = shares
                    cash -= target_per_stock

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
    'strategy': 'Gradual (Replace 3)',
    'annual_return': annual_return,
    'max_drawdown': drawdown,
    'sharpe_ratio': sharpe,
    'final_value': final
})


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL RESULTS - PARTIAL REBALANCING COMPARISON")
print("="*80)
print()
print(f"{'Strategy':<30} {'Annual Return':<15} {'Max DD':<12} {'Sharpe':<10} {'Final Value':<15}")
print("-"*85)

for r in results:
    print(f"{r['strategy']:<30} {r['annual_return']:>6.2f}%         {r['max_drawdown']:>6.2f}%      {r['sharpe_ratio']:>5.2f}     ${r['final_value']:>12,.0f}")

print()
print("="*80)

# Find best
best_idx = max(range(len(results)), key=lambda i: results[i]['annual_return'])
best = results[best_idx]

baseline = results[0]  # Full Liquidation

print(f"🏆 WINNER: {best['strategy']}")
print(f"   Annual Return: {best['annual_return']:.2f}%")
print(f"   Max Drawdown: {best['max_drawdown']:.2f}%")
print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
print(f"\n   Improvement vs Full Liquidation:")
print(f"   Annual Return: +{best['annual_return'] - baseline['annual_return']:.2f}%")
print(f"   Max Drawdown: {best['max_drawdown'] - baseline['max_drawdown']:.2f}%")
print(f"   Sharpe Ratio: +{best['sharpe_ratio'] - baseline['sharpe_ratio']:.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("-"*80)
print("• Partial rebalancing reduces unnecessary trading")
print("• Keeps winning positions longer (momentum)")
print("• Lower transaction costs (not modeled in backtest)")
print("• Better tax efficiency (more long-term gains)")
print("• All strategies use same adaptive regime (fair comparison)")
print("="*80)
