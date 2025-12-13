"""
PHASE 1 OPTIMIZATION TEST
Quick test of low-hanging fruit optimizations
"""

import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src', 'backtest'))

from portfolio_bot_demo import PortfolioRotationBot

def run_test(name, **kwargs):
    """Run a quick backtest"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    bot = PortfolioRotationBot(data_dir='sp500_data/daily')
    bot.prepare_data()
    bot.score_all_stocks()

    portfolio_df = bot.backtest_with_bear_protection(**kwargs)

    if portfolio_df is not None and len(portfolio_df) > 0:
        initial = portfolio_df['value'].iloc[0]
        final = portfolio_df['value'].iloc[-1]
        years = len(portfolio_df) / 252
        annual_return = (((final / initial) ** (1/years)) - 1) * 100

        # Max drawdown
        rolling_max = portfolio_df['value'].expanding().max()
        drawdown = ((portfolio_df['value'] - rolling_max) / rolling_max * 100).min()

        print(f"Annual Return: {annual_return:.1f}%")
        print(f"Max Drawdown: {drawdown:.1f}%")
        print(f"Final Value: ${final:,.0f}")

        return annual_return, drawdown, final
    return None, None, None

print("\n" + "="*60)
print("PHASE 1 OPTIMIZATION TESTS")
print("="*60)

# Baseline
print("\n🔷 BASELINE")
r1, d1, f1 = run_test(
    "Current: 10% bull, 70% bear, Top 10",
    top_n=10, rebalance_freq='M',
    bear_cash_reserve=0.70, bull_cash_reserve=0.10
)

# Test 1: Lower bull cash
print("\n🔷 TEST 1: LOWER BULL CASH")
r2, d2, f2 = run_test(
    "5% bull cash",
    top_n=10, rebalance_freq='M',
    bear_cash_reserve=0.70, bull_cash_reserve=0.05
)

# Test 2: Top 7 stocks
print("\n🔷 TEST 2: CONCENTRATED (TOP 7)")
r3, d3, f3 = run_test(
    "Top 7 stocks, 5% bull, 70% bear",
    top_n=7, rebalance_freq='M',
    bear_cash_reserve=0.70, bull_cash_reserve=0.05
)

# Test 3: Lower bear cash
print("\n🔷 TEST 3: LESS DEFENSIVE")
r4, d4, f4 = run_test(
    "60% bear cash (vs 70%)",
    top_n=10, rebalance_freq='M',
    bear_cash_reserve=0.60, bull_cash_reserve=0.05
)

# Test 4: Combined optimization
print("\n🔷 TEST 4: OPTIMIZED COMBO")
r5, d5, f5 = run_test(
    "Top 7, 5% bull, 60% bear",
    top_n=7, rebalance_freq='M',
    bear_cash_reserve=0.60, bull_cash_reserve=0.05
)

# Summary
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)

results = [
    ("Baseline (10% bull, 70% bear, Top 10)", r1, d1),
    ("Test 1: 5% bull cash", r2, d2),
    ("Test 2: Top 7 stocks", r3, d3),
    ("Test 3: 60% bear cash", r4, d4),
    ("Test 4: Optimized combo", r5, d5),
]

for name, ret, dd in results:
    if ret:
        improvement = ret - r1 if r1 else 0
        print(f"{name:40} | {ret:6.1f}% | {dd:6.1f}% | +{improvement:4.1f}%")

print("\n" + "="*60)
best_idx = max(range(len(results)), key=lambda i: results[i][1] if results[i][1] else 0)
best_name, best_ret, best_dd = results[best_idx]

print(f"🏆 WINNER: {best_name}")
print(f"   Annual Return: {best_ret:.1f}%")
print(f"   Max Drawdown: {best_dd:.1f}%")
print(f"   Improvement: +{best_ret - r1:.1f}% over baseline")
print("="*60 + "\n")
