"""
COMPREHENSIVE STRATEGY OPTIMIZATION
Goal: 20%+ annual return AND profitable every year
"""
from portfolio_bot_demo import PortfolioRotationBot
import pandas as pd

def test_strategy(description, bot, backtest_func, **kwargs):
    """Test a strategy and return metrics"""
    portfolio_df = backtest_func(**kwargs)

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = portfolio_df_copy.groupby('year')['value'].apply(
        lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 0 else 0
    )

    # Metrics
    final_value = portfolio_df['value'].iloc[-1]
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((final_value / 100000) ** (1/years) - 1) * 100

    all_years_positive = all(yearly_returns > 0)
    worst_year = yearly_returns.min()

    return {
        'description': description,
        'annual_return': annual_return,
        'worst_year': worst_year,
        'all_years_positive': all_years_positive,
        'yearly_returns': yearly_returns
    }

print("="*80)
print("FINDING STRATEGY: 20%+ Annual Return + Profitable Every Year")
print("="*80)

bot = PortfolioRotationBot()
bot.prepare_data()
bot.score_all_stocks()

results = []

print("\n🔍 Testing strategies...")

# 1. Current baseline
print("\n1. Current strategy (20% cash, Top 10, Monthly)")
r = test_strategy("Current (baseline)", bot, bot.backtest, top_n=10)
results.append(r)
print(f"   → Annual: {r['annual_return']:.1f}% | Worst Yr: {r['worst_year']:.1f}% | All +: {r['all_years_positive']}")

# 2. Higher cash reserves
for cash in [0.30, 0.40, 0.50, 0.60, 0.70]:
    desc = f"{int(cash*100)}% cash reserve"
    print(f"\n2. {desc}")
    r = test_strategy(desc, bot, bot.backtest_with_cash, top_n=10, cash_reserve=cash)
    results.append(r)
    print(f"   → Annual: {r['annual_return']:.1f}% | Worst Yr: {r['worst_year']:.1f}% | All +: {r['all_years_positive']}")

# 3. More diversification
for n in [15, 20, 30, 50]:
    desc = f"Top {n} stocks"
    print(f"\n3. {desc}")
    r = test_strategy(desc, bot, bot.backtest, top_n=n)
    results.append(r)
    print(f"   → Annual: {r['annual_return']:.1f}% | Worst Yr: {r['worst_year']:.1f}% | All +: {r['all_years_positive']}")

# 4. Combo: High cash + diversification
for cash in [0.50, 0.60]:
    for n in [20, 30]:
        desc = f"{int(cash*100)}% cash + Top {n}"
        print(f"\n4. {desc}")
        r = test_strategy(desc, bot, bot.backtest_with_cash, top_n=n, cash_reserve=cash)
        results.append(r)
        print(f"   → Annual: {r['annual_return']:.1f}% | Worst Yr: {r['worst_year']:.1f}% | All +: {r['all_years_positive']}")

print("\n" + "="*80)
print("RESULTS RANKING")
print("="*80)

# Sort: 1) All years positive, 2) Annual return >= 20%, 3) Best worst year
results_sorted = sorted(results,
                       key=lambda x: (x['all_years_positive'],
                                     x['annual_return'] >= 20,
                                     x['worst_year'],
                                     x['annual_return']),
                       reverse=True)

print(f"\n{'#':<4} {'Strategy':<30} {'Annual':<10} {'Worst Yr':<12} {'Goal Met?'}")
print("-"*80)

for i, r in enumerate(results_sorted, 1):
    all_pos = '✅' if r['all_years_positive'] else '❌'
    goal_met = '🎯' if (r['annual_return'] >= 20 and r['all_years_positive']) else '❌'

    print(f"{i:<4} {r['description']:<30} {r['annual_return']:>6.1f}%   {r['worst_year']:>6.1f}%    {goal_met} {all_pos}")

print("\n" + "="*80)
print("GOAL: 20%+ annual return AND all years positive")
print("="*80)

# Find best that meets goal
best_meeting_goal = [r for r in results_sorted if r['annual_return'] >= 20 and r['all_years_positive']]

if best_meeting_goal:
    best = best_meeting_goal[0]
    print(f"\n✅ FOUND: {best['description']}")
    print(f"   Annual Return: {best['annual_return']:.1f}%")
    print(f"   Worst Year: {best['worst_year']:.1f}%")
    print(f"\n   Year-by-year:")
    for year, ret in best['yearly_returns'].items():
        print(f"     {year}: {ret:>6.1f}%")
else:
    print("\n⚠️  No strategy met BOTH goals (20% + all years positive)")
    print("\nBest compromise:")
    best = results_sorted[0]
    print(f"   {best['description']}")
    print(f"   Annual: {best['annual_return']:.1f}% | Worst Yr: {best['worst_year']:.1f}%")

print("\n" + "="*80)
