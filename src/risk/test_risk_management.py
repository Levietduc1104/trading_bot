"""
TEST RISK MANAGEMENT STRATEGIES
Goal: 20%+ annual return + ALL years positive

Testing different combinations of:
- Stop-loss: 3%, 5%, 7%, 10%
- Take-profit: 10%, 15%, 20%, 25%
- Cash reserve: 20%, 30%, 40%
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.backtest.portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("RISK MANAGEMENT OPTIMIZATION")
print("Goal: 20%+ Annual Return + ALL Years Positive")
print("="*80)

bot = PortfolioRotationBot()
bot.prepare_data()
bot.score_all_stocks()

results = []

print("\n🔍 Testing risk management configurations...")

# Test matrix
configs = [
    # (stop_loss, take_profit, cash_reserve, description)
    (5, 15, 0.20, "5% stop | 15% profit | 20% cash"),
    (5, 20, 0.20, "5% stop | 20% profit | 20% cash"),
    (5, 25, 0.20, "5% stop | 25% profit | 20% cash"),

    (3, 15, 0.20, "3% stop | 15% profit | 20% cash"),
    (7, 15, 0.20, "7% stop | 15% profit | 20% cash"),
    (10, 20, 0.20, "10% stop | 20% profit | 20% cash"),

    (5, 15, 0.30, "5% stop | 15% profit | 30% cash"),
    (5, 15, 0.40, "5% stop | 15% profit | 40% cash"),

    (3, 20, 0.30, "3% stop | 20% profit | 30% cash"),
    (5, 20, 0.30, "5% stop | 20% profit | 30% cash"),
    (7, 20, 0.30, "7% stop | 20% profit | 30% cash"),

    (5, 25, 0.30, "5% stop | 25% profit | 30% cash"),
    (7, 25, 0.30, "7% stop | 25% profit | 30% cash"),

    (3, 15, 0.40, "3% stop | 15% profit | 40% cash"),
    (5, 20, 0.40, "5% stop | 20% profit | 40% cash"),
]

for i, (stop, profit, cash, desc) in enumerate(configs, 1):
    print(f"\n{i}/{len(configs)}. {desc}")

    try:
        pf = bot.backtest_with_risk_management(
            top_n=10,
            stop_loss_pct=stop,
            take_profit_pct=profit,
            cash_reserve=cash
        )

        # Calculate metrics
        pf_copy = pf.copy()
        pf_copy['year'] = pf_copy.index.year
        yearly_returns = pf_copy.groupby('year')['value'].apply(
            lambda x: ((x.iloc[-1] / x.iloc[0]) - 1) * 100 if len(x) > 0 else 0
        )

        final_value = pf['value'].iloc[-1]
        years = (pf.index[-1] - pf.index[0]).days / 365.25
        annual_return = ((final_value / 100000) ** (1/years) - 1) * 100

        all_positive = all(yearly_returns > 0)
        worst_year = yearly_returns.min()

        results.append({
            'description': desc,
            'stop_loss': stop,
            'take_profit': profit,
            'cash_reserve': cash,
            'annual_return': annual_return,
            'worst_year': worst_year,
            'all_years_positive': all_positive,
            'yearly_returns': yearly_returns
        })

        print(f"   → Annual: {annual_return:.1f}% | Worst Yr: {worst_year:.1f}% | All +: {all_positive}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("RESULTS RANKING")
print("="*80)

# Sort: 1) All years positive, 2) Annual >= 20%, 3) Best worst year, 4) Highest annual
results_sorted = sorted(results,
                       key=lambda x: (x['all_years_positive'],
                                     x['annual_return'] >= 20,
                                     x['worst_year'],
                                     x['annual_return']),
                       reverse=True)

print(f"\n{'#':<4} {'Strategy':<45} {'Annual':<10} {'Worst Yr':<12} {'Goal?'}")
print("-"*80)

for i, r in enumerate(results_sorted, 1):
    goal_met = '🎯' if (r['annual_return'] >= 20 and r['all_years_positive']) else '❌'
    print(f"{i:<4} {r['description']:<45} {r['annual_return']:>6.1f}%   {r['worst_year']:>6.1f}%    {goal_met}")

print("\n" + "="*80)
print("GOAL: 20%+ annual return AND all years positive")
print("="*80)

# Find winners
winners = [r for r in results_sorted if r['annual_return'] >= 20 and r['all_years_positive']]

if winners:
    print(f"\n✅ FOUND {len(winners)} WINNING STRATEGY(IES)!")

    for r in winners[:3]:
        print(f"\n{'='*80}")
        print(f"✨ {r['description']}")
        print(f"Annual Return: {r['annual_return']:.1f}%")
        print(f"Worst Year: {r['worst_year']:.1f}%")
        print(f"\nConfiguration:")
        print(f"  - Stop-Loss: {r['stop_loss']}%")
        print(f"  - Take-Profit: {r['take_profit']}%")
        print(f"  - Cash Reserve: {r['cash_reserve']*100:.0f}%")
        print(f"\nYear-by-year:")
        for year, ret in r['yearly_returns'].items():
            status = '✅' if ret > 0 else '❌'
            print(f"  {year}: {ret:>7.1f}%  {status}")
else:
    print("\n⚠️  No strategy achieved BOTH goals")

    # Show best compromise
    print("\nBest strategies:")

    all_pos = [r for r in results_sorted if r['all_years_positive']]
    if all_pos:
        print("\n📊 All years positive (but < 20%):")
        for r in all_pos[:3]:
            print(f"  {r['description']}: {r['annual_return']:.1f}% annual, {r['worst_year']:.1f}% worst")

    print("\n📈 Highest returns (but some losses):")
    for r in results_sorted[:3]:
        print(f"  {r['description']}: {r['annual_return']:.1f}% annual, {r['worst_year']:.1f}% worst")

print("\n" + "="*80)
