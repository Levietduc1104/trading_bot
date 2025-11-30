"""
SWING TRADING OPTIMIZATION
Goal: 20%+ annual return + profitable every year

Swing trading advantages:
- More frequent rebalancing (catch recoveries faster)
- Stop-loss protection (cut losses early)
- Better for V-shape markets (COVID 2020)
- Exit bear markets faster (2022)
"""
from portfolio_bot_demo import PortfolioRotationBot

def test_swing(description, bot, rebalance_days, top_n, cash_reserve, stop_loss_pct):
    """Test a swing trading configuration"""

    portfolio_df = bot.backtest_swing_trading(
        top_n=top_n,
        rebalance_days=rebalance_days,
        cash_reserve=cash_reserve,
        stop_loss_pct=stop_loss_pct
    )

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
        'yearly_returns': yearly_returns,
        'config': {
            'rebalance_days': rebalance_days,
            'top_n': top_n,
            'cash_reserve': cash_reserve,
            'stop_loss_pct': stop_loss_pct
        }
    }

print("="*80)
print("SWING TRADING OPTIMIZATION")
print("Goal: 20%+ Annual Return + ALL Years Positive")
print("="*80)

bot = PortfolioRotationBot()
bot.prepare_data()
bot.score_all_stocks()

results = []

print("\n🔍 Testing swing trading configurations...")

# Test matrix
test_configs = [
    # (rebalance_days, top_n, cash_reserve, stop_loss_pct, description)
    (5, 10, 0.30, 5, "Weekly | Top 10 | 30% cash | 5% stop"),
    (5, 10, 0.40, 5, "Weekly | Top 10 | 40% cash | 5% stop"),
    (5, 10, 0.50, 5, "Weekly | Top 10 | 50% cash | 5% stop"),
    (5, 10, 0.40, 3, "Weekly | Top 10 | 40% cash | 3% stop"),
    (5, 10, 0.40, 7, "Weekly | Top 10 | 40% cash | 7% stop"),

    (3, 10, 0.40, 5, "3-day | Top 10 | 40% cash | 5% stop"),
    (7, 10, 0.40, 5, "7-day | Top 10 | 40% cash | 5% stop"),
    (10, 10, 0.40, 5, "10-day | Top 10 | 40% cash | 5% stop"),

    (5, 15, 0.40, 5, "Weekly | Top 15 | 40% cash | 5% stop"),
    (5, 20, 0.40, 5, "Weekly | Top 20 | 40% cash | 5% stop"),

    (5, 10, 0.35, 4, "Weekly | Top 10 | 35% cash | 4% stop"),
    (5, 10, 0.45, 4, "Weekly | Top 10 | 45% cash | 4% stop"),

    (1, 10, 0.50, 3, "DAILY | Top 10 | 50% cash | 3% stop"),
    (1, 10, 0.40, 5, "DAILY | Top 10 | 40% cash | 5% stop"),
    (1, 15, 0.40, 5, "DAILY | Top 15 | 40% cash | 5% stop"),
]

for i, (rebal, n, cash, stop, desc) in enumerate(test_configs, 1):
    print(f"\n{i}. {desc}")
    try:
        r = test_swing(desc, bot, rebal, n, cash, stop)
        results.append(r)
        print(f"   → Annual: {r['annual_return']:.1f}% | Worst Yr: {r['worst_year']:.1f}% | All +: {r['all_years_positive']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*80)
print("RESULTS RANKING")
print("="*80)

# Sort by: 1) All years positive, 2) Annual >= 20%, 3) Best worst year, 4) Highest annual
results_sorted = sorted(results,
                       key=lambda x: (x['all_years_positive'],
                                     x['annual_return'] >= 20,
                                     x['worst_year'],
                                     x['annual_return']),
                       reverse=True)

print(f"\n{'#':<4} {'Strategy':<50} {'Annual':<10} {'Worst Yr':<12} {'Goal?'}")
print("-"*80)

for i, r in enumerate(results_sorted, 1):
    all_pos = '✅' if r['all_years_positive'] else '❌'
    goal_met = '🎯' if (r['annual_return'] >= 20 and r['all_years_positive']) else '❌'

    print(f"{i:<4} {r['description']:<50} {r['annual_return']:>6.1f}%   {r['worst_year']:>6.1f}%    {goal_met}")

print("\n" + "="*80)
print("GOAL: 20%+ annual return AND all years positive")
print("="*80)

# Find best that meets goal
best_meeting_goal = [r for r in results_sorted if r['annual_return'] >= 20 and r['all_years_positive']]

if best_meeting_goal:
    print("\n✅ FOUND WINNING STRATEGY!")
    for r in best_meeting_goal[:3]:  # Show top 3
        print(f"\n{'='*80}")
        print(f"Strategy: {r['description']}")
        print(f"Annual Return: {r['annual_return']:.1f}%")
        print(f"Worst Year: {r['worst_year']:.1f}%")
        print(f"\nConfiguration:")
        print(f"  - Rebalance every {r['config']['rebalance_days']} days")
        print(f"  - Hold top {r['config']['top_n']} stocks")
        print(f"  - Keep {r['config']['cash_reserve']*100:.0f}% in cash")
        print(f"  - Stop-loss at {r['config']['stop_loss_pct']}%")
        print(f"\nYear-by-year performance:")
        for year, ret in r['yearly_returns'].items():
            status = '✅' if ret > 0 else '❌'
            print(f"  {year}: {ret:>7.1f}%  {status}")
else:
    print("\n⚠️  No strategy achieved BOTH goals")
    print("\nBest strategies:")

    # Show strategies with all years positive
    all_pos = [r for r in results_sorted if r['all_years_positive']]
    if all_pos:
        print("\n📊 Strategies with ALL years positive:")
        for r in all_pos[:3]:
            print(f"  - {r['description']}")
            print(f"    Annual: {r['annual_return']:.1f}% | Worst: {r['worst_year']:.1f}%")

    # Show strategies with highest returns
    print("\n📈 Strategies with highest returns:")
    for r in results_sorted[:3]:
        print(f"  - {r['description']}")
        print(f"    Annual: {r['annual_return']:.1f}% | Worst: {r['worst_year']:.1f}%")

print("\n" + "="*80)
