"""
TEST BEAR MARKET PROTECTION STRATEGY
Focus: REDUCE LOSSES during crises (2021, 2022)
Then optimize for returns

Strategy: When SPY < 200-day MA → Go defensive (70% cash)
         When SPY > 200-day MA → Aggressive (20% cash)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backtest'))
from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("BEAR MARKET PROTECTION TESTING")
print("Goal: Reduce losses during 2021-2022, protect capital first")
print("="*80)

# Initialize bot
data_dir = os.path.join(os.path.dirname(__file__), 'sp500_data', 'daily')
bot = PortfolioRotationBot(data_dir=data_dir)
bot.prepare_data()
bot.score_all_stocks()

# Test different bear/bull cash allocations
configs = [
    ('M', 0.70, 0.20, "Monthly, 70% bear / 20% bull"),
    ('M', 0.80, 0.20, "Monthly, 80% bear / 20% bull (very defensive)"),
    ('M', 0.60, 0.20, "Monthly, 60% bear / 20% bull"),
    ('M', 0.70, 0.30, "Monthly, 70% bear / 30% bull"),
    ('M', 0.70, 0.10, "Monthly, 70% bear / 10% bull (aggressive bull)"),
    ('W', 0.70, 0.20, "Weekly, 70% bear / 20% bull"),
    ('W', 0.80, 0.20, "Weekly, 80% bear / 20% bull"),
    ('W', 0.60, 0.30, "Weekly, 60% bear / 30% bull"),
]

results = []

for idx, (freq, bear_cash, bull_cash, desc) in enumerate(configs, 1):
    print(f"\n{'='*80}")
    print(f"CONFIG {idx}/{len(configs)}: {desc}")
    print(f"{'='*80}")

    try:
        portfolio_df = bot.backtest_with_bear_protection(
            top_n=10,
            rebalance_freq=freq,
            bear_cash_reserve=bear_cash,
            bull_cash_reserve=bull_cash
        )

        # Extract results
        final_value = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

        # Max drawdown
        cummax = portfolio_df['value'].cummax()
        drawdown = ((portfolio_df['value'] - cummax) / cummax * 100)
        max_drawdown = drawdown.min()

        # Yearly returns
        portfolio_df_copy = portfolio_df.copy()
        portfolio_df_copy['year'] = portfolio_df_copy.index.year
        yearly_returns = {}
        for year in portfolio_df_copy['year'].unique():
            year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
            if len(year_data) > 1:
                year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
                yearly_returns[year] = year_return

        all_positive = all(r > 0 for r in yearly_returns.values())
        min_year = min(yearly_returns.values()) if yearly_returns else 0
        year_2021 = yearly_returns.get(2021, 0)
        year_2022 = yearly_returns.get(2022, 0)

        results.append({
            'desc': desc,
            'annual': annual_return,
            'all_positive': all_positive,
            'min_year': min_year,
            'max_dd': max_drawdown,
            'year_2021': year_2021,
            'year_2022': year_2022,
            'freq': freq,
            'bear_cash': bear_cash,
            'bull_cash': bull_cash
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY - BEAR MARKET PROTECTION RESULTS")
print("="*80)

results.sort(key=lambda x: (x['all_positive'], x['annual']), reverse=True)

print(f"\n{'Rank':<6}{'Annual':<10}{'All+':<6}{'2021':<10}{'2022':<10}{'Max DD':<10}{'Description':<45}")
print("-"*100)

for i, r in enumerate(results, 1):
    all_pos_emoji = "✅" if r['all_positive'] else "❌"
    goal_emoji = "🎯" if r['annual'] >= 20 and r['all_positive'] else "⭐" if r['all_positive'] else ""
    print(f"{i:<6}{r['annual']:>6.1f}%   {all_pos_emoji:<6}{r['year_2021']:>6.1f}%   {r['year_2022']:>6.1f}%   {r['max_dd']:>6.1f}%   {r['desc']:<45}{goal_emoji}")

# Best result
if results:
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION (Priority: All years positive)")
    print("="*80)
    best = results[0]
    print(f"Description: {best['desc']}")
    print(f"Annual Return: {best['annual']:.1f}%")
    print(f"All Years Positive: {best['all_positive']}")
    print(f"Max Drawdown: {best['max_dd']:.1f}%")
    print(f"\nCrisis Years:")
    print(f"  2021: {best['year_2021']:.1f}%")
    print(f"  2022: {best['year_2022']:.1f}%")

    if best['annual'] >= 20 and best['all_positive']:
        print("\n🎯🎯🎯 GOAL ACHIEVED! 🎯🎯🎯")
    elif best['all_positive']:
        print(f"\n✅ SUCCESS! All years positive including 2021 & 2022!")
        print(f"   Now need to optimize to reach 20% annual (currently {best['annual']:.1f}%)")
    else:
        print(f"\n⚠️  Still have negative years")

# Compare to baseline
print("\n" + "="*80)
print("COMPARISON TO BASELINE")
print("="*80)
print(f"Monthly rebalancing (baseline): 15.8% annual, 2021: -11.4%, 2022: -25.7%")
if results:
    print(f"Best bear protection: {results[0]['annual']:.1f}% annual, 2021: {results[0]['year_2021']:.1f}%, 2022: {results[0]['year_2022']:.1f}%")
    print(f"\nImprovement in crisis years:")
    print(f"  2021: {results[0]['year_2021'] - (-11.4):+.1f}%")
    print(f"  2022: {results[0]['year_2022'] - (-25.7):+.1f}%")

print("\n" + "="*80)
