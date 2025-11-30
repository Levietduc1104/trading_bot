"""
SWING TRADING CONFIGURATION TESTING
Goal: 20%+ annual return + ALL years positive

Testing combinations:
- Rebalance: 1, 3, 5, 7 days
- Cash reserve: 30%, 40%, 50%
- Stop-loss: 3%, 5%, 7%
- Take-profit: 10%, 15%, 20%
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backtest'))
from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("SWING TRADING OPTIMIZATION")
print("Goal: 20%+ Annual Return + ALL Years Positive")
print("="*80)

# Initialize bot
data_dir = os.path.join(os.path.dirname(__file__), 'sp500_data', 'daily')
bot = PortfolioRotationBot(data_dir=data_dir)
bot.prepare_data()
bot.score_all_stocks()

# Track best result
best_config = None
best_annual = 0
best_all_positive = False

# Configuration grid
configs = [
    # (rebalance_days, cash_reserve, stop_loss, take_profit, description)
    (5, 0.4, 5, 15, "Weekly, 40% cash, 5% SL, 15% TP"),
    (5, 0.5, 5, 15, "Weekly, 50% cash, 5% SL, 15% TP"),
    (5, 0.3, 5, 15, "Weekly, 30% cash, 5% SL, 15% TP"),
    (3, 0.4, 5, 15, "Twice/week, 40% cash, 5% SL, 15% TP"),
    (7, 0.4, 5, 15, "Every 7 days, 40% cash, 5% SL, 15% TP"),
    (5, 0.4, 3, 15, "Weekly, 40% cash, 3% SL, 15% TP (tight)"),
    (5, 0.4, 7, 15, "Weekly, 40% cash, 7% SL, 15% TP (loose)"),
    (5, 0.4, 5, 10, "Weekly, 40% cash, 5% SL, 10% TP"),
    (5, 0.4, 5, 20, "Weekly, 40% cash, 5% SL, 20% TP"),
    (1, 0.5, 5, 15, "Daily, 50% cash, 5% SL, 15% TP"),
    (1, 0.4, 3, 10, "Daily, 40% cash, 3% SL, 10% TP (aggressive)"),
    (3, 0.5, 3, 20, "Twice/week, 50% cash, 3% SL, 20% TP"),
]

results = []

for idx, (rebal, cash, sl, tp, desc) in enumerate(configs, 1):
    print(f"\n{'='*80}")
    print(f"CONFIG {idx}/{len(configs)}: {desc}")
    print(f"{'='*80}")

    try:
        portfolio_df = bot.backtest_swing_trading(
            top_n=10,
            rebalance_days=rebal,
            cash_reserve=cash,
            stop_loss_pct=sl,
            take_profit_pct=tp
        )

        # Extract results from last run
        final_value = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

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

        results.append({
            'desc': desc,
            'annual': annual_return,
            'all_positive': all_positive,
            'min_year': min_year,
            'rebal': rebal,
            'cash': cash,
            'sl': sl,
            'tp': tp
        })

        # Track best
        if annual_return > best_annual and all_positive:
            best_config = (rebal, cash, sl, tp, desc)
            best_annual = annual_return
            best_all_positive = True
        elif annual_return > best_annual and not best_all_positive:
            best_config = (rebal, cash, sl, tp, desc)
            best_annual = annual_return

    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY - ALL CONFIGURATIONS")
print("="*80)

# Sort by annual return
results.sort(key=lambda x: x['annual'], reverse=True)

print(f"\n{'Rank':<6}{'Annual':<10}{'All+':<6}{'Min Year':<10}{'Description':<50}")
print("-"*80)

for i, r in enumerate(results, 1):
    all_pos_emoji = "✅" if r['all_positive'] else "❌"
    goal_emoji = "🎯" if r['annual'] >= 20 and r['all_positive'] else ""
    print(f"{i:<6}{r['annual']:>6.1f}%   {all_pos_emoji:<6}{r['min_year']:>6.1f}%   {r['desc']:<50}{goal_emoji}")

# Best overall
if best_config:
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION")
    print("="*80)
    rebal, cash, sl, tp, desc = best_config
    print(f"Description: {desc}")
    print(f"Annual Return: {best_annual:.1f}%")
    print(f"All Years Positive: {best_all_positive}")
    print(f"\nParameters:")
    print(f"  - Rebalance every {rebal} days")
    print(f"  - Cash reserve: {cash*100:.0f}%")
    print(f"  - Stop-loss: {sl}%")
    print(f"  - Take-profit: {tp}%")

    if best_annual >= 20 and best_all_positive:
        print("\n🎯🎯🎯 GOAL ACHIEVED! 🎯🎯🎯")
    else:
        print(f"\n⚠️  Not quite there yet. Best: {best_annual:.1f}%")
        print("Try adjusting parameters or testing more configurations.")

print("\n" + "="*80)
