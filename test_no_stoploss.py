"""
TEST FASTER REBALANCING WITHOUT STOP-LOSS/TAKE-PROFIT
Hypothesis: Stop-loss and take-profit are hurting performance
Let's try just faster momentum rotation (weekly/biweekly)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'backtest'))
from portfolio_bot_demo import PortfolioRotationBot

print("="*80)
print("TESTING FASTER REBALANCING (NO STOP-LOSS/TAKE-PROFIT)")
print("="*80)

# Initialize bot
data_dir = os.path.join(os.path.dirname(__file__), 'sp500_data', 'daily')
bot = PortfolioRotationBot(data_dir=data_dir)
bot.prepare_data()
bot.score_all_stocks()

# Test with NO stop-loss/take-profit (set to very high values so they never trigger)
configs = [
    (5, 0.20, 1000, 1000, "Weekly, 20% cash, NO SL/TP"),
    (5, 0.30, 1000, 1000, "Weekly, 30% cash, NO SL/TP"),
    (5, 0.40, 1000, 1000, "Weekly, 40% cash, NO SL/TP"),
    (10, 0.20, 1000, 1000, "Biweekly, 20% cash, NO SL/TP"),
    (10, 0.30, 1000, 1000, "Biweekly, 30% cash, NO SL/TP"),
    (15, 0.20, 1000, 1000, "Every 15 days, 20% cash, NO SL/TP"),
    (3, 0.20, 1000, 1000, "Every 3 days, 20% cash, NO SL/TP"),
    (3, 0.30, 1000, 1000, "Every 3 days, 30% cash, NO SL/TP"),
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

        # Extract results
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
            'cash': cash
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY - FASTER REBALANCING (NO SL/TP)")
print("="*80)

results.sort(key=lambda x: x['annual'], reverse=True)

print(f"\n{'Rank':<6}{'Annual':<10}{'All+':<6}{'Min Year':<10}{'Description':<50}")
print("-"*80)

for i, r in enumerate(results, 1):
    all_pos_emoji = "✅" if r['all_positive'] else "❌"
    goal_emoji = "🎯" if r['annual'] >= 20 and r['all_positive'] else ""
    print(f"{i:<6}{r['annual']:>6.1f}%   {all_pos_emoji:<6}{r['min_year']:>6.1f}%   {r['desc']:<50}{goal_emoji}")

# Compare to baseline
print("\n" + "="*80)
print("COMPARISON TO BASELINE")
print("="*80)
print(f"Monthly rebalancing (baseline): 15.8% annual")
if results:
    best = results[0]
    print(f"Best faster rebalancing: {best['annual']:.1f}% annual")
    print(f"Improvement: {best['annual'] - 15.8:+.1f}%")

print("\n" + "="*80)
