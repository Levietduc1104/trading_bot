"""
Validate Tier 2 Strategy (2006-2024)
Compare FA-enhanced vs pure momentum on period with full FA data coverage
"""
import sys
import logging
from datetime import datetime
from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from src.strategies.v31_enhanced import V31EnhancedStrategy
from src.strategies.v31_tier2_growth_scoring import V31Tier2GrowthScoringStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(portfolio_df, spy_df, strategy_name):
    """Calculate performance metrics"""
    if len(portfolio_df) == 0:
        return None

    # Align dates
    common_dates = portfolio_df.index.intersection(spy_df.index)
    portfolio_returns = portfolio_df.loc[common_dates, 'value'].pct_change().dropna()
    spy_returns = spy_df.loc[common_dates, 'close'].pct_change().dropna()

    # Calculate metrics
    total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0] - 1) * 100
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100

    # Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = (portfolio_returns.mean() / portfolio_returns.std() * (252 ** 0.5)) if portfolio_returns.std() > 0 else 0

    # SPY comparison
    spy_total_return = (spy_df.loc[common_dates, 'close'].iloc[-1] / spy_df.loc[common_dates, 'close'].iloc[0] - 1) * 100
    spy_annual_return = ((1 + spy_total_return/100) ** (1/years) - 1) * 100
    alpha = annual_return - spy_annual_return

    return {
        'strategy': strategy_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'spy_annual': spy_annual_return,
        'alpha': alpha,
        'final_value': portfolio_df['value'].iloc[-1],
        'years': years
    }


def main():
    print("=" * 80)
    print("TIER 2 VALIDATION: 2006-2024 (Full FA Data Coverage)")
    print("=" * 80)
    print()

    START_YEAR = 2006
    END_YEAR = 2024

    # Initialize bot
    print("📊 Loading data...")
    bot = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=100000
    )
    bot.load_all_stocks()

    if 'SPY' not in bot.stocks_data:
        print("❌ ERROR: SPY data not found")
        return

    spy_df = bot.stocks_data['SPY']
    print(f"✅ Loaded {len(bot.stocks_data)} stocks")
    print()

    # Test 1: Baseline V31 (Pure Momentum)
    print("=" * 80)
    print("TEST 1: V31 Enhanced (Pure Momentum - No FA Scoring)")
    print("=" * 80)
    print()

    bot1 = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=100000
    )
    bot1.load_all_stocks()

    strategy1 = V31EnhancedStrategy(
        bot=bot1,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=True
    )

    portfolio1 = strategy1.run_backtest(start_year=START_YEAR, end_year=END_YEAR)
    metrics1 = calculate_metrics(portfolio1, spy_df, "V31 Enhanced (Pure Momentum)")

    if metrics1:
        print()
        print(f"📈 Results:")
        print(f"   Annual Return: {metrics1['annual_return']:.2f}%")
        print(f"   Total Return: {metrics1['total_return']:.2f}%")
        print(f"   Max Drawdown: {metrics1['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {metrics1['sharpe_ratio']:.2f}")
        print(f"   Alpha vs SPY: {metrics1['alpha']:+.2f}%")
        print(f"   Final Value: ${metrics1['final_value']:,.2f}")
        print(f"   Total Costs: ${strategy1.total_costs:,.2f}")
        print(f"   Trades: {len(strategy1.trade_history)}")

    print()

    # Test 2: Tier 2 (70% Momentum + 30% FA Growth)
    print("=" * 80)
    print("TEST 2: Tier 2 Growth Scoring (70% Momentum + 30% FA)")
    print("=" * 80)
    print()

    bot2 = PortfolioRotationBot(
        data_dir='sp500_data/stock_data_1990_2024',
        initial_capital=100000
    )
    bot2.load_all_stocks()

    strategy2 = V31Tier2GrowthScoringStrategy(
        bot=bot2,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=True,
        enable_growth_scoring=True,
        momentum_weight=0.70,
        growth_weight=0.30
    )

    portfolio2 = strategy2.run_backtest(start_year=START_YEAR, end_year=END_YEAR)
    metrics2 = calculate_metrics(portfolio2, spy_df, "Tier 2 Growth Scoring")

    if metrics2:
        print()
        print(f"📈 Results:")
        print(f"   Annual Return: {metrics2['annual_return']:.2f}%")
        print(f"   Total Return: {metrics2['total_return']:.2f}%")
        print(f"   Max Drawdown: {metrics2['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {metrics2['sharpe_ratio']:.2f}")
        print(f"   Alpha vs SPY: {metrics2['alpha']:+.2f}%")
        print(f"   Final Value: ${metrics2['final_value']:,.2f}")
        print(f"   Total Costs: ${strategy2.total_costs:,.2f}")
        print(f"   Trades: {len(strategy2.trade_history)}")

    print()

    # Comparison
    if metrics1 and metrics2:
        print("=" * 80)
        print("COMPARISON: FA Scoring Impact")
        print("=" * 80)
        print()
        print(f"  Metric                        V31 Enhanced    Tier 2      Improvement")
        print(f"  " + "-" * 75)
        print(f"  Annual Return                 {metrics1['annual_return']:7.2f}%      {metrics2['annual_return']:7.2f}%   {metrics2['annual_return']-metrics1['annual_return']:+7.2f}%")
        print(f"  Max Drawdown                  {metrics1['max_drawdown']:7.2f}%      {metrics2['max_drawdown']:7.2f}%   {metrics2['max_drawdown']-metrics1['max_drawdown']:+7.2f}%")
        print(f"  Sharpe Ratio                  {metrics1['sharpe_ratio']:7.2f}       {metrics2['sharpe_ratio']:7.2f}    {metrics2['sharpe_ratio']-metrics1['sharpe_ratio']:+7.2f}")
        print(f"  Alpha vs SPY                  {metrics1['alpha']:+7.2f}%      {metrics2['alpha']:+7.2f}%   {metrics2['alpha']-metrics1['alpha']:+7.2f}%")
        print(f"  Final Value                   ${metrics1['final_value']:>10,.0f}  ${metrics2['final_value']:>10,.0f}  ${metrics2['final_value']-metrics1['final_value']:+10,.0f}")
        print()

        # Verdict
        improvement_pct = ((metrics2['final_value'] / metrics1['final_value']) - 1) * 100
        print(f"🎯 FA SCORING VALUE:")
        print(f"   Tier 2 outperformed baseline by {improvement_pct:+.2f}% ({metrics2['annual_return']-metrics1['annual_return']:+.2f}% annual)")

        if improvement_pct > 5:
            print(f"   ✅ SIGNIFICANT IMPROVEMENT - FA scoring adds substantial value")
        elif improvement_pct > 0:
            print(f"   ⚠️  MODEST IMPROVEMENT - FA scoring adds some value")
        else:
            print(f"   ❌ NO IMPROVEMENT - FA scoring not helping")

        print()

        # Period analysis
        print(f"📅 PERIOD ANALYSIS ({START_YEAR}-{END_YEAR}):")
        print(f"   Duration: {metrics1['years']:.1f} years")
        print(f"   Market events covered:")
        print(f"     • 2008 Financial Crisis ✅")
        print(f"     • 2011 European Debt Crisis ✅")
        print(f"     • 2015-2016 Market Correction ✅")
        print(f"     • 2018 Q4 Selloff ✅")
        print(f"     • 2020 COVID Crash ✅")
        print(f"     • 2022 Bear Market ✅")
        print(f"   FA Data Coverage: 100% (all stocks have data)")
        print()

    # Save results
    print("💾 Saving results...")

    # Create report
    with open('output/reports/tier2_fa_impact_2006_2024.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TIER 2 FA SCORING IMPACT ANALYSIS (2006-2024)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Period: {START_YEAR}-{END_YEAR} ({metrics1['years']:.1f} years)\n")
        f.write(f"Initial Capital: $100,000\n\n")

        f.write("-" * 80 + "\n")
        f.write("BASELINE: V31 Enhanced (Pure Momentum)\n")
        f.write("-" * 80 + "\n")
        if metrics1:
            for key, value in metrics1.items():
                if key == 'strategy':
                    continue
                if isinstance(value, float):
                    if 'return' in key or 'drawdown' in key or 'alpha' in key or 'spy' in key:
                        f.write(f"  {key:20s}: {value:7.2f}%\n")
                    else:
                        f.write(f"  {key:20s}: {value:7.2f}\n")
                else:
                    f.write(f"  {key:20s}: {value}\n")
        f.write(f"  Total Costs: ${strategy1.total_costs:,.2f}\n")
        f.write(f"  Trades: {len(strategy1.trade_history)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("TIER 2: FA-Enhanced Growth Scoring (70% Momentum + 30% FA)\n")
        f.write("-" * 80 + "\n")
        if metrics2:
            for key, value in metrics2.items():
                if key == 'strategy':
                    continue
                if isinstance(value, float):
                    if 'return' in key or 'drawdown' in key or 'alpha' in key or 'spy' in key:
                        f.write(f"  {key:20s}: {value:7.2f}%\n")
                    else:
                        f.write(f"  {key:20s}: {value:7.2f}\n")
                else:
                    f.write(f"  {key:20s}: {value}\n")
        f.write(f"  Total Costs: ${strategy2.total_costs:,.2f}\n")
        f.write(f"  Trades: {len(strategy2.trade_history)}\n\n")

        if metrics1 and metrics2:
            improvement_pct = ((metrics2['final_value'] / metrics1['final_value']) - 1) * 100
            f.write("-" * 80 + "\n")
            f.write("FA SCORING IMPACT\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Annual Return Improvement: {metrics2['annual_return']-metrics1['annual_return']:+.2f}%\n")
            f.write(f"  Drawdown Change: {metrics2['max_drawdown']-metrics1['max_drawdown']:+.2f}%\n")
            f.write(f"  Sharpe Improvement: {metrics2['sharpe_ratio']-metrics1['sharpe_ratio']:+.2f}\n")
            f.write(f"  Total Value Improvement: {improvement_pct:+.2f}%\n")
            f.write(f"  Dollar Improvement: ${metrics2['final_value']-metrics1['final_value']:+,.2f}\n\n")

    print("✅ Report saved to: output/reports/tier2_fa_impact_2006_2024.txt")
    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
