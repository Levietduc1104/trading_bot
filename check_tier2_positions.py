"""
Check Tier 2 Buy/Sell Signals
==============================

Utility to show current Tier 2 strategy recommendations:
- What stocks to BUY
- What stocks to SELL (if currently holding)
- Target allocations
- Execution plan for quarterly rebalancing

Usage:
    python3 check_tier2_positions.py
    python3 check_tier2_positions.py --holdings AAPL,MSFT,GOOGL,NVDA,AMZN,META,TSLA,BRK.B,V,JPM
    python3 check_tier2_positions.py --date 2024-03-31
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from src.strategies.v31_tier2_growth_scoring import V31Tier2GrowthScoringStrategy


def check_positions(current_holdings=None, target_date=None, capital=100000):
    """
    Check current Tier 2 recommendations by running backtest

    Args:
        current_holdings: List of currently held tickers
        target_date: Target date to check (default: latest)
        capital: Portfolio capital for allocation calculation
    """
    print("\n" + "="*80)
    print("TIER 2 GROWTH SCORING - BUY/SELL SIGNALS")
    print("="*80)
    print(f"\nStrategy: V31 Tier 2 (70% Momentum + 30% Growth Fundamentals)")
    print(f"Portfolio: 70% Top 3 Mega-caps + 30% Top 7 Momentum")
    print(f"Capital: ${capital:,.0f}\n")

    # Initialize and run backtest to get current recommendations
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=capital)
    bot.prepare_data()

    # Determine date range
    if target_date:
        end_year = pd.to_datetime(target_date).year
    else:
        # Get latest date from SPY
        spy_df = bot.stocks_data.get('SPY')
        if spy_df is not None:
            latest_date = spy_df.index[-1]
            end_year = latest_date.year
            target_date = latest_date.strftime('%Y-%m-%d')
        else:
            end_year = datetime.now().year
            target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Target Date: {target_date}")
    print(f"Analyzing data through: {end_year}\n")

    # Run strategy to get recommendations
    strategy = V31Tier2GrowthScoringStrategy(
        bot=bot,
        use_transaction_costs=True,
        broker='interactive_brokers',
        enable_covered_calls=True,
        enable_growth_scoring=True,
        momentum_weight=0.70,
        growth_weight=0.30
    )

    # Run backtest (quick, just to get final holdings)
    print("Calculating current recommendations...")
    portfolio_df = strategy.run_backtest(start_year=end_year, end_year=end_year)

    # Get final holdings from the backtest
    # The strategy stores holdings internally
    if hasattr(strategy, 'holdings') and strategy.holdings:
        # Get the last set of holdings
        recommended_tickers = sorted(list(strategy.holdings.keys()))
    else:
        print("⚠️  Could not determine recommended positions from backtest")
        print("    Try running: python3 src/core/execution.py --strategy v31_growth --start 2024 --end 2024")
        return

    print("\n" + "─" * 80)
    print(f"RECOMMENDED PORTFOLIO ({len(recommended_tickers)} stocks)")
    print("─" * 80)

    # Classify as mega-caps vs momentum (based on typical Tier 2 allocation)
    # Top 3 by market cap are mega-caps, rest are momentum
    market_caps = {}
    for ticker in recommended_tickers:
        if ticker in bot.stocks_data:
            df = bot.stocks_data[ticker]
            if len(df) > 0:
                # Get market cap from bot
                market_cap = bot.get_market_cap(ticker, df.index[-1]) if hasattr(bot, 'get_market_cap') else 0
                market_caps[ticker] = market_cap

    # Sort by market cap to identify mega-caps
    sorted_by_mcap = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
    megacap_tickers = [t for t, _ in sorted_by_mcap[:3]]
    momentum_tickers = [t for t in recommended_tickers if t not in megacap_tickers]

    # Get current prices
    prices = {}
    for ticker in recommended_tickers:
        if ticker in bot.stocks_data:
            df = bot.stocks_data[ticker]
            if len(df) > 0:
                prices[ticker] = df['close'].iloc[-1]

    # Calculate allocations
    megacap_allocation = capital * 0.70
    momentum_allocation = capital * 0.30

    per_megacap = megacap_allocation / 3 if len(megacap_tickers) == 3 else 0
    per_momentum = momentum_allocation / len(momentum_tickers) if len(momentum_tickers) > 0 else 0

    # Display mega-caps
    print("\n📊 TOP 3 MEGA-CAPS (70% allocation):")
    print(f"{'Ticker':<8} {'Price':<12} {'Target $':<15} {'Shares':<10} {'Status'}")
    print("─" * 80)

    for ticker in megacap_tickers:
        price = prices.get(ticker, 0)
        shares = int(per_megacap / price) if price > 0 else 0
        status = ""
        if current_holdings:
            if ticker in current_holdings:
                status = "✓ HOLD"
            else:
                status = "🟢 BUY"
        print(f"{ticker:<8} ${price:>10.2f} ${per_megacap:>13,.0f} {shares:>8}   {status}")

    # Display momentum stocks
    print("\n📈 TOP MOMENTUM STOCKS (30% allocation):")
    print(f"{'Ticker':<8} {'Price':<12} {'Target $':<15} {'Shares':<10} {'Status'}")
    print("─" * 80)

    for ticker in momentum_tickers:
        price = prices.get(ticker, 0)
        shares = int(per_momentum / price) if price > 0 else 0
        status = ""
        if current_holdings:
            if ticker in current_holdings:
                status = "✓ HOLD"
            else:
                status = "🟢 BUY"
        print(f"{ticker:<8} ${price:>10.2f} ${per_momentum:>13,.0f} {shares:>8}   {status}")

    # Determine buy/sell/hold
    if current_holdings:
        current_set = set(current_holdings)
        recommended_set = set(recommended_tickers)

        to_buy = recommended_set - current_set
        to_sell = current_set - recommended_set
        to_hold = current_set & recommended_set

        # Show sell recommendations
        if len(to_sell) > 0:
            print("\n🔴 SELL (exit these positions):")
            print(f"{'Ticker':<8} {'Reason'}")
            print("─" * 80)
            for ticker in sorted(to_sell):
                print(f"{ticker:<8} Not in current top 10 recommendations")

        # Summary
        print("\n" + "─" * 80)
        print("EXECUTION SUMMARY")
        print("─" * 80)
        print(f"\n🟢 BUY:  {len(to_buy)} positions - {', '.join(sorted(to_buy)) if to_buy else 'None'}")
        print(f"🔴 SELL: {len(to_sell)} positions - {', '.join(sorted(to_sell)) if to_sell else 'None'}")
        print(f"✓ HOLD: {len(to_hold)} positions - {', '.join(sorted(to_hold)) if to_hold else 'None'}")
    else:
        print("\n" + "─" * 80)
        print("EXECUTION PLAN (Starting Fresh)")
        print("─" * 80)
        print(f"\n🟢 BUY all {len(recommended_tickers)} positions:")
        print(f"   Mega-caps: {', '.join(megacap_tickers)}")
        print(f"   Momentum: {', '.join(momentum_tickers)}")

    # Allocation summary
    print("\n" + "─" * 80)
    print("ALLOCATION SUMMARY")
    print("─" * 80)
    print(f"\nTotal Capital: ${capital:,.0f}")
    print(f"\nMega-cap (70%): ${megacap_allocation:,.0f}")
    print(f"  → ${per_megacap:,.0f} per stock (3 stocks)")
    print(f"\nMomentum (30%): ${momentum_allocation:,.0f}")
    print(f"  → ${per_momentum:,.0f} per stock ({len(momentum_tickers)} stocks)")

    print("\n" + "=" * 80)
    print("\n💡 TIP: Run this before each quarterly rebalancing (Jan 1, Apr 1, Jul 1, Oct 1)")
    print("💡 TIP: Use --holdings to compare vs your current portfolio\n")

    return {
        'recommended': recommended_tickers,
        'megacaps': megacap_tickers,
        'momentum': momentum_tickers,
        'buy': list(to_buy) if current_holdings else recommended_tickers,
        'sell': list(to_sell) if current_holdings else [],
        'hold': list(to_hold) if current_holdings else []
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check Tier 2 Buy/Sell Signals')
    parser.add_argument('--holdings', type=str, default=None,
                       help='Current holdings (comma-separated), e.g., AAPL,MSFT,GOOGL')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date (YYYY-MM-DD), default: latest')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Portfolio capital (default: 100000)')

    args = parser.parse_args()

    # Parse current holdings
    current_holdings = None
    if args.holdings:
        current_holdings = [x.strip().upper() for x in args.holdings.split(',')]
        print(f"\n📋 Your Current Holdings: {', '.join(current_holdings)}")

    # Get recommendations
    check_positions(
        current_holdings=current_holdings,
        target_date=args.date,
        capital=args.capital
    )


if __name__ == '__main__':
    main()
