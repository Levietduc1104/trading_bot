"""
V30 Strategy Live Trading - Main Execution Script
==================================================
Runs V30 Vol-Weighted strategy for live trading on Alpaca

Strategy:
- 70% Top 3 Mega-Caps (volatility-weighted)
- 30% Top 2 Momentum stocks (volatility-weighted)
- VIX-based cash reserves (5%-70%)
- 15% trailing stops
- Position limits: 10%-25% per stock

Usage:
    python v30_live_trading.py           # Run with default settings
    python v30_live_trading.py --debug   # Run with debug mode
"""

import sys
import os
from datetime import datetime
import pandas as pd
import argparse
import json
import os
from datetime import datetime, timedelta


# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v30_bot_adapter import V30LiveBot
from src.strategies.v30_vol_weighted import V30VolWeightedStrategy


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)



# ============================================================
# REBALANCE DATE CHECKING
# ============================================================

REBALANCE_FILE = 'last_rebalance.json'
REBALANCE_DAYS = 63  # Quarterly (63 trading days ~= 3 months)

def should_rebalance():
    """
    Check if enough time has passed since last rebalance
    
    Returns:
        bool: True if rebalancing is needed, False otherwise
    """
    if not os.path.exists(REBALANCE_FILE):
        print("\n✅ First rebalance - no previous rebalance found")
        return True
    
    try:
        with open(REBALANCE_FILE, 'r') as f:
            data = json.load(f)
            last_date = datetime.fromisoformat(data['date'])
        
        days_since = (datetime.now() - last_date).days
        
        if days_since >= REBALANCE_DAYS:
            print(f"\n✅ {days_since} days since last rebalance (>={REBALANCE_DAYS} days)")
            print(f"   Quarterly rebalancing is due")
            return True
        else:
            print(f"\n⏸️  REBALANCING NOT NEEDED")
            print(f"   Last rebalanced: {last_date.strftime('%Y-%m-%d')} ({days_since} days ago)")
            print(f"   V30 rebalances quarterly: every {REBALANCE_DAYS} days")
            print(f"   Next rebalance in: {REBALANCE_DAYS - days_since} days")
            print(f"\n   Your positions are protected with 15% trailing stops")
            print(f"   No action needed - just let the strategy work\!")
            return False
            
    except Exception as e:
        print(f"\n⚠️  Error reading last rebalance date: {e}")
        print("   Proceeding with rebalance to be safe")
        return True

def mark_rebalanced():
    """Mark that rebalancing was done today"""
    try:
        with open(REBALANCE_FILE, 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"\n✅ Rebalance date recorded: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"   Next rebalance: ~{(datetime.now() + timedelta(days=REBALANCE_DAYS)).strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"\n⚠️  Could not save rebalance date: {e}")


def run_v30_live(debug=True, rebalance_frequency='quarterly'):
    """
    Run V30 strategy for live trading

    Args:
        debug: Enable debug mode
        rebalance_frequency: 'weekly', 'monthly', or 'quarterly'
    """
    print_header("V30 VOL-WEIGHTED STRATEGY - LIVE TRADING")

    print(f"\n{'Strategy Overview:'}")
    print(f"  - Volatility-weighted position sizing")
    print(f"  - 70% Top 3 Mega-Caps / 30% Top 2 Momentum")
    print(f"  - VIX-based cash reserves (5%-70%)")
    print(f"  - 15% trailing stops")
    print(f"  - Position limits: 10%-25%")
    print(f"  - Rebalance: {rebalance_frequency}")

    print(f"\n{'Historical Performance (2015-2024):'}")
    print(f"  - Annual Return: 14.9%")
    print(f"  - Max Drawdown: -21.9%")
    print(f"  - Sharpe Ratio: 1.02")

    # Initialize bot
    print_header("STEP 1: INITIALIZE BOT")
    bot = V30LiveBot(
        paper_trading=True,
        lookback_days=252,  # 1 year of data
        debug=debug
    )

    # Load market data
    
    # ============================================================
    # CHECK IF REBALANCING IS NEEDED
    # ============================================================
    print_header("REBALANCE CHECK")
    
    if not should_rebalance():
        print("\n" + "="*60)
        print("⏸️  NO REBALANCING NEEDED - SHOWING CURRENT STATUS")
        print("="*60)
        
        # Show current positions
        try:
            positions = bot.broker.get_positions()
            account = bot.broker.get_account_info()
            
            print(f"\n{'='*70}")
            print("CURRENT PORTFOLIO STATUS")
            print(f"{'='*70}")
            print(f"\nPortfolio Value:      ${account['portfolio_value']:>12,.2f}")
            print(f"Cash Available:       ${account['cash']:>12,.2f}")
            print(f"P/L Today:            ${account.get('equity', 0) - account.get('last_equity', account.get('equity', 0)):>12,.2f}")
            
            if not positions.empty:
                print(f"\nActive Positions:     {len(positions)}")
                print(f"\n{'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'Value':>12} {'P/L':>10} {'P/L %':>8}")
                print("-" * 70)
                
                total_value = 0
                total_pl = 0
                
                for _, pos in positions.iterrows():
                    symbol = pos['symbol']
                    qty = int(pos['qty'])
                    entry = float(pos['avg_entry_price'])
                    current = float(pos['current_price'])
                    value = float(pos['market_value'])
                    pl = float(pos.get('unrealized_pl', 0))
                    pl_pct = float(pos.get('unrealized_plpc', 0)) * 100
                    
                    # Status indicator
                    status = "🟢" if pl_pct > 0 else "🔴" if pl_pct < 0 else "⚪"
                    
                    print(f"{status} {symbol:<6} {qty:>6} ${entry:>9.2f} ${current:>9.2f} ${value:>11,.2f} ${pl:>9,.2f} {pl_pct:>7.2f}%")
                    
                    total_value += value
                    total_pl += pl
                
                print("-" * 70)
                total_pl_pct = (total_pl / (total_value - total_pl)) * 100 if (total_value - total_pl) > 0 else 0
                print(f"{'TOTAL':<8} {'':<6} {'':>10} {'':>10} ${total_value:>11,.2f} ${total_pl:>9,.2f} {total_pl_pct:>7.2f}%")
            else:
                print("\nNo positions currently held")
            
            # Check and SET trailing stops if missing
            print(f"\n{'='*70}")
            print("TRAILING STOP STATUS")
            print(f"{'='*70}")
            
            try:
                orders = bot.broker.api.list_orders(status='open')
                trailing_stops = [o for o in orders if o.type == 'trailing_stop']
                
                if trailing_stops:
                    print(f"\nActive Trailing Stops: {len(trailing_stops)}")
                    for order in trailing_stops:
                        trail_pct = f"{float(order.trail_percent)*100:.1f}%" if hasattr(order, 'trail_percent') and order.trail_percent else "N/A"
                        print(f"  • {order.symbol}: {order.qty} shares @ {trail_pct} trail")
                    
                    # Check if all positions have stops
                    positions_with_stops = {o.symbol for o in trailing_stops}
                    all_positions = {pos['symbol'] for _, pos in positions.iterrows()}
                    missing_stops = all_positions - positions_with_stops
                    
                    if missing_stops:
                        print(f"\n⚠️  WARNING: {len(missing_stops)} position(s) WITHOUT trailing stops:")
                        for symbol in missing_stops:
                            print(f"     - {symbol}")
                        
                        print(f"\n🔧 Setting missing trailing stops...")
                        for symbol in missing_stops:
                            pos = positions[positions['symbol'] == symbol].iloc[0]
                            qty = int(pos['qty'])
                            order_id = bot.broker.place_trailing_stop_order(symbol, qty, trail_percent=0.15)
                            if order_id:
                                print(f"     ✅ {symbol}: {qty} shares @ 15.0% trail")
                        
                        print(f"\n✅ All positions now protected\!")
                    else:
                        print(f"\n✅ All {len(positions)} positions are protected")
                        
                else:
                    print("\n⚠️  No active trailing stops found")
                    print(f"\n🔧 Setting 15% trailing stops for all {len(positions)} positions...")
                    
                    success_count = bot.broker.set_trailing_stops_for_portfolio(trail_percent=0.15)
                    
                    if success_count > 0:
                        print(f"\n✅ Successfully set {success_count}/{len(positions)} trailing stops")
                        print(f"   All positions are now protected\!")
                    else:
                        print(f"\n⚠️  Failed to set trailing stops")
                        print("   Run manually: python3 set_trailing_stops.py")
                        
            except Exception as e:
                print(f"\n⚠️  Error managing trailing stops: {e}")
                print("   Run manually: python3 set_trailing_stops.py")
            
            print(f"\n{'='*70}")
            print("✅ PORTFOLIO IS ACTIVE - NO ACTION NEEDED")
            print(f"{'='*70}")
            print("\nWhat's happening:")
            print("  ✅ Positions are being monitored 24/7")
            print("  ✅ Trailing stops will trigger automatically if needed")
            print("  ✅ Next quarterly rebalance in 63 days")
            print(f"\n{'='*70}\n")
            
        except Exception as e:
            print(f"\n⚠️  Could not fetch positions: {e}")
            print("\nTo check manually:")
            print("  python3 monitor_positions.py")
            print("\n" + "="*60 + "\n")
        
        return
    
    print("\n✅ Proceeding with quarterly rebalancing...")
    

    print_header("STEP 2: LOAD MARKET DATA")
    bot.load_market_data()

    if not bot.stocks_data:
        print("\n❌ ERROR: No stock data loaded. Cannot proceed.")
        print("Check your network connection to Alpaca API.")
        return

    # Initialize V30 strategy
    print_header("STEP 3: INITIALIZE V30 STRATEGY")

    v30_config = {
        'megacap_allocation': 0.70,
        'num_megacap': 3,
        'num_momentum': 2,
        'trailing_stop': 0.15,
        'max_portfolio_dd': 0.25,
        'vix_crisis': 35,
        'num_top_megacaps': 7,
        'lookback_trading_value': 20,
        'vol_lookback': 20,
        'min_position_size': 0.10,
        'max_position_size': 0.25,
        'rebalance_frequency': rebalance_frequency
    }

    if debug:
        print(f"\n[DEBUG] V30 Configuration:")
        for key, value in v30_config.items():
            print(f"[DEBUG]   {key}: {value}")

    strategy = V30VolWeightedStrategy(
        bot=bot,
        config=v30_config,
        use_transaction_costs=False  # Alpaca handles transaction costs
    )

    # Get current date and VIX
    print_header("STEP 4: ANALYZE MARKET CONDITIONS")

    current_date = pd.Timestamp.now().normalize()
    current_vix = bot.get_current_vix()

    print(f"\nCurrent Date: {current_date.date()}")
    print(f"Current VIX: {current_vix:.2f}")

    # Calculate VIX-based cash reserve
    vix_cash_reserve = strategy.get_vix_cash_reserve(current_vix)
    print(f"VIX Cash Reserve: {vix_cash_reserve*100:.1f}%")

    # Get account info
    account = bot.broker.get_account_info()
    portfolio_value = account['portfolio_value']
    investable_amount = portfolio_value * (1 - vix_cash_reserve)

    print(f"\nPortfolio Value: ${portfolio_value:,.2f}")
    print(f"Cash Reserve: ${portfolio_value * vix_cash_reserve:,.2f}")
    print(f"Investable Amount: ${investable_amount:,.2f}")

    # Identify mega-caps
    print_header("STEP 5: IDENTIFY MEGA-CAPS")

    if debug:
        print(f"\n[DEBUG] Identifying top {v30_config['num_top_megacaps']} mega-caps by trading volume...")

    megacaps = strategy.identify_megacaps(
        current_date,
        top_n=v30_config['num_top_megacaps']
    )

    print(f"\nTop {len(megacaps)} Mega-Caps:")
    for i, ticker in enumerate(megacaps, 1):
        if ticker in bot.stocks_data:
            latest_price = bot.stocks_data[ticker]['close'].iloc[-1]
            print(f"  {i}. {ticker:6} - ${latest_price:.2f}")

    # Calculate momentum scores for mega-caps
    print_header("STEP 6: SELECT TOP MEGA-CAPS BY MOMENTUM")

    if debug:
        print(f"\n[DEBUG] Calculating 6-month momentum for mega-caps...")

    megacap_scores = []
    for ticker in megacaps:
        if ticker in bot.stocks_data:
            df = bot.stocks_data[ticker]
            if len(df) >= 126:  # Need 6 months
                returns_6m = (df['close'].iloc[-1] / df['close'].iloc[-126]) - 1
                megacap_scores.append((ticker, returns_6m))
                if debug:
                    print(f"[DEBUG]   {ticker}: {returns_6m*100:+.2f}%")

    megacap_scores.sort(key=lambda x: x[1], reverse=True)
    top_megacaps = megacap_scores[:v30_config['num_megacap']]

    print(f"\nSelected Top {len(top_megacaps)} Mega-Caps (70% allocation):")
    for ticker, score in top_megacaps:
        print(f"  {ticker:6} - 6M Return: {score*100:+.2f}%")

    # Find top momentum stocks (non-megacaps)
    print_header("STEP 7: SELECT TOP MOMENTUM STOCKS")

    if debug:
        print(f"\n[DEBUG] Finding top {v30_config['num_momentum']} momentum stocks from non-megacaps...")

    momentum_scores = []
    for ticker, df in bot.stocks_data.items():
        if ticker not in megacaps and ticker not in ['SPY', 'VIX']:
            if len(df) >= 126:
                returns_6m = (df['close'].iloc[-1] / df['close'].iloc[-126]) - 1
                momentum_scores.append((ticker, returns_6m))

    momentum_scores.sort(key=lambda x: x[1], reverse=True)
    top_momentum = momentum_scores[:v30_config['num_momentum']]

    print(f"\nSelected Top {len(top_momentum)} Momentum Stocks (30% allocation):")
    for ticker, score in top_momentum:
        print(f"  {ticker:6} - 6M Return: {score*100:+.2f}%")

    # Calculate volatility-weighted positions
    print_header("STEP 8: CALCULATE VOLATILITY-WEIGHTED POSITIONS")

    megacap_capital = investable_amount * v30_config['megacap_allocation']
    momentum_capital = investable_amount * (1 - v30_config['megacap_allocation'])

    if debug:
        print(f"\n[DEBUG] Mega-cap capital: ${megacap_capital:,.2f}")
        print(f"[DEBUG] Momentum capital: ${momentum_capital:,.2f}")

    # Get volatility-weighted allocations
    megacap_allocations = strategy.calculate_vol_weighted_positions(
        top_megacaps, current_date, megacap_capital
    )
    momentum_allocations = strategy.calculate_vol_weighted_positions(
        top_momentum, current_date, momentum_capital
    )

    # Combine allocations
    all_allocations = {**megacap_allocations, **momentum_allocations}

    print(f"\nTarget Positions (Volatility-Weighted):")
    print(f"{'Ticker':<8} {'Amount':>12} {'% Portfolio':>12} {'Type':<10}")
    print("-" * 50)

    total_allocated = 0
    for ticker, amount in sorted(all_allocations.items(), key=lambda x: x[1], reverse=True):
        pct = (amount / portfolio_value) * 100
        stock_type = "Mega-Cap" if ticker in [t for t, _ in top_megacaps] else "Momentum"
        print(f"{ticker:<8} ${amount:>10,.2f} {pct:>10.2f}% {stock_type:<10}")
        total_allocated += amount

    print("-" * 50)
    print(f"{'Total':<8} ${total_allocated:>10,.2f} {(total_allocated/portfolio_value)*100:>10.2f}%")

    # Convert to target weights for broker
    target_positions = {
        ticker: amount / portfolio_value
        for ticker, amount in all_allocations.items()
    }

    # Execute rebalance
    print_header("STEP 9: EXECUTE REBALANCE")

    print(f"\nRebalancing portfolio...")
    print(f"  Target positions: {len(target_positions)}")
    print(f"  Cash reserve: {vix_cash_reserve*100:.1f}%")

    try:
        bot.broker.rebalance_portfolio(
            target_positions=target_positions,
            cash_reserve=vix_cash_reserve
        )
        print(f"\n✓ Rebalance executed successfully!")


        # Set 15% trailing stops on all positions (V30 requirement)
        print(f"\n" + "="*60)
        print("SETTING TRAILING STOPS (15%)")
        print("="*60)
        
        try:
            success_count = bot.broker.set_trailing_stops_for_portfolio(trail_percent=0.15)
            if success_count > 0:
                print(f"\n✓ Trailing stops active - {success_count} positions protected\!")
            else:
                print(f"\n⚠️  No positions to protect (100% cash)")
        except Exception as e:
            print(f"\n⚠️  Could not set trailing stops: {e}")
            print("   Run manually: python3 set_trailing_stops.py")
        
        # Record rebalance date
        mark_rebalanced()

    except Exception as e:
        print(f"\n❌ Rebalance failed: {e}")
        if debug:
            import traceback
            print(f"\n[DEBUG] Full error:")
            traceback.print_exc()

    # Summary
    print_header("EXECUTION COMPLETE")

    print(f"\nSummary:")
    print(f"  Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Invested: ${total_allocated:,.2f} ({(total_allocated/portfolio_value)*100:.1f}%)")
    print(f"  Cash Reserve: ${portfolio_value * vix_cash_reserve:,.2f} ({vix_cash_reserve*100:.1f}%)")
    print(f"  Positions: {len(target_positions)}")
    print(f"  Rebalance Frequency: {rebalance_frequency}")

    print(f"\nNext Steps:")
    print(f"  - Monitor positions after market open")
    print(f"  - Check trailing stops daily")
    print(f"  - Rebalance {rebalance_frequency}")
    print(f"  - Adjust for VIX changes")

    print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='V30 Live Trading Bot')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--rebalance', type=str, default='quarterly',
                       choices=['weekly', 'monthly', 'quarterly'],
                       help='Rebalance frequency (default: quarterly)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 V30 LIVE TRADING BOT")
    print("="*60)
    print("\nConfiguration:")
    print(f"  Debug Mode: {args.debug}")
    print(f"  Rebalance: {args.rebalance}")
    print(f"  Paper Trading: Yes")

    input("\nPress ENTER to start...")

    run_v30_live(debug=args.debug, rebalance_frequency=args.rebalance)


if __name__ == "__main__":
    main()
