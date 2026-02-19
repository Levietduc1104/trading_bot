"""
Export Tier 2 Strategy Recommendations to CSV
==============================================

Creates a CSV file with current buy/sell/hold recommendations for Tier 2 strategy.

Output CSV columns:
- date: Recommendation date
- ticker: Stock symbol
- type: Mega-cap or Momentum
- action: BUY, SELL, or HOLD
- target_allocation_dollars: Dollar amount to allocate
- recommended_shares: Number of shares to buy
- current_price: Stock price
- market_cap: Company market capitalization
- momentum_score: 20-day momentum score
- growth_score: Fundamental growth score (if available)
- combined_score: Final combined score (70% momentum + 30% growth)

Usage:
    python3 export_tier2_recommendations.py
    python3 export_tier2_recommendations.py --holdings AAPL,MSFT,GOOGL --output my_trades.csv
    python3 export_tier2_recommendations.py --date 2024-03-31 --capital 50000
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from src.data.historical_fa_data_loader import HistoricalFADataLoader


def calculate_momentum_score(df, use_date):
    """Calculate 20-day momentum score"""
    if len(df) < 20:
        return 0

    # Get data up to use_date
    df = df[df.index <= use_date]
    if len(df) < 20:
        return 0

    # 20-day return
    current_price = df['close'].iloc[-1]
    price_20d_ago = df['close'].iloc[-20]

    if price_20d_ago <= 0:
        return 0

    return_20d = (current_price / price_20d_ago - 1) * 100
    return return_20d


def calculate_growth_score(ticker, use_date, fa_loader):
    """Calculate fundamental growth score"""
    fa_data = fa_loader.get_metrics_as_of(ticker, use_date)

    if not fa_data:
        return 50  # Neutral score

    score = 0

    # 1. Profitability (25 points)
    roe = fa_data.get('returnOnEquity', 0)
    if roe > 0.25:
        score += 25
    elif roe > 0.20:
        score += 20
    elif roe > 0.15:
        score += 15
    elif roe > 0.10:
        score += 10
    elif roe > 0.05:
        score += 5

    # 2. Margins (20 points)
    operating_margin = fa_data.get('operatingProfitMargin', 0)
    net_margin = fa_data.get('netProfitMargin', 0)

    if operating_margin > 0.20 and net_margin > 0.15:
        score += 20
    elif operating_margin > 0.15 and net_margin > 0.10:
        score += 15
    elif operating_margin > 0.10 and net_margin > 0.05:
        score += 10
    elif operating_margin > 0.05 and net_margin > 0:
        score += 5

    # 3. Cash Generation (20 points)
    fcf_yield = fa_data.get('freeCashFlowYield', 0)
    income_quality = fa_data.get('incomeQuality', 0)

    if fcf_yield > 0.10:
        score += 12
    elif fcf_yield > 0.07:
        score += 9
    elif fcf_yield > 0.05:
        score += 6
    elif fcf_yield > 0.03:
        score += 3

    if income_quality > 1.0:
        score += 8
    elif income_quality > 0.8:
        score += 5
    elif income_quality > 0.5:
        score += 3

    # 4. Capital Efficiency (20 points)
    roic = fa_data.get('returnOnInvestedCapital', 0)
    roce = fa_data.get('returnOnCapitalEmployed', 0)

    if roic > 0.20 or roce > 0.20:
        score += 20
    elif roic > 0.15 or roce > 0.15:
        score += 15
    elif roic > 0.10 or roce > 0.10:
        score += 10
    elif roic > 0.05 or roce > 0.05:
        score += 5

    # 5. Financial Health (15 points)
    current_ratio = fa_data.get('currentRatio', 0)
    debt_equity = fa_data.get('debtToEquity', 999)

    if current_ratio > 1.5 and debt_equity < 1.0:
        score += 15
    elif current_ratio > 1.2 and debt_equity < 2.0:
        score += 10
    elif current_ratio > 1.0 and debt_equity < 3.0:
        score += 5

    return score


def get_tier2_recommendations(bot, fa_loader, use_date, capital=100000, momentum_weight=0.70, growth_weight=0.30):
    """
    Get Tier 2 recommendations by scoring all stocks

    Returns:
        DataFrame with columns: ticker, type, momentum_score, growth_score,
                                combined_score, market_cap, price,
                                target_allocation, recommended_shares
    """
    print(f"\nScoring all stocks as of {use_date}...")

    results = []

    for ticker in bot.stocks_data.keys():
        if ticker == 'SPY':
            continue

        df = bot.stocks_data[ticker]
        if len(df) < 20:
            continue

        # Get data up to use_date
        df_slice = df[df.index <= use_date]
        if len(df_slice) < 20:
            continue

        # Calculate scores
        momentum_score = calculate_momentum_score(df, use_date)
        growth_score = calculate_growth_score(ticker, use_date, fa_loader)

        # Combined score: 70% momentum + 30% growth
        combined_score = (momentum_weight * momentum_score) + (growth_weight * growth_score)

        # Get market cap and price
        market_cap = bot.get_market_cap(ticker, use_date) if hasattr(bot, 'get_market_cap') else 0
        price = df_slice['close'].iloc[-1]

        results.append({
            'ticker': ticker,
            'momentum_score': momentum_score,
            'growth_score': growth_score,
            'combined_score': combined_score,
            'market_cap': market_cap,
            'price': price
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by combined score
    results_df = results_df.sort_values('combined_score', ascending=False)

    print(f"Scored {len(results_df)} stocks")

    # Select top stocks
    # First, get mega-caps (largest by market cap among top scorers)
    top_candidates = results_df.head(50)  # Top 50 by combined score

    # Sort top candidates by market cap to get mega-caps
    megacaps_df = top_candidates.nlargest(3, 'market_cap').copy()
    megacaps_df['type'] = 'Mega-cap'

    # Get remaining top 7 by combined score (excluding mega-caps)
    remaining = results_df[~results_df['ticker'].isin(megacaps_df['ticker'])]
    momentum_df = remaining.head(7).copy()
    momentum_df['type'] = 'Momentum'

    # Combine
    selected_df = pd.concat([megacaps_df, momentum_df], ignore_index=True)

    # Calculate allocations
    megacap_allocation = capital * 0.70
    momentum_allocation = capital * 0.30

    per_megacap = megacap_allocation / 3
    per_momentum = momentum_allocation / 7

    selected_df['target_allocation_dollars'] = selected_df['type'].apply(
        lambda x: per_megacap if x == 'Mega-cap' else per_momentum
    )

    selected_df['recommended_shares'] = (
        selected_df['target_allocation_dollars'] / selected_df['price']
    ).astype(int)

    return selected_df


def export_recommendations(current_holdings=None, target_date=None, capital=100000,
                           output_file='output/tier2_recommendations.csv'):
    """
    Export Tier 2 recommendations to CSV

    Args:
        current_holdings: List of currently held tickers (optional)
        target_date: Target date (default: latest available)
        capital: Portfolio capital
        output_file: Output CSV file path
    """
    print("\n" + "="*80)
    print("TIER 2 STRATEGY - EXPORT RECOMMENDATIONS TO CSV")
    print("="*80)
    print(f"\nStrategy: V31 Tier 2 (70% Momentum + 30% Growth Fundamentals)")
    print(f"Portfolio: 70% Top 3 Mega-caps + 30% Top 7 Momentum")
    print(f"Capital: ${capital:,.0f}\n")

    # Initialize bot and load data
    data_dir = os.path.join(project_root, 'sp500_data', 'stock_data_1990_2024')
    bot = PortfolioRotationBot(data_dir=data_dir, initial_capital=capital)
    bot.prepare_data()

    # Load FA data
    fa_loader = HistoricalFADataLoader()
    fa_loader.load_all()

    # Determine date
    if target_date:
        use_date = pd.to_datetime(target_date)
    else:
        spy_df = bot.stocks_data.get('SPY')
        if spy_df is not None:
            use_date = spy_df.index[-1]
        else:
            use_date = pd.Timestamp.now()

    print(f"Target Date: {use_date.strftime('%Y-%m-%d')}\n")

    # Get recommendations
    recommendations_df = get_tier2_recommendations(bot, fa_loader, use_date, capital)

    # Add action column
    if current_holdings:
        current_set = set(current_holdings)
        recommended_set = set(recommendations_df['ticker'].tolist())

        def get_action(ticker):
            if ticker in current_set:
                return 'HOLD'
            else:
                return 'BUY'

        recommendations_df['action'] = recommendations_df['ticker'].apply(get_action)

        # Add SELL recommendations for holdings not in recommended list
        sells = []
        for ticker in current_holdings:
            if ticker not in recommended_set:
                sells.append({
                    'ticker': ticker,
                    'type': 'N/A',
                    'action': 'SELL',
                    'momentum_score': 0,
                    'growth_score': 0,
                    'combined_score': 0,
                    'market_cap': 0,
                    'price': 0,
                    'target_allocation_dollars': 0,
                    'recommended_shares': 0
                })

        if sells:
            sells_df = pd.DataFrame(sells)
            recommendations_df = pd.concat([recommendations_df, sells_df], ignore_index=True)
    else:
        recommendations_df['action'] = 'BUY'

    # Add date column
    recommendations_df['date'] = use_date.strftime('%Y-%m-%d')

    # Reorder columns
    columns = ['date', 'ticker', 'type', 'action', 'target_allocation_dollars',
               'recommended_shares', 'price', 'market_cap', 'momentum_score',
               'growth_score', 'combined_score']

    recommendations_df = recommendations_df[columns]

    # Sort: BUY first (mega-caps, then momentum), then HOLD, then SELL
    sort_order = {'BUY': 0, 'HOLD': 1, 'SELL': 2}
    type_order = {'Mega-cap': 0, 'Momentum': 1, 'N/A': 2}

    recommendations_df['action_rank'] = recommendations_df['action'].map(sort_order)
    recommendations_df['type_rank'] = recommendations_df['type'].map(type_order)
    recommendations_df = recommendations_df.sort_values(['action_rank', 'type_rank', 'combined_score'],
                                                        ascending=[True, True, False])
    recommendations_df = recommendations_df.drop(['action_rank', 'type_rank'], axis=1)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    recommendations_df.to_csv(output_file, index=False)

    print(f"\n✅ Saved {len(recommendations_df)} recommendations to: {output_file}")

    # Display summary
    print("\n" + "─"*80)
    print("SUMMARY")
    print("─"*80)

    buy_count = len(recommendations_df[recommendations_df['action'] == 'BUY'])
    hold_count = len(recommendations_df[recommendations_df['action'] == 'HOLD'])
    sell_count = len(recommendations_df[recommendations_df['action'] == 'SELL'])

    print(f"\n🟢 BUY:  {buy_count} positions")
    print(f"✓ HOLD: {hold_count} positions")
    print(f"🔴 SELL: {sell_count} positions")

    print(f"\nTotal Recommended: {buy_count + hold_count} stocks")
    print(f"Mega-caps (70%): {len(recommendations_df[recommendations_df['type'] == 'Mega-cap'])}")
    print(f"Momentum (30%): {len(recommendations_df[recommendations_df['type'] == 'Momentum'])}")

    print("\n" + "="*80)

    return recommendations_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export Tier 2 Recommendations to CSV')
    parser.add_argument('--holdings', type=str, default=None,
                       help='Current holdings (comma-separated), e.g., AAPL,MSFT,GOOGL')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date (YYYY-MM-DD), default: latest')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Portfolio capital (default: 100000)')
    parser.add_argument('--output', type=str, default='output/tier2_recommendations.csv',
                       help='Output CSV file path (default: output/tier2_recommendations.csv)')

    args = parser.parse_args()

    # Parse current holdings
    current_holdings = None
    if args.holdings:
        current_holdings = [x.strip().upper() for x in args.holdings.split(',')]
        print(f"\n📋 Your Current Holdings: {', '.join(current_holdings)}")

    # Export recommendations
    export_recommendations(
        current_holdings=current_holdings,
        target_date=args.date,
        capital=args.capital,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
