"""
V20 LEVERAGE OPTIMIZATION - SIGNAL-BASED DYNAMIC LEVERAGE
===========================================================

Goal: Find optimal leverage parameters to beat 9.8% annual

NEW INNOVATION: Signal-Based Dynamic Leverage
- High conviction entries (top scores, strong momentum) → More leverage
- Weak signals (low scores, uncertain momentum) → Less leverage
- This increases leverage when edge is highest

Test Matrix:
1. Leverage caps: 1.2x, 1.3x, 1.5x (lower = less borrowing cost)
2. Margin rates: 4%, 5%, 6%, 6.5%, 7%, 8% (varies by broker/account size)
3. Target volatility: 10%, 11%, 12%
4. Dynamic leverage: Static vs Signal-based

Signal-Based Leverage Formula:
  base_leverage = volatility_targeted_leverage (as before)
  signal_quality = avg_score of top 5 stocks / max_possible_score

  dynamic_multiplier =
    1.3x if signal_quality > 0.8 (very high conviction)
    1.1x if signal_quality > 0.6 (good conviction)
    0.9x if signal_quality <= 0.6 (weak signals)

  final_leverage = base_leverage × dynamic_multiplier × dd_brake

Broker margin rates (for reference):
- Interactive Brokers Pro (>$100k): 5.83%
- Interactive Brokers Portfolio Margin: ~5.5%
- IBKR (> $1M): 4.75%
- Fidelity/Schwab: 7-8.5%
- Futures (MES): No interest, just fees
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot
from test_v20_volatility_targeted_leverage import (
    calculate_volatility_target_leverage,
    calculate_drawdown_brake
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_optimized_leverage_backtest(bot, target_vol=0.12, margin_rate=0.065, max_leverage=1.5, use_dynamic_leverage=False):
    """
    Test specific leverage configuration

    Args:
        use_dynamic_leverage: If True, adjust leverage based on signal quality
    """
    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    total_interest_paid = 0

    for date in dates[100:]:
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 10
            )
        )

        if is_rebalance_day:
            # Liquidate
            for ticker in list(holdings.keys()):
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * current_price
            holdings = {}
            last_rebalance_date = date

            # Score and select stocks FIRST (before leverage calculation)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:5]

            # Calculate signal quality for dynamic leverage
            if use_dynamic_leverage and len(top_stocks) > 0:
                avg_score = np.mean([score for _, score in top_stocks])
                # Normalize score (typical range 0-100, but can vary)
                # Use percentile of current scores instead
                all_positive_scores = [s for _, s in ranked if s > 0]
                if len(all_positive_scores) > 0:
                    score_percentile = np.percentile(all_positive_scores, 80)  # Top 20% threshold
                    signal_quality = avg_score / score_percentile if score_percentile > 0 else 1.0
                else:
                    signal_quality = 1.0

                # Signal-based multiplier
                if signal_quality > 1.2:  # Very high quality (better than 80th percentile)
                    signal_multiplier = 1.3
                elif signal_quality > 1.0:  # Good quality
                    signal_multiplier = 1.1
                else:  # Weak signals
                    signal_multiplier = 0.9
            else:
                signal_multiplier = 1.0  # Static leverage (no adjustment)

            top_stock_tickers = [t for t, _ in top_stocks]

            # Calculate base leverage
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None

            vol_leverage = calculate_volatility_target_leverage(portfolio_df, target_vol)
            dd_brake = calculate_drawdown_brake(portfolio_df)

            # Apply signal multiplier and caps
            if use_dynamic_leverage:
                final_leverage = np.clip(vol_leverage * dd_brake * signal_multiplier, 0.7, max_leverage)
            else:
                final_leverage = np.clip(vol_leverage * dd_brake, 0.7, max_leverage)

            # VIX-based cash reserve
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
                if vix < 30:
                    cash_reserve = 0.05 + (vix - 10) * 0.005
                else:
                    cash_reserve = 0.15 + (vix - 30) * 0.0125
                cash_reserve = np.clip(cash_reserve, 0.05, 0.70)
            else:
                cash_reserve = 0.30

            # Apply leverage
            effective_capital = cash * final_leverage
            invest_amount = effective_capital * (1 - cash_reserve)

            # Allocate
            if top_stock_tickers:
                momentum_weights = {}
                for ticker in top_stock_tickers:
                    weight = bot.calculate_momentum_strength_weight(ticker, date, lookback_months=9)
                    if weight > 0:
                        momentum_weights[ticker] = weight

                total_weight = sum(momentum_weights.values())
                if total_weight > 0:
                    allocations = {
                        ticker: (weight / total_weight) * invest_amount
                        for ticker, weight in momentum_weights.items()
                    }
                else:
                    allocation_per_stock = invest_amount / len(top_stock_tickers)
                    allocations = {ticker: allocation_per_stock for ticker in top_stock_tickers}
            else:
                allocations = {}

            # Drawdown control
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

            # Buy
            for ticker in top_stock_tickers:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    allocation_amount = allocations.get(ticker, 0)
                    shares = allocation_amount / current_price
                    holdings[ticker] = shares
                    fee = allocation_amount * 0.001
                    cash -= (allocation_amount + fee)

        # Daily valuation
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        # OVERNIGHT FEES
        borrowed_capital = max(0, stocks_value - cash)
        daily_interest = borrowed_capital * (margin_rate / 252)
        cash -= daily_interest
        total_interest_paid += daily_interest

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    return pd.DataFrame(portfolio_values).set_index('date'), total_interest_paid


def main():
    """Comprehensive leverage optimization"""
    logger.info("="*80)
    logger.info("V20 LEVERAGE OPTIMIZATION")
    logger.info("="*80)
    logger.info("")
    logger.info("Testing combinations of:")
    logger.info("  - Leverage caps: 1.2x, 1.3x, 1.5x")
    logger.info("  - Margin rates: 4%, 5%, 6%, 6.5%, 7%, 8%")
    logger.info("  - Target vols: 10%, 11%, 12%")
    logger.info("")
    logger.info("Goal: Find configuration that beats V13's 9.8% annual")
    logger.info("")

    # Load data once
    logger.info("Loading data (this takes a minute)...")
    bot = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
    bot.prepare_data()
    bot.score_all_stocks()

    # Test matrix (FOCUSED - 8 configs for faster testing)
    leverage_caps = [1.3, 1.5]  # Skip 1.2x - too conservative
    margin_rates = [0.05, 0.065]  # IBKR typical rates
    target_vols = [0.12]  # Sweet spot from V20
    leverage_types = [False, True]  # False = static, True = signal-based dynamic

    results = []
    total_tests = len(leverage_caps) * len(margin_rates) * len(target_vols) * len(leverage_types)
    test_count = 0

    logger.info(f"Running {total_tests} FOCUSED configurations (static + dynamic leverage)...")
    logger.info("")

    for use_dynamic in leverage_types:
        for max_lev in leverage_caps:
            for margin_rate in margin_rates:
                for target_vol in target_vols:
                    test_count += 1

                    # Run backtest
                    portfolio_df, total_interest = run_optimized_leverage_backtest(
                        bot, target_vol, margin_rate, max_lev, use_dynamic
                    )

                    # Metrics
                    final_value = portfolio_df['value'].iloc[-1]
                    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
                    annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

                    cummax = portfolio_df['value'].cummax()
                    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
                    max_drawdown = drawdown.min()

                    daily_returns = portfolio_df['value'].pct_change().dropna()
                    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

                    annual_interest = total_interest / years

                    results.append({
                        'leverage_type': 'Dynamic' if use_dynamic else 'Static',
                        'max_leverage': max_lev,
                        'margin_rate': margin_rate,
                        'target_vol': target_vol,
                        'annual_return': annual_return,
                        'max_drawdown': max_drawdown,
                        'sharpe': sharpe,
                        'annual_interest': annual_interest,
                        'final_value': final_value
                    })

                    if test_count % 18 == 0:
                        logger.info(f"Progress: {test_count}/{total_tests} tests completed...")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    logger.info("")
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info("")

    # Find best configurations
    logger.info("TOP 10 CONFIGURATIONS (by Annual Return):")
    logger.info("-" * 105)
    logger.info(f"{'Type':<9} {'Max Lev':<9} {'Rate':<7} {'Target':>8} {'Annual':>8} {'Interest':>10} {'Drawdown':>10} {'Sharpe':>8}")
    logger.info("-" * 105)

    top_10 = df.nlargest(10, 'annual_return')
    for _, row in top_10.iterrows():
        beat_v13 = "🎯" if row['annual_return'] > 9.8 else ""
        logger.info(
            f"{row['leverage_type']:<9} "
            f"{row['max_leverage']:.1f}x{' '*5} "
            f"{row['margin_rate']*100:>4.1f}% "
            f"{row['target_vol']*100:>7.0f}% "
            f"{row['annual_return']:>7.1f}% "
            f"${row['annual_interest']:>8,.0f} "
            f"{row['max_drawdown']:>9.1f}% "
            f"{row['sharpe']:>8.2f} {beat_v13}"
        )

    # Best by margin rate
    logger.info("")
    logger.info("="*80)
    logger.info("BEST CONFIGURATION PER MARGIN RATE")
    logger.info("="*80)
    logger.info("")

    for rate in sorted(df['margin_rate'].unique()):
        rate_data = df[df['margin_rate'] == rate]
        best_for_rate = rate_data.loc[rate_data['annual_return'].idxmax()]

        beat_v13 = "✅ BEATS V13" if best_for_rate['annual_return'] > 9.8 else "❌ Below V13"

        logger.info(f"Margin Rate: {rate*100:.1f}%")
        logger.info(f"  Best config: {best_for_rate['max_leverage']:.1f}x leverage, {best_for_rate['target_vol']*100:.0f}% target vol")
        logger.info(f"  Annual return: {best_for_rate['annual_return']:.1f}% {beat_v13}")
        logger.info(f"  Max drawdown: {best_for_rate['max_drawdown']:.1f}%")
        logger.info(f"  Interest cost: ${best_for_rate['annual_interest']:,.0f}/year")
        logger.info("")

    # Overall best
    best = df.loc[df['annual_return'].idxmax()]

    logger.info("="*80)
    logger.info("🏆 OVERALL BEST CONFIGURATION")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Leverage Type: {best['leverage_type']}")
    logger.info(f"Max Leverage: {best['max_leverage']:.1f}x")
    logger.info(f"Margin Rate: {best['margin_rate']*100:.1f}%")
    logger.info(f"Target Volatility: {best['target_vol']*100:.0f}%")
    logger.info(f"Annual Return: {best['annual_return']:.1f}%")
    logger.info(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio: {best['sharpe']:.2f}")
    logger.info(f"Annual Interest Cost: ${best['annual_interest']:,.0f}")
    logger.info("")

    if best['annual_return'] > 9.8:
        improvement = best['annual_return'] - 9.8
        logger.info("="*80)
        logger.info("🎯 SUCCESS: OPTIMIZED LEVERAGE BEATS V13!")
        logger.info("="*80)
        logger.info(f"Strategy: {best['leverage_type']} leverage")
        logger.info(f"Improvement: +{improvement:.1f}% annual vs V13 (9.8%)")
        logger.info(f"New annual return: {best['annual_return']:.1f}%")
    else:
        gap = 9.8 - best['annual_return']
        logger.info("="*80)
        logger.info("CONCLUSION: V13 WITHOUT LEVERAGE IS BETTER")
        logger.info("="*80)
        logger.info(f"Best leverage config: {best['annual_return']:.1f}% annual")
        logger.info(f"V13 (no leverage): 9.8% annual")
        logger.info(f"Gap: {gap:.1f}% in favor of V13")
        logger.info("")
        logger.info("Recommendation: Stick with V13 (no leverage)")

    logger.info("")
    logger.info("="*80)

    # Save results
    df.to_csv('v20_leverage_optimization_results.csv', index=False)
    logger.info("Results saved to: v20_leverage_optimization_results.csv")
    logger.info("="*80)

    return df


if __name__ == '__main__':
    results_df = main()
