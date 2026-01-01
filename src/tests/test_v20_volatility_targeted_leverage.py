"""
V20: VOLATILITY-TARGETED LEVERAGE (Conservative)
=================================================

Goal: Scale 9.8% to 10%+ by targeting constant risk, not signals

Core Principle:
- Keep portfolio risk constant across time
- Leverage up in calm markets, down in volatile markets
- Hard caps prevent blow-ups
- Drawdown brake for safety

Algorithm:
1. Measure realized portfolio volatility (20-day rolling)
2. Target: 12% annual volatility (conservative for stocks)
3. Leverage = target / realized (capped 0.7 to 1.5)
4. Drawdown brake reduces leverage when underwater
5. Apply at portfolio level (scales entire position)

Why it works:
- Calm markets (low vol) → leverage ~1.3-1.5x
- Volatile markets (high vol) → leverage ~0.7-1.0x
- Crisis → leverage drops automatically
- Never exceeds 1.5x (conservative cap)

Expected:
- 9.8% * ~1.2 avg leverage = 11-12% annual
- Max drawdown: -25% to -30%
- Sharpe: Similar to V13 (~1.0)
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def calculate_volatility_target_leverage(portfolio_df, target_vol=0.12, vol_window=20):
    """
    Calculate volatility-targeted leverage

    Args:
        portfolio_df: DataFrame with 'value' column
        target_vol: Target annual volatility (default 12%)
        vol_window: Rolling window for vol calculation (default 20 days)

    Returns:
        float: Leverage multiplier (0.7 to 1.5)
    """
    if portfolio_df is None or len(portfolio_df) < vol_window + 1:
        return 1.0  # No leverage until sufficient history

    # Calculate realized volatility (rolling 20-day)
    returns = portfolio_df['value'].pct_change()
    realized_vol = returns.tail(vol_window).std() * np.sqrt(252)

    if realized_vol == 0 or np.isnan(realized_vol):
        return 1.0

    # Raw leverage from volatility targeting
    raw_leverage = target_vol / realized_vol

    # Apply HARD caps
    leverage = np.clip(raw_leverage, 0.7, 1.5)

    return leverage


def calculate_drawdown_brake(portfolio_df):
    """
    Reduce leverage during drawdowns (safety layer)

    Drawdown rules:
    - DD < 10%: 1.0x (normal)
    - DD 10-15%: 0.8x (reduce)
    - DD 15-20%: 0.6x (defensive)
    - DD >= 20%: 0.4x (maximum defense)

    Args:
        portfolio_df: DataFrame with 'value' column

    Returns:
        float: Drawdown multiplier (0.4 to 1.0)
    """
    if portfolio_df is None or len(portfolio_df) < 2:
        return 1.0

    current_value = portfolio_df['value'].iloc[-1]
    peak_value = portfolio_df['value'].max()

    drawdown_pct = ((current_value - peak_value) / peak_value) * 100

    if drawdown_pct > -10:
        return 1.0
    elif drawdown_pct > -15:
        return 0.8
    elif drawdown_pct > -20:
        return 0.6
    else:
        return 0.4


def run_volatility_leveraged_backtest(bot, target_vol=0.12, margin_rate=0.065):
    """
    V20: Backtest with volatility-targeted leverage + OVERNIGHT FINANCING FEES

    Leverage is applied at PORTFOLIO LEVEL:
    - Signals stay the same
    - Rankings stay the same
    - Only capital scaling changes
    - Includes realistic margin interest costs (overnight fees)

    Args:
        margin_rate: Annual margin interest rate (default 6.5%)
                     Interactive Brokers: ~5.8-6.8%
                     TD Ameritrade/Schwab: ~7.0-8.5%
    """
    logger.info(f"Target volatility: {target_vol*100:.0f}%")
    logger.info("Leverage caps: 0.7x to 1.5x")
    logger.info("Drawdown brake: Active")
    logger.info(f"Margin rate: {margin_rate*100:.1f}% annual (overnight fees included)")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    leverage_history = []
    total_interest_paid = 0  # Track total margin interest

    for date in dates[100:]:
        # Monthly rebalancing (day 7-10)
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 10
            )
        )

        if is_rebalance_day:
            # Liquidate holdings
            for ticker in list(holdings.keys()):
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    cash += holdings[ticker] * current_price
            holdings = {}

            last_rebalance_date = date

            # Create portfolio DataFrame
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None

            # STEP 1: Calculate volatility-targeted leverage
            vol_leverage = calculate_volatility_target_leverage(portfolio_df, target_vol)

            # STEP 2: Apply drawdown brake
            dd_brake = calculate_drawdown_brake(portfolio_df)

            # STEP 3: Final leverage = vol_leverage * dd_brake
            final_leverage = vol_leverage * dd_brake

            leverage_history.append({
                'date': date,
                'vol_leverage': vol_leverage,
                'dd_brake': dd_brake,
                'final_leverage': final_leverage
            })

            # Score stocks (unchanged from V13)
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            # Get top 5 stocks (unchanged)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [t for t, s in ranked if s > 0][:5]

            # Calculate VIX-based cash reserve (unchanged from V13)
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

            # V20: APPLY LEVERAGE AT PORTFOLIO LEVEL
            # This effectively borrows capital when leverage > 1.0
            effective_capital = cash * final_leverage
            invest_amount = effective_capital * (1 - cash_reserve)

            # Calculate allocations with momentum weighting (unchanged)
            if top_stocks:
                momentum_weights = {}
                for ticker in top_stocks:
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
                    allocation_per_stock = invest_amount / len(top_stocks)
                    allocations = {ticker: allocation_per_stock for ticker in top_stocks}
            else:
                allocations = {}

            # Apply V12 drawdown control (unchanged)
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

            # Buy stocks
            for ticker in top_stocks:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    allocation_amount = allocations.get(ticker, 0)
                    shares = allocation_amount / current_price
                    holdings[ticker] = shares
                    fee = allocation_amount * 0.001
                    cash -= (allocation_amount + fee)

        # Calculate daily portfolio value
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        # CALCULATE OVERNIGHT FINANCING FEES (margin interest)
        # Borrowed capital = amount we're leveraged beyond our cash
        equity = cash + stocks_value  # Current total equity
        borrowed_capital = max(0, stocks_value - cash)  # Only pay interest if stocks > cash

        # Daily interest cost = borrowed × (annual_rate / 252 trading days)
        daily_interest = borrowed_capital * (margin_rate / 252)

        # Deduct interest from cash (cost of leverage)
        cash -= daily_interest
        total_interest_paid += daily_interest

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    return pd.DataFrame(portfolio_values).set_index('date'), pd.DataFrame(leverage_history).set_index('date'), total_interest_paid


def main():
    """Test V20 with different target volatilities"""
    logger.info("="*80)
    logger.info("V20: VOLATILITY-TARGETED LEVERAGE")
    logger.info("="*80)
    logger.info("")
    logger.info("Strategy:")
    logger.info("  Base: V13 with 5 stocks (9.8% annual)")
    logger.info("  Add: Volatility-targeted leverage (conservative)")
    logger.info("")
    logger.info("Leverage formula:")
    logger.info("  1. raw_leverage = target_vol / realized_vol")
    logger.info("  2. leverage = clip(raw_leverage, 0.7, 1.5)")
    logger.info("  3. dd_brake = 1.0 → 0.4 based on drawdown")
    logger.info("  4. final_leverage = leverage × dd_brake")
    logger.info("")
    logger.info("Goal: Push 9.8% to 10%+ with controlled risk")
    logger.info("")

    # Test different target volatilities
    logger.info("="*80)
    logger.info("TESTING TARGET VOLATILITIES: 10%, 12%, 13%")
    logger.info("="*80)

    results = []

    for target_vol in [0.10, 0.12, 0.13]:
        logger.info("")
        logger.info(f"Testing {target_vol*100:.0f}% target volatility...")

        bot = PortfolioRotationBot(
            data_dir='sp500_data/daily',
            initial_capital=100000
        )

        bot.prepare_data()
        bot.score_all_stocks()

        portfolio_df, leverage_df, total_interest_paid = run_volatility_leveraged_backtest(bot, target_vol)

        # Calculate metrics
        final_value = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

        # Interest costs
        annual_interest_cost = total_interest_paid / years
        interest_cost_pct = (total_interest_paid / bot.initial_capital) * 100

        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        realized_vol = daily_returns.std() * np.sqrt(252) * 100

        # Average leverage used
        avg_leverage = leverage_df['final_leverage'].mean()
        max_leverage_used = leverage_df['final_leverage'].max()
        min_leverage_used = leverage_df['final_leverage'].min()

        # Yearly returns
        portfolio_df['year'] = portfolio_df.index.year
        yearly_returns = {}
        for year in portfolio_df['year'].unique():
            year_data = portfolio_df[portfolio_df['year'] == year]
            if len(year_data) > 1:
                year_return = (year_data['value'].iloc[-1] / year_data['value'].iloc[0] - 1) * 100
                yearly_returns[year] = year_return

        positive_years = sum(1 for r in yearly_returns.values() if r > 0)
        total_years = len(yearly_returns)

        results.append({
            'target_vol': target_vol,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'realized_vol': realized_vol,
            'avg_leverage': avg_leverage,
            'max_leverage': max_leverage_used,
            'min_leverage': min_leverage_used,
            'total_interest_paid': total_interest_paid,
            'annual_interest_cost': annual_interest_cost,
            'interest_cost_pct': interest_cost_pct,
            'positive_years': positive_years,
            'total_years': total_years,
            'yearly_returns': yearly_returns
        })

        logger.info(f"  Annual: {annual_return:.1f}%, DD: {max_drawdown:.1f}%, Sharpe: {sharpe:.2f}")
        logger.info(f"  Avg leverage: {avg_leverage:.2f}x (range: {min_leverage_used:.2f}x to {max_leverage_used:.2f}x)")
        logger.info(f"  Interest cost: ${annual_interest_cost:,.0f}/year ({interest_cost_pct:.1f}% of capital)")


    # RESULTS SUMMARY
    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info("")

    logger.info(f"{'Target Vol':<12} {'Annual':>8} {'Avg Lev':>9} {'Drawdown':>10} {'Sharpe':>8} {'Win Rate':>10}")
    logger.info("-" * 72)

    for r in results:
        status = " 🎯" if r['annual_return'] >= 10.0 else ""
        logger.info(
            f"{r['target_vol']*100:>3.0f}%{' '*8} "
            f"{r['annual_return']:>7.1f}% "
            f"{r['avg_leverage']:>8.2f}x "
            f"{r['max_drawdown']:>9.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['positive_years']}/{r['total_years']:>2}"
            f"{status}"
        )

    # BEST RESULT
    best = max(results, key=lambda x: x['annual_return'])

    logger.info("")
    logger.info("="*80)
    logger.info("BEST CONFIGURATION")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Target Volatility: {best['target_vol']*100:.0f}%")
    logger.info(f"Annual Return: {best['annual_return']:.1f}%")
    logger.info(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    logger.info(f"Sharpe Ratio: {best['sharpe']:.2f}")
    logger.info(f"Realized Vol: {best['realized_vol']:.1f}%")
    logger.info(f"Avg Leverage: {best['avg_leverage']:.2f}x")
    logger.info(f"Leverage Range: {best['min_leverage']:.2f}x to {best['max_leverage']:.2f}x")
    logger.info(f"Win Rate: {best['positive_years']}/{best['total_years']} ({best['positive_years']/best['total_years']*100:.0f}%)")
    logger.info("")
    logger.info(f"Financing Costs:")
    logger.info(f"  Total interest paid: ${best['total_interest_paid']:,.0f}")
    logger.info(f"  Annual interest cost: ${best['annual_interest_cost']:,.0f}/year")
    logger.info(f"  Interest % of capital: {best['interest_cost_pct']:.1f}%")
    logger.info(f"  Margin rate: 6.5% annual")

    if best['annual_return'] >= 10.0:
        logger.info("")
        logger.info("="*80)
        logger.info("🎯 GOAL ACHIEVED: 10%+ ANNUAL RETURN!")
        logger.info("="*80)
        logger.info(f"V20 delivers {best['annual_return']:.1f}% annual with controlled leverage")
    else:
        logger.info("")
        logger.info(f"Best: {best['annual_return']:.1f}% (Goal: 10%+, Gap: {10.0 - best['annual_return']:.1f}%)")

    # YEARLY BREAKDOWN
    logger.info("")
    logger.info("="*80)
    logger.info(f"YEARLY RETURNS - BEST ({best['target_vol']*100:.0f}% target)")
    logger.info("="*80)

    for year in sorted(best['yearly_returns'].keys()):
        ret = best['yearly_returns'][year]
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:>6.1f}% {status}")

    # COMPARISON
    logger.info("")
    logger.info("="*80)
    logger.info("COMPARISON: V13 vs V20")
    logger.info("="*80)
    logger.info("")
    logger.info("V13 (5 stocks, no leverage):")
    logger.info("  Annual: 9.8%, DD: -19.1%, Sharpe: 1.07, Leverage: 1.0x")
    logger.info("")
    logger.info(f"V20 ({best['target_vol']*100:.0f}% vol target, capped leverage):")
    logger.info(f"  Annual: {best['annual_return']:.1f}%, DD: {best['max_drawdown']:.1f}%, Sharpe: {best['sharpe']:.2f}, Avg Lev: {best['avg_leverage']:.2f}x")
    logger.info("")

    improvement = best['annual_return'] - 9.8
    logger.info(f"Improvement: +{improvement:.1f}% annual return")
    logger.info("")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    results = main()
