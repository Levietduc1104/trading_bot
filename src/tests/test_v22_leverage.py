"""
V22 + LEVERAGE: TWO APPROACHES
================================

Goal: Add leverage to V22-Sqrt (10.2% baseline) to reach 12-14% annual

Approach #1: FUTURES LEVERAGE (70% stocks + 30% MES futures)
- Deploy 70% in V22 stock picking
- Deploy 30% in MES futures (S&P 500 exposure)
- Total leverage: ~1.4x
- Cost: $200/year (futures fees only, NO interest!)

Approach #2: ADAPTIVE MARGIN LEVERAGE (score-based)
- Use margin leverage ONLY when avg score > 90 (high conviction)
- Max leverage: 1.3x
- Cost: 6.5% annual interest on borrowed amount
- Drawdown brake still applies

Expected:
- Futures: 12-14% annual (minimal costs)
- Adaptive: 11-12% annual (selective leverage)
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


def calculate_kelly_weights_sqrt(scores):
    """
    Calculate position weights using Square Root method (best from V22)

    Args:
        scores: List of (ticker, score) tuples

    Returns:
        dict: {ticker: weight}
    """
    tickers = [t for t, s in scores]
    score_values = [s for t, s in scores]

    sqrt_scores = [np.sqrt(max(0, s)) for s in score_values]
    total_sqrt = sum(sqrt_scores)

    if total_sqrt > 0:
        weights = {ticker: np.sqrt(max(0, score)) / total_sqrt for ticker, score in scores}
    else:
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    return weights


def run_v22_futures_leverage(bot, stock_allocation=0.70, futures_return_multiplier=1.0):
    """
    V22 + FUTURES LEVERAGE

    Strategy:
    - 70% capital in stocks (V22 Kelly-weighted)
    - 30% capital in MES futures (S&P 500 exposure)
    - Futures cost: $200/year fixed (no interest!)

    Args:
        stock_allocation: Percentage of capital in stocks (0.7 = 70%)
        futures_return_multiplier: How to calculate futures returns (1.0 = track SPY)
    """
    logger.info(f"Stock allocation: {stock_allocation*100:.0f}%, Futures: {(1-stock_allocation)*100:.0f}%")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    # Load SPY for futures tracking
    spy_data = bot.stocks_data.get('SPY')
    if spy_data is None:
        logger.error("SPY data not found - cannot simulate futures!")
        return None, 0

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    total_futures_fees = 0

    # Split capital
    stock_capital = bot.initial_capital * stock_allocation
    futures_capital = bot.initial_capital * (1 - stock_allocation)
    futures_entry_spy = None

    for date in dates[100:]:
        is_rebalance_day = (
            last_rebalance_date is None or
            (
                (date.year, date.month) != (last_rebalance_date.year, last_rebalance_date.month) and
                7 <= date.day <= 10
            )
        )

        if is_rebalance_day:
            # Liquidate stocks
            for ticker in list(holdings.keys()):
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    stock_capital += holdings[ticker] * current_price
            holdings = {}
            last_rebalance_date = date

            # Score and select stocks
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

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

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': stock_capital + futures_capital})
                continue

            # VIX-based cash reserve
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            invest_amount = stock_capital * (1 - cash_reserve)

            # Kelly weights (Square Root method)
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)
            allocations = {ticker: invest_amount * weight for ticker, weight in kelly_weights.items()}

            # Drawdown control
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {ticker: amount * drawdown_multiplier for ticker, amount in allocations.items()}

            # Buy stocks
            for ticker, _ in top_stocks:
                df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
                if len(df_at_date) > 0:
                    current_price = df_at_date.iloc[-1]['close']
                    allocation_amount = allocations.get(ticker, 0)
                    shares = allocation_amount / current_price
                    holdings[ticker] = shares
                    fee = allocation_amount * 0.001
                    stock_capital -= (allocation_amount + fee)

            # Set futures entry point
            spy_at_date = spy_data[spy_data.index <= date]
            if len(spy_at_date) > 0:
                futures_entry_spy = spy_at_date.iloc[-1]['close']

            # Monthly futures fee
            total_futures_fees += 200 / 12  # $200/year = $16.67/month

        # Daily valuation
        # Stock portfolio
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        # Futures portfolio (tracks SPY)
        spy_at_date = spy_data[spy_data.index <= date]
        if len(spy_at_date) > 0 and futures_entry_spy is not None:
            current_spy = spy_at_date.iloc[-1]['close']
            spy_return = (current_spy - futures_entry_spy) / futures_entry_spy
            futures_value = futures_capital * (1 + spy_return)
        else:
            futures_value = futures_capital

        total_stock_value = stock_capital + stocks_value
        total_value = total_stock_value + futures_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    return portfolio_df, total_futures_fees


def run_v22_adaptive_margin_leverage(bot, score_threshold=90, max_leverage=1.3, margin_rate=0.065):
    """
    V22 + ADAPTIVE MARGIN LEVERAGE

    Strategy:
    - Use Kelly position sizing (Square Root)
    - Apply leverage ONLY when avg score > threshold
    - Max leverage: 1.3x
    - Drawdown brake still applies
    - Pay 6.5% interest on borrowed amount
    """
    logger.info(f"Score threshold: {score_threshold}, Max leverage: {max_leverage:.1f}x, Margin rate: {margin_rate*100:.1f}%")

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None
    total_interest_paid = 0
    leverage_history = []

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

            # Score stocks
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

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

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # Calculate average score of top 5
            avg_score = np.mean([score for _, score in top_stocks])

            # Adaptive leverage based on score quality
            if avg_score >= score_threshold:
                target_leverage = max_leverage
            else:
                target_leverage = 1.0  # No leverage if scores are weak

            # Drawdown brake
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                current_value = portfolio_df['value'].iloc[-1]
                peak_value = portfolio_df['value'].max()
                drawdown_pct = ((current_value - peak_value) / peak_value) * 100

                if drawdown_pct > -10:
                    dd_brake = 1.0
                elif drawdown_pct > -15:
                    dd_brake = 0.8
                elif drawdown_pct > -20:
                    dd_brake = 0.6
                else:
                    dd_brake = 0.4
            else:
                dd_brake = 1.0

            final_leverage = target_leverage * dd_brake
            final_leverage = np.clip(final_leverage, 1.0, max_leverage)

            leverage_history.append({
                'date': date,
                'avg_score': avg_score,
                'target_leverage': target_leverage,
                'final_leverage': final_leverage
            })

            # VIX cash reserve
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            # Apply leverage
            effective_capital = cash * final_leverage
            invest_amount = effective_capital * (1 - cash_reserve)

            # Kelly weights
            kelly_weights = calculate_kelly_weights_sqrt(top_stocks)
            allocations = {ticker: invest_amount * weight for ticker, weight in kelly_weights.items()}

            # Buy stocks
            for ticker, _ in top_stocks:
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

        # Calculate overnight fees
        borrowed_capital = max(0, stocks_value - cash)
        daily_interest = borrowed_capital * (margin_rate / 252)
        cash -= daily_interest
        total_interest_paid += daily_interest

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    leverage_df = pd.DataFrame(leverage_history)

    return portfolio_df, leverage_df, total_interest_paid


def main():
    """Test V22 + Leverage (both approaches)"""
    logger.info("="*80)
    logger.info("V22 + LEVERAGE TESTING")
    logger.info("="*80)
    logger.info("")
    logger.info("Baseline: V22-Sqrt Kelly sizing = 10.2% annual")
    logger.info("")
    logger.info("Testing 2 leverage approaches:")
    logger.info("  #1: Futures (70% stocks + 30% MES) - NO interest fees")
    logger.info("  #2: Adaptive Margin (score-based, 1.3x max) - 6.5% interest")
    logger.info("")
    logger.info("Goal: Reach 12-14% annual return")
    logger.info("")

    results = []

    # Approach #1: Futures Leverage
    logger.info("="*80)
    logger.info("APPROACH #1: FUTURES LEVERAGE")
    logger.info("="*80)
    logger.info("")

    bot = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
    bot.prepare_data()
    bot.score_all_stocks()

    portfolio_df, futures_fees = run_v22_futures_leverage(bot, stock_allocation=0.70)

    if portfolio_df is not None:
        final_value = portfolio_df['value'].iloc[-1]
        years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
        annual_return = ((final_value / bot.initial_capital) ** (1/years) - 1) * 100

        cummax = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        daily_returns = portfolio_df['value'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        annual_fees = futures_fees / years

        results.append({
            'approach': 'Futures Leverage',
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'annual_cost': annual_fees,
            'final_value': final_value
        })

        logger.info(f"  Annual Return: {annual_return:.1f}%")
        logger.info(f"  Max Drawdown: {max_drawdown:.1f}%")
        logger.info(f"  Sharpe: {sharpe:.2f}")
        logger.info(f"  Annual Cost: ${annual_fees:.0f}/year")
        logger.info("")

    # Approach #2: Adaptive Margin Leverage
    logger.info("="*80)
    logger.info("APPROACH #2: ADAPTIVE MARGIN LEVERAGE")
    logger.info("="*80)
    logger.info("")

    bot2 = PortfolioRotationBot(data_dir='sp500_data/daily', initial_capital=100000)
    bot2.prepare_data()
    bot2.score_all_stocks()

    portfolio_df2, leverage_df, total_interest = run_v22_adaptive_margin_leverage(
        bot2, score_threshold=90, max_leverage=1.3, margin_rate=0.065
    )

    final_value2 = portfolio_df2['value'].iloc[-1]
    years2 = (portfolio_df2.index[-1] - portfolio_df2.index[0]).days / 365.25
    annual_return2 = ((final_value2 / bot2.initial_capital) ** (1/years2) - 1) * 100

    cummax2 = portfolio_df2['value'].cummax()
    drawdown2 = (portfolio_df2['value'] - cummax2) / cummax2 * 100
    max_drawdown2 = drawdown2.min()

    daily_returns2 = portfolio_df2['value'].pct_change().dropna()
    sharpe2 = (daily_returns2.mean() / daily_returns2.std()) * np.sqrt(252)

    annual_interest = total_interest / years2

    # Leverage stats
    avg_leverage = leverage_df['final_leverage'].mean() if len(leverage_df) > 0 else 1.0
    pct_leveraged = (leverage_df['final_leverage'] > 1.0).sum() / len(leverage_df) * 100 if len(leverage_df) > 0 else 0

    results.append({
        'approach': 'Adaptive Margin',
        'annual_return': annual_return2,
        'max_drawdown': max_drawdown2,
        'sharpe': sharpe2,
        'annual_cost': annual_interest,
        'final_value': final_value2,
        'avg_leverage': avg_leverage,
        'pct_leveraged': pct_leveraged
    })

    logger.info(f"  Annual Return: {annual_return2:.1f}%")
    logger.info(f"  Max Drawdown: {max_drawdown2:.1f}%")
    logger.info(f"  Sharpe: {sharpe2:.2f}")
    logger.info(f"  Annual Interest: ${annual_interest:,.0f}/year")
    logger.info(f"  Avg Leverage: {avg_leverage:.2f}x")
    logger.info(f"  Leveraged: {pct_leveraged:.1f}% of time")
    logger.info("")

    # COMPARISON
    logger.info("")
    logger.info("="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)
    logger.info("")
    logger.info(f"{'Strategy':<25} {'Annual':>8} {'Drawdown':>10} {'Sharpe':>8} {'Cost/Year':>11} {'Status'}")
    logger.info("-" * 80)

    logger.info(f"{'V13 (Baseline)':<25} {'9.8%':>8} {'-19.1%':>10} {'1.07':>8} {'$0':>11}")
    logger.info(f"{'V22-Sqrt (Kelly)':<25} {'10.2%':>8} {'-15.2%':>10} {'1.11':>8} {'$0':>11} {'✅'}")

    for r in results:
        beat = "🎯" if r['annual_return'] >= 12.0 else "✅" if r['annual_return'] > 10.2 else ""
        logger.info(
            f"{r['approach']:<25} "
            f"{r['annual_return']:>7.1f}% "
            f"{r['max_drawdown']:>9.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"${r['annual_cost']:>10,.0f} {beat}"
        )

    best = max(results, key=lambda x: x['annual_return'])

    logger.info("")
    logger.info("="*80)
    logger.info("WINNER")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Best approach: {best['approach']}")
    logger.info(f"Annual return: {best['annual_return']:.1f}%")
    logger.info(f"Improvement over V13: +{best['annual_return'] - 9.8:.1f}%")
    logger.info(f"Improvement over V22: +{best['annual_return'] - 10.2:.1f}%")
    logger.info("")

    if best['annual_return'] >= 12.0:
        logger.info("="*80)
        logger.info("SUCCESS: 12%+ ANNUAL RETURN ACHIEVED!")
        logger.info("="*80)
        logger.info(f"Strategy: V22-Sqrt + {best['approach']}")
        logger.info(f"Return: {best['annual_return']:.1f}% annual")
    else:
        logger.info(f"Close! Achieved {best['annual_return']:.1f}% (target was 12%+)")

    logger.info("")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    results = main()
