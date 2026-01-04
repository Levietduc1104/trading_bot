"""
V27: REGIME-BASED PORTFOLIO SIZE
=================================

PROBLEM: V22 uses FIXED 5-stock portfolio in all market conditions
- Bull markets: Could concentrate more (3 stocks)
- Bear markets: Should diversify more (7-10 stocks)

SOLUTION: Vary portfolio size based on market regime

Portfolio Size Rules:
---------------------
1. STRONG BULL (VIX < 15, SPY > MA200):          3 stocks (high conviction)
2. BULL (VIX 15-20, SPY > MA200):                4 stocks
3. NORMAL (VIX 20-30, mixed signals):            5 stocks (V22 baseline)
4. VOLATILE (VIX 30-40, SPY < MA200):            7 stocks (more diversification)
5. CRISIS (VIX > 40):                            10 stocks (maximum diversification)

Why This Works:
- Economic intuition: Concentrate when confident, diversify when uncertain
- No ML overfitting: Simple rule-based logic
- Adaptive but mechanical
- Academic backing: Kelly criterion suggests varying position size with edge

Expected Impact: +0.5% to +1.2% annual return
Risk: Low (just varying top_n parameter)

Test Type: INDEPENDENT - Does not modify main codebase
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.backtest.portfolio_bot_demo import PortfolioRotationBot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RegimeBasedPortfolioBot(PortfolioRotationBot):
    """
    Enhanced bot with regime-based portfolio sizing

    Varies portfolio size (3-10 stocks) based on market regime
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime_history = []  # Track regime changes

    def determine_portfolio_size(self, date):
        """
        Determine optimal portfolio size based on current market regime

        Uses VIX and SPY vs MA200 to classify regime:
        - Strong Bull: 3 stocks (concentrate when confident)
        - Bull: 4 stocks
        - Normal: 5 stocks (V22 baseline)
        - Volatile: 7 stocks (diversify when uncertain)
        - Crisis: 10 stocks (maximum diversification)

        Args:
            date: Current date

        Returns:
            tuple: (top_n: int, regime_name: str, reason: str)
        """
        # Get VIX level
        vix = 20.0  # Default
        if self.vix_data is not None:
            vix_at_date = self.vix_data[self.vix_data.index <= date]
            if len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']

        # Get SPY vs MA200
        spy_vs_ma200_pct = 0.0
        spy_data = self.stocks_data.get('SPY')
        if spy_data is not None:
            spy_at_date = spy_data[spy_data.index <= date]
            if len(spy_at_date) >= 200:
                spy_price = spy_at_date.iloc[-1]['close']
                spy_ma200 = spy_at_date['close'].tail(200).mean()
                spy_vs_ma200_pct = (spy_price / spy_ma200 - 1) * 100

        # Classify regime and determine portfolio size
        if vix < 15 and spy_vs_ma200_pct > 5:
            # STRONG BULL: Low fear, strong uptrend
            top_n = 3
            regime = "STRONG_BULL"
            reason = f"VIX={vix:.1f} (<15), SPY {spy_vs_ma200_pct:+.1f}% vs MA200 (>5%)"

        elif vix < 20 and spy_vs_ma200_pct > 2:
            # BULL: Moderate fear, decent uptrend
            top_n = 4
            regime = "BULL"
            reason = f"VIX={vix:.1f} (<20), SPY {spy_vs_ma200_pct:+.1f}% vs MA200 (>2%)"

        elif vix > 40:
            # CRISIS: Extreme fear regardless of trend
            top_n = 10
            regime = "CRISIS"
            reason = f"VIX={vix:.1f} (>40) - CRISIS MODE"

        elif vix > 30 or spy_vs_ma200_pct < -5:
            # VOLATILE/BEAR: High fear or strong downtrend
            top_n = 7
            regime = "VOLATILE"
            reason = f"VIX={vix:.1f}, SPY {spy_vs_ma200_pct:+.1f}% vs MA200 - Volatile/Bear"

        else:
            # NORMAL: Mid-range VIX, mixed signals
            top_n = 5
            regime = "NORMAL"
            reason = f"VIX={vix:.1f}, SPY {spy_vs_ma200_pct:+.1f}% vs MA200 - Normal"

        return top_n, regime, reason


def run_v27_backtest(bot):
    """
    Run V27 backtest with regime-based portfolio sizing

    Same as V22 but dynamically adjusts top_n based on regime
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING V27 BACKTEST (Regime-Based Portfolio Size)")
    logger.info("="*80)

    first_ticker = list(bot.stocks_data.keys())[0]
    dates = bot.stocks_data[first_ticker].index

    portfolio_values = []
    holdings = {}
    cash = bot.initial_capital
    last_rebalance_date = None

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

            # Get VIX
            vix_at_date = bot.vix_data[bot.vix_data.index <= date] if bot.vix_data is not None else None
            if vix_at_date is not None and len(vix_at_date) > 0:
                vix = vix_at_date.iloc[-1]['close']
            else:
                vix = 20

            # 🆕 DETERMINE PORTFOLIO SIZE BASED ON REGIME
            top_n, regime, reason = bot.determine_portfolio_size(date)

            # Record regime
            bot.regime_history.append({
                'date': date,
                'regime': regime,
                'top_n': top_n,
                'vix': vix,
                'reason': reason
            })

            # Score stocks
            current_scores = {}
            for ticker, df in bot.stocks_data.items():
                df_at_date = df[df.index <= date]
                if len(df_at_date) >= 100:
                    try:
                        current_scores[ticker] = bot.score_stock(ticker, df_at_date)
                    except:
                        pass

            # Get top N stocks (variable N based on regime)
            ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            top_stocks = [(t, s) for t, s in ranked if s > 0][:top_n]

            if not top_stocks:
                portfolio_values.append({'date': date, 'value': cash})
                continue

            # VIX-based cash reserve (same as V22)
            if vix < 30:
                cash_reserve = 0.05 + (vix - 10) * 0.005
            else:
                cash_reserve = 0.15 + (vix - 30) * 0.0125
            cash_reserve = np.clip(cash_reserve, 0.05, 0.70)

            invest_amount = cash * (1 - cash_reserve)

            # Kelly position sizing (same as V22)
            kelly_weights = {}
            tickers = [t for t, s in top_stocks]
            scores = [s for t, s in top_stocks]
            sqrt_scores = [np.sqrt(max(0, s)) for s in scores]
            total_sqrt = sum(sqrt_scores)

            if total_sqrt > 0:
                kelly_weights = {
                    ticker: np.sqrt(max(0, score)) / total_sqrt
                    for ticker, score in top_stocks
                }
            else:
                kelly_weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

            allocations = {
                ticker: invest_amount * weight
                for ticker, weight in kelly_weights.items()
            }

            # Drawdown control (same as V22)
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else None
            if portfolio_df is not None and len(portfolio_df) > 1:
                drawdown_multiplier = bot.calculate_drawdown_multiplier(portfolio_df)
                allocations = {
                    ticker: amount * drawdown_multiplier
                    for ticker, amount in allocations.items()
                }

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

        # Calculate daily portfolio value
        stocks_value = 0
        for ticker, shares in holdings.items():
            df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
            if len(df_at_date) > 0:
                current_price = df_at_date.iloc[-1]['close']
                stocks_value += shares * current_price

        total_value = cash + stocks_value
        portfolio_values.append({'date': date, 'value': total_value})

    portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
    return portfolio_df


def calculate_metrics(portfolio_df, initial_capital):
    """Calculate performance metrics"""
    final_value = portfolio_df['value'].iloc[-1]

    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]
    years = (end_date - start_date).days / 365.25
    annual_return = (((final_value / initial_capital) ** (1 / years)) - 1) * 100

    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    returns = portfolio_df['value'].pct_change()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

    # Yearly returns
    portfolio_df_copy = portfolio_df.copy()
    portfolio_df_copy['year'] = portfolio_df_copy.index.year
    yearly_returns = {}
    for year in portfolio_df_copy['year'].unique():
        year_data = portfolio_df_copy[portfolio_df_copy['year'] == year]
        if len(year_data) > 0:
            year_start = year_data['value'].iloc[0]
            year_end = year_data['value'].iloc[-1]
            year_return = (year_end - year_start) / year_start * 100
            yearly_returns[year] = year_return

    return {
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_value': final_value,
        'yearly_returns': yearly_returns
    }


def main():
    """Main test function"""
    logger.info("="*80)
    logger.info("V27: REGIME-BASED PORTFOLIO SIZE TEST")
    logger.info("="*80)
    logger.info("")
    logger.info("Innovation: Vary portfolio size (3-10 stocks) based on market regime")
    logger.info("")
    logger.info("Portfolio Size Rules:")
    logger.info("  • STRONG BULL (VIX<15, SPY>>MA200):    3 stocks (concentrate)")
    logger.info("  • BULL (VIX<20, SPY>MA200):            4 stocks")
    logger.info("  • NORMAL (VIX 20-30):                  5 stocks (V22 baseline)")
    logger.info("  • VOLATILE (VIX 30-40, SPY<MA200):     7 stocks (diversify)")
    logger.info("  • CRISIS (VIX>40):                     10 stocks (max diversification)")
    logger.info("")
    logger.info("Expected: +0.5% to +1.2% annual improvement")
    logger.info("Why: Concentrate when confident, diversify when uncertain")
    logger.info("")

    # Initialize bot
    bot = RegimeBasedPortfolioBot(
        data_dir='sp500_data/daily',
        initial_capital=100000
    )

    logger.info("Loading stock data...")
    bot.prepare_data()

    logger.info("Scoring stocks...")
    bot.score_all_stocks()

    # Run V27 backtest
    portfolio_v27 = run_v27_backtest(bot)
    metrics_v27 = calculate_metrics(portfolio_v27, bot.initial_capital)

    # Display results
    logger.info("\n" + "="*80)
    logger.info("V27 RESULTS (Regime-Based Portfolio Size)")
    logger.info("="*80)
    logger.info(f"Annual Return:   {metrics_v27['annual_return']:.2f}%")
    logger.info(f"Max Drawdown:    {metrics_v27['max_drawdown']:.2f}%")
    logger.info(f"Sharpe Ratio:    {metrics_v27['sharpe_ratio']:.2f}")
    logger.info(f"Final Value:     ${metrics_v27['final_value']:,.0f}")
    logger.info("")

    positive_years = sum(1 for r in metrics_v27['yearly_returns'].values() if r > 0)
    total_years = len(metrics_v27['yearly_returns'])
    logger.info(f"Win Rate: {positive_years}/{total_years} ({positive_years/total_years*100:.0f}%)")

    logger.info("\nYearly Returns:")
    for year in sorted(metrics_v27['yearly_returns'].keys()):
        ret = metrics_v27['yearly_returns'][year]
        status = "✅" if ret > 0 else "❌"
        logger.info(f"  {year}: {ret:>6.1f}% {status}")

    # Regime analysis
    logger.info("\n" + "="*80)
    logger.info("REGIME DISTRIBUTION ANALYSIS")
    logger.info("="*80)

    regime_df = pd.DataFrame(bot.regime_history)
    if len(regime_df) > 0:
        regime_counts = regime_df['regime'].value_counts()
        logger.info(f"\nRegime Frequency (out of {len(regime_df)} rebalances):")
        for regime, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            avg_top_n = regime_df[regime_df['regime'] == regime]['top_n'].mean()
            logger.info(f"  {regime:15s}: {count:3d} ({pct:5.1f}%) - Avg {avg_top_n:.1f} stocks")

        logger.info(f"\nAverage Portfolio Size: {regime_df['top_n'].mean():.1f} stocks")
        logger.info(f"Portfolio Size Range: {regime_df['top_n'].min()}-{regime_df['top_n'].max()} stocks")

    # Comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: V22 vs V27")
    logger.info("="*80)
    logger.info("")
    logger.info("V22 Baseline (fixed 5 stocks):")
    logger.info("  Annual: 8.67%, DD: -19.20%, Sharpe: 0.96")
    logger.info("")
    logger.info("V27 (regime-based portfolio size):")
    logger.info(f"  Annual: {metrics_v27['annual_return']:.2f}%, DD: {metrics_v27['max_drawdown']:.2f}%, Sharpe: {metrics_v27['sharpe_ratio']:.2f}")
    logger.info("")

    improvement = metrics_v27['annual_return'] - 8.67
    dd_improvement = metrics_v27['max_drawdown'] - (-19.20)
    sharpe_improvement = metrics_v27['sharpe_ratio'] - 0.96

    if improvement > 0:
        logger.info(f"✅ IMPROVEMENT: +{improvement:.2f}% annual return")
    else:
        logger.info(f"❌ REGRESSION: {improvement:.2f}% annual return")

    if dd_improvement > 0:
        logger.info(f"⚠️  Drawdown worse by {dd_improvement:.2f}%")
    else:
        logger.info(f"✅ Drawdown better by {abs(dd_improvement):.2f}%")

    if sharpe_improvement > 0:
        logger.info(f"✅ Sharpe improved by +{sharpe_improvement:.2f}")
    else:
        logger.info(f"❌ Sharpe declined by {sharpe_improvement:.2f}")

    # Final recommendation
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION")
    logger.info("="*80)

    if improvement >= 0.5:
        logger.info("✅ SUCCESS: Regime-based portfolio size improves performance!")
        logger.info("   Recommendation: INTEGRATE into production")
        logger.info("")
        logger.info("   Benefits:")
        logger.info("     • Concentrates in bull markets (higher returns)")
        logger.info("     • Diversifies in bear markets (lower risk)")
        logger.info("     • Simple rule-based logic (no overfitting)")
        logger.info("     • Economic intuition (Kelly principle)")
    elif improvement >= 0.2:
        logger.info("⚠️  MARGINAL: Small improvement")
        logger.info("   Recommendation: CONSIDER integrating")
    elif improvement >= -0.2:
        logger.info("⚠️  NEUTRAL: No meaningful change")
        logger.info("   Recommendation: OPTIONAL - no strong case either way")
    else:
        logger.info("❌ REGRESSION: Regime-based sizing hurts performance")
        logger.info("   Recommendation: DO NOT integrate, keep fixed 5 stocks")

    logger.info("\n" + "="*80)
    logger.info(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    return metrics_v27


if __name__ == '__main__':
    metrics = main()
