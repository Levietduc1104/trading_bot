"""
Phase 2 ML: Comprehensive Feature Engineering
Extracts ALL 111 fundamental features + 20 technical features
Designed for minimal overfitting risk
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MLFeatureExtractor:
    """
    Extract all available features for ML stock ranking
    Total: ~130 features (111 FA + 20 technical)
    """

    def __init__(self):
        self.technical_features = []
        self.fundamental_features = []
        self.all_feature_names = []

    def extract_features(self, ticker: str, date: pd.Timestamp, bot, fa_data: Optional[Dict] = None,
                         fa_loader=None) -> Optional[Dict]:
        """
        Extract all features for a stock at a given date

        Returns:
            Dict with features or None if insufficient data
        """
        features = {}

        # Check if we have price data
        if ticker not in bot.stocks_data:
            return None

        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
        if len(df_at_date) < 120:  # Need at least 120 days of data
            return None

        # =================================================================
        # PART 1: TECHNICAL FEATURES (20 features)
        # =================================================================
        tech_features = self._extract_technical_features(df_at_date, ticker, date, bot)
        if tech_features is None:
            return None
        features.update(tech_features)

        # =================================================================
        # PART 2: FUNDAMENTAL FEATURES (111 features from FMP)
        # =================================================================
        if fa_data:
            fa_features = self._extract_fundamental_features(fa_data)
            features.update(fa_features)
        else:
            # If no FA data, return None (we need both technical + fundamental)
            return None

        # =================================================================
        # PART 3: PREMIUM FEATURES (earnings, analyst, income trends)
        # =================================================================
        if fa_loader is not None:
            premium = self._extract_premium_features(ticker, date, fa_loader)
            features.update(premium)

        # =================================================================
        # PART 4: MACRO REGIME FEATURES (7 features)
        # =================================================================
        macro = self._extract_macro_features(date, bot)
        features.update(macro)

        return features

    def _extract_technical_features(self, df_at_date: pd.DataFrame, ticker: str, date: pd.Timestamp, bot) -> Optional[Dict]:
        """Extract 20 technical indicators"""
        features = {}

        current_price = df_at_date['close'].iloc[-1]

        # 1-5: Returns at multiple timeframes
        for period in [5, 10, 20, 60, 120]:
            if len(df_at_date) >= period:
                ret = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-period] - 1) * 100
                features[f'return_{period}d'] = ret

        # 6: Momentum acceleration
        if len(df_at_date) >= 60:
            ret_20 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1)
            ret_60 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1)
            features['momentum_acceleration'] = (ret_20 - ret_60 / 3) * 100

        # 7-8: Price to moving averages
        if len(df_at_date) >= 20:
            sma_20 = df_at_date['close'].tail(20).mean()
            features['price_to_sma20'] = (current_price / sma_20 - 1) * 100
        if len(df_at_date) >= 50:
            sma_50 = df_at_date['close'].tail(50).mean()
            features['price_to_sma50'] = (current_price / sma_50 - 1) * 100

        # 9-10: Volatility
        if len(df_at_date) >= 20:
            returns_20 = df_at_date['close'].tail(20).pct_change().dropna()
            features['volatility_20d'] = returns_20.std() * np.sqrt(252) * 100
        if len(df_at_date) >= 60:
            returns_60 = df_at_date['close'].tail(60).pct_change().dropna()
            features['volatility_60d'] = returns_60.std() * np.sqrt(252) * 100

        # 11-12: Volume features
        if 'volume' in df_at_date.columns and len(df_at_date) >= 20:
            avg_vol_20 = df_at_date['volume'].tail(20).mean()
            current_vol = df_at_date['volume'].iloc[-1]
            if avg_vol_20 > 0:
                features['volume_ratio'] = current_vol / avg_vol_20
                vol_trend = df_at_date['volume'].tail(5).mean() / avg_vol_20
                features['volume_trend'] = vol_trend
            else:
                features['volume_ratio'] = 1.0
                features['volume_trend'] = 1.0

        # 13: RSI
        if len(df_at_date) >= 14:
            delta = df_at_date['close'].diff()
            gain = (delta.where(delta > 0, 0)).tail(14).mean()
            loss = (-delta.where(delta < 0, 0)).tail(14).mean()
            if loss != 0:
                rs = gain / loss
                features['rsi_14'] = 100 - (100 / (1 + rs))
            else:
                features['rsi_14'] = 100

        # 14: 52-week price position
        if len(df_at_date) >= 252:
            high_52w = df_at_date['close'].tail(252).max()
            low_52w = df_at_date['close'].tail(252).min()
            if high_52w != low_52w:
                features['price_position_52w'] = (current_price - low_52w) / (high_52w - low_52w) * 100
            else:
                features['price_position_52w'] = 50.0

        # 15-16: Relative strength vs SPY
        if 'SPY' in bot.stocks_data:
            spy_df = bot.stocks_data['SPY'][bot.stocks_data['SPY'].index <= date]
            if len(spy_df) >= 60 and len(df_at_date) >= 60:
                stock_ret_60 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1)
                spy_ret_60 = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-60] - 1)
                features['relative_strength_60d'] = (stock_ret_60 - spy_ret_60) * 100

            if len(spy_df) >= 20 and len(df_at_date) >= 20:
                stock_ret_20 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1)
                spy_ret_20 = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-20] - 1)
                features['relative_strength_20d'] = (stock_ret_20 - spy_ret_20) * 100

        # 17: Beta (market sensitivity)
        spy_df = None
        if 'SPY' in bot.stocks_data:
            spy_df = bot.stocks_data['SPY'][bot.stocks_data['SPY'].index <= date]

        if spy_df is not None and len(spy_df) >= 60 and len(df_at_date) >= 60:
            stock_returns = df_at_date['close'].tail(60).pct_change().dropna()
            spy_returns   = spy_df['close'].tail(60).pct_change().dropna()
            common_dates  = stock_returns.index.intersection(spy_returns.index)
            if len(common_dates) >= 30:
                stock_aligned = stock_returns.loc[common_dates]
                spy_aligned   = spy_returns.loc[common_dates]
                covariance    = np.cov(stock_aligned, spy_aligned)[0][1]
                spy_variance  = np.var(spy_aligned)
                if spy_variance > 0:
                    features['beta'] = covariance / spy_variance

        close = df_at_date['close']
        n     = len(df_at_date)

        # ── TREND (5 new) ──────────────────────────────────────────────────────

        # 18: Price / SMA200
        if n >= 200:
            features['price_to_sma200'] = (current_price / close.tail(200).mean() - 1) * 100

        # 19: EMA12 / EMA26 ratio (MACD signal proxy)
        if n >= 26:
            ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
            if ema26 != 0:
                features['ema_ratio_12_26'] = (ema12 / ema26 - 1) * 100

        # 20: MACD histogram (ema12 - ema26 - signal9)
        if n >= 35:
            macd_line   = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            features['macd_histogram'] = float(macd_line.iloc[-1] - signal_line.iloc[-1])

        # 21: SMA50 / SMA200 — golden/death cross
        if n >= 200:
            sma50  = close.tail(50).mean()
            sma200 = close.tail(200).mean()
            if sma200 != 0:
                features['sma50_sma200_ratio'] = (sma50 / sma200 - 1) * 100

        # 22: 250-day return (1-year momentum)
        if n >= 250:
            features['return_250d'] = (close.iloc[-1] / close.iloc[-250] - 1) * 100

        # ── MEAN REVERSION (3 new) ─────────────────────────────────────────────

        # 23: Bollinger Band position  (0=lower band, 50=mid, 100=upper)
        if n >= 20:
            mid  = close.tail(20).mean()
            std  = close.tail(20).std()
            if std > 0:
                upper = mid + 2 * std
                lower = mid - 2 * std
                features['bb_position'] = float(np.clip((current_price - lower) / (upper - lower) * 100, 0, 100))
                features['bb_width']    = float((upper - lower) / mid * 100)  # band width as % of price

        # 25: Distance from 52-week high (drawdown from peak)
        if n >= 252:
            high_52w = close.tail(252).max()
            features['dist_from_52w_high'] = (current_price / high_52w - 1) * 100  # always <= 0

        # ── VOLATILITY (3 new) ─────────────────────────────────────────────────

        # 26: ATR-14 normalised by price
        if n >= 15 and 'high' in df_at_date.columns and 'low' in df_at_date.columns:
            high = df_at_date['high'].tail(15)
            low  = df_at_date['low'].tail(15)
            prev_close = df_at_date['close'].shift(1).tail(15)
            tr = pd.concat([high - low,
                            (high - prev_close).abs(),
                            (low  - prev_close).abs()], axis=1).max(axis=1)
            atr14 = tr.tail(14).mean()
            if current_price > 0:
                features['atr_pct'] = float(atr14 / current_price * 100)

        # 27: Volatility regime — short vol / long vol ratio
        if n >= 60:
            vol20 = close.tail(20).pct_change().dropna().std()
            vol60 = close.tail(60).pct_change().dropna().std()
            if vol60 > 0:
                features['vol_regime'] = float(vol20 / vol60)  # >1 = vol expanding, <1 = contracting

        # 28: Max intra-period drawdown (20d)
        if n >= 20:
            roll_max = close.tail(20).cummax()
            dd_series = (close.tail(20) - roll_max) / roll_max * 100
            features['max_dd_20d'] = float(dd_series.min())

        # ── VOLUME (3 new) ─────────────────────────────────────────────────────

        if 'volume' in df_at_date.columns:
            vol_series = df_at_date['volume']

            # 29: OBV trend (slope of OBV over last 20 days, normalised)
            if n >= 20:
                price_chg = close.diff()
                obv = (np.sign(price_chg) * vol_series).tail(20).cumsum()
                if obv.std() > 0:
                    obv_norm = (obv - obv.mean()) / obv.std()
                    # slope via last vs first normalised value
                    features['obv_trend'] = float(obv_norm.iloc[-1] - obv_norm.iloc[0])

            # 30: Volume z-score (current vol vs 60d avg)
            if n >= 60:
                vol_mean = vol_series.tail(60).mean()
                vol_std  = vol_series.tail(60).std()
                if vol_std > 0:
                    features['volume_zscore'] = float((vol_series.iloc[-1] - vol_mean) / vol_std)

            # 31: Price-volume divergence (price up but volume down = weak move)
            if n >= 10:
                price_ret_5 = (close.iloc[-1] / close.iloc[-5] - 1)
                vol_ret_5   = (vol_series.tail(5).mean() / vol_series.tail(10).mean() - 1) if vol_series.tail(10).mean() > 0 else 0
                features['price_vol_divergence'] = float(price_ret_5 * 100 - vol_ret_5 * 100)

        # ── MOMENTUM OSCILLATORS (3 new) ──────────────────────────────────────

        # 32: Williams %R (14-period)
        if n >= 14 and 'high' in df_at_date.columns and 'low' in df_at_date.columns:
            high14 = df_at_date['high'].tail(14).max()
            low14  = df_at_date['low'].tail(14).min()
            if high14 != low14:
                features['williams_r'] = float((high14 - current_price) / (high14 - low14) * -100)

        # 33: Stochastic %K (14-period)
        if n >= 14 and 'high' in df_at_date.columns and 'low' in df_at_date.columns:
            high14 = df_at_date['high'].tail(14).max()
            low14  = df_at_date['low'].tail(14).min()
            if high14 != low14:
                features['stochastic_k'] = float((current_price - low14) / (high14 - low14) * 100)

        # 34: Rate of change 10d
        if n >= 10:
            features['roc_10d'] = float((close.iloc[-1] / close.iloc[-10] - 1) * 100)

        # ── PATTERN / STRUCTURE (3 new) ───────────────────────────────────────

        # 35: Consecutive up days (momentum confirmation)
        if n >= 10:
            streak = 0
            daily_chg = close.diff().tail(10)
            for chg in reversed(daily_chg.values):
                if chg > 0:
                    streak += 1
                else:
                    break
            features['up_streak'] = float(streak)

        # 36: Gap ratio (today open vs yesterday close)
        if n >= 2 and 'open' in df_at_date.columns:
            prev_c = df_at_date['close'].iloc[-2]
            curr_o = df_at_date['open'].iloc[-1]
            if prev_c > 0:
                features['gap_ratio'] = float((curr_o / prev_c - 1) * 100)

        # 37: Downside deviation (semi-deviation, 60d)
        if n >= 60:
            rets60 = close.tail(60).pct_change().dropna()
            down   = rets60[rets60 < 0]
            features['downside_dev'] = float(down.std() * np.sqrt(252) * 100) if len(down) > 1 else 0.0

        # ── RELATIVE STRENGTH (2 new) ─────────────────────────────────────────

        # 38: 120d relative strength vs SPY
        if spy_df is not None and n >= 120 and len(spy_df) >= 120:
            stock_ret_120 = (close.iloc[-1] / close.iloc[-120] - 1)
            spy_ret_120   = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-120] - 1)
            features['relative_strength_120d'] = float((stock_ret_120 - spy_ret_120) * 100)

        # 39: Return consistency — % of months positive over last 12 months
        if n >= 252:
            monthly_rets = close.tail(252).resample('ME').last().pct_change().dropna()
            if len(monthly_rets) >= 6:
                features['pct_months_positive'] = float((monthly_rets > 0).mean() * 100)

        return features

    def _extract_fundamental_features(self, fa_data: Dict) -> Dict:
        """
        Extract ALL 111 fundamental features from FMP data

        Financial Ratios (64) + Key Metrics (47) = 111 features
        """
        features = {}

        # ============================================================
        # FINANCIAL RATIOS (64 fields)
        # ============================================================

        # Profitability margins (12)
        features['gross_profit_margin'] = fa_data.get('grossProfitMargin', 0) * 100
        features['operating_profit_margin'] = fa_data.get('operatingProfitMargin', 0) * 100
        features['net_profit_margin'] = fa_data.get('netProfitMargin', 0) * 100
        features['pretax_profit_margin'] = fa_data.get('pretaxProfitMargin', 0) * 100
        features['bottom_line_profit_margin'] = fa_data.get('bottomLineProfitMargin', 0) * 100
        features['continuous_ops_profit_margin'] = fa_data.get('continuousOperationsProfitMargin', 0) * 100
        features['ebit_margin'] = fa_data.get('ebitMargin', 0) * 100
        features['ebitda_margin'] = fa_data.get('ebitdaMargin', 0) * 100
        features['net_income_per_ebt'] = fa_data.get('netIncomePerEBT', 0)
        features['ebt_per_ebit'] = fa_data.get('ebtPerEbit', 0)
        features['effective_tax_rate'] = fa_data.get('effectiveTaxRate', 0) * 100
        features['financial_leverage_ratio'] = fa_data.get('financialLeverageRatio', 0)

        # Liquidity ratios (4)
        features['current_ratio'] = fa_data.get('currentRatio', 0)
        features['quick_ratio'] = fa_data.get('quickRatio', 0)
        features['cash_ratio'] = fa_data.get('cashRatio', 0)
        features['operating_cf_ratio'] = fa_data.get('operatingCashFlowRatio', 0)

        # Leverage ratios (8)
        features['debt_to_equity'] = fa_data.get('debtToEquityRatio', 0)
        features['debt_to_assets'] = fa_data.get('debtToAssetsRatio', 0)
        features['debt_to_capital'] = fa_data.get('debtToCapitalRatio', 0)
        features['debt_to_market_cap'] = fa_data.get('debtToMarketCap', 0)
        features['long_term_debt_to_capital'] = fa_data.get('longTermDebtToCapitalRatio', 0)
        features['interest_coverage'] = fa_data.get('interestCoverageRatio', 0)
        features['debt_service_coverage'] = fa_data.get('debtServiceCoverageRatio', 0)
        features['solvency_ratio'] = fa_data.get('solvencyRatio', 0)

        # Efficiency ratios (6)
        features['asset_turnover'] = fa_data.get('assetTurnover', 0)
        features['fixed_asset_turnover'] = fa_data.get('fixedAssetTurnover', 0)
        features['inventory_turnover'] = fa_data.get('inventoryTurnover', 0)
        features['receivables_turnover'] = fa_data.get('receivablesTurnover', 0)
        features['payables_turnover'] = fa_data.get('payablesTurnover', 0)
        features['working_capital_turnover'] = fa_data.get('workingCapitalTurnoverRatio', 0)

        # Valuation ratios (10)
        features['pe_ratio'] = fa_data.get('priceToEarningsRatio', 0)
        features['price_to_book'] = fa_data.get('priceToBookRatio', 0)
        features['price_to_sales'] = fa_data.get('priceToSalesRatio', 0)
        features['price_to_fair_value'] = fa_data.get('priceToFairValue', 0)
        features['price_to_fcf'] = fa_data.get('priceToFreeCashFlowRatio', 0)
        features['price_to_ocf'] = fa_data.get('priceToOperatingCashFlowRatio', 0)
        features['peg_ratio'] = fa_data.get('priceToEarningsGrowthRatio', 0)
        features['forward_peg_ratio'] = fa_data.get('forwardPriceToEarningsGrowthRatio', 0)
        features['ev_multiple'] = fa_data.get('enterpriseValueMultiple', 0)
        features['dividend_yield'] = fa_data.get('dividendYield', 0) * 100

        # Cash flow coverage (5)
        features['ocf_coverage'] = fa_data.get('operatingCashFlowCoverageRatio', 0)
        features['short_term_ocf_coverage'] = fa_data.get('shortTermOperatingCashFlowCoverageRatio', 0)
        features['capex_coverage'] = fa_data.get('capitalExpenditureCoverageRatio', 0)
        features['dividend_capex_coverage'] = fa_data.get('dividendPaidAndCapexCoverageRatio', 0)
        features['fcf_to_ocf'] = fa_data.get('freeCashFlowOperatingCashFlowRatio', 0)

        # Per-share metrics (11)
        features['book_value_per_share'] = fa_data.get('bookValuePerShare', 0)
        features['tangible_book_per_share'] = fa_data.get('tangibleBookValuePerShare', 0)
        features['cash_per_share'] = fa_data.get('cashPerShare', 0)
        features['revenue_per_share'] = fa_data.get('revenuePerShare', 0)
        features['net_income_per_share'] = fa_data.get('netIncomePerShare', 0)
        features['dividend_per_share'] = fa_data.get('dividendPerShare', 0)
        features['fcf_per_share'] = fa_data.get('freeCashFlowPerShare', 0)
        features['ocf_per_share'] = fa_data.get('operatingCashFlowPerShare', 0)
        features['capex_per_share'] = fa_data.get('capexPerShare', 0)
        features['interest_debt_per_share'] = fa_data.get('interestDebtPerShare', 0)
        features['shareholders_equity_per_share'] = fa_data.get('shareholdersEquityPerShare', 0)

        # Dividend metrics (2)
        features['dividend_payout_ratio'] = fa_data.get('dividendPayoutRatio', 0) * 100
        features['dividend_yield_pct'] = fa_data.get('dividendYieldPercentage', 0)

        # Cash flow ratios (2)
        features['ocf_sales_ratio'] = fa_data.get('operatingCashFlowSalesRatio', 0)
        features['fcf_operating_cf_ratio'] = fa_data.get('freeCashFlowOperatingCashFlowRatio', 0)

        # ============================================================
        # KEY METRICS (47 fields)
        # ============================================================

        # Return metrics (6)
        features['roe'] = fa_data.get('returnOnEquity', 0) * 100
        features['roa'] = fa_data.get('returnOnAssets', 0) * 100
        features['roic'] = fa_data.get('returnOnInvestedCapital', 0) * 100
        features['roce'] = fa_data.get('returnOnCapitalEmployed', 0) * 100
        features['rota'] = fa_data.get('returnOnTangibleAssets', 0) * 100
        features['operating_roa'] = fa_data.get('operatingReturnOnAssets', 0) * 100

        # Cash flow quality metrics (6)
        features['fcf_yield'] = fa_data.get('freeCashFlowYield', 0) * 100
        features['fcf_to_equity'] = fa_data.get('freeCashFlowToEquity', 0)
        features['fcf_to_firm'] = fa_data.get('freeCashFlowToFirm', 0)
        features['income_quality'] = fa_data.get('incomeQuality', 0)  # KEY METRIC
        features['ev_to_fcf'] = fa_data.get('evToFreeCashFlow', 0)
        features['ev_to_ocf'] = fa_data.get('evToOperatingCashFlow', 0)

        # Working capital metrics (9)
        features['working_capital'] = fa_data.get('workingCapital', 0)
        features['cash_conversion_cycle'] = fa_data.get('cashConversionCycle', 0)  # KEY METRIC
        features['operating_cycle'] = fa_data.get('operatingCycle', 0)
        features['days_sales_outstanding'] = fa_data.get('daysOfSalesOutstanding', 0)
        features['days_inventory_outstanding'] = fa_data.get('daysOfInventoryOutstanding', 0)
        features['days_payables_outstanding'] = fa_data.get('daysOfPayablesOutstanding', 0)
        features['avg_receivables'] = fa_data.get('averageReceivables', 0)
        features['avg_inventory'] = fa_data.get('averageInventory', 0)
        features['avg_payables'] = fa_data.get('averagePayables', 0)

        # Capex metrics (3)
        features['capex_to_revenue'] = fa_data.get('capexToRevenue', 0)
        features['capex_to_ocf'] = fa_data.get('capexToOperatingCashFlow', 0)  # KEY METRIC
        features['capex_to_depreciation'] = fa_data.get('capexToDepreciation', 0)

        # Enterprise value metrics (5)
        features['enterprise_value'] = fa_data.get('enterpriseValue', 0)
        features['ev_to_sales'] = fa_data.get('evToSales', 0)
        features['ev_to_ebitda'] = fa_data.get('evToEBITDA', 0)
        features['ev_to_fcf_metrics'] = fa_data.get('evToFreeCashFlow', 0)
        features['ev_to_ocf_metrics'] = fa_data.get('evToOperatingCashFlow', 0)
        features['net_debt_to_ebitda'] = fa_data.get('netDebtToEBITDA', 0)

        # Valuation metrics (4)
        features['graham_number'] = fa_data.get('grahamNumber', 0)
        features['graham_net_net'] = fa_data.get('grahamNetNet', 0)
        features['earnings_yield'] = fa_data.get('earningsYield', 0) * 100
        features['net_current_asset_value'] = fa_data.get('netCurrentAssetValue', 0)

        # Asset quality (3)
        features['tangible_asset_value'] = fa_data.get('tangibleAssetValue', 0)
        features['intangibles_to_assets'] = fa_data.get('intangiblesToTotalAssets', 0) * 100
        features['invested_capital'] = fa_data.get('investedCapital', 0)

        # Expense ratios (3)
        features['rd_to_revenue'] = fa_data.get('researchAndDevelopementToRevenue', 0) * 100
        features['sga_to_revenue'] = fa_data.get('salesGeneralAndAdministrativeToRevenue', 0) * 100
        features['stock_comp_to_revenue'] = fa_data.get('stockBasedCompensationToRevenue', 0) * 100

        # Tax and burden metrics (2)
        features['tax_burden'] = fa_data.get('taxBurden', 0)
        features['interest_burden'] = fa_data.get('interestBurden', 0)

        # Market data (1)
        features['market_cap'] = fa_data.get('marketCap', 0)

        return features

    def _extract_premium_features(self, ticker: str, date: pd.Timestamp, fa_loader) -> Dict:
        """
        Extract ~22 forward-looking and stock-specific features from premium FMP data.

        Original 15 features:
          - earnings_beat_pct        : (actual - estimated) / |estimated| EPS surprise %
          - revenue_surprise_pct     : (actual - estimated) / estimated revenue surprise %
          - eps_beat_3q_avg          : avg EPS beat % over last 3 quarters
          - eps_trend_3q             : slope of EPS actuals over last 3 quarters (improving?)
          - analyst_eps_revision     : change in forward EPS estimate vs prior quarter
          - analyst_rev_revision     : change in forward revenue estimate vs prior quarter
          - revenue_growth_yoy       : YoY revenue growth % (most recent vs 4q ago)
          - revenue_growth_qoq       : QoQ revenue growth %
          - gross_margin_change      : change in gross margin vs 4 quarters ago (pp)
          - net_income_growth_yoy    : YoY net income growth %
          - fcf_margin               : free cash flow / revenue %
          - fcf_growth_yoy           : YoY FCF growth %
          - debt_change_yoy          : YoY change in total debt (positive = more debt)
          - cash_change_yoy          : YoY change in cash & equivalents
          - capex_to_revenue         : capex / revenue % (investment intensity)

        NEW stock-picking features (7):
          - analyst_dispersion       : (epsHigh - epsLow) / |epsAvg| — consensus vs uncertainty
          - num_analysts_eps         : number of analysts covering EPS — institutional interest proxy
          - forward_eps_growth       : forward epsAvg vs trailing EPS — growth expectation
          - eps_beat_streak          : consecutive quarters beating EPS estimate (0–8)
          - eps_acceleration         : current EPS surprise % minus prior quarter surprise %
          - operating_leverage       : operating income growth / revenue growth — margin expansion
          - earnings_quality         : OCF / net income — cash earnings vs reported earnings
        """
        f = {}

        # ── Earnings surprises ────────────────────────────────────────────
        earn_rec = fa_loader.get_earnings_at_date(ticker, date)
        if earn_rec:
            eps_actual    = earn_rec.get('epsActual')
            eps_estimated = earn_rec.get('epsEstimated')
            rev_actual    = earn_rec.get('revenueActual')
            rev_estimated = earn_rec.get('revenueEstimated')

            if eps_actual is not None and eps_estimated not in (None, 0):
                f['earnings_beat_pct'] = (eps_actual - eps_estimated) / abs(eps_estimated) * 100
            else:
                f['earnings_beat_pct'] = 0.0

            if rev_actual is not None and rev_estimated not in (None, 0):
                f['revenue_surprise_pct'] = (rev_actual - rev_estimated) / abs(rev_estimated) * 100
            else:
                f['revenue_surprise_pct'] = 0.0
        else:
            f['earnings_beat_pct'] = 0.0
            f['revenue_surprise_pct'] = 0.0

        # Last 3 quarters of earnings beats — average surprise and trend
        date_str = date.strftime('%Y-%m-%d')
        earn_history = []
        if ticker in fa_loader.earnings_date_idx:
            dates = fa_loader.earnings_date_idx[ticker]
            from bisect import bisect_left
            idx = bisect_left(dates, date_str)
            if idx >= len(dates):
                idx = len(dates) - 1
            elif dates[idx] > date_str and idx > 0:
                idx -= 1
            src = {r['date']: r for r in fa_loader.earnings.get(ticker, [])}
            for i in range(idx, max(-1, idx - 3), -1):
                if i >= 0 and dates[i] in src:
                    r = src[dates[i]]
                    ea = r.get('epsActual')
                    ee = r.get('epsEstimated')
                    if ea is not None and ee not in (None, 0):
                        earn_history.append((ea - ee) / abs(ee) * 100)
                    elif ea is not None:
                        earn_history.append(0.0)

        f['eps_beat_3q_avg'] = float(np.mean(earn_history)) if earn_history else 0.0

        # Slope of EPS actuals over last 3 quarters
        if len(earn_history) >= 2:
            f['eps_trend_3q'] = earn_history[0] - earn_history[-1]   # positive = improving
        else:
            f['eps_trend_3q'] = 0.0

        # ── Analyst estimate revisions ────────────────────────────────────
        analyst_rec = fa_loader.get_analyst_at_date(ticker, date)
        if analyst_rec and ticker in fa_loader.analyst_date_idx:
            dates = fa_loader.analyst_date_idx[ticker]
            from bisect import bisect_left
            idx = bisect_left(dates, date_str)
            if idx >= len(dates):
                idx = len(dates) - 1
            elif dates[idx] > date_str and idx > 0:
                idx -= 1
            # Get prior record (one step back)
            prior_rec = None
            if idx > 0:
                src = {r['date']: r for r in fa_loader.analyst.get(ticker, [])}
                prior_date = dates[idx - 1]
                prior_rec = src.get(prior_date)

            cur_eps = analyst_rec.get('epsEstimated') or analyst_rec.get('epAvg')
            cur_rev = analyst_rec.get('revenueEstimated') or analyst_rec.get('revenueAvg')
            if prior_rec:
                prior_eps = prior_rec.get('epsEstimated') or prior_rec.get('epAvg')
                prior_rev = prior_rec.get('revenueEstimated') or prior_rec.get('revenueAvg')
                if cur_eps is not None and prior_eps not in (None, 0):
                    f['analyst_eps_revision'] = (cur_eps - prior_eps) / abs(prior_eps) * 100
                else:
                    f['analyst_eps_revision'] = 0.0
                if cur_rev is not None and prior_rev not in (None, 0):
                    f['analyst_rev_revision'] = (cur_rev - prior_rev) / abs(prior_rev) * 100
                else:
                    f['analyst_rev_revision'] = 0.0
            else:
                f['analyst_eps_revision'] = 0.0
                f['analyst_rev_revision'] = 0.0
        else:
            f['analyst_eps_revision'] = 0.0
            f['analyst_rev_revision'] = 0.0

        # ── Income statement trends ───────────────────────────────────────
        income_hist = fa_loader.get_income_history(ticker, date, n_quarters=5)

        if income_hist:
            rev_now = income_hist[0].get('revenue') or 0
            ni_now  = income_hist[0].get('netIncome') or 0
            gp_now  = income_hist[0].get('grossProfit') or 0

            # YoY growth (q vs q-4)
            if len(income_hist) >= 5:
                rev_4q   = income_hist[4].get('revenue') or 0
                ni_4q    = income_hist[4].get('netIncome') or 0
                gp_4q    = income_hist[4].get('grossProfit') or 0
                f['revenue_growth_yoy']     = (rev_now / rev_4q - 1) * 100 if rev_4q != 0 else 0.0
                f['net_income_growth_yoy']  = (ni_now  / ni_4q  - 1) * 100 if ni_4q  != 0 else 0.0
                gm_now = (gp_now / rev_now * 100) if rev_now != 0 else 0
                gm_4q  = (gp_4q  / rev_4q  * 100) if rev_4q  != 0 else 0
                f['gross_margin_change'] = gm_now - gm_4q
            else:
                f['revenue_growth_yoy']    = 0.0
                f['net_income_growth_yoy'] = 0.0
                f['gross_margin_change']   = 0.0

            # QoQ growth
            if len(income_hist) >= 2:
                rev_1q = income_hist[1].get('revenue') or 0
                f['revenue_growth_qoq'] = (rev_now / rev_1q - 1) * 100 if rev_1q != 0 else 0.0
            else:
                f['revenue_growth_qoq'] = 0.0
        else:
            f['revenue_growth_yoy']    = 0.0
            f['revenue_growth_qoq']    = 0.0
            f['gross_margin_change']   = 0.0
            f['net_income_growth_yoy'] = 0.0

        # ── Cash flow features ────────────────────────────────────────────
        cf_rec  = fa_loader.get_cashflow_at_date(ticker, date)
        inc_rec = fa_loader.get_income_at_date(ticker, date)

        if cf_rec and inc_rec:
            net_income = inc_rec.get('netIncome') or 0
            revenue    = inc_rec.get('revenue')   or 0
            capex      = cf_rec.get('capitalExpenditure') or 0
            ocf        = cf_rec.get('operatingCashFlow')  or 0
            fcf        = ocf + capex   # capex is negative in statements

            f['fcf_margin']       = (fcf / revenue * 100)   if revenue != 0 else 0.0
            f['capex_to_revenue'] = (abs(capex) / revenue * 100) if revenue != 0 else 0.0
        else:
            f['fcf_margin']       = 0.0
            f['capex_to_revenue'] = 0.0

        # FCF YoY growth — need cashflow history (reuse income_history pattern)
        cf_now_val  = (cf_rec.get('operatingCashFlow', 0) or 0) + (cf_rec.get('capitalExpenditure', 0) or 0) if cf_rec else 0
        if ticker in fa_loader.cashflow_date_idx:
            from bisect import bisect_left
            dates  = fa_loader.cashflow_date_idx[ticker]
            idx    = bisect_left(dates, date_str)
            if idx >= len(dates): idx = len(dates) - 1
            elif dates[idx] > date_str and idx > 0: idx -= 1
            src = {r['date']: r for r in fa_loader.cashflow.get(ticker, [])}
            if idx >= 4 and dates[idx-4] in src:
                r4q = src[dates[idx-4]]
                fcf_4q = (r4q.get('operatingCashFlow', 0) or 0) + (r4q.get('capitalExpenditure', 0) or 0)
                f['fcf_growth_yoy'] = (cf_now_val / fcf_4q - 1) * 100 if fcf_4q != 0 else 0.0
            else:
                f['fcf_growth_yoy'] = 0.0
        else:
            f['fcf_growth_yoy'] = 0.0

        # ── Balance sheet changes ─────────────────────────────────────────
        bal_rec = fa_loader.get_balance_at_date(ticker, date)
        if bal_rec and ticker in fa_loader.balance_date_idx:
            from bisect import bisect_left
            dates = fa_loader.balance_date_idx[ticker]
            idx   = bisect_left(dates, date_str)
            if idx >= len(dates): idx = len(dates) - 1
            elif dates[idx] > date_str and idx > 0: idx -= 1
            src = {r['date']: r for r in fa_loader.balance.get(ticker, [])}

            debt_now = (bal_rec.get('totalDebt') or bal_rec.get('longTermDebt') or 0)
            cash_now = (bal_rec.get('cashAndCashEquivalents') or 0)

            if idx >= 4 and dates[idx-4] in src:
                r4q      = src[dates[idx-4]]
                debt_4q  = (r4q.get('totalDebt') or r4q.get('longTermDebt') or 0)
                cash_4q  = (r4q.get('cashAndCashEquivalents') or 0)
                rev_ref  = (inc_rec.get('revenue') or 0) if inc_rec else 0
                if rev_ref != 0:
                    f['debt_change_yoy'] = (debt_now - debt_4q) / abs(rev_ref) * 100
                    f['cash_change_yoy'] = (cash_now - cash_4q) / abs(rev_ref) * 100
                else:
                    f['debt_change_yoy'] = 0.0
                    f['cash_change_yoy'] = 0.0
            else:
                f['debt_change_yoy'] = 0.0
                f['cash_change_yoy'] = 0.0
        else:
            f['debt_change_yoy'] = 0.0
            f['cash_change_yoy'] = 0.0

        # ── NEW: Analyst dispersion, coverage, forward growth ─────────────
        analyst_rec = fa_loader.get_analyst_at_date(ticker, date)
        if analyst_rec:
            eps_avg  = analyst_rec.get('epsAvg')  or analyst_rec.get('epsEstimated') or 0
            eps_high = analyst_rec.get('epsHigh') or 0
            eps_low  = analyst_rec.get('epsLow')  or 0
            num_an   = analyst_rec.get('numAnalystsEps') or 0

            # Dispersion: how much analysts disagree (normalised by magnitude)
            if abs(eps_avg) > 0:
                f['analyst_dispersion'] = (eps_high - eps_low) / abs(eps_avg)
            else:
                f['analyst_dispersion'] = 0.0

            f['num_analysts_eps'] = float(num_an)

            # Forward EPS growth: forward consensus vs trailing EPS
            inc_rec_now = fa_loader.get_income_at_date(ticker, date)
            trailing_eps = 0.0
            if inc_rec_now:
                trailing_eps = float(inc_rec_now.get('epsDiluted') or
                                     inc_rec_now.get('eps') or 0)
            if abs(trailing_eps) > 0.01 and eps_avg != 0:
                f['forward_eps_growth'] = (eps_avg - trailing_eps) / abs(trailing_eps) * 100
            else:
                f['forward_eps_growth'] = 0.0
        else:
            f['analyst_dispersion']  = 0.0
            f['num_analysts_eps']    = 0.0
            f['forward_eps_growth']  = 0.0

        # ── NEW: EPS beat streak and acceleration ─────────────────────────
        date_str = date.strftime('%Y-%m-%d')
        beat_history = []   # list of (beat_bool, surprise_pct) newest first
        if ticker in fa_loader.earnings_date_idx:
            from bisect import bisect_left
            dates_list = fa_loader.earnings_date_idx[ticker]
            idx = bisect_left(dates_list, date_str)
            if idx >= len(dates_list): idx = len(dates_list) - 1
            elif dates_list[idx] > date_str and idx > 0: idx -= 1
            src = {r['date']: r for r in fa_loader.earnings.get(ticker, [])}
            for i in range(idx, max(-1, idx - 8), -1):
                if i >= 0 and dates_list[i] in src:
                    r = src[dates_list[i]]
                    ea = r.get('epsActual')
                    ee = r.get('epsEstimated')
                    if ea is not None and ee not in (None, 0):
                        surp = (ea - ee) / abs(ee) * 100
                        beat_history.append((ea >= ee, surp))
                    elif ea is not None:
                        beat_history.append((False, 0.0))

        # Consecutive beat streak (newest first)
        streak = 0
        for beat, _ in beat_history:
            if beat:
                streak += 1
            else:
                break
        f['eps_beat_streak'] = float(streak)

        # EPS surprise acceleration: most recent vs prior quarter
        if len(beat_history) >= 2:
            f['eps_acceleration'] = beat_history[0][1] - beat_history[1][1]
        else:
            f['eps_acceleration'] = 0.0

        # ── NEW: Operating leverage ────────────────────────────────────────
        # Operating income growth / revenue growth — positive = margin expansion
        income_hist_5 = fa_loader.get_income_history(ticker, date, n_quarters=5)
        f['operating_leverage'] = 0.0
        if income_hist_5 and len(income_hist_5) >= 5:
            rev_now = income_hist_5[0].get('revenue') or 0
            rev_4q  = income_hist_5[4].get('revenue') or 0
            oi_now  = income_hist_5[0].get('operatingIncome') or 0
            oi_4q   = income_hist_5[4].get('operatingIncome') or 0
            rev_growth = (rev_now / rev_4q - 1) if rev_4q != 0 else 0
            oi_growth  = (oi_now  / oi_4q  - 1) if oi_4q  != 0 and oi_4q > 0 else 0
            if abs(rev_growth) > 0.001:
                f['operating_leverage'] = oi_growth / abs(rev_growth)

        # ── NEW: Earnings quality (OCF / Net Income) ──────────────────────
        cf_rec  = fa_loader.get_cashflow_at_date(ticker, date)
        inc_rec = fa_loader.get_income_at_date(ticker, date)
        f['earnings_quality'] = 0.0  # default: unknown
        if cf_rec and inc_rec:
            ocf = cf_rec.get('operatingCashFlow') or 0
            ni  = inc_rec.get('netIncome') or 0
            if abs(ni) > 0:
                eq = ocf / abs(ni)
                # Clip to [-3, 5] to avoid extreme values
                f['earnings_quality'] = float(np.clip(eq, -3.0, 5.0))

        # Replace any NaN/inf/non-numeric with 0
        for k, v in f.items():
            try:
                fv = float(v)
                f[k] = 0.0 if (np.isnan(fv) or np.isinf(fv)) else fv
            except (TypeError, ValueError):
                f[k] = 0.0

        return f

    def _extract_macro_features(self, date: pd.Timestamp, bot) -> Dict:
        """
        Extract 7 macro regime features from SPY and VIX.
        All derived from data already loaded in bot — no new downloads needed.

        Features:
          vix_level           : current VIX reading
          vix_roc_20d         : VIX 20-day rate of change (rising = stress building)
          vix_roc_5d          : VIX 5-day rate of change (spike detector)
          spy_ma200_ratio     : SPY / 200-day MA  (>1 = bull, <1 = bear)
          spy_ma50_ratio      : SPY / 50-day MA   (faster signal)
          spy_trend_strength  : 50-day MA / 200-day MA (golden/death cross)
          market_stress       : composite score = vix_roc_20d - 100*(spy_ma200_ratio-1)
                                positive = stress, negative = calm
        """
        f = {}

        # ── VIX features ─────────────────────────────────────────────────
        vix_val = 20.0  # neutral default
        vix_20d_ago = 20.0
        vix_5d_ago  = 20.0

        # Use bot.vix_data if available, otherwise fall back to stocks_data['VIX']
        vix_series = None
        if bot.vix_data is not None:
            vix_hist = bot.vix_data[bot.vix_data.index <= date]
            if len(vix_hist) >= 1:
                vix_series = vix_hist['close']
        elif 'VIX' in bot.stocks_data:
            vix_raw = bot.stocks_data['VIX']
            vix_hist = vix_raw[vix_raw.index <= date]
            if len(vix_hist) >= 1:
                vix_series = vix_hist['close']

        if vix_series is not None and len(vix_series) >= 1:
            vix_val = float(vix_series.iloc[-1])
            if len(vix_series) >= 21:
                vix_20d_ago = float(vix_series.iloc[-21])
            if len(vix_series) >= 6:
                vix_5d_ago = float(vix_series.iloc[-6])

        f['vix_level']   = vix_val
        f['vix_roc_20d'] = (vix_val / vix_20d_ago - 1) * 100 if vix_20d_ago > 0 else 0.0
        f['vix_roc_5d']  = (vix_val / vix_5d_ago  - 1) * 100 if vix_5d_ago  > 0 else 0.0

        # ── SPY trend features ────────────────────────────────────────────
        spy_ma200_ratio    = 1.0   # neutral defaults
        spy_ma50_ratio     = 1.0
        spy_trend_strength = 1.0

        if 'SPY' in bot.stocks_data:
            spy = bot.stocks_data['SPY']
            spy_hist = spy[spy.index <= date]
            if len(spy_hist) >= 200:
                current  = float(spy_hist['close'].iloc[-1])
                ma200    = float(spy_hist['close'].tail(200).mean())
                spy_ma200_ratio = current / ma200 if ma200 > 0 else 1.0
                if len(spy_hist) >= 50:
                    ma50 = float(spy_hist['close'].tail(50).mean())
                    spy_ma50_ratio     = current / ma50 if ma50 > 0 else 1.0
                    spy_trend_strength = ma50    / ma200 if ma200 > 0 else 1.0

        f['spy_ma200_ratio']    = (spy_ma200_ratio    - 1) * 100   # % above/below MA200
        f['spy_ma50_ratio']     = (spy_ma50_ratio     - 1) * 100   # % above/below MA50
        f['spy_trend_strength'] = (spy_trend_strength - 1) * 100   # golden/death cross signal

        # ── Composite stress score ────────────────────────────────────────
        # Positive = stress (high VIX rising + SPY below MA200)
        # Negative = calm  (low VIX falling + SPY above MA200)
        f['market_stress'] = f['vix_roc_20d'] - f['spy_ma200_ratio']

        # Sanitize
        for k, v in f.items():
            try:
                fv = float(v)
                f[k] = 0.0 if (np.isnan(fv) or np.isinf(fv)) else fv
            except (TypeError, ValueError):
                f[k] = 0.0

        return f

    def get_feature_count(self) -> Dict[str, int]:
        """Get feature counts by category"""
        return {
            'technical': 42,
            'fundamental_ratios': 64,
            'fundamental_metrics': 47,
            'premium': 22,
            'macro': 7,
            'total': 182
        }
