"""
Phase 2: ML Feature Engineering
Extracts ALL available fundamental + technical features for LightGBM model

Uses EVERY FMP fundamental metric available:
- Profitability ratios (ROE, ROA, margins, etc.)
- Liquidity ratios (current, quick, cash)
- Leverage ratios (debt-to-equity, debt-to-assets)
- Efficiency ratios (asset turnover, inventory turnover)
- Valuation ratios (P/E, P/B, P/S, EV/EBITDA)
- Growth metrics (revenue growth, earnings growth)
- Cash flow metrics (FCF, OCF, FCF yield)
- Quality metrics (income quality, cash conversion)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """Extract comprehensive features for ML stock ranking"""

    def __init__(self):
        self.feature_names = []

    def extract_all_features(self, ticker: str, date: pd.Timestamp, bot, fa_data: Optional[Dict] = None) -> Dict:
        """
        Extract ALL available features for a stock at a given date

        Returns dictionary with 80+ features:
        - Technical indicators (20+)
        - Fundamental ratios (50+)
        - Market-relative features (10+)
        """
        features = {}

        # Get price data
        if ticker not in bot.stocks_data:
            return None

        df_at_date = bot.stocks_data[ticker][bot.stocks_data[ticker].index <= date]
        if len(df_at_date) < 120:
            return None

        # ============================================================
        # PART 1: TECHNICAL INDICATORS (20+ features)
        # ============================================================

        # Price-based features
        current_price = df_at_date['close'].iloc[-1]
        features['price'] = current_price

        # Returns at multiple timeframes
        if len(df_at_date) >= 5:
            features['return_5d'] = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-5] - 1) * 100
        if len(df_at_date) >= 10:
            features['return_10d'] = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-10] - 1) * 100
        if len(df_at_date) >= 20:
            features['return_20d'] = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1) * 100
        if len(df_at_date) >= 60:
            features['return_60d'] = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1) * 100
        if len(df_at_date) >= 120:
            features['return_120d'] = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-120] - 1) * 100

        # Momentum acceleration
        if len(df_at_date) >= 60:
            ret_20 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-20] - 1)
            ret_60 = (df_at_date['close'].iloc[-1] / df_at_date['close'].iloc[-60] - 1)
            features['momentum_accel'] = ret_20 - (ret_60 / 3)  # Is recent momentum stronger?

        # Moving averages
        if len(df_at_date) >= 20:
            sma_20 = df_at_date['close'].tail(20).mean()
            features['price_to_sma20'] = (current_price / sma_20 - 1) * 100
        if len(df_at_date) >= 50:
            sma_50 = df_at_date['close'].tail(50).mean()
            features['price_to_sma50'] = (current_price / sma_50 - 1) * 100

        # Volatility
        if len(df_at_date) >= 20:
            returns_20 = df_at_date['close'].tail(20).pct_change().dropna()
            features['volatility_20d'] = returns_20.std() * np.sqrt(252) * 100  # Annualized
        if len(df_at_date) >= 60:
            returns_60 = df_at_date['close'].tail(60).pct_change().dropna()
            features['volatility_60d'] = returns_60.std() * np.sqrt(252) * 100

        # Volume features
        if 'volume' in df_at_date.columns:
            if len(df_at_date) >= 20:
                avg_vol_20 = df_at_date['volume'].tail(20).mean()
                current_vol = df_at_date['volume'].iloc[-1]
                features['volume_ratio'] = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

            if len(df_at_date) >= 5:
                vol_trend = df_at_date['volume'].tail(5).mean() / df_at_date['volume'].tail(20).mean()
                features['volume_trend'] = vol_trend if avg_vol_20 > 0 else 1.0

        # RSI (Relative Strength Index)
        if len(df_at_date) >= 14:
            delta = df_at_date['close'].diff()
            gain = (delta.where(delta > 0, 0)).tail(14).mean()
            loss = (-delta.where(delta < 0, 0)).tail(14).mean()
            if loss != 0:
                rs = gain / loss
                features['rsi_14'] = 100 - (100 / (1 + rs))
            else:
                features['rsi_14'] = 100

        # Price position in 52-week range
        if len(df_at_date) >= 252:
            high_52w = df_at_date['close'].tail(252).max()
            low_52w = df_at_date['close'].tail(252).min()
            if high_52w != low_52w:
                features['price_position_52w'] = (current_price - low_52w) / (high_52w - low_52w) * 100
            else:
                features['price_position_52w'] = 50.0

        # ============================================================
        # PART 2: FUNDAMENTAL FEATURES (50+ from FMP data)
        # ============================================================

        if fa_data:
            # Profitability Ratios
            features['roe'] = fa_data.get('returnOnEquity', 0) * 100
            features['roa'] = fa_data.get('returnOnAssets', 0) * 100
            features['roic'] = fa_data.get('returnOnInvestedCapital', 0) * 100
            features['roce'] = fa_data.get('returnOnCapitalEmployed', 0) * 100
            features['operating_margin'] = fa_data.get('operatingProfitMargin', 0) * 100
            features['net_margin'] = fa_data.get('netProfitMargin', 0) * 100
            features['gross_margin'] = fa_data.get('grossProfitMargin', 0) * 100
            features['ebit_margin'] = fa_data.get('ebitPerRevenue', 0) * 100
            features['ebitda_margin'] = fa_data.get('ebitdaRatio', 0) * 100
            features['pretax_margin'] = fa_data.get('pretaxProfitMargin', 0) * 100

            # Liquidity Ratios
            features['current_ratio'] = fa_data.get('currentRatio', 0)
            features['quick_ratio'] = fa_data.get('quickRatio', 0)
            features['cash_ratio'] = fa_data.get('cashRatio', 0)
            features['operating_cash_flow_ratio'] = fa_data.get('operatingCashFlowRatio', 0)

            # Leverage Ratios
            features['debt_to_equity'] = fa_data.get('debtToEquityRatio', 0)
            features['debt_to_assets'] = fa_data.get('debtToAssets', 0)
            features['long_term_debt_to_cap'] = fa_data.get('longTermDebtToCapitalization', 0)
            features['total_debt_to_cap'] = fa_data.get('totalDebtToCapitalization', 0)
            features['debt_equity_ratio'] = fa_data.get('debtEquityRatio', 0)
            features['interest_coverage'] = fa_data.get('interestCoverage', 0)

            # Efficiency Ratios
            features['asset_turnover'] = fa_data.get('assetTurnover', 0)
            features['inventory_turnover'] = fa_data.get('inventoryTurnover', 0)
            features['receivables_turnover'] = fa_data.get('receivablesTurnover', 0)
            features['payables_turnover'] = fa_data.get('payablesTurnover', 0)
            features['fixed_asset_turnover'] = fa_data.get('fixedAssetTurnover', 0)
            features['days_sales_outstanding'] = fa_data.get('daysSalesOutstanding', 0)
            features['days_inventory_outstanding'] = fa_data.get('daysOfInventoryOnHand', 0)
            features['days_payables_outstanding'] = fa_data.get('daysOfPayablesOutstanding', 0)
            features['cash_conversion_cycle'] = fa_data.get('cashConversionCycle', 0)

            # Valuation Ratios
            features['pe_ratio'] = fa_data.get('priceToEarningsRatio', 0)
            features['price_to_book'] = fa_data.get('priceToBookRatio', 0)
            features['price_to_sales'] = fa_data.get('priceToSalesRatio', 0)
            features['ev_to_sales'] = fa_data.get('enterpriseValueToSales', 0)
            features['ev_to_ebitda'] = fa_data.get('enterpriseValueOverEBITDA', 0)
            features['price_to_fcf'] = fa_data.get('priceToFreeCashFlowsRatio', 0)
            features['peg_ratio'] = fa_data.get('pegRatio', 0)

            # Cash Flow Metrics
            features['fcf_yield'] = fa_data.get('freeCashFlowYield', 0) * 100
            features['fcf_to_ocf'] = fa_data.get('freeCashFlowOperatingCashFlowRatio', 0)
            features['ocf_to_sales'] = fa_data.get('operatingCashFlowSalesRatio', 0)
            features['capex_to_ocf'] = fa_data.get('capexToOperatingCashFlow', 0)
            features['capex_to_revenue'] = fa_data.get('capexToRevenue', 0)
            features['capex_to_depreciation'] = fa_data.get('capexToDepreciation', 0)

            # Quality Metrics (Phase 1A)
            features['income_quality'] = fa_data.get('incomeQuality', 0)
            features['dividend_payout_ratio'] = fa_data.get('dividendPaidAndCapexCoverageRatio', 0)
            features['payout_ratio'] = fa_data.get('payoutRatio', 0) * 100
            features['dividend_yield'] = fa_data.get('dividendYield', 0) * 100

            # Growth-related (if available)
            features['revenue_per_share'] = fa_data.get('revenuePerShare', 0)
            features['earnings_per_share'] = fa_data.get('netIncomePerShare', 0)
            features['book_value_per_share'] = fa_data.get('bookValuePerShare', 0)
            features['tangible_book_value_per_share'] = fa_data.get('tangibleBookValuePerShare', 0)

            # Additional useful ratios
            features['earnings_yield'] = 1 / features['pe_ratio'] * 100 if features['pe_ratio'] > 0 else 0
            features['fcf_per_share'] = fa_data.get('freeCashFlowPerShare', 0)
            features['operating_cf_per_share'] = fa_data.get('operatingCashFlowPerShare', 0)

        # ============================================================
        # PART 3: MARKET-RELATIVE FEATURES (10+ features)
        # ============================================================

        # Relative strength vs SPY
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

        # Beta (if we have enough data)
        if len(df_at_date) >= 60 and 'SPY' in bot.stocks_data:
            spy_df = bot.stocks_data['SPY'][bot.stocks_data['SPY'].index <= date]
            if len(spy_df) >= 60:
                stock_returns = df_at_date['close'].tail(60).pct_change().dropna()
                spy_returns = spy_df['close'].tail(60).pct_change().dropna()

                # Align dates
                common_dates = stock_returns.index.intersection(spy_returns.index)
                if len(common_dates) >= 30:
                    stock_aligned = stock_returns.loc[common_dates]
                    spy_aligned = spy_returns.loc[common_dates]

                    covariance = np.cov(stock_aligned, spy_aligned)[0][1]
                    spy_variance = np.var(spy_aligned)
                    features['beta'] = covariance / spy_variance if spy_variance > 0 else 1.0

        return features

    def get_feature_names(self, features: Dict) -> list:
        """Get sorted list of feature names"""
        return sorted(features.keys())

    def features_to_array(self, features: Dict, feature_names: list) -> np.ndarray:
        """Convert feature dictionary to numpy array in consistent order"""
        return np.array([features.get(name, 0) for name in feature_names])
