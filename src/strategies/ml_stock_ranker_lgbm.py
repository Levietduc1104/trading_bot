"""
LightGBM-Based Stock Ranking
Enhanced ML model using gradient boosting instead of RandomForest
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    logger.error("LightGBM not available. Install: pip install lightgbm")
    LGBM_AVAILABLE = False


class LGBMStockRanker:
    """Machine Learning Stock Ranking with LightGBM"""

    def __init__(self, config=None):
        self.config = config or {}
        self.lookback_months = 36
        self.forward_return_days = 21
        self.retrain_frequency = 3

        # LightGBM parameters (optimized for stock prediction)
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.get('num_leaves', 31),  # Default 31, lower = less overfitting
            'max_depth': self.config.get('max_depth', 5),      # Limit tree depth
            'learning_rate': self.config.get('learning_rate', 0.05),  # Lower = more stable
            'n_estimators': self.config.get('n_estimators', 100),
            'min_child_samples': self.config.get('min_child_samples', 20),  # Regularization
            'subsample': self.config.get('subsample', 0.8),    # Use 80% of data per tree
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),  # Use 80% of features
            'reg_alpha': self.config.get('reg_alpha', 0.1),    # L1 regularization
            'reg_lambda': self.config.get('reg_lambda', 0.1),  # L2 regularization
            'random_state': 42,
            'verbose': -1,  # Suppress warnings
            'force_col_wise': True,
        }

        self.model = None
        self.feature_names = None
        self.last_train_date = None
        self.feature_importance = None

    def extract_features(self, ticker: str, df: pd.DataFrame, spy_df: pd.DataFrame = None, metadata: Dict = None) -> Dict:
        """Extract features from stock data (technical + fundamental)"""
        if len(df) < 200:
            return None

        latest = df.iloc[-1]
        features = {}

        close = latest['close']
        ema_13 = latest['ema_13']
        ema_34 = latest['ema_34']
        ema_89 = latest['ema_89']

        # EMA features
        features['price_to_ema13'] = close / ema_13 if ema_13 > 0 else 1.0
        features['price_to_ema34'] = close / ema_34 if ema_34 > 0 else 1.0
        features['price_to_ema89'] = close / ema_89 if ema_89 > 0 else 1.0
        features['ema13_to_ema34'] = ema_13 / ema_34 if ema_34 > 0 else 1.0
        features['ema34_to_ema89'] = ema_34 / ema_89 if ema_89 > 0 else 1.0

        # Momentum features
        features['roc_20'] = latest['roc_20']
        features['roc_60'] = latest.get('roc_60', 0)
        features['rsi'] = latest['rsi']

        if len(df) >= 5:
            features['roc_5'] = ((close / df.iloc[-5]['close']) - 1) * 100
        else:
            features['roc_5'] = 0

        if len(df) >= 10:
            features['roc_10'] = ((close / df.iloc[-10]['close']) - 1) * 100
        else:
            features['roc_10'] = 0

        # Volatility
        atr = latest['atr']
        features['atr_pct'] = (atr / close) * 100 if close > 0 else 0

        # 52-week high distance
        if len(df) >= 252:
            high_52w = df['high'].iloc[-252:].max()
            features['dist_52w_high'] = ((close / high_52w) - 1) * 100
        else:
            features['dist_52w_high'] = -50

        # Relative strength vs SPY
        if spy_df is not None and len(spy_df) >= 60 and len(df) >= 60:
            stock_ret = (close / df.iloc[-60]['close'] - 1) * 100
            spy_ret = (spy_df.iloc[-1]['close'] / spy_df.iloc[-60]['close'] - 1) * 100
            features['rel_strength_60'] = stock_ret - spy_ret
        else:
            features['rel_strength_60'] = 0

        # ====== MARKET REGIME FEATURES (NEW) ======
        if spy_df is not None and len(spy_df) >= 200:
            spy_close = spy_df.iloc[-1]['close']

            # SPY vs 200-day MA (bull/bear detector)
            spy_200ma = spy_df['close'].rolling(200).mean().iloc[-1]
            if spy_200ma > 0:
                features['spy_to_200ma'] = ((spy_close / spy_200ma) - 1) * 100
                features['bull_market'] = 1 if spy_close > spy_200ma else 0
            else:
                features['spy_to_200ma'] = 0
                features['bull_market'] = 1

            # SPY momentum (market trend strength)
            if len(spy_df) >= 20:
                spy_20d_ret = ((spy_close / spy_df.iloc[-20]['close']) - 1) * 100
                features['spy_momentum_20d'] = spy_20d_ret
            else:
                features['spy_momentum_20d'] = 0
        else:
            features['spy_to_200ma'] = 0
            features['bull_market'] = 1
            features['spy_momentum_20d'] = 0

        # ====== VOLUME FEATURES (NEW) ======
        if 'volume' in df.columns and len(df) >= 50:
            volume = latest['volume']

            # Volume ratio: current vs 20-day average (volume surge detector)
            vol_20ma = df['volume'].rolling(20).mean().iloc[-1]
            if vol_20ma > 0:
                features['volume_ratio'] = volume / vol_20ma
            else:
                features['volume_ratio'] = 1.0

            # Volume trend: 10-day avg vs 50-day avg (sustained volume change)
            vol_10ma = df['volume'].rolling(10).mean().iloc[-1]
            vol_50ma = df['volume'].rolling(50).mean().iloc[-1]
            if vol_50ma > 0:
                features['volume_trend'] = vol_10ma / vol_50ma
            else:
                features['volume_trend'] = 1.0

            # Dollar volume: liquidity measure (in billions)
            features['dollar_volume'] = (close * volume) / 1e9

            # Volume surge flag: >2x average volume
            features['volume_surge'] = 1 if features['volume_ratio'] > 2.0 else 0

            # Volume consistency: std/mean (lower = more consistent)
            vol_std = df['volume'].rolling(20).std().iloc[-1]
            if vol_20ma > 0:
                features['volume_consistency'] = vol_std / vol_20ma
            else:
                features['volume_consistency'] = 1.0
        else:
            # No volume data - use defaults
            features['volume_ratio'] = 1.0
            features['volume_trend'] = 1.0
            features['dollar_volume'] = 1.0
            features['volume_surge'] = 0
            features['volume_consistency'] = 1.0

        # ====== FUNDAMENTAL FEATURES (NEW) ======
        if metadata:
            # Calculate P/E ratio from current price and EPS
            # Note: This is a PROXY because we use current EPS for all historical dates
            # It works for relative ranking (high vs low P/E stocks)
            eps = metadata.get('eps', None)
            # Convert to float if it's a string
            if eps is not None:
                try:
                    eps = float(eps)
                except (ValueError, TypeError):
                    eps = None

            if eps and eps > 0:
                pe_calculated = close / eps  # Current price / Current EPS
                # Handle invalid P/E (inf or nan)
                if np.isfinite(pe_calculated) and pe_calculated > 0:
                    features['pe_ratio'] = pe_calculated
                    features['log_pe'] = np.log(pe_calculated + 1)  # Log transform for better distribution
                    features['pe_percentile'] = min(pe_calculated / 50.0, 2.0)  # Normalize: 50 = 100%
                else:
                    features['pe_ratio'] = 30
                    features['log_pe'] = np.log(31)
                    features['pe_percentile'] = 0.6
                features['eps'] = eps
                features['has_earnings'] = 1
            else:
                # No earnings data available - use defaults
                features['pe_ratio'] = 30  # Market average default
                features['log_pe'] = np.log(31)
                features['pe_percentile'] = 0.6
                features['eps'] = 0
                features['has_earnings'] = 0

            # Risk metrics
            beta = metadata.get('beta', None)
            if beta is not None:
                try:
                    beta = float(beta)
                    if np.isfinite(beta):
                        features['beta'] = beta
                    else:
                        features['beta'] = 1.0
                except (ValueError, TypeError):
                    features['beta'] = 1.0
            else:
                features['beta'] = 1.0

            # Income metrics
            div_yield = metadata.get('dividend_yield', None)
            if div_yield is not None:
                try:
                    div_yield = float(div_yield)
                    if np.isfinite(div_yield):
                        features['dividend_yield'] = div_yield
                    else:
                        features['dividend_yield'] = 0
                except (ValueError, TypeError):
                    features['dividend_yield'] = 0
            else:
                features['dividend_yield'] = 0

            # Market cap (size factor)
            mcap = metadata.get('market_cap', None)
            if mcap is not None:
                try:
                    mcap = float(mcap)
                    if np.isfinite(mcap) and mcap > 0:
                        features['log_mcap'] = np.log(mcap + 1)  # Log transform
                        # Categorize: mega (>200B), large (50-200B), mid (<50B)
                        features['is_megacap'] = 1 if mcap > 200e9 else 0
                    else:
                        features['log_mcap'] = np.log(100e9)
                        features['is_megacap'] = 0
                except (ValueError, TypeError):
                    features['log_mcap'] = np.log(100e9)
                    features['is_megacap'] = 0
            else:
                features['log_mcap'] = np.log(100e9)  # Default ~100B
                features['is_megacap'] = 0

            # Additional fundamental metrics from metadata
            # Annual volatility (risk metric)
            annual_vol = metadata.get('annual_volatility_pct', None)
            if annual_vol is not None:
                try:
                    annual_vol = float(annual_vol)
                    if np.isfinite(annual_vol) and annual_vol > 0:
                        features['annual_volatility'] = annual_vol
                    else:
                        features['annual_volatility'] = 30.0  # Market average
                except (ValueError, TypeError):
                    features['annual_volatility'] = 30.0
            else:
                features['annual_volatility'] = 30.0

            # Price vs analyst target (valuation signal)
            target_price = metadata.get('target_est', None)
            if target_price is not None:
                try:
                    target_price = float(target_price)
                    if np.isfinite(target_price) and target_price > 0:
                        features['price_to_target'] = close / target_price
                        features['upside_to_target'] = ((target_price / close) - 1) * 100
                    else:
                        features['price_to_target'] = 1.0
                        features['upside_to_target'] = 0
                except (ValueError, TypeError):
                    features['price_to_target'] = 1.0
                    features['upside_to_target'] = 0
            else:
                features['price_to_target'] = 1.0
                features['upside_to_target'] = 0

            # Derived features
            # PEG proxy: PE / momentum (if high momentum, higher PE is justified)
            if features['pe_ratio'] > 0 and features['roc_60'] != 0:
                features['pe_to_momentum'] = features['pe_ratio'] / abs(features['roc_60'] + 1)
            else:
                features['pe_to_momentum'] = 1.0
        else:
            # No metadata available - use neutral defaults
            features['pe_ratio'] = 30
            features['log_pe'] = np.log(31)
            features['pe_percentile'] = 0.6
            features['eps'] = 0
            features['has_earnings'] = 0
            features['beta'] = 1.0
            features['dividend_yield'] = 0
            features['log_mcap'] = np.log(100e9)
            features['is_megacap'] = 0
            features['annual_volatility'] = 30.0
            features['price_to_target'] = 1.0
            features['upside_to_target'] = 0
            features['pe_to_momentum'] = 1.0

        return features

    def calculate_forward_return(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate forward return - THIS IS THE LABEL"""
        if idx + self.forward_return_days >= len(df):
            return None
        curr = df.iloc[idx]['close']
        future = df.iloc[idx + self.forward_return_days]['close']
        return ((future / curr) - 1) * 100

    def prepare_training_data(self, stock_data: Dict, spy_df: pd.DataFrame, end_date: pd.Timestamp, metadata: Dict = None) -> Tuple:
        """Prepare training data with reduced temporal correlation"""
        all_features = []
        all_labels = []

        # Sample every 63 days (3 months) to reduce temporal correlation
        sample_interval = self.config.get('sample_interval', 63)

        for ticker, df in stock_data.items():
            df_train = df[df.index <= end_date].copy()
            if len(df_train) < 200:
                continue

            # Sample at wider intervals to reduce overfitting
            sample_indices = range(200, len(df_train) - self.forward_return_days, sample_interval)

            # Get metadata for this ticker
            ticker_metadata = metadata.get(ticker) if metadata else None

            for idx in sample_indices:
                df_slice = df_train.iloc[:idx+1]
                spy_slice = spy_df[spy_df.index <= df_train.iloc[idx].name] if spy_df is not None else None

                features = self.extract_features(ticker, df_slice, spy_slice, ticker_metadata)
                if features is None:
                    continue

                fwd_ret = self.calculate_forward_return(df_train, idx)
                if fwd_ret is None:
                    continue

                all_features.append(features)
                all_labels.append(fwd_ret)

        if len(all_features) == 0:
            raise ValueError("No training data")

        features_df = pd.DataFrame(all_features)
        labels_array = np.array(all_labels)
        self.feature_names = features_df.columns.tolist()

        logger.info(f"Training data: {len(features_df)} samples, {len(self.feature_names)} features")
        return features_df, labels_array

    def train(self, stock_data: Dict, spy_df: pd.DataFrame, train_end_date: pd.Timestamp, metadata: Dict = None):
        """Train LightGBM model with validation to prevent overfitting"""
        logger.info(f"Training LightGBM model up to {train_end_date.date()}")
        X, y = self.prepare_training_data(stock_data, spy_df, train_end_date, metadata)

        # Time-based train/validation split (80/20) to detect overfitting
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0)  # Suppress iteration logs
        ]

        self.model = lgb.train(
            self.lgbm_params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )

        # Validate to check overfitting
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        # Calculate R² scores
        train_r2 = 1 - (np.sum((y_train - train_pred)**2) / np.sum((y_train - y_train.mean())**2))
        val_r2 = 1 - (np.sum((y_val - val_pred)**2) / np.sum((y_val - y_val.mean())**2))

        # Calculate RMSE
        train_rmse = np.sqrt(np.mean((y_train - train_pred)**2))
        val_rmse = np.sqrt(np.mean((y_val - val_pred)**2))

        logger.info(f"Model trained successfully")
        logger.info(f"  Train R²: {train_r2:.3f}  RMSE: {train_rmse:.2f}")
        logger.info(f"  Val R²:   {val_r2:.3f}  RMSE: {val_rmse:.2f}")

        # Get feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importance()))
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  Top features: {', '.join([f[0] for f in top_features])}")

        # Warn if severe overfitting detected
        if train_r2 - val_r2 > 0.2:
            logger.warning(f"⚠️  Overfitting detected! Train-Val gap: {train_r2 - val_r2:.3f}")

        self.last_train_date = train_end_date
        return self.model

    def predict_scores(self, stock_features: Dict[str, Dict]) -> Dict[str, float]:
        """Predict scores for stocks"""
        if self.model is None:
            raise ValueError("Model not trained")

        scores = {}
        for ticker, features in stock_features.items():
            feature_df = pd.DataFrame([features])[self.feature_names]
            score = self.model.predict(feature_df)[0]
            scores[ticker] = score

        return scores

    def should_retrain(self, current_date):
        """Check if model should be retrained"""
        if self.last_train_date is None:
            return True

        months_diff = (current_date.year - self.last_train_date.year) * 12 + \
                     (current_date.month - self.last_train_date.month)

        return months_diff >= self.config.get('retrain_months', 6)
