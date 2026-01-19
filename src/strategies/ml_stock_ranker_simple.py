"""
Simple ML-Based Stock Ranking using sklearn RandomForest
No external dependencies beyond what's already installed
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.error("sklearn not available")
    SKLEARN_AVAILABLE = False


class MLStockRanker:
    """Machine Learning Stock Ranking with RandomForest"""

    def __init__(self, config=None):
        self.config = config or {}
        self.lookback_months = 36
        self.forward_return_days = 21
        self.retrain_frequency = 3

        # Default RF parameters (can be overridden by config)
        self.rf_params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 4),
            'min_samples_split': self.config.get('min_samples_split', 20),
            'min_samples_leaf': self.config.get('min_samples_leaf', 10),
            'max_features': self.config.get('max_features', 'sqrt'),
            'random_state': 42,
            'n_jobs': -1
        }

        self.model = None
        self.feature_names = None
        self.last_train_date = None

    def extract_features(self, ticker: str, df: pd.DataFrame, spy_df: pd.DataFrame = None) -> Dict:
        """Extract features from stock data"""
        if len(df) < 200:
            return None

        latest = df.iloc[-1]
        features = {}

        close = latest['close']
        ema_13 = latest['ema_13']
        ema_34 = latest['ema_34']
        ema_89 = latest['ema_89']

        features['price_to_ema13'] = close / ema_13 if ema_13 > 0 else 1.0
        features['price_to_ema34'] = close / ema_34 if ema_34 > 0 else 1.0
        features['price_to_ema89'] = close / ema_89 if ema_89 > 0 else 1.0
        features['ema13_to_ema34'] = ema_13 / ema_34 if ema_34 > 0 else 1.0
        features['ema34_to_ema89'] = ema_34 / ema_89 if ema_89 > 0 else 1.0

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

        atr = latest['atr']
        features['atr_pct'] = (atr / close) * 100 if close > 0 else 0

        if len(df) >= 252:
            high_52w = df['high'].iloc[-252:].max()
            features['dist_52w_high'] = ((close / high_52w) - 1) * 100
        else:
            features['dist_52w_high'] = -50

        if spy_df is not None and len(spy_df) >= 60 and len(df) >= 60:
            stock_ret = (close / df.iloc[-60]['close'] - 1) * 100
            spy_ret = (spy_df.iloc[-1]['close'] / spy_df.iloc[-60]['close'] - 1) * 100
            features['rel_strength_60'] = stock_ret - spy_ret
        else:
            features['rel_strength_60'] = 0

        return features

    def calculate_forward_return(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate forward return"""
        if idx + self.forward_return_days >= len(df):
            return None
        curr = df.iloc[idx]['close']
        future = df.iloc[idx + self.forward_return_days]['close']
        return ((future / curr) - 1) * 100

    def prepare_training_data(self, stock_data: Dict, spy_df: pd.DataFrame, end_date: pd.Timestamp) -> Tuple:
        """Prepare training data with reduced temporal correlation"""
        all_features = []
        all_labels = []

        # Sample every 63 days (3 months) instead of 21 days to reduce temporal correlation
        sample_interval = self.config.get('sample_interval', 63)

        for ticker, df in stock_data.items():
            df_train = df[df.index <= end_date].copy()
            if len(df_train) < 200:
                continue

            # Sample at wider intervals to reduce overfitting
            sample_indices = range(200, len(df_train) - self.forward_return_days, sample_interval)

            for idx in sample_indices:
                df_slice = df_train.iloc[:idx+1]
                spy_slice = spy_df[spy_df.index <= df_train.iloc[idx].name] if spy_df is not None else None

                features = self.extract_features(ticker, df_slice, spy_slice)
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

    def train(self, stock_data: Dict, spy_df: pd.DataFrame, train_end_date: pd.Timestamp):
        """Train model with validation to prevent overfitting"""
        logger.info(f"Training model up to {train_end_date.date()}")
        X, y = self.prepare_training_data(stock_data, spy_df, train_end_date)

        # Time-based train/validation split (80/20) to detect overfitting
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train model
        self.model = RandomForestRegressor(**self.rf_params)
        self.model.fit(X_train, y_train)

        # Validate to check overfitting
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        logger.info(f"Model trained successfully")
        logger.info(f"  Train R²: {train_score:.3f}")
        logger.info(f"  Val R²:   {val_score:.3f}")

        # Warn if severe overfitting detected
        if train_score - val_score > 0.2:
            logger.warning(f"⚠️  Overfitting detected! Train-Val gap: {train_score - val_score:.3f}")

        self.last_train_date = train_end_date
        return self.model

    def predict_scores(self, stock_features: Dict[str, Dict]) -> Dict[str, float]:
        """Predict scores"""
        if self.model is None:
            raise ValueError("Model not trained")

        scores = {}
        for ticker, features in stock_features.items():
            if features is None:
                scores[ticker] = -999
                continue
            try:
                feat_df = pd.DataFrame([features])[self.feature_names]
                pred = self.model.predict(feat_df)[0]
                scores[ticker] = pred
            except:
                scores[ticker] = -999
        return scores

    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        """Check if should retrain"""
        if self.last_train_date is None:
            return True
        months = (current_date.year - self.last_train_date.year) * 12 + \
                 (current_date.month - self.last_train_date.month)
        return months >= self.retrain_frequency
