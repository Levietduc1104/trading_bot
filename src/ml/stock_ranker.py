"""
Phase 2 ML: LightGBM Stock Ranker with Maximum Overfitting Prevention

Anti-Overfitting Measures:
1. Feature Selection - Use only top 40-50 features by importance
2. L1/L2 Regularization - lambda_l1=0.5, lambda_l2=0.5
3. Conservative Tree Parameters - max_depth=4, num_leaves=15
4. High Min Samples - min_child_samples=100 (prevents tiny splits)
5. Feature/Bagging Fraction - 0.8 (dropout)
6. Early Stopping - Stop if validation doesn't improve for 50 rounds
7. Walk-Forward Validation - Always test on future data
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import RobustScaler
import warnings
import pickle
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLStockRanker:
    """
    LightGBM-based stock ranker with aggressive overfitting prevention

    Key Features:
    - Train/test split by time (walk-forward)
    - Feature importance selection
    - Strong regularization
    - Conservative hyperparameters
    """

    def __init__(self, n_features_to_select=50, look_ahead_days=21):
        """
        Initialize ML ranker

        Args:
            n_features_to_select: Max features to use (reduces overfitting)
            look_ahead_days: Days to predict forward return (target variable)
        """
        self.n_features_to_select = n_features_to_select
        self.look_ahead_days = look_ahead_days

        # Model and preprocessing
        self.model = None
        self.scaler = RobustScaler()  # Robust to outliers
        self.selected_features = None
        self.feature_importance = None

        # Training metadata
        self.train_metrics = {}
        self.validation_metrics = {}

        # Anti-overfitting hyperparameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',

            # Conservative tree structure
            'num_leaves': 15,  # Low (prevents complex trees)
            'max_depth': 4,  # Shallow trees

            # Regularization
            'lambda_l1': 0.5,  # L1 regularization
            'lambda_l2': 0.5,  # L2 regularization
            'min_child_samples': 100,  # High minimum (prevents overfitting to small groups)
            'min_child_weight': 0.01,  # Minimum sum of instance weight

            # Stochastic elements (dropout)
            'feature_fraction': 0.8,  # Use 80% of features per tree
            'bagging_fraction': 0.8,  # Use 80% of data per iteration
            'bagging_freq': 5,  # Bagging frequency

            # Learning rate
            'learning_rate': 0.01,  # Very conservative (slower learning = less overfitting)
            'num_boost_round': 1000,  # Many trees with small learning rate

            # Other
            'verbose': -1,
            'seed': 42,
            'deterministic': True,
        }

    def prepare_training_data(self, features_df: pd.DataFrame, returns_df: pd.DataFrame,
                             train_start: str, train_end: str,
                             val_start: str, val_end: str) -> Tuple:
        """
        Prepare training and validation data with temporal split

        Args:
            features_df: DataFrame with features (index=date, columns=tickers)
            returns_df: DataFrame with forward returns
            train_start, train_end: Training period
            val_start, val_end: Validation period

        Returns:
            X_train, y_train, X_val, y_val
        """
        # Filter to training period
        train_features = features_df.loc[train_start:train_end]
        train_returns = returns_df.loc[train_start:train_end]

        # Filter to validation period
        val_features = features_df.loc[val_start:val_end]
        val_returns = returns_df.loc[val_start:val_end]

        # Flatten to samples
        X_train = []
        y_train = []
        X_val = []
        y_val = []

        # Training data
        for date in train_features.index:
            for ticker in train_features.columns:
                feature_dict = train_features.loc[date, ticker]
                if feature_dict is not None and not pd.isna(train_returns.loc[date, ticker]):
                    X_train.append(list(feature_dict.values()))
                    y_train.append(train_returns.loc[date, ticker])

        # Validation data
        for date in val_features.index:
            for ticker in val_features.columns:
                feature_dict = val_features.loc[date, ticker]
                if feature_dict is not None and not pd.isna(val_returns.loc[date, ticker]):
                    X_val.append(list(feature_dict.values()))
                    y_val.append(val_returns.loc[date, ticker])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        return X_train, y_train, X_val, y_val

    def select_features_by_importance(self, X_train: np.ndarray, y_train: np.ndarray,
                                     feature_names: List[str]) -> List[str]:
        """
        Select top N features by training a simple model and using feature importance

        This reduces overfitting by eliminating noisy/irrelevant features
        """
        logger.info(f"Selecting top {self.n_features_to_select} features from {len(feature_names)}...")

        # Train a quick model for feature importance
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

        # Simple model just for feature selection
        simple_params = {
            'objective': 'regression',
            'num_leaves': 15,
            'max_depth': 3,
            'learning_rate': 0.05,
            'verbose': -1
        }

        # Train without early stopping for feature selection
        model_for_selection = lgb.train(
            simple_params,
            dtrain,
            num_boost_round=100
        )

        # Get feature importance
        importance = model_for_selection.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Select top N features
        selected = feature_importance_df.head(self.n_features_to_select)['feature'].tolist()

        logger.info(f"✅ Selected {len(selected)} features")
        logger.info(f"   Top 10: {selected[:10]}")

        self.feature_importance = feature_importance_df
        return selected

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             feature_names: List[str]) -> Dict:
        """
        Train LightGBM model with overfitting prevention

        Returns:
            Dict with training metrics
        """
        logger.info(f"Training ML model on {len(X_train)} samples...")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Features: {X_train.shape[1]}")

        # Feature selection (CRITICAL for preventing overfitting)
        if X_train.shape[1] > self.n_features_to_select:
            self.selected_features = self.select_features_by_importance(X_train, y_train, feature_names)

            # Filter to selected features
            feature_indices = [feature_names.index(f) for f in self.selected_features]
            X_train = X_train[:, feature_indices]
            X_val = X_val[:, feature_indices]
            feature_names = self.selected_features
        else:
            self.selected_features = feature_names

        # Scale features (helps with regularization)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val_scaled, label=y_val, feature_name=feature_names, reference=dtrain)

        # Train with early stopping (CRITICAL for preventing overfitting)
        logger.info("Training with early stopping...")
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['num_boost_round'],
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=150),  # Stop if no improvement for 150 rounds
                lgb.log_evaluation(period=100)  # Log every 100 rounds
            ]
        )

        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))

        train_corr = np.corrcoef(train_pred, y_train)[0, 1]
        val_corr = np.corrcoef(val_pred, y_val)[0, 1]

        # Overfitting check (train vs validation difference)
        overfitting_ratio = val_rmse / train_rmse

        metrics = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_corr': train_corr,
            'val_corr': val_corr,
            'overfitting_ratio': overfitting_ratio,
            'best_iteration': self.model.best_iteration,
            'n_features_used': len(self.selected_features)
        }

        self.train_metrics = metrics

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Train RMSE: {train_rmse:.4f}")
        logger.info(f"Val RMSE:   {val_rmse:.4f}")
        logger.info(f"Train Corr: {train_corr:.4f}")
        logger.info(f"Val Corr:   {val_corr:.4f}")
        logger.info(f"Overfitting Ratio: {overfitting_ratio:.3f}")

        if overfitting_ratio > 1.3:
            logger.warning(f"⚠️  OVERFITTING DETECTED! Validation RMSE is {overfitting_ratio:.1f}x training RMSE")
        elif overfitting_ratio > 1.15:
            logger.warning(f"⚠️  Moderate overfitting. Validation RMSE is {overfitting_ratio:.1f}x training RMSE")
        else:
            logger.info(f"✅ Good generalization. Validation RMSE is {overfitting_ratio:.1f}x training RMSE")

        logger.info(f"Best iteration: {self.model.best_iteration}")
        logger.info(f"Features used: {len(self.selected_features)}")
        logger.info(f"{'='*60}\n")

        return metrics

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict forward return for a stock given its features

        Args:
            features: Dict of feature name -> value

        Returns:
            Predicted forward return (%)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Extract features in correct order
        if self.selected_features:
            feature_values = [features.get(f, 0) for f in self.selected_features]
        else:
            feature_values = list(features.values())

        # Scale
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict
        pred = self.model.predict(X_scaled)[0]
        return pred

    def get_feature_importance(self, top_n=20) -> pd.DataFrame:
        """Get top N most important features"""
        if self.model is None:
            return None

        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    def save_model(self, path: str):
        """Save model and all associated objects to disk"""
        if self.model:
            # Save LightGBM model
            self.model.save_model(path)

            # Save scaler, selected features, and metrics as pickle
            metadata_path = path.replace('.txt', '_metadata.pkl')
            metadata = {
                'scaler': self.scaler,
                'selected_features': self.selected_features,
                'train_metrics': self.train_metrics,
                'feature_importance': self.feature_importance,
                'n_features_to_select': self.n_features_to_select,
                'look_ahead_days': self.look_ahead_days
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"✅ Model saved to {path}")
            logger.info(f"✅ Metadata saved to {metadata_path}")

    def load_model(self, path: str):
        """Load model and all associated objects from disk"""
        # Load LightGBM model
        self.model = lgb.Booster(model_file=path)

        # Load scaler, selected features, and metrics
        metadata_path = path.replace('.txt', '_metadata.pkl')
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.scaler = metadata['scaler']
            self.selected_features = metadata['selected_features']
            self.train_metrics = metadata.get('train_metrics', {})
            self.feature_importance = metadata.get('feature_importance', None)
            self.n_features_to_select = metadata.get('n_features_to_select', 50)
            self.look_ahead_days = metadata.get('look_ahead_days', 21)

            logger.info(f"✅ Model loaded from {path}")
            logger.info(f"✅ Metadata loaded from {metadata_path}")
        except FileNotFoundError:
            logger.warning(f"⚠️  Metadata file not found: {metadata_path}")
            logger.warning("⚠️  Creating new scaler - predictions may not work correctly!")
            self.scaler = RobustScaler()
            self.selected_features = None
