"""
XGBoost Baseline Model

Implements XGBoost regression model for volatility prediction.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from typing import Optional, Dict, Tuple, List
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Please install it: pip install xgboost")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostVolatilityModel:
    """XGBoost volatility prediction model"""
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model
        
        Args:
            params: XGBoost parameter dictionary (overrides other parameters if provided)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install it: pip install xgboost")
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        logger.info(f"XGBoost model initialized with parameters: {self.params}")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_volatility_log_return_abs',
        exclude_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable
        
        Args:
            df: Feature DataFrame
            target_col: Target variable column name
            exclude_cols: List of columns to exclude
        
        Returns:
            (Feature DataFrame, Target variable Series)
        """
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'stock_symbol', 'has_reddit_data', 'has_stock_data']
        
        # Add target variable to exclusion list
        exclude_cols = exclude_cols + [col for col in df.columns if col.startswith('target_')]
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Features prepared: {len(feature_cols)} features, {len(X)} samples")
        logger.info(f"Target variable statistics: mean={y.mean():.6f}, std={y.std():.6f}")
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds: Number of rounds for early stopping
            verbose: Whether to show training progress
        """
        logger.info("Starting XGBoost model training...")
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Create model (XGBoost 3.x: early_stopping_rounds goes in constructor)
        model_params = self.params.copy()
        if eval_set and early_stopping_rounds:
            model_params['early_stopping_rounds'] = early_stopping_rounds
        self.model = xgb.XGBRegressor(**model_params)
        
        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Model training completed")
        logger.info(f"Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.params['n_estimators']}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call train() first")
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X: Feature DataFrame
            y: Ground truth target values
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        }
        
        # Directional accuracy (predicting increase/decrease in volatility)
        # Note: Cannot use pandas index alignment for comparison (causes many NaNs), use position-aligned numpy diff
        y_arr = np.asarray(y)
        y_diff = np.diff(y_arr)
        y_pred_diff = np.diff(np.asarray(y_pred))
        if len(y_diff) > 0:
            directional_accuracy = (np.sign(y_diff) == np.sign(y_pred_diff)).mean() * 100
        else:
            directional_accuracy = 0.0
        metrics['directional_accuracy'] = directional_accuracy
        
        logger.info("Model evaluation results:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Directional accuracy: {metrics['directional_accuracy']:.2f}%")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            top_n: Return top N important features
        
        Returns:
            Feature importance DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
        
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str):
        """
        Save the model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostVolatilityModel':
        """
        Load the model
        
        Args:
            filepath: Path to the model file
        
        Returns:
            XGBoostVolatilityModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance


if __name__ == '__main__':
    # Test code
    import sys
    import os
    
    # Add src directory to path
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Load feature data
    data_path = 'data/processed/features_GME.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"Data loaded: {len(df)} records")
        
        # Create model
        model = XGBoostVolatilityModel(
            n_estimators=50,  # Use smaller value for testing
            max_depth=5,
            learning_rate=0.1
        )
        
        # 准备特征
        X, y = model.prepare_features(df)
        
        # 划分训练集和测试集（按时间顺序）
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 进一步划分验证集
        val_split_idx = int(len(X_train) * 0.8)
        X_train_final, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
        y_train_final, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
        
        # 训练
        model.train(X_train_final, y_train_final, X_val, y_val)
        
        # 评估
        train_metrics = model.evaluate(X_train_final, y_train_final)
        test_metrics = model.evaluate(X_test, y_test)
        
        # 特征重要性
        importance = model.get_feature_importance(top_n=10)
        logger.info("\n前10个重要特征:")
        logger.info(importance)
        
        # 保存模型
        model_path = 'models/xgboost_GME_test.pkl'
        model.save(model_path)
        
        logger.info("测试完成！")
    else:
        logger.warning(f"测试数据文件不存在: {data_path}")


