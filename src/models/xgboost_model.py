"""
XGBoost基线模型

实现XGBoost回归模型用于波动率预测。
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostVolatilityModel:
    """XGBoost波动率预测模型"""
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        初始化XGBoost模型
        
        Args:
            params: XGBoost参数字典（如果提供则覆盖其他参数）
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            random_state: 随机种子
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install it: pip install xgboost")
        
        # 默认参数
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
        
        logger.info(f"XGBoost模型初始化完成，参数: {self.params}")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_volatility_log_return_abs',
        exclude_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和目标变量
        
        Args:
            df: 特征DataFrame
            target_col: 目标变量列名
            exclude_cols: 要排除的列名列表
        
        Returns:
            (特征DataFrame, 目标变量Series)
        """
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'stock_symbol', 'has_reddit_data', 'has_stock_data']
        
        # 添加目标变量到排除列表
        exclude_cols = exclude_cols + [col for col in df.columns if col.startswith('target_')]
        
        # 选择特征列
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 删除包含NaN的行
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"准备特征: {len(feature_cols)} 个特征, {len(X)} 个样本")
        logger.info(f"目标变量统计: mean={y.mean():.6f}, std={y.std():.6f}")
        
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
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
        """
        logger.info("开始训练XGBoost模型...")
        
        # 创建模型
        self.model = xgb.XGBRegressor(**self.params)
        
        # 准备验证集
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # 训练
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=verbose
        )
        
        # 计算特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("模型训练完成")
        logger.info(f"最佳迭代次数: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.params['n_estimators']}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征DataFrame
        
        Returns:
            预测值数组
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征DataFrame
            y: 真实目标值
        
        Returns:
            评估指标字典
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        }
        
        # 方向准确率（预测波动率增加/减少的方向）
        y_diff = y.diff()
        y_pred_diff = pd.Series(y_pred).diff()
        directional_accuracy = (np.sign(y_diff) == np.sign(y_pred_diff)).mean() * 100
        metrics['directional_accuracy'] = directional_accuracy
        
        logger.info("模型评估结果:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  方向准确率: {metrics['directional_accuracy']:.2f}%")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            top_n: 返回前N个重要特征
        
        Returns:
            特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("模型尚未训练")
        
        return self.feature_importance.head(top_n)
    
    def save(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和元数据
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostVolatilityModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        
        Returns:
            XGBoostVolatilityModel实例
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建模型实例
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.feature_importance = model_data['feature_importance']
        
        logger.info(f"模型已从 {filepath} 加载")
        
        return instance


if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    
    # 添加src目录到路径
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 加载特征数据
    data_path = 'data/processed/features_GME.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"加载数据: {len(df)} 条记录")
        
        # 创建模型
        model = XGBoostVolatilityModel(
            n_estimators=50,  # 测试时使用较小的值
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


