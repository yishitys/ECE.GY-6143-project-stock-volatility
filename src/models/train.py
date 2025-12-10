"""
训练主脚本

支持数据划分、交叉验证、超参数调优和模型训练。
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import sys

# 添加src目录到路径
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.xgboost_model import XGBoostVolatilityModel

try:
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_data_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    按时间顺序划分数据集（时间序列数据必须保持时间顺序）
    
    Args:
        X: 特征DataFrame
        y: 目标变量Series
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    logger.info(f"数据划分完成:")
    logger.info(f"  训练集: {len(X_train)} 样本 ({len(X_train)/n*100:.1f}%)")
    logger.info(f"  验证集: {len(X_val)} 样本 ({len(X_val)/n*100:.1f}%)")
    logger.info(f"  测试集: {len(X_test)} 样本 ({len(X_test)/n*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def time_series_cross_validation(
    model: XGBoostVolatilityModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Dict[str, List[float]]:
    """
    时间序列交叉验证
    
    Args:
        model: 模型实例
        X: 特征DataFrame
        y: 目标变量Series
        n_splits: 折数
    
    Returns:
        每折的评估指标
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, skipping cross-validation")
        return {}
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    logger.info(f"开始 {n_splits} 折时间序列交叉验证...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"Fold {fold + 1}/{n_splits}")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # 训练模型
        model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, verbose=False)
        
        # 评估
        metrics = model.evaluate(X_val_fold, y_val_fold)
        
        cv_results['rmse'].append(metrics['rmse'])
        cv_results['mae'].append(metrics['mae'])
        cv_results['r2'].append(metrics['r2'])
    
    # 计算平均指标
    logger.info("\n交叉验证结果:")
    logger.info(f"  RMSE: {np.mean(cv_results['rmse']):.6f} ± {np.std(cv_results['rmse']):.6f}")
    logger.info(f"  MAE: {np.mean(cv_results['mae']):.6f} ± {np.std(cv_results['mae']):.6f}")
    logger.info(f"  R²: {np.mean(cv_results['r2']):.4f} ± {np.std(cv_results['r2']):.4f}")
    
    return cv_results


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = 'random',
    n_iter: int = 20
) -> Dict:
    """
    超参数调优
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        method: 调优方法 ('grid' 或 'random')
        n_iter: 随机搜索的迭代次数
    
    Returns:
        最佳参数字典
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, using default parameters")
        return {}
    
    logger.info(f"开始超参数调优 (方法: {method})...")
    
    # 参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    # 创建基础模型
    base_model = XGBoostVolatilityModel()
    
    if method == 'grid':
        # 网格搜索（可能很慢）
        search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
    else:
        # 随机搜索
        search = RandomizedSearchCV(
            base_model.model,
            param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    search.fit(X_train, y_train)
    
    best_params = search.best_params_
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳交叉验证分数: {-search.best_score_:.6f}")
    
    return best_params


def train_model(
    stock_symbol: str,
    features_path: str = None,
    target_col: str = 'target_volatility_log_return_abs',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    model_params: Optional[Dict] = None,
    do_cross_validation: bool = False,
    do_hyperparameter_tuning: bool = False,
    save_model: bool = True,
    model_output_dir: str = 'models'
) -> Tuple[XGBoostVolatilityModel, Dict]:
    """
    完整的模型训练流程
    
    Args:
        stock_symbol: 股票代码
        features_path: 特征数据路径（如果为None则自动生成）
        target_col: 目标变量列名
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        model_params: 模型参数字典
        do_cross_validation: 是否进行交叉验证
        do_hyperparameter_tuning: 是否进行超参数调优
        save_model: 是否保存模型
        model_output_dir: 模型输出目录
    
    Returns:
        (训练好的模型, 评估结果字典)
    """
    logger.info("="*60)
    logger.info(f"开始训练模型: {stock_symbol}")
    logger.info("="*60)
    
    # 加载特征数据
    if features_path is None:
        features_path = f'data/processed/features_{stock_symbol}.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特征数据文件不存在: {features_path}")
    
    logger.info(f"加载特征数据: {features_path}")
    df = pd.read_csv(features_path, parse_dates=['timestamp'])
    logger.info(f"加载了 {len(df)} 条记录")
    
    # 创建模型
    if model_params is None:
        model_params = {}
    
    model = XGBoostVolatilityModel(**model_params)
    
    # 准备特征
    X, y = model.prepare_features(df, target_col=target_col)
    
    # 划分数据集
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
        X, y, train_ratio, val_ratio, test_ratio
    )
    
    # 超参数调优（可选）
    if do_hyperparameter_tuning:
        best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val)
        model = XGBoostVolatilityModel(**best_params)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(
            X, y, train_ratio, val_ratio, test_ratio
        )
    
    # 交叉验证（可选）
    if do_cross_validation:
        time_series_cross_validation(model, X_train, y_train, n_splits=5)
    
    # 训练模型
    logger.info("\n开始训练最终模型...")
    model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=10)
    
    # 评估
    logger.info("\n训练集评估:")
    train_metrics = model.evaluate(X_train, y_train)
    
    logger.info("\n验证集评估:")
    val_metrics = model.evaluate(X_val, y_val)
    
    logger.info("\n测试集评估:")
    test_metrics = model.evaluate(X_test, y_test)
    
    # 特征重要性
    importance = model.get_feature_importance(top_n=20)
    logger.info("\n前20个重要特征:")
    logger.info(importance.to_string())
    
    # 保存模型
    if save_model:
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_output_dir, f'xgboost_{stock_symbol}.pkl')
        model.save(model_path)
    
    # 汇总结果
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'feature_importance': importance.to_dict('records'),
        'model_params': model.params
    }
    
    logger.info("\n" + "="*60)
    logger.info("模型训练完成！")
    logger.info("="*60)
    
    return model, results


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('--symbol', type=str, default='GME', help='股票代码')
    parser.add_argument('--features', type=str, default=None, help='特征数据路径')
    parser.add_argument('--cv', action='store_true', help='进行交叉验证')
    parser.add_argument('--tune', action='store_true', help='进行超参数调优')
    
    args = parser.parse_args()
    
    model, results = train_model(
        stock_symbol=args.symbol,
        features_path=args.features,
        do_cross_validation=args.cv,
        do_hyperparameter_tuning=args.tune
    )
    
    logger.info("训练完成！")


