"""
模型评估模块

计算各种评估指标：RMSE, MAE, R², MAPE, 方向准确率等。
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path

try:
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
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


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算回归评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        评估指标字典
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for evaluation")
    
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # 移除NaN值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        logger.warning("没有有效数据用于评估")
        return {}
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    
    # 方向准确率
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = (np.sign(y_true_diff) == np.sign(y_pred_diff)).mean() * 100
        metrics['directional_accuracy'] = directional_accuracy
    else:
        metrics['directional_accuracy'] = 0.0
    
    # 相关系数
    if len(y_true) > 1:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['correlation'] = correlation
    else:
        metrics['correlation'] = 0.0
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率（用于计算AUC）
    
    Returns:
        评估指标字典
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for evaluation")
    
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    if y_pred_proba is not None:
        y_pred_proba = y_pred_proba[:min_len]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # AUC（如果有概率预测）
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if len(np.unique(y_true)) == 2:
                # 二分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                # 多分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except Exception as e:
            logger.warning(f"无法计算AUC: {e}")
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'regression',
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    评估模型（自动选择回归或分类指标）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        task_type: 任务类型 ('regression' 或 'classification')
        y_pred_proba: 预测概率（仅用于分类任务）
    
    Returns:
        评估指标字典
    """
    if task_type == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    elif task_type == 'classification':
        return calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    else:
        raise ValueError(f"未知的任务类型: {task_type}")


def print_metrics(metrics: Dict[str, float], task_type: str = 'regression'):
    """
    打印评估指标
    
    Args:
        metrics: 评估指标字典
        task_type: 任务类型
    """
    logger.info("="*60)
    logger.info("模型评估结果")
    logger.info("="*60)
    
    if task_type == 'regression':
        logger.info(f"RMSE: {metrics.get('rmse', 0):.6f}")
        logger.info(f"MAE: {metrics.get('mae', 0):.6f}")
        logger.info(f"R²: {metrics.get('r2', 0):.4f}")
        logger.info(f"MAPE: {metrics.get('mape', 0):.2f}%")
        logger.info(f"方向准确率: {metrics.get('directional_accuracy', 0):.2f}%")
        logger.info(f"相关系数: {metrics.get('correlation', 0):.4f}")
    else:
        logger.info(f"准确率: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"精确率: {metrics.get('precision', 0):.4f}")
        logger.info(f"召回率: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1分数: {metrics.get('f1', 0):.4f}")
        logger.info(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")


def save_evaluation_report(
    metrics: Dict[str, float],
    output_path: str,
    task_type: str = 'regression',
    model_name: str = 'Model',
    stock_symbol: str = 'GME'
):
    """
    保存评估报告
    
    Args:
        metrics: 评估指标字典
        output_path: 输出文件路径
        task_type: 任务类型
        model_name: 模型名称
        stock_symbol: 股票代码
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if task_type == 'regression':
        report = f"""# 模型评估报告

**模型**: {model_name}
**股票代码**: {stock_symbol}
**任务类型**: 回归（波动率预测）
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 评估指标

| 指标 | 数值 |
|------|------|
| RMSE (均方根误差) | {metrics.get('rmse', 0):.6f} |
| MAE (平均绝对误差) | {metrics.get('mae', 0):.6f} |
| R² (决定系数) | {metrics.get('r2', 0):.4f} |
| MAPE (平均绝对百分比误差) | {metrics.get('mape', 0):.2f}% |
| 方向准确率 | {metrics.get('directional_accuracy', 0):.2f}% |
| 相关系数 | {metrics.get('correlation', 0):.4f} |

---

## 说明

- **RMSE**: 预测值与真实值之间的均方根误差，越小越好
- **MAE**: 预测值与真实值之间的平均绝对误差，越小越好
- **R²**: 决定系数，表示模型解释的方差比例，越接近1越好
- **MAPE**: 平均绝对百分比误差，越小越好
- **方向准确率**: 预测波动率变化方向的准确率，越高越好
- **相关系数**: 预测值与真实值之间的线性相关系数，越接近1越好
"""
    else:
        report = f"""# 模型评估报告

**模型**: {model_name}
**股票代码**: {stock_symbol}
**任务类型**: 分类
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 评估指标

| 指标 | 数值 |
|------|------|
| 准确率 | {metrics.get('accuracy', 0):.4f} |
| 精确率 | {metrics.get('precision', 0):.4f} |
| 召回率 | {metrics.get('recall', 0):.4f} |
| F1分数 | {metrics.get('f1', 0):.4f} |
| ROC-AUC | {metrics.get('roc_auc', 0):.4f} |

---

## 混淆矩阵

{np.array(metrics.get('confusion_matrix', [])).__str__()}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"评估报告已保存到: {output_path}")


if __name__ == '__main__':
    # 测试代码
    # 生成示例数据
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    # 计算指标
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # 打印结果
    print_metrics(metrics, task_type='regression')
    
    # 保存报告
    save_evaluation_report(metrics, 'results/evaluation_report_test.md', 
                          model_name='Test Model', stock_symbol='GME')

