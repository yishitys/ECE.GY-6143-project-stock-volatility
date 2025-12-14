"""
结果可视化模块

生成预测结果的可视化图表。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, Dict
from pathlib import Path

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    output_path: str = 'results/figures/predictions_vs_actual.png',
    title: str = '预测值 vs 实际值'
):
    """
    绘制预测值与实际值的对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        timestamps: 时间戳（可选）
        output_path: 输出文件路径
        title: 图表标题
    """
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    if timestamps is not None:
        timestamps = timestamps[:min_len]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if timestamps is not None:
        ax.plot(timestamps, y_true, label='实际值', alpha=0.7, linewidth=1.5)
        ax.plot(timestamps, y_pred, label='预测值', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('时间')
    else:
        ax.plot(y_true, label='实际值', alpha=0.7, linewidth=1.5)
        ax.plot(y_pred, label='预测值', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('样本索引')
    
    ax.set_ylabel('波动率')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"预测对比图已保存到: {output_path}")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = 'results/figures/residuals.png',
    title: str = '残差分析'
):
    """
    绘制残差分析图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        output_path: 输出文件路径
        title: 图表标题
    """
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 残差时间序列
    axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('样本索引')
    axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差时间序列')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差分布
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('残差')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('残差分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差 vs 预测值
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('残差 vs 预测值')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q图
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q图（正态性检验）')
        axes[1, 1].grid(True, alpha=0.3)
    except Exception as e:
        # SciPy 可能未安装；保持其它子图可用
        axes[1, 1].axis('off')
        axes[1, 1].text(
            0.5, 0.5,
            f"Q-Q图不可用（缺少scipy或错误）\n{e}",
            ha='center', va='center', fontsize=10
        )
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"残差分析图已保存到: {output_path}")


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = 'results/figures/error_distribution.png',
    title: str = '误差分布'
):
    """
    绘制误差分布图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        output_path: 输出文件路径
        title: 图表标题
    """
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绝对误差分布
    axes[0].hist(abs_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=np.mean(abs_errors), color='r', linestyle='--', 
                   label=f'均值: {np.mean(abs_errors):.6f}')
    axes[0].set_xlabel('绝对误差')
    axes[0].set_ylabel('频数')
    axes[0].set_title('绝对误差分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 相对误差分布
    relative_errors = abs_errors / (np.abs(y_true) + 1e-8) * 100
    axes[1].hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(relative_errors), color='r', linestyle='--',
                   label=f'均值: {np.mean(relative_errors):.2f}%')
    axes[1].set_xlabel('相对误差 (%)')
    axes[1].set_ylabel('频数')
    axes[1].set_title('相对误差分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"误差分布图已保存到: {output_path}")


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: str = 'results/figures/feature_importance.png',
    title: str = '特征重要性'
):
    """
    绘制特征重要性图
    
    Args:
        feature_importance: 特征重要性DataFrame（包含'feature'和'importance'列）
        top_n: 显示前N个特征
        output_path: 输出文件路径
        title: 图表标题
    """
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('重要性')
    ax.set_title(title)
    ax.invert_yaxis()  # 最重要的在顶部
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"特征重要性图已保存到: {output_path}")


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str = 'results/figures/model_comparison.png',
    title: str = '模型性能对比'
):
    """
    绘制模型性能对比图
    
    Args:
        results: 模型结果字典，格式为 {model_name: {metric_name: value}}
        output_path: 输出文件路径
        title: 图表标题
    """
    # 提取指标
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    
    # 创建DataFrame
    comparison_df = pd.DataFrame(results).T
    
    # 绘制对比图
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        comparison_df[metric].plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_title(metric)
        ax.set_ylabel('数值')
        ax.set_xlabel('模型')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"模型对比图已保存到: {output_path}")


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    
    # 生成示例数据
    n = 100
    y_true = np.random.randn(n)
    y_pred = y_true + np.random.randn(n) * 0.1
    timestamps = pd.date_range('2021-01-01', periods=n, freq='H')
    
    # 绘制各种图表
    plot_predictions_vs_actual(y_true, y_pred, timestamps)
    plot_residuals(y_true, y_pred)
    plot_error_distribution(y_true, y_pred)
    
    # 特征重要性示例
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.random.rand(20)
    }).sort_values('importance', ascending=False)
    
    plot_feature_importance(feature_importance)
    
    logger.info("可视化测试完成！")


