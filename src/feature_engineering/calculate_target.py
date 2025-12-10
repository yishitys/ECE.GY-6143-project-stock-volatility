"""
目标变量计算模块

计算下一小时的波动率作为预测目标。
支持多种波动率定义方式。
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_log_return(close_prices: pd.Series) -> pd.Series:
    """
    计算对数收益率
    
    Args:
        close_prices: 收盘价序列
    
    Returns:
        对数收益率序列
    """
    return np.log(close_prices / close_prices.shift(1))


def calculate_volatility_target(
    df: pd.DataFrame,
    method: str = 'log_return_abs',
    close_col: str = None,
    high_col: str = None,
    low_col: str = None,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    计算下一小时的波动率作为目标变量
    
    Args:
        df: 包含股票价格数据的DataFrame
        method: 波动率计算方法
            - 'log_return_abs': 下一小时对数收益率的绝对值
            - 'price_range': 下一小时的价格范围 (high - low) / close
            - 'realized_vol': 下一小时的已实现波动率（标准差）
        close_col: 收盘价列名，如果为None则自动检测（格式：{SYMBOL}_close）
        high_col: 最高价列名，如果为None则自动检测
        low_col: 最低价列名，如果为None则自动检测
        timestamp_col: 时间戳列名
    
    Returns:
        添加了目标变量的DataFrame
    """
    df = df.copy()
    
    # 确保按时间排序
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # 自动检测列名
    if close_col is None:
        close_cols = [col for col in df.columns if col.endswith('_close')]
        if close_cols:
            close_col = close_cols[0]
        else:
            raise ValueError("无法找到收盘价列，请指定close_col参数")
    
    if high_col is None:
        high_cols = [col for col in df.columns if col.endswith('_high')]
        if high_cols:
            high_col = high_cols[0]
        else:
            high_col = None
    
    if low_col is None:
        low_cols = [col for col in df.columns if col.endswith('_low')]
        if low_cols:
            low_col = low_cols[0]
        else:
            low_col = None
    
    logger.info(f"使用列: close={close_col}, high={high_col}, low={low_col}")
    logger.info(f"计算方法: {method}")
    
    # 计算目标变量
    if method == 'log_return_abs':
        # 方法1: 下一小时对数收益率的绝对值
        next_close = df[close_col].shift(-1)
        current_close = df[close_col]
        target = np.abs(np.log(next_close / current_close))
        target_name = 'target_volatility_log_return_abs'
    
    elif method == 'price_range':
        # 方法2: 下一小时的价格范围
        if high_col is None or low_col is None:
            raise ValueError("price_range方法需要high和low列")
        next_high = df[high_col].shift(-1)
        next_low = df[low_col].shift(-1)
        next_close = df[close_col].shift(-1)
        target = (next_high - next_low) / next_close
        target_name = 'target_volatility_price_range'
    
    elif method == 'realized_vol':
        # 方法3: 下一小时的已实现波动率（使用滚动窗口的标准差）
        # 这里简化为下一小时的对数收益率的标准差
        # 由于只有一小时数据，我们使用下一小时的对数收益率
        next_close = df[close_col].shift(-1)
        current_close = df[close_col]
        returns = np.log(next_close / current_close)
        # 使用滚动窗口计算标准差（窗口大小=1，即单点）
        target = returns.abs()  # 简化版本，实际应该是滚动std
        target_name = 'target_volatility_realized'
    
    else:
        raise ValueError(f"未知的波动率计算方法: {method}")
    
    # 添加目标变量到DataFrame
    df[target_name] = target
    
    # 最后一行没有下一小时数据，设置为NaN
    df.loc[df.index[-1], target_name] = np.nan
    
    logger.info(f"目标变量 '{target_name}' 已计算完成")
    logger.info(f"有效值数量: {df[target_name].notna().sum()} / {len(df)}")
    
    return df


def calculate_multiple_targets(
    df: pd.DataFrame,
    methods: list = ['log_return_abs', 'price_range'],
    **kwargs
) -> pd.DataFrame:
    """
    计算多种目标变量
    
    Args:
        df: 包含股票价格数据的DataFrame
        methods: 要计算的波动率方法列表
        **kwargs: 传递给calculate_volatility_target的其他参数
    
    Returns:
        添加了多个目标变量的DataFrame
    """
    df = df.copy()
    
    for method in methods:
        try:
            df = calculate_volatility_target(df, method=method, **kwargs)
            logger.info(f"成功计算目标变量: {method}")
        except Exception as e:
            logger.warning(f"计算目标变量 {method} 时出错: {e}")
    
    return df


def analyze_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'target_volatility_log_return_abs'
) -> Dict:
    """
    分析目标变量的分布
    
    Args:
        df: 包含目标变量的DataFrame
        target_col: 目标变量列名
    
    Returns:
        包含统计信息的字典
    """
    if target_col not in df.columns:
        raise ValueError(f"目标变量列 '{target_col}' 不存在")
    
    target = df[target_col].dropna()
    
    if len(target) == 0:
        logger.warning("目标变量没有有效值")
        return {}
    
    stats = {
        'count': len(target),
        'mean': float(target.mean()),
        'median': float(target.median()),
        'std': float(target.std()),
        'min': float(target.min()),
        'max': float(target.max()),
        'q25': float(target.quantile(0.25)),
        'q75': float(target.quantile(0.75)),
        'missing_count': int(df[target_col].isna().sum()),
        'missing_rate': float(df[target_col].isna().sum() / len(df))
    }
    
    logger.info(f"目标变量 '{target_col}' 统计信息:")
    logger.info(f"  有效值数量: {stats['count']}")
    logger.info(f"  均值: {stats['mean']:.6f}")
    logger.info(f"  中位数: {stats['median']:.6f}")
    logger.info(f"  标准差: {stats['std']:.6f}")
    logger.info(f"  最小值: {stats['min']:.6f}")
    logger.info(f"  最大值: {stats['max']:.6f}")
    logger.info(f"  缺失值数量: {stats['missing_count']} ({stats['missing_rate']*100:.2f}%)")
    
    return stats


def save_target_analysis_report(
    stats: Dict,
    output_path: str,
    target_col: str = 'target_volatility_log_return_abs'
):
    """
    保存目标变量分析报告
    
    Args:
        stats: 统计信息字典
        output_path: 输出文件路径
        target_col: 目标变量列名
    """
    report = f"""# 目标变量分析报告

**目标变量**: {target_col}
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 统计信息

| 指标 | 数值 |
|------|------|
| 有效值数量 | {stats.get('count', 0):,} |
| 缺失值数量 | {stats.get('missing_count', 0):,} |
| 缺失率 | {stats.get('missing_rate', 0)*100:.2f}% |

### 描述性统计

| 指标 | 数值 |
|------|------|
| 均值 | {stats.get('mean', 0):.6f} |
| 中位数 | {stats.get('median', 0):.6f} |
| 标准差 | {stats.get('std', 0):.6f} |
| 最小值 | {stats.get('min', 0):.6f} |
| 最大值 | {stats.get('max', 0):.6f} |
| 25%分位数 | {stats.get('q25', 0):.6f} |
| 75%分位数 | {stats.get('q75', 0):.6f} |

---

## 说明

- 目标变量表示下一小时的波动率
- 最后一条记录没有下一小时数据，因此目标变量为NaN
- 建议在模型训练时删除包含NaN的行
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"目标变量分析报告已保存到: {output_path}")


if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    
    # 添加src目录到路径
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 加载测试数据
    data_path = 'data/processed/merged_data_GME.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"加载数据: {len(df)} 条记录")
        
        # 计算目标变量
        df = calculate_multiple_targets(df, methods=['log_return_abs', 'price_range'])
        
        # 分析目标变量
        stats = analyze_target_distribution(df, 'target_volatility_log_return_abs')
        
        # 保存报告
        report_path = 'data/processed/target_analysis_report.md'
        save_target_analysis_report(stats, report_path)
        
        # 保存更新后的数据
        output_path = 'data/processed/merged_data_GME_with_target.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"已保存包含目标变量的数据到: {output_path}")
    else:
        logger.warning(f"测试数据文件不存在: {data_path}")


