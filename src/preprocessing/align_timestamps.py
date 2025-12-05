"""
时间戳对齐模块

将Reddit数据和股票价格数据按小时对齐，处理时区差异和交易时间窗口。
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def group_reddit_by_hour(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    将Reddit数据按小时分组
    
    Args:
        df: 包含时间戳列的Reddit数据DataFrame
        timestamp_col: 时间戳列名，默认为'timestamp'
    
    Returns:
        按小时聚合的DataFrame，包含：
        - timestamp: 小时级时间戳（UTC）
        - post_count: 该小时的帖子数量
        - total_comments: 该小时的总评论数
        - total_score: 该小时的总分数
        - unique_authors: 该小时的唯一作者数
        - posts: 该小时的所有帖子ID列表（可选）
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
    
    if timestamp_col not in df.columns:
        logger.error(f"Column '{timestamp_col}' not found in DataFrame")
        return df
    
    logger.info(f"Grouping {len(df)} Reddit posts by hour...")
    
    # 确保时间戳是datetime类型
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 按小时分组（向下取整到小时）
    df['hour'] = df[timestamp_col].dt.floor('h')
    
    # 聚合统计信息
    agg_dict = {
        'id': 'count',  # 帖子数量
    }
    
    # 添加可选的聚合列
    if 'num_comments' in df.columns:
        agg_dict['num_comments'] = 'sum'
    if 'score' in df.columns:
        agg_dict['score'] = 'sum'
    if 'author' in df.columns:
        agg_dict['author'] = 'nunique'
    
    # 执行聚合
    grouped = df.groupby('hour').agg(agg_dict)
    
    # 重命名列
    new_columns = ['post_count']
    if 'num_comments' in df.columns:
        new_columns.append('total_comments')
    if 'score' in df.columns:
        new_columns.append('total_score')
    if 'author' in df.columns:
        new_columns.append('unique_authors')
    
    grouped.columns = new_columns
    
    # 重置索引，将hour作为列
    grouped = grouped.reset_index()
    grouped.rename(columns={'hour': 'timestamp'}, inplace=True)
    
    # 确保时间戳有时区信息（UTC）
    if grouped['timestamp'].dt.tz is None:
        grouped['timestamp'] = grouped['timestamp'].dt.tz_localize('UTC')
    else:
        grouped['timestamp'] = grouped['timestamp'].dt.tz_convert('UTC')
    
    logger.info(f"Grouped into {len(grouped)} hourly intervals")
    logger.info(f"Time range: {grouped['timestamp'].min()} to {grouped['timestamp'].max()}")
    
    return grouped


def align_stock_to_hourly(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    确保股票数据按小时对齐
    
    Args:
        df: 股票价格数据DataFrame
        timestamp_col: 时间戳列名，默认为'timestamp'
    
    Returns:
        按小时对齐的股票数据DataFrame
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
    
    if timestamp_col not in df.columns:
        logger.error(f"Column '{timestamp_col}' not found in DataFrame")
        return df
    
    logger.info(f"Aligning {len(df)} stock records to hourly intervals...")
    
    # 创建副本
    df_aligned = df.copy()
    
    # 确保时间戳是datetime类型
    df_aligned[timestamp_col] = pd.to_datetime(df_aligned[timestamp_col])
    
    # 按小时分组（向下取整到小时）
    df_aligned['hour'] = df_aligned[timestamp_col].dt.floor('h')
    
    # 对于每个小时，选择最后一个数据点（收盘价）
    # 或者使用平均值/第一个值，取决于需求
    # 这里使用最后一个值（收盘价）
    df_aligned = df_aligned.sort_values(timestamp_col)
    
    # 按小时分组，取每小时的最后一个记录
    grouped = df_aligned.groupby('hour').last().reset_index()
    
    # 删除原来的timestamp列（如果存在），避免重复
    if timestamp_col in grouped.columns and timestamp_col != 'hour':
        grouped = grouped.drop(columns=[timestamp_col])
    
    # 重命名hour列为timestamp
    grouped = grouped.rename(columns={'hour': 'timestamp'})
    
    # 确保时间戳有时区信息（UTC）
    if 'timestamp' in grouped.columns:
        ts_series = grouped['timestamp']
        if isinstance(ts_series, pd.Series):
            if ts_series.dt.tz is None:
                grouped['timestamp'] = ts_series.dt.tz_localize('UTC')
            else:
                grouped['timestamp'] = ts_series.dt.tz_convert('UTC')
    
    logger.info(f"Aligned to {len(grouped)} hourly intervals")
    if 'timestamp' in grouped.columns and len(grouped) > 0:
        ts_min = grouped['timestamp'].min()
        ts_max = grouped['timestamp'].max()
        logger.info(f"Time range: {ts_min} to {ts_max}")
    
    return grouped


def create_hourly_index(
    start_date: str,
    end_date: str,
    timezone: str = 'UTC'
) -> pd.DatetimeIndex:
    """
    创建完整的小时级时间戳索引
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)，包含该日期
        timezone: 时区，默认为'UTC'
    
    Returns:
        小时级DatetimeIndex
    """
    start = pd.to_datetime(start_date).tz_localize(timezone)
    end = pd.to_datetime(end_date).tz_localize(timezone) + pd.Timedelta(days=1)
    
    # 创建每小时的时间戳
    hourly_index = pd.date_range(start=start, end=end, freq='H', tz=timezone)
    
    logger.info(f"Created hourly index: {len(hourly_index)} hours from {start} to {end}")
    
    return hourly_index


def filter_trading_hours(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    过滤出交易时间的数据（美国股市：9:30 AM - 4:00 PM EST，转换为UTC）
    
    注意：这个函数假设数据已经是UTC时区
    
    Args:
        df: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
    
    Returns:
        过滤后的DataFrame（仅包含交易时间）
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 转换为EST时区以检查交易时间
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC')
    
    # 转换为EST
    df['est_time'] = df[timestamp_col].dt.tz_convert('US/Eastern')
    
    # 交易时间：9:30 AM - 4:00 PM EST
    # 获取小时和分钟
    df['hour'] = df['est_time'].dt.hour
    df['minute'] = df['est_time'].dt.minute
    
    # 过滤交易时间
    # 9:30 AM (hour=9, minute=30) 到 4:00 PM (hour=16, minute=0)
    mask = (
        ((df['hour'] == 9) & (df['minute'] >= 30)) |
        ((df['hour'] >= 10) & (df['hour'] < 16)) |
        ((df['hour'] == 16) & (df['minute'] == 0))
    )
    
    # 同时过滤掉周末
    df['weekday'] = df['est_time'].dt.weekday
    mask = mask & (df['weekday'] < 5)  # 0-4 是周一到周五
    
    filtered_df = df[mask].copy()
    
    # 删除临时列
    filtered_df = filtered_df.drop(columns=['est_time', 'hour', 'minute', 'weekday'])
    
    logger.info(f"Filtered to trading hours: {len(filtered_df)} records (from {len(df)} total)")
    
    return filtered_df

