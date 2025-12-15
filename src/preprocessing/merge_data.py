"""
Data Merging Module

Merges Reddit discussion volume data and stock price data, aligned by hourly timestamp.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_reddit_and_stock(
    reddit_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    stock_symbol: str,
    merge_type: str = 'inner',
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Merge Reddit discussion volume data and stock price data
    
    Args:
        reddit_df: Reddit data DataFrame (already aggregated by hour)
        stock_df: Stock price data DataFrame (already aligned by hour)
        stock_symbol: Stock symbol
        merge_type: Merge type, 'inner' (inner join) or 'left' (left join), default is 'inner'
        timestamp_col: Name of timestamp column, default is 'timestamp'
    
    Returns:
        Merged DataFrame
    """
    if reddit_df is None or reddit_df.empty:
        logger.warning("Empty Reddit DataFrame provided")
        return pd.DataFrame()
    
    if stock_df is None or stock_df.empty:
        logger.warning("Empty stock DataFrame provided")
        return pd.DataFrame()
    
    logger.info(f"Merging Reddit data ({len(reddit_df)} hours) with {stock_symbol} stock data ({len(stock_df)} hours)...")
    
    # Ensure timestamp column exists and is correctly formatted
    if timestamp_col not in reddit_df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found in Reddit DataFrame")
        return pd.DataFrame()
    
    if timestamp_col not in stock_df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found in stock DataFrame")
        return pd.DataFrame()
    
    # Ensure timestamp is datetime type
    reddit_df = reddit_df.copy()
    stock_df = stock_df.copy()
    
    reddit_df[timestamp_col] = pd.to_datetime(reddit_df[timestamp_col])
    stock_df[timestamp_col] = pd.to_datetime(stock_df[timestamp_col])
    
    # Ensure timezone consistency (UTC)
    if reddit_df[timestamp_col].dt.tz is None:
        reddit_df[timestamp_col] = reddit_df[timestamp_col].dt.tz_localize('UTC')
    else:
        reddit_df[timestamp_col] = reddit_df[timestamp_col].dt.tz_convert('UTC')
    
    if stock_df[timestamp_col].dt.tz is None:
        stock_df[timestamp_col] = stock_df[timestamp_col].dt.tz_localize('UTC')
    else:
        stock_df[timestamp_col] = stock_df[timestamp_col].dt.tz_convert('UTC')
    
    # Rename stock data columns, add stock symbol prefix
    stock_columns = {}
    for col in stock_df.columns:
        if col != timestamp_col:
            stock_columns[col] = f"{stock_symbol}_{col}"
    
    stock_df_renamed = stock_df.rename(columns=stock_columns)
    
    # Perform merge
    if merge_type == 'inner':
        merged_df = pd.merge(
            reddit_df,
            stock_df_renamed,
            on=timestamp_col,
            how='inner'
        )
    elif merge_type == 'left':
        merged_df = pd.merge(
            reddit_df,
            stock_df_renamed,
            on=timestamp_col,
            how='left'
        )
    else:
        logger.error(f"Invalid merge_type: {merge_type}. Use 'inner' or 'left'")
        return pd.DataFrame()
    
    logger.info(f"Merged data: {len(merged_df)} hours")
    logger.info(f"Time range: {merged_df[timestamp_col].min()} to {merged_df[timestamp_col].max()}")
    
    # Add stock symbol column
    merged_df['stock_symbol'] = stock_symbol
    
    return merged_df


def handle_missing_values(df: pd.DataFrame, fill_method: str = 'forward') -> pd.DataFrame:
    """
    Handle missing values in merged data
    
    Args:
        df: Merged DataFrame
        fill_method: Fill method
            - 'forward': Forward fill
            - 'backward': Backward fill
            - 'zero': Fill with 0
            - 'mean': Fill with mean
            - 'drop': Drop rows with missing values
    
    Returns:
        Processed DataFrame
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Count missing values
    missing_before = df.isnull().sum().sum()
    logger.info(f"Missing values before handling: {missing_before}")
    
    if missing_before == 0:
        logger.info("No missing values to handle")
        return df
    
    # Process missing values by column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if fill_method == 'forward':
        df[numeric_columns] = df[numeric_columns].ffill()
    elif fill_method == 'backward':
        df[numeric_columns] = df[numeric_columns].bfill()
    elif fill_method == 'zero':
        df[numeric_columns] = df[numeric_columns].fillna(0)
    elif fill_method == 'mean':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif fill_method == 'drop':
        df = df.dropna()
    else:
        logger.warning(f"Unknown fill_method: {fill_method}, using forward fill")
        df[numeric_columns] = df[numeric_columns].ffill()
    
    # Count missing values after handling
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after handling: {missing_after}")
    
    return df


def add_data_availability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加数据可用性标志
    
    Args:
        df: 合并后的DataFrame
    
    Returns:
        添加了标志列的DataFrame
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 检查Reddit数据可用性
    if 'post_count' in df.columns:
        df['has_reddit_data'] = (df['post_count'] > 0).astype(int)
    else:
        df['has_reddit_data'] = 0
    
    # 检查股票数据可用性（假设有close列）
    stock_price_cols = [col for col in df.columns if '_close' in col or 'close' in col.lower()]
    if stock_price_cols:
        # 使用第一个找到的价格列
        price_col = stock_price_cols[0]
        df['has_stock_data'] = df[price_col].notna().astype(int)
    else:
        df['has_stock_data'] = 0
    
    logger.info(f"Data availability flags added")
    logger.info(f"  Hours with Reddit data: {df['has_reddit_data'].sum()}")
    logger.info(f"  Hours with stock data: {df['has_stock_data'].sum()}")
    
    return df


def validate_merged_data(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> bool:
    """
    验证合并后数据的完整性
    
    Args:
        df: 合并后的DataFrame
        timestamp_col: 时间戳列名
    
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("Validation failed: Empty DataFrame")
        return False
    
    logger.info("\n" + "="*50)
    logger.info("Merged Data Validation Report")
    logger.info("="*50)
    
    # 基本统计
    logger.info(f"Total records: {len(df)}")
    
    # 检查时间戳
    if timestamp_col not in df.columns:
        logger.error(f"Timestamp column '{timestamp_col}' not found")
        return False
    
    min_ts = df[timestamp_col].min()
    max_ts = df[timestamp_col].max()
    logger.info(f"Time range: {min_ts} to {max_ts}")
    
    # 检查时间戳是否连续（可选）
    df_sorted = df.sort_values(timestamp_col)
    time_diffs = df_sorted[timestamp_col].diff()
    expected_diff = pd.Timedelta(hours=1)
    
    # 允许一些时间间隔不是正好1小时（由于交易时间等）
    non_hourly = (time_diffs != expected_diff).sum() - 1  # 减去第一个NaN
    if non_hourly > 0:
        logger.info(f"Note: {non_hourly} time intervals are not exactly 1 hour apart (expected for trading hours)")
    
    # 检查必需的列
    required_columns = [timestamp_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # 检查缺失值
    logger.info("\nMissing values by column:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # 检查数据列
    if 'post_count' in df.columns:
        logger.info(f"\nReddit data statistics:")
        logger.info(f"  Total posts: {df['post_count'].sum()}")
        logger.info(f"  Average posts per hour: {df['post_count'].mean():.2f}")
        logger.info(f"  Hours with posts: {(df['post_count'] > 0).sum()}")
        logger.info(f"  Hours without posts: {(df['post_count'] == 0).sum()}")
    
    # 检查股票价格列
    price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
    if price_cols:
        logger.info(f"\nStock data statistics:")
        for col in price_cols[:3]:  # 只显示前3个
            if df[col].notna().sum() > 0:
                logger.info(f"  {col}: {df[col].notna().sum()} non-null values")
    
    logger.info("="*50)
    
    return True

