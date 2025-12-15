"""
Feature Aggregation Module

Aggregates post-level embedding vectors into hourly-level features.
Supports multiple aggregation strategies: mean pooling, weighted mean, max pooling, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mean_pooling(embeddings: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mean pooling: Calculate average of embedding vectors
    
    Args:
        embeddings: Embedding vector array, shape (n_posts, embedding_dim)
        weights: Weight array, shape (n_posts,), if None use uniform weights
    
    Returns:
        Aggregated embedding vector, shape (embedding_dim,)
    """
    if embeddings is None or len(embeddings) == 0:
        return None
    
    if weights is None:
        return np.mean(embeddings, axis=0)
    else:
        # Weighted average
        weights = weights / (weights.sum() + 1e-8)  # Normalize weights
        return np.average(embeddings, axis=0, weights=weights)


def max_pooling(embeddings: np.ndarray) -> np.ndarray:
    """
    Max pooling: Take the maximum value of each dimension
    
    Args:
        embeddings: Embedding vector array, shape (n_posts, embedding_dim)
    
    Returns:
        Aggregated embedding vector, shape (embedding_dim,)
    """
    if embeddings is None or len(embeddings) == 0:
        return None
    
    return np.max(embeddings, axis=0)


def weighted_mean_pooling(
    embeddings: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Weighted mean pooling
    
    Args:
        embeddings: Embedding vector array
        weights: Weight array
    
    Returns:
        Aggregated embedding vector
    """
    return mean_pooling(embeddings, weights)


def aggregate_embeddings_by_hour(
    posts_df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    embedding_prefix: str = 'embedding_',
    aggregation_method: str = 'mean',
    weight_col: Optional[str] = None,
    post_id_col: str = 'id'
) -> pd.DataFrame:
    """
    Aggregate post-level embeddings by hour
    
    Args:
        posts_df: DataFrame containing embeddings
        timestamp_col: Timestamp column name
        embedding_prefix: Embedding column name prefix (e.g., 'embedding_')
        aggregation_method: Aggregation method ('mean', 'weighted_mean', 'max')
        weight_col: Weight column name (for weighted_mean), e.g., 'score' or 'num_comments'
        post_id_col: Post ID column name
    
    Returns:
        Hourly aggregated DataFrame with aggregated embedding features
    """
    if posts_df is None or posts_df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
    
    # Find all embedding columns
    embedding_cols = [col for col in posts_df.columns if col.startswith(embedding_prefix)]
    if not embedding_cols:
        raise ValueError(f"No embedding columns found with prefix '{embedding_prefix}'")
    
    logger.info(f"Found {len(embedding_cols)} embedding columns")
    logger.info(f"Aggregation method: {aggregation_method}")
    
    # Ensure timestamp column exists and is datetime type
    if timestamp_col not in posts_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    
    df = posts_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Group by hour
    df['hour'] = df[timestamp_col].dt.floor('H')
    
    # Aggregate embeddings
    aggregated_data = []
    
    for hour, group in df.groupby('hour'):
        hour_data = {'timestamp': hour}
        
        # Extract all embeddings for this hour
        embeddings = group[embedding_cols].values
        
        # Calculate based on aggregation method
        if aggregation_method == 'mean':
            aggregated_embedding = mean_pooling(embeddings)
        elif aggregation_method == 'weighted_mean':
            if weight_col is None:
                logger.warning("weight_col not specified, falling back to mean pooling")
                aggregated_embedding = mean_pooling(embeddings)
            else:
                if weight_col not in group.columns:
                    logger.warning(f"weight_col '{weight_col}' not found, falling back to mean pooling")
                    aggregated_embedding = mean_pooling(embeddings)
                else:
                    weights = group[weight_col].values
                    aggregated_embedding = weighted_mean_pooling(embeddings, weights)
        elif aggregation_method == 'max':
            aggregated_embedding = max_pooling(embeddings)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # 将聚合后的嵌入向量添加到结果中
        if aggregated_embedding is not None:
            for i, col in enumerate(embedding_cols):
                hour_data[col] = aggregated_embedding[i]
        
        # 添加其他统计信息
        hour_data['post_count'] = len(group)
        if weight_col and weight_col in group.columns:
            hour_data[f'total_{weight_col}'] = group[weight_col].sum()
        
        aggregated_data.append(hour_data)
    
    result_df = pd.DataFrame(aggregated_data)
    logger.info(f"Aggregated {len(posts_df)} posts into {len(result_df)} hours")
    
    return result_df


def combine_aggregated_features(
    aggregated_embeddings: pd.DataFrame,
    reddit_stats: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    合并聚合后的嵌入特征和Reddit统计特征
    
    Args:
        aggregated_embeddings: 聚合后的嵌入特征DataFrame
        reddit_stats: Reddit统计特征DataFrame（包含post_count, total_comments等）
        timestamp_col: 时间戳列名
    
    Returns:
        合并后的DataFrame
    """
    if aggregated_embeddings is None or aggregated_embeddings.empty:
        logger.warning("Empty aggregated embeddings DataFrame")
        return reddit_stats.copy() if reddit_stats is not None else pd.DataFrame()
    
    if reddit_stats is None or reddit_stats.empty:
        logger.warning("Empty reddit stats DataFrame")
        return aggregated_embeddings.copy()
    
    # 确保时间戳列格式一致
    for df in [aggregated_embeddings, reddit_stats]:
        if timestamp_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 合并
    merged_df = pd.merge(
        reddit_stats,
        aggregated_embeddings,
        on=timestamp_col,
        how='outer',
        suffixes=('', '_embedding')
    )
    
    # 处理重复列（如果reddit_stats中已有post_count，保留原来的）
    if 'post_count_embedding' in merged_df.columns and 'post_count' in merged_df.columns:
        # 如果post_count为空，用post_count_embedding填充
        merged_df['post_count'] = merged_df['post_count'].fillna(merged_df['post_count_embedding'])
        merged_df = merged_df.drop(columns=['post_count_embedding'])
    
    # 按时间排序
    merged_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)
    
    logger.info(f"Combined features: {len(merged_df)} hours")
    logger.info(f"Embedding columns: {len([col for col in merged_df.columns if 'embedding' in col])}")
    
    return merged_df


def handle_missing_hours(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    embedding_prefix: str = 'embedding_',
    fill_method: str = 'zero'
) -> pd.DataFrame:
    """
    处理缺失的小时数据（无帖子的时间）
    
    Args:
        df: 特征DataFrame
        timestamp_col: 时间戳列名
        embedding_prefix: 嵌入列名前缀
        fill_method: 填充方法
            - 'zero': 使用零向量
            - 'forward': 前向填充
            - 'backward': 后向填充
    
    Returns:
        处理后的DataFrame
    """
    df = df.copy()
    
    embedding_cols = [col for col in df.columns if col.startswith(embedding_prefix)]
    
    if not embedding_cols:
        return df
    
    # 检查哪些行的嵌入向量全为零或缺失
    embedding_data = df[embedding_cols]
    missing_mask = (embedding_data == 0).all(axis=1) | embedding_data.isna().all(axis=1)
    
    missing_count = missing_mask.sum()
    if missing_count == 0:
        logger.info("No missing hours found")
        return df
    
    logger.info(f"Found {missing_count} hours with missing embeddings")
    
    if fill_method == 'zero':
        # 使用零向量填充（已经是零向量，无需操作）
        logger.info("Using zero vectors for missing hours")
    elif fill_method == 'forward':
        # 前向填充
        df[embedding_cols] = df[embedding_cols].ffill()
        logger.info("Using forward fill for missing hours")
    elif fill_method == 'backward':
        # 后向填充
        df[embedding_cols] = df[embedding_cols].bfill()
        logger.info("Using backward fill for missing hours")
    else:
        logger.warning(f"Unknown fill method: {fill_method}, using zero vectors")
    
    return df


if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    
    # 添加src目录到路径
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 这里需要先有嵌入数据才能测试
    logger.info("This module requires embedding data to test.")
    logger.info("Please run generate_embeddings.py first to create embeddings.")


