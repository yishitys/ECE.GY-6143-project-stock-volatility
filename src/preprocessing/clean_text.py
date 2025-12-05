"""
Reddit文本清洗模块

清理Reddit文本内容，移除URL、特殊字符、多余空格等。
"""

import pandas as pd
import re
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text_content(text: str) -> str:
    """
    清洗单个文本内容
    
    Args:
        text: 原始文本字符串
    
    Returns:
        清洗后的文本字符串
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 移除Reddit特定的标记
    text = re.sub(r'\[removed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[deleted\]', '', text, flags=re.IGNORECASE)
    
    # 移除特殊字符（保留字母、数字、基本标点）
    # 保留：字母、数字、空格、基本标点符号
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]', ' ', text)
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空格
    text = text.strip()
    
    return text


def clean_reddit_data(df: pd.DataFrame, text_column: str = 'text_content') -> pd.DataFrame:
    """
    批量清洗Reddit数据
    
    Args:
        df: 包含文本列的DataFrame
        text_column: 要清洗的文本列名，默认为'text_content'
    
    Returns:
        清洗后的DataFrame，添加了'text_cleaned'列
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
    
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found in DataFrame")
        return df
    
    logger.info(f"Cleaning text content for {len(df)} posts...")
    
    # 创建副本以避免修改原始数据
    df_cleaned = df.copy()
    
    # 清洗文本内容
    df_cleaned['text_cleaned'] = df_cleaned[text_column].apply(clean_text_content)
    
    # 统计清洗结果
    original_lengths = df_cleaned[text_column].str.len()
    cleaned_lengths = df_cleaned['text_cleaned'].str.len()
    
    avg_original = original_lengths.mean()
    avg_cleaned = cleaned_lengths.mean()
    
    logger.info(f"Average text length: {avg_original:.1f} -> {avg_cleaned:.1f} characters")
    
    # 移除完全为空的内容（虽然之前已经过滤过，但这里再次确保）
    empty_count = (df_cleaned['text_cleaned'].str.len() == 0).sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} posts with empty text after cleaning")
        # 可以选择移除或保留这些行，这里保留但标记为空
    
    return df_cleaned


def remove_stopwords(text: str, stopwords: Optional[list] = None) -> str:
    """
    移除停用词（可选功能）
    
    Args:
        text: 文本字符串
        stopwords: 停用词列表，如果为None则不处理
    
    Returns:
        移除停用词后的文本
    """
    if not stopwords or not text:
        return text
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)


def normalize_text(text: str) -> str:
    """
    标准化文本（可选功能）
    
    - 转换为小写
    - 标准化空格
    
    Args:
        text: 文本字符串
    
    Returns:
        标准化后的文本
    """
    if not text:
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 标准化空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


