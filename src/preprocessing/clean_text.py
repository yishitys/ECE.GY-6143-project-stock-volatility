"""
Reddit Text Cleaning Module

Cleans Reddit text content by removing URLs, special characters, excess whitespace, etc.
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
    Clean a single text content
    
    Args:
        text: Original text string
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove Reddit-specific markers
    text = re.sub(r'\[removed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[deleted\]', '', text, flags=re.IGNORECASE)
    
    # Remove special characters (keep letters, numbers, basic punctuation)
    # Keep: letters, numbers, spaces, basic punctuation symbols
    text = re.sub(r'[^\w\s.,!?;:()'\-'\"'  "]+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def clean_reddit_data(df: pd.DataFrame, text_column: str = 'text_content') -> pd.DataFrame:
    """
    Batch clean Reddit data
    
    Args:
        df: DataFrame containing text column
        text_column: Name of text column to clean, default is 'text_content'
    
    Returns:
        Cleaned DataFrame with added 'text_cleaned' column
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
    
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found in DataFrame")
        return df
    
    logger.info(f"Cleaning text content for {len(df)} posts...")
    
    # Create a copy to avoid modifying the original data
    df_cleaned = df.copy()
    
    # Clean text content
    df_cleaned['text_cleaned'] = df_cleaned[text_column].apply(clean_text_content)
    
    # Statistics from cleaning
    original_lengths = df_cleaned[text_column].str.len()
    cleaned_lengths = df_cleaned['text_cleaned'].str.len()
    
    avg_original = original_lengths.mean()
    avg_cleaned = cleaned_lengths.mean()
    
    logger.info(f"Average text length: {avg_original:.1f} -> {avg_cleaned:.1f} characters")
    
    # Remove completely empty content (already filtered before, but ensure here)
    empty_count = (df_cleaned['text_cleaned'].str.len() == 0).sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} posts with empty text after cleaning")
        # 可以选择移除或保留这些行，这里保留但标记为空
    
    return df_cleaned


def remove_stopwords(text: str, stopwords: Optional[list] = None) -> str:
    """
    Remove stopwords (optional feature)
    
    Args:
        text: Text string
        stopwords: List of stopwords, if None then no processing
    
    Returns:
        Text with stopwords removed
    """
    if not stopwords or not text:
        return text
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)


def normalize_text(text: str) -> str:
    """
    Normalize text (optional feature)
    
    - Convert to lowercase
    - Normalize whitespace
    
    Args:
        text: Text string
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text



