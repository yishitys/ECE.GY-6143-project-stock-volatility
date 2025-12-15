"""
Feature Engineering Main Pipeline

Integrates all feature engineering steps:
1. Load merged data
2. Calculate target variables
3. Generate text embeddings (if not yet generated)
4. Aggregate text features
5. Calculate technical indicators
6. Merge all features
7. Handle missing values
8. Save final feature dataset
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List
import sys

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from feature_engineering.calculate_target import (
    calculate_multiple_targets,
    analyze_target_distribution,
    save_target_analysis_report
)
from feature_engineering.generate_embeddings import (
    generate_embeddings_for_reddit_data,
    EmbeddingGenerator
)
from feature_engineering.aggregate_features import (
    aggregate_embeddings_by_hour,
    combine_aggregated_features,
    handle_missing_hours
)
from feature_engineering.technical_indicators import (
    calculate_technical_indicators
)
from data_loading.load_reddit_data import load_multiple_subreddits
from preprocessing.clean_text import clean_reddit_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_merged_data(merged_data_path: str) -> pd.DataFrame:
    """
    Load merged data
    
    Args:
        merged_data_path: Merged data file path
    
    Returns:
        Merged data DataFrame
    """
    if not os.path.exists(merged_data_path):
        raise FileNotFoundError(f"Merged data file does not exist: {merged_data_path}")
    
    logger.info(f"Loading merged data: {merged_data_path}")
    df = pd.read_csv(merged_data_path, parse_dates=['timestamp'])
    logger.info(f"Loaded {len(df)} records")
    
    return df


def load_reddit_posts_for_embeddings(
    stock_symbol: str,
    subreddits: List[str] = None,
    reddit_data_dir: str = 'data/raw',
    cleaned_reddit_path: str = 'data/processed/reddit_cleaned.csv',
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31'
) -> pd.DataFrame:
    """
    Load Reddit posts data for embedding generation
    
    Args:
        stock_symbol: Stock symbol
        subreddits: List of subreddits
        reddit_data_dir: Reddit data directory
        cleaned_reddit_path: Cleaned Reddit data path
        start_date: Start date
        end_date: End date
    
    Returns:
        Reddit posts DataFrame
    """
    # Try to load cleaned data
    if os.path.exists(cleaned_reddit_path):
        logger.info(f"Loading cleaned Reddit data: {cleaned_reddit_path}")
        df = pd.read_csv(cleaned_reddit_path, parse_dates=['timestamp'], low_memory=False)
        logger.info(f"Loaded {len(df)} posts")
        return df
    
    # Otherwise load from raw data
    logger.info("Loading Reddit posts from raw data...")
    if subreddits is None:
        subreddits = [
            'stocks', 'wallstreetbets', 'investing', 'stockmarket',
            'options', 'pennystocks', 'gme'
        ]
    
    df = load_multiple_subreddits(
        subreddits=subreddits,
        data_dir=reddit_data_dir,
        start_date=start_date,
        end_date=end_date,
        prefer_h5=True
    )
    
    if df.empty:
        raise ValueError("Cannot load Reddit data")
    
    # Clean text
    df = clean_reddit_data(df, text_column='text_content')
    
    return df


def build_feature_pipeline(
    stock_symbol: str,
    merged_data_path: str = None,
    reddit_data_dir: str = 'data/raw',
    output_dir: str = 'data/processed',
    embedding_model: str = 'all-MiniLM-L6-v2',
    aggregation_method: str = 'mean',
    generate_embeddings: bool = True,
    use_cached_embeddings: bool = True
) -> pd.DataFrame:
    """
    Build complete feature engineering pipeline
    
    Args:
        stock_symbol: Stock code
        merged_data_path: Path to merged data (auto-generated if None)
        reddit_data_dir: Reddit data directory
        output_dir: Output directory
        embedding_model: Embedding model name
        aggregation_method: Aggregation method ('mean', 'weighted_mean', 'max')
        generate_embeddings: Whether to generate embeddings (if cache doesn't exist)
        use_cached_embeddings: Whether to use cached embeddings
    
    Returns:
        DataFrame containing all features
    """
    logger.info("="*60)
    logger.info(f"Starting feature engineering pipeline: {stock_symbol}")
    logger.info("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load merged data
    if merged_data_path is None:
        merged_data_path = os.path.join(output_dir, f'merged_data_{stock_symbol}.csv')
    
    df = load_merged_data(merged_data_path)
    
    # Step 2: Calculate target variables
    logger.info("\n" + "="*60)
    logger.info("Step 2: Calculate target variables")
    logger.info("="*60)
    df = calculate_multiple_targets(df, methods=['log_return_abs', 'price_range'])
    
    # 分析目标变量
    target_col = 'target_volatility_log_return_abs'
    if target_col in df.columns:
        stats = analyze_target_distribution(df, target_col)
        report_path = os.path.join(output_dir, f'target_analysis_{stock_symbol}.md')
        save_target_analysis_report(stats, report_path, target_col)
    
    # Step 3: Generate text embeddings (if needed)
    embedding_df = None
    if generate_embeddings or not use_cached_embeddings:
        logger.info("\n" + "="*60)
        logger.info("Step 3: Generate text embeddings")
        logger.info("="*60)
        
        # Load Reddit posts
        reddit_posts = load_reddit_posts_for_embeddings(
            stock_symbol=stock_symbol,
            reddit_data_dir=reddit_data_dir
        )
        
        # Generate embeddings
        cache_file = os.path.join(output_dir, 'embeddings', 
                                 f'embeddings_{embedding_model.replace("/", "_")}.pkl')
        
        # Check if embeddings need to be generated and notify in advance
        if not (use_cached_embeddings and os.path.exists(cache_file)):
            logger.info("\n" + "!"*60)
            logger.info("⚠️  IMPORTANT: About to start generating text embeddings")
            logger.info("!"*60)
            logger.info(f"📊 Number of posts to process: {len(reddit_posts):,}")
            logger.info(f"🤖 Model used: {embedding_model}")
            logger.info(f"⏱️  Estimated time: 30 minutes to several hours (depends on hardware)")
            logger.info(f"💾 Embeddings will be cached to: {cache_file}")
            logger.info("📝 Starting embedding generation...")
            logger.info("!"*60 + "\n")
        
        if use_cached_embeddings and os.path.exists(cache_file):
            logger.info(f"Using cached embeddings: {cache_file}")
            try:
                embedding_df = pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Cannot load cache, regenerating: {e}")
                embedding_df = generate_embeddings_for_reddit_data(
                    reddit_df=reddit_posts,
                    model_name=embedding_model,
                    text_col='text_cleaned',
                    output_dir=os.path.join(output_dir, 'embeddings'),
                    cache_file=cache_file
                )
        else:
            embedding_df = generate_embeddings_for_reddit_data(
                reddit_df=reddit_posts,
                model_name=embedding_model,
                text_col='text_cleaned',
                output_dir=os.path.join(output_dir, 'embeddings'),
                cache_file=cache_file
            )
    
    # Step 4: Aggregate text features
    if embedding_df is not None and not embedding_df.empty:
        logger.info("\n" + "="*60)
        logger.info("Step 4: Aggregate text features")
        logger.info("="*60)
        
        aggregated_embeddings = aggregate_embeddings_by_hour(
            posts_df=embedding_df,
            timestamp_col='timestamp',
            embedding_prefix='embedding_',
            aggregation_method=aggregation_method,
            weight_col='score' if aggregation_method == 'weighted_mean' else None
        )
        
        # Merge aggregated embeddings into original DataFrame
        # Only extract embedding columns (excluding Reddit stats as original df already has them)
        embedding_cols = [col for col in aggregated_embeddings.columns if col.startswith('embedding_') or col == 'timestamp']
        embeddings_only = aggregated_embeddings[embedding_cols].copy()
        
        # Merge embedding features into original DataFrame, preserving all original columns (including stock price data)
        df = pd.merge(
            df,
            embeddings_only,
            on='timestamp',
            how='left'
        )
        
        # Handle missing hours
        df = handle_missing_hours(
            df,
            timestamp_col='timestamp',
            embedding_prefix='embedding_',
            fill_method='zero'
        )
    
    # Step 5: Calculate technical indicators
    logger.info("\n" + "="*60)
    logger.info("Step 5: Calculate technical indicators")
    logger.info("="*60)
    df = calculate_technical_indicators(df)
    
    # Step 6: Handle missing values
    logger.info("\n" + "="*60)
    logger.info("Step 6: Handle missing values")
    logger.info("="*60)
    
    # Remove rows where target variable is NaN (last row)
    initial_len = len(df)
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if target_cols:
        df = df.dropna(subset=target_cols)
        logger.info(f"Removed {initial_len - len(df)} rows (target variable is NaN)")
    
    # Fill other missing values (using forward fill)
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        # Use forward fill for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    # Step 7: Save final feature dataset
    logger.info("\n" + "="*60)
    logger.info("Step 7: Save feature dataset")
    logger.info("="*60)
    
    output_path = os.path.join(output_dir, f'features_{stock_symbol}.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Feature dataset saved to: {output_path}")
    logger.info(f"Final feature count: {len(df.columns)}")
    logger.info(f"Final record count: {len(df)}")
    
    # 生成特征报告
    generate_feature_report(df, stock_symbol, output_dir)
    
    return df


def generate_feature_report(df: pd.DataFrame, stock_symbol: str, output_dir: str):
    """
    Generate feature engineering report
    
    Args:
        df: Features DataFrame
        stock_symbol: Stock symbol
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, f'feature_report_{stock_symbol}.md')
    
    # Count feature types
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    reddit_cols = ['post_count', 'total_comments', 'total_score', 'unique_authors']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    technical_cols = [col for col in df.columns if col not in 
                      embedding_cols + reddit_cols + target_cols + ['timestamp', 'stock_symbol', 
                                                                     'has_reddit_data', 'has_stock_data']]
    
    report = f"""# Feature Engineering Report

**Stock Symbol**: {stock_symbol}
**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Data Overview

| Metric | Value |
|--------|-------|
| Total Records | {len(df):,} |
| Total Features | {len(df.columns)} |
| Time Range | {df['timestamp'].min()} to {df['timestamp'].max()} |

---

## Feature Classification

### 1. Text Embedding Features
- **Count**: {len(embedding_cols)}
- **Dimension**: {len(embedding_cols)} dimensions
- **Description**: Text embedding vectors generated using sentence-transformers

### 2. Reddit Statistics Features
- **Count**: {len([col for col in reddit_cols if col in df.columns])}
- **Features**: {', '.join([col for col in reddit_cols if col in df.columns])}

### 3. Technical Indicator Features
- **Count**: {len(technical_cols)}
- **Main Types**: 
  - Return features
  - Moving average features
  - RSI, MACD indicators
  - Volatility features
  - Volume features
  - Lag features
  - Rolling statistics features

### 4. Target Variables
- **Count**: {len(target_cols)}
- **Variables**: {', '.join(target_cols)}

---

## Feature List

### Text Embedding Features
{chr(10).join([f'- {col}' for col in embedding_cols[:10]])}
... (total {len(embedding_cols)} features)

### Reddit Statistics Features
{chr(10).join([f'- {col}' for col in reddit_cols if col in df.columns])}

### Technical Indicator Features (Example)
{chr(10).join([f'- {col}' for col in technical_cols[:20]])}
... (total {len(technical_cols)} features)

---

## Data Quality

| Check Item | Result |
|------------|--------|
| Total Missing Values | {df.isnull().sum().sum()} |
| Target Variable Missing | {df[target_cols].isnull().sum().sum() if target_cols else 0} |
| Duplicate Records | {df.duplicated().sum()} |

---

## Notes

- All features are sorted by timestamp
- Records with NaN target variables have been removed
- Other missing values have been handled using forward fill
- Text embeddings generated using {embedding_model if 'embedding_model' in locals() else 'all-MiniLM-L6-v2'} model
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Feature engineering report saved to: {report_path}")


if __name__ == '__main__':
    # Test code
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering main pipeline')
    parser.add_argument('--symbol', type=str, default='GME', help='Stock symbol')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2', 
                       help='Embedding model name')
    parser.add_argument('--aggregation', type=str, default='mean', 
                       choices=['mean', 'weighted_mean', 'max'],
                       help='Aggregation method')
    parser.add_argument('--no-embeddings', action='store_true', 
                       help='Skip embedding generation (use existing features only)')
    
    args = parser.parse_args()
    
    df = build_feature_pipeline(
        stock_symbol=args.symbol,
        embedding_model=args.embedding_model,
        aggregation_method=args.aggregation,
        generate_embeddings=not args.no_embeddings
    )
    
    logger.info("Feature engineering pipeline completed!")


