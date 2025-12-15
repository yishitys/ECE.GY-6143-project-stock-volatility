"""
Main Preprocessing Script

Integrates all preprocessing steps:
1. Load Reddit data
2. Clean text
3. Load stock price data
4. Align timestamps
5. Merge data
6. Save results
"""

import pandas as pd
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 导入数据加载模块
import sys
# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_loading.load_reddit_data import load_multiple_subreddits
from data_loading.load_stock_data import download_stock_data

# Import preprocessing modules
from .clean_text import clean_reddit_data
from .align_timestamps import group_reddit_by_hour, align_stock_to_hourly
from .merge_data import (
    merge_reddit_and_stock,
    handle_missing_values,
    add_data_availability_flags,
    validate_merged_data
)
from .generate_report import generate_preprocessing_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_clean_reddit_data(
    subreddits: list,
    data_dir: str = 'data/raw',
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31',
    output_dir: str = 'data/processed',
    save_cleaned: bool = True
) -> pd.DataFrame:
    """
    Load and clean Reddit data
    
    Args:
        subreddits: List of subreddits
        data_dir: Reddit data directory
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        save_cleaned: Whether to save cleaned data
    
    Returns:
        Cleaned Reddit data DataFrame
    """
    logger.info("="*60)
    logger.info("Step 1: Loading and cleaning Reddit data")
    logger.info("="*60)
    
    # Load Reddit data
    reddit_df = load_multiple_subreddits(
        subreddits=subreddits,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        prefer_h5=True
    )
    
    if reddit_df.empty:
        logger.error("No Reddit data loaded")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(reddit_df)} Reddit posts")
    
    # Clean text
    reddit_df = clean_reddit_data(reddit_df, text_column='text_content')
    
    # Save cleaned data (optional)
    if save_cleaned:
        os.makedirs(output_dir, exist_ok=True)
        cleaned_path = os.path.join(output_dir, 'reddit_cleaned.csv')
        reddit_df.to_csv(cleaned_path, index=False)
        logger.info(f"Saved cleaned Reddit data to {cleaned_path}")
    
    return reddit_df


def load_stock_data_for_symbol(
    symbol: str,
    data_dir: str = 'data/stock_prices',
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31'
) -> pd.DataFrame:
    """
    Load stock price data
    
    Args:
        symbol: Stock symbol
        data_dir: Stock data directory
        start_date: Start date
        end_date: End date
    
    Returns:
        Stock price data DataFrame
    """
    logger.info("="*60)
    logger.info(f"Step 2: Loading stock data for {symbol}")
    logger.info("="*60)
    
    # Try to load from local file
    stock_file = os.path.join(data_dir, f"{symbol}_2021.csv")
    
    if os.path.exists(stock_file):
        logger.info(f"Loading {symbol} data from {stock_file}")
        stock_df = pd.read_csv(stock_file)
        stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'])
        
        # Ensure timezone is UTC
        if stock_df['timestamp'].dt.tz is None:
            stock_df['timestamp'] = stock_df['timestamp'].dt.tz_localize('UTC')
        else:
            stock_df['timestamp'] = stock_df['timestamp'].dt.tz_convert('UTC')
        
        logger.info(f"Loaded {len(stock_df)} stock records")
        return stock_df
    else:
        logger.warning(f"Stock file not found: {stock_file}")
        logger.info("Attempting to download from API...")
        
        # If file doesn't exist, try to download
        stock_df = download_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if stock_df is not None and not stock_df.empty:
            # Save downloaded data
            os.makedirs(data_dir, exist_ok=True)
            stock_df.to_csv(stock_file, index=False)
            logger.info(f"Saved downloaded data to {stock_file}")
        
        return stock_df if stock_df is not None else pd.DataFrame()


def preprocess_data(
    stock_symbol: str,
    subreddits: list = None,
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31',
    reddit_data_dir: str = 'data/raw',
    stock_data_dir: str = 'data/stock_prices',
    output_dir: str = 'data/processed',
    merge_type: str = 'inner',
    fill_method: str = 'forward'
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    
    Args:
        stock_symbol: Stock symbol
        subreddits: List of subreddits, uses default list if None
        start_date: Start date
        end_date: End date
        reddit_data_dir: Reddit data directory
        stock_data_dir: Stock data directory
        output_dir: Output directory
        merge_type: Merge type ('inner' or 'left')
        fill_method: Missing value fill method
    
    Returns:
        Merged DataFrame
    """
    logger.info("="*60)
    logger.info(f"Starting preprocessing for {stock_symbol}")
    logger.info("="*60)
    
    # Default subreddit list
    if subreddits is None:
        subreddits = [
            'stocks', 'wallstreetbets', 'investing', 'stockmarket',
            'options', 'pennystocks', 'gme'
        ]
    
    # Step 1: Load and clean Reddit data
    reddit_df = load_and_clean_reddit_data(
        subreddits=subreddits,
        data_dir=reddit_data_dir,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        save_cleaned=True
    )
    
    if reddit_df.empty:
        logger.error("Failed to load Reddit data")
        return pd.DataFrame()
    
    # Step 2: Group Reddit data by hour
    logger.info("="*60)
    logger.info("Step 3: Grouping Reddit data by hour")
    logger.info("="*60)
    
    reddit_hourly = group_reddit_by_hour(reddit_df, timestamp_col='timestamp')
    
    if reddit_hourly.empty:
        logger.error("Failed to group Reddit data by hour")
        return pd.DataFrame()
    
    # Step 3: Load stock price data
    stock_df = load_stock_data_for_symbol(
        symbol=stock_symbol,
        data_dir=stock_data_dir,
        start_date=start_date,
        end_date=end_date
    )
    
    if stock_df.empty:
        logger.error(f"Failed to load stock data for {stock_symbol}")
        return pd.DataFrame()
    
    # Step 4: Align stock data to hourly intervals
    logger.info("="*60)
    logger.info("Step 4: Aligning stock data to hourly intervals")
    logger.info("="*60)
    
    stock_hourly = align_stock_to_hourly(stock_df, timestamp_col='timestamp')
    
    if stock_hourly.empty:
        logger.error("Failed to align stock data to hourly intervals")
        return pd.DataFrame()
    
    # Step 5: Merge data
    logger.info("="*60)
    logger.info("Step 5: Merging Reddit and stock data")
    logger.info("="*60)
    
    merged_df = merge_reddit_and_stock(
        reddit_df=reddit_hourly,
        stock_df=stock_hourly,
        stock_symbol=stock_symbol,
        merge_type=merge_type
    )
    
    if merged_df.empty:
        logger.error("Failed to merge data")
        return pd.DataFrame()
    
    # Step 6: Add data availability flags
    merged_df = add_data_availability_flags(merged_df)
    
    # Step 7: Handle missing values
    logger.info("="*60)
    logger.info("Step 6: Handling missing values")
    logger.info("="*60)
    
    merged_df = handle_missing_values(merged_df, fill_method=fill_method)
    
    # Step 8: Validate data
    logger.info("="*60)
    logger.info("Step 7: Validating merged data")
    logger.info("="*60)
    
    is_valid = validate_merged_data(merged_df, timestamp_col='timestamp')
    
    if not is_valid:
        logger.warning("Data validation found issues, but continuing...")
    
    # Step 9: Save results
    logger.info("="*60)
    logger.info("Step 8: Saving merged data")
    logger.info("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"merged_data_{stock_symbol}.csv")
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged data to {output_file}")
    
    # Step 10: Generate preprocessing report
    logger.info("="*60)
    logger.info("Step 9: Generating preprocessing report")
    logger.info("="*60)
    
    report_path = generate_preprocessing_report(
        merged_df=merged_df,
        stock_symbol=stock_symbol,
        output_dir=output_dir,
        timestamp_col='timestamp'
    )
    
    if report_path:
        logger.info(f"Report generated: {report_path}")
    
    logger.info("="*60)
    logger.info("Preprocessing completed successfully!")
    logger.info("="*60)
    
    return merged_df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Preprocess Reddit and stock data')
    parser.add_argument(
        '--stock',
        type=str,
        required=True,
        help='Stock symbol (e.g., GME, AMC, TSLA)'
    )
    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=None,
        help='List of subreddits (e.g., stocks wallstreetbets). Uses default list if not specified'
    )
    parser.add_argument(
        '--start-date',
        default='2021-01-01',
        help='Start date (YYYY-MM-DD), default: 2021-01-01'
    )
    parser.add_argument(
        '--end-date',
        default='2021-12-31',
        help='End date (YYYY-MM-DD), default: 2021-12-31'
    )
    parser.add_argument(
        '--reddit-data-dir',
        default='data/raw',
        help='Reddit data directory, default: data/raw'
    )
    parser.add_argument(
        '--stock-data-dir',
        default='data/stock_prices',
        help='Stock data directory, default: data/stock_prices'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory, default: data/processed'
    )
    parser.add_argument(
        '--merge-type',
        choices=['inner', 'left'],
        default='inner',
        help='Merge type: inner (inner join) or left (left join), default: inner'
    )
    parser.add_argument(
        '--fill-method',
        choices=['forward', 'backward', 'zero', 'mean', 'drop'],
        default='forward',
        help='Missing value fill method, default: forward'
    )
    
    args = parser.parse_args()
    
    # Execute preprocessing
    merged_df = preprocess_data(
        stock_symbol=args.stock,
        subreddits=args.subreddits,
        start_date=args.start_date,
        end_date=args.end_date,
        reddit_data_dir=args.reddit_data_dir,
        stock_data_dir=args.stock_data_dir,
        output_dir=args.output_dir,
        merge_type=args.merge_type,
        fill_method=args.fill_method
    )
    
    if merged_df.empty:
        logger.error("Preprocessing failed")
        return 1
    
    logger.info(f"\nPreprocessing completed. Output: {args.output_dir}/merged_data_{args.stock}.csv")
    return 0


if __name__ == '__main__':
    exit(main())

