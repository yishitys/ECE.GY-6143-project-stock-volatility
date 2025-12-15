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
    加载并清洗Reddit数据
    
    Args:
        subreddits: 子版块列表
        data_dir: Reddit数据目录
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        save_cleaned: 是否保存清洗后的数据
    
    Returns:
        清洗后的Reddit数据DataFrame
    """
    logger.info("="*60)
    logger.info("Step 1: Loading and cleaning Reddit data")
    logger.info("="*60)
    
    # 加载Reddit数据
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
    
    # 清洗文本
    reddit_df = clean_reddit_data(reddit_df, text_column='text_content')
    
    # 保存清洗后的数据（可选）
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
    加载股票价格数据
    
    Args:
        symbol: 股票代码
        data_dir: 股票数据目录
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        股票价格数据DataFrame
    """
    logger.info("="*60)
    logger.info(f"Step 2: Loading stock data for {symbol}")
    logger.info("="*60)
    
    # 尝试从本地文件加载
    stock_file = os.path.join(data_dir, f"{symbol}_2021.csv")
    
    if os.path.exists(stock_file):
        logger.info(f"Loading {symbol} data from {stock_file}")
        stock_df = pd.read_csv(stock_file)
        stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'])
        
        # 确保时区是UTC
        if stock_df['timestamp'].dt.tz is None:
            stock_df['timestamp'] = stock_df['timestamp'].dt.tz_localize('UTC')
        else:
            stock_df['timestamp'] = stock_df['timestamp'].dt.tz_convert('UTC')
        
        logger.info(f"Loaded {len(stock_df)} stock records")
        return stock_df
    else:
        logger.warning(f"Stock file not found: {stock_file}")
        logger.info("Attempting to download from API...")
        
        # 如果文件不存在，尝试下载
        stock_df = download_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if stock_df is not None and not stock_df.empty:
            # 保存下载的数据
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
    完整的预处理流程
    
    Args:
        stock_symbol: 股票代码
        subreddits: Reddit子版块列表，如果为None则使用默认列表
        start_date: 开始日期
        end_date: 结束日期
        reddit_data_dir: Reddit数据目录
        stock_data_dir: 股票数据目录
        output_dir: 输出目录
        merge_type: 合并类型（'inner'或'left'）
        fill_method: 缺失值填充方法
    
    Returns:
        合并后的DataFrame
    """
    logger.info("="*60)
    logger.info(f"Starting preprocessing for {stock_symbol}")
    logger.info("="*60)
    
    # 默认子版块列表
    if subreddits is None:
        subreddits = [
            'stocks', 'wallstreetbets', 'investing', 'stockmarket',
            'options', 'pennystocks', 'gme'
        ]
    
    # Step 1: 加载并清洗Reddit数据
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
    
    # Step 2: 按小时聚合Reddit数据
    logger.info("="*60)
    logger.info("Step 3: Grouping Reddit data by hour")
    logger.info("="*60)
    
    reddit_hourly = group_reddit_by_hour(reddit_df, timestamp_col='timestamp')
    
    if reddit_hourly.empty:
        logger.error("Failed to group Reddit data by hour")
        return pd.DataFrame()
    
    # Step 3: 加载股票价格数据
    stock_df = load_stock_data_for_symbol(
        symbol=stock_symbol,
        data_dir=stock_data_dir,
        start_date=start_date,
        end_date=end_date
    )
    
    if stock_df.empty:
        logger.error(f"Failed to load stock data for {stock_symbol}")
        return pd.DataFrame()
    
    # Step 4: 对齐股票数据到小时级
    logger.info("="*60)
    logger.info("Step 4: Aligning stock data to hourly intervals")
    logger.info("="*60)
    
    stock_hourly = align_stock_to_hourly(stock_df, timestamp_col='timestamp')
    
    if stock_hourly.empty:
        logger.error("Failed to align stock data to hourly intervals")
        return pd.DataFrame()
    
    # Step 5: 合并数据
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
    
    # Step 6: 添加数据可用性标志
    merged_df = add_data_availability_flags(merged_df)
    
    # Step 7: 处理缺失值
    logger.info("="*60)
    logger.info("Step 6: Handling missing values")
    logger.info("="*60)
    
    merged_df = handle_missing_values(merged_df, fill_method=fill_method)
    
    # Step 8: 验证数据
    logger.info("="*60)
    logger.info("Step 7: Validating merged data")
    logger.info("="*60)
    
    is_valid = validate_merged_data(merged_df, timestamp_col='timestamp')
    
    if not is_valid:
        logger.warning("Data validation found issues, but continuing...")
    
    # Step 9: 保存结果
    logger.info("="*60)
    logger.info("Step 8: Saving merged data")
    logger.info("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"merged_data_{stock_symbol}.csv")
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged data to {output_file}")
    
    # Step 10: 生成预处理报告
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
    """主函数"""
    parser = argparse.ArgumentParser(description='预处理Reddit和股票数据')
    parser.add_argument(
        '--stock',
        type=str,
        required=True,
        help='股票代码（例如：GME, AMC, TSLA）'
    )
    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=None,
        help='Reddit子版块列表（例如：stocks wallstreetbets）。如果不指定，将使用默认列表'
    )
    parser.add_argument(
        '--start-date',
        default='2021-01-01',
        help='开始日期 (YYYY-MM-DD)，默认：2021-01-01'
    )
    parser.add_argument(
        '--end-date',
        default='2021-12-31',
        help='结束日期 (YYYY-MM-DD)，默认：2021-12-31'
    )
    parser.add_argument(
        '--reddit-data-dir',
        default='data/raw',
        help='Reddit数据目录，默认：data/raw'
    )
    parser.add_argument(
        '--stock-data-dir',
        default='data/stock_prices',
        help='股票数据目录，默认：data/stock_prices'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='输出目录，默认：data/processed'
    )
    parser.add_argument(
        '--merge-type',
        choices=['inner', 'left'],
        default='inner',
        help='合并类型：inner（内连接）或left（左连接），默认：inner'
    )
    parser.add_argument(
        '--fill-method',
        choices=['forward', 'backward', 'zero', 'mean', 'drop'],
        default='forward',
        help='缺失值填充方法，默认：forward'
    )
    
    args = parser.parse_args()
    
    # 执行预处理
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

