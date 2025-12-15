"""
Reddit Data Loading Script

Loads Reddit submission data from local CSV/H5 files, filters invalid posts, parses timestamps, and prepares for feature engineering and model training.
Time range: Full year 2021 (2021-01-01 to 2021-12-31)
"""

import pandas as pd
import os
from typing import List, Optional
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认的子版块列表（根据data/raw目录中的实际子版块）
DEFAULT_SUBREDDITS = [
    'stocks', 'wallstreetbets', 'investing', 'stockmarket',
    'options', 'pennystocks', 'finance', 'forex',
    'personalfinance', 'robinhood', 'gme'
]


def load_reddit_subreddit(
    subreddit: str,
    data_dir: str = 'data/raw',
    prefer_h5: bool = True
) -> Optional[pd.DataFrame]:
    """
    从CSV或H5文件加载单个子版块的Reddit数据
    
    Args:
        subreddit: 子版块名称
        data_dir: 数据目录路径
        prefer_h5: 是否优先使用H5格式（更快）
    
    Returns:
        DataFrame包含原始Reddit数据，如果文件不存在则返回None
    """
    subreddit_dir = os.path.join(data_dir, subreddit)
    
    if not os.path.exists(subreddit_dir):
        logger.warning(f"Subreddit directory not found: {subreddit_dir}")
        return None
    
    # 尝试加载H5格式
    h5_path = os.path.join(subreddit_dir, 'submissions_reddit.h5')
    csv_path = os.path.join(subreddit_dir, 'submissions_reddit.csv')
    
    if prefer_h5 and os.path.exists(h5_path):
        try:
            logger.info(f"Loading {subreddit} from H5 format...")
            # 尝试使用pytables读取H5
            df = pd.read_hdf(h5_path, key='df')
            logger.info(f"Successfully loaded {len(df)} records from {h5_path}")
            return df
        except ImportError:
            logger.warning("pytables not available, falling back to CSV")
        except Exception as e:
            logger.warning(f"Error loading H5 file: {e}, falling back to CSV")
    
    # 回退到CSV格式
    if os.path.exists(csv_path):
        try:
            logger.info(f"Loading {subreddit} from CSV format...")
            df = pd.read_csv(csv_path, low_memory=False)
            logger.info(f"Successfully loaded {len(df)} records from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return None
    else:
        logger.warning(f"Neither H5 nor CSV file found for {subreddit}")
        return None


def filter_reddit_posts(
    df: pd.DataFrame,
    min_score: Optional[int] = None
) -> pd.DataFrame:
    """
    过滤Reddit帖子，移除已删除/移除的帖子和空内容
    
    Args:
        df: 原始Reddit数据DataFrame
        min_score: 最小score阈值（可选）
    
    Returns:
        过滤后的DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    original_count = len(df)
    logger.info(f"Starting with {original_count} posts")
    
    # 过滤已删除/移除的帖子
    if 'removed' in df.columns and 'deleted' in df.columns:
        df = df[(df['removed'] == 0) & (df['deleted'] == 0)]
        logger.info(f"After filtering removed/deleted: {len(df)} posts")
    elif 'removed' in df.columns:
        df = df[df['removed'] == 0]
        logger.info(f"After filtering removed: {len(df)} posts")
    elif 'deleted' in df.columns:
        df = df[df['deleted'] == 0]
        logger.info(f"After filtering deleted: {len(df)} posts")
    
    # 过滤空文本内容
    if 'title' in df.columns and 'selftext' in df.columns:
        # 保留至少title或selftext有内容的帖子
        df = df[
            (df['title'].notna() & (df['title'].astype(str).str.strip() != '')) |
            (df['selftext'].notna() & (df['selftext'].astype(str).str.strip() != ''))
        ]
        logger.info(f"After filtering empty content: {len(df)} posts")
    
    # 可选：过滤低质量帖子（基于score）
    if min_score is not None and 'score' in df.columns:
        df = df[df['score'] >= min_score]
        logger.info(f"After filtering by min_score ({min_score}): {len(df)} posts")
    
    filtered_count = original_count - len(df)
    logger.info(f"Filtered out {filtered_count} posts ({filtered_count/original_count*100:.1f}%)")
    
    return df


def parse_reddit_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    解析Reddit时间戳，转换为UTC时区的datetime
    
    Args:
        df: 包含'created'列的DataFrame
    
    Returns:
        添加了'timestamp'列的DataFrame
    """
    if df is None or df.empty:
        return df
    
    if 'created' not in df.columns:
        logger.error("'created' column not found in DataFrame")
        return df
    
    # 将created列转换为datetime
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['created'], errors='coerce')
    
    # 检查是否有无效的时间戳
    invalid_timestamps = df['timestamp'].isna().sum()
    if invalid_timestamps > 0:
        logger.warning(f"Found {invalid_timestamps} invalid timestamps, removing those rows")
        df = df[df['timestamp'].notna()]
    
    # 假设时间戳已经是UTC格式（根据数据样本判断）
    # 如果没有时区信息，添加UTC时区
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    
    logger.info(f"Parsed timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def extract_text_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取并合并文本内容（title + selftext）
    
    Args:
        df: 包含'title'和'selftext'列的DataFrame
    
    Returns:
        添加了'text_content'列的DataFrame
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 处理缺失值，将NaN转换为空字符串
    title = df['title'].fillna('').astype(str) if 'title' in df.columns else pd.Series([''] * len(df))
    selftext = df['selftext'].fillna('').astype(str) if 'selftext' in df.columns else pd.Series([''] * len(df))
    
    # 合并title和selftext，用空格分隔
    df['text_content'] = (title + ' ' + selftext).str.strip()
    
    # 移除完全为空的内容（虽然之前已经过滤过，但这里再次确保）
    df = df[df['text_content'].str.len() > 0]
    
    logger.info(f"Extracted text content for {len(df)} posts")
    
    return df


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31'
) -> pd.DataFrame:
    """
    按日期范围过滤数据
    
    Args:
        df: 包含'timestamp'列的DataFrame
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)，包含该日期
    
    Returns:
        过滤后的DataFrame
    """
    if df is None or df.empty:
        return df
    
    if 'timestamp' not in df.columns:
        logger.error("'timestamp' column not found in DataFrame")
        return df
    
    # 转换为datetime
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)  # 包含结束日期
    
    # 过滤日期范围
    mask = (df['timestamp'] >= start) & (df['timestamp'] < end)
    df_filtered = df[mask].copy()
    
    logger.info(f"Filtered by date range [{start_date}, {end_date}]: {len(df_filtered)} posts")
    
    return df_filtered


def load_multiple_subreddits(
    subreddits: List[str],
    data_dir: str = 'data/raw',
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31',
    min_score: Optional[int] = None,
    prefer_h5: bool = True
) -> pd.DataFrame:
    """
    批量加载多个子版块的数据并合并
    
    Args:
        subreddits: 子版块名称列表
        data_dir: 数据目录路径
        start_date: 开始日期
        end_date: 结束日期
        min_score: 最小score阈值（可选）
        prefer_h5: 是否优先使用H5格式
    
    Returns:
        合并后的DataFrame，包含所有子版块的数据
    """
    all_data = []
    
    for subreddit in subreddits:
        logger.info(f"\nProcessing subreddit: {subreddit}")
        
        # 加载数据
        df = load_reddit_subreddit(subreddit, data_dir, prefer_h5)
        
        if df is None or df.empty:
            logger.warning(f"No data loaded for {subreddit}")
            continue
        
        # 过滤帖子
        df = filter_reddit_posts(df, min_score)
        
        if df.empty:
            logger.warning(f"No valid posts after filtering for {subreddit}")
            continue
        
        # 解析时间戳
        df = parse_reddit_timestamps(df)
        
        if df.empty:
            logger.warning(f"No valid timestamps for {subreddit}")
            continue
        
        # 提取文本内容
        df = extract_text_content(df)
        
        if df.empty:
            logger.warning(f"No valid text content for {subreddit}")
            continue
        
        # 按日期范围过滤
        df = filter_by_date_range(df, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data in date range for {subreddit}")
            continue
        
        # 添加子版块标识
        df['subreddit'] = subreddit
        
        all_data.append(df)
        logger.info(f"Successfully processed {len(df)} posts from {subreddit}")
    
    if not all_data:
        logger.warning("No data loaded from any subreddit")
        return pd.DataFrame()
    
    # 合并所有子版块的数据
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nTotal combined posts: {len(combined_df)}")
    
    return combined_df


def save_processed_reddit_data(
    df: pd.DataFrame,
    output_path: str,
    output_format: str = 'csv'
):
    """
    保存处理后的Reddit数据
    
    Args:
        df: 处理后的DataFrame
        output_path: 输出文件路径
        output_format: 输出格式 ('csv' 或 'h5')
    """
    if df is None or df.empty:
        logger.warning("No data to save")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 选择要保存的列
    columns_to_save = [
        'id', 'subreddit', 'timestamp', 'title', 'selftext', 'text_content',
        'score', 'num_comments', 'upvote_ratio', 'author'
    ]
    
    # 只保存存在的列
    available_columns = [col for col in columns_to_save if col in df.columns]
    df_to_save = df[available_columns].copy()
    
    # 保存数据
    if output_format.lower() == 'h5':
        try:
            df_to_save.to_hdf(output_path, key='df', mode='w', format='table')
            logger.info(f"Saved {len(df_to_save)} records to {output_path} (H5 format)")
        except ImportError:
            logger.warning("pytables not available, saving as CSV instead")
            csv_path = output_path.replace('.h5', '.csv')
            df_to_save.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df_to_save)} records to {csv_path} (CSV format)")
        except Exception as e:
            logger.error(f"Error saving H5 file: {e}, saving as CSV instead")
            csv_path = output_path.replace('.h5', '.csv')
            df_to_save.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df_to_save)} records to {csv_path} (CSV format)")
    else:
        df_to_save.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_to_save)} records to {output_path} (CSV format)")


def validate_reddit_data(df: pd.DataFrame) -> bool:
    """
    验证Reddit数据的完整性
    
    Args:
        df: 处理后的DataFrame
    
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning("Validation failed: Empty dataframe")
        return False
    
    logger.info("\n" + "="*50)
    logger.info("Data Validation Report")
    logger.info("="*50)
    
    # 基本统计
    logger.info(f"Total posts: {len(df)}")
    
    # 检查必需的列
    required_columns = ['id', 'subreddit', 'timestamp', 'text_content']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # 验证时间戳范围
    if 'timestamp' in df.columns:
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        logger.info(f"Timestamp range: {min_ts} to {max_ts}")
    
    # 检查文本内容非空
    if 'text_content' in df.columns:
        empty_text = (df['text_content'].isna() | (df['text_content'].str.len() == 0)).sum()
        if empty_text > 0:
            logger.warning(f"Found {empty_text} posts with empty text content")
    
    # 子版块分布
    if 'subreddit' in df.columns:
        subreddit_counts = df['subreddit'].value_counts()
        logger.info(f"\nSubreddit distribution:")
        for subreddit, count in subreddit_counts.items():
            logger.info(f"  {subreddit}: {count} posts ({count/len(df)*100:.1f}%)")
    
    # 日期分布
    if 'timestamp' in df.columns:
        df['date'] = df['timestamp'].dt.date
        date_counts = df['date'].value_counts().sort_index()
        logger.info(f"\nDate range: {date_counts.index.min()} to {date_counts.index.max()}")
        logger.info(f"Total unique dates: {len(date_counts)}")
    
    # 检查缺失值
    logger.info(f"\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        logger.info(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    logger.info("="*50)
    
    return True


def get_available_subreddits(data_dir: str = 'data/raw') -> List[str]:
    """
    获取可用的子版块列表
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        可用子版块名称列表
    """
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return []
    
    subreddits = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # 检查是否有CSV或H5文件
            csv_file = os.path.join(item_path, 'submissions_reddit.csv')
            h5_file = os.path.join(item_path, 'submissions_reddit.h5')
            if os.path.exists(csv_file) or os.path.exists(h5_file):
                subreddits.append(item)
    
    return sorted(subreddits)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='加载和预处理Reddit数据')
    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=None,
        help='要加载的子版块列表（例如：stocks wallstreetbets）。如果不指定，将加载所有可用子版块'
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
        '--min-score',
        type=int,
        default=None,
        help='最小score阈值（可选，默认不过滤）'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='输出目录，默认：data/processed'
    )
    parser.add_argument(
        '--output-file',
        default='reddit_data.csv',
        help='输出文件名，默认：reddit_data.csv'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'h5'],
        default='csv',
        help='输出格式：csv或h5，默认：csv'
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Reddit数据目录，默认：data/raw'
    )
    parser.add_argument(
        '--list-subreddits',
        action='store_true',
        help='列出所有可用的子版块并退出'
    )
    
    args = parser.parse_args()
    
    # 如果只是列出子版块
    if args.list_subreddits:
        subreddits = get_available_subreddits(args.data_dir)
        print(f"\nAvailable subreddits ({len(subreddits)}):")
        for subreddit in subreddits:
            print(f"  - {subreddit}")
        return
    
    # 确定要加载的子版块列表
    if args.subreddits:
        subreddits = args.subreddits
    else:
        subreddits = get_available_subreddits(args.data_dir)
        if not subreddits:
            logger.error(f"No subreddits found in {args.data_dir}")
            return
    
    logger.info(f"Will load data from {len(subreddits)} subreddits: {', '.join(subreddits)}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # 加载数据
    df = load_multiple_subreddits(
        subreddits=subreddits,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        min_score=args.min_score,
        prefer_h5=True
    )
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    # 验证数据
    if not validate_reddit_data(df):
        logger.warning("Data validation failed, but continuing...")
    
    # 保存数据
    output_path = os.path.join(args.output_dir, args.output_file)
    if args.format == 'h5' and not args.output_file.endswith('.h5'):
        output_path = output_path.replace('.csv', '.h5')
    elif args.format == 'csv' and not args.output_file.endswith('.csv'):
        output_path = output_path.replace('.h5', '.csv')
    
    save_processed_reddit_data(df, output_path, args.format)
    
    logger.info("\nReddit data loading completed successfully!")


if __name__ == '__main__':
    main()

