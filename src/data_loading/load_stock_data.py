"""
股票价格数据收集脚本

从Yahoo Finance获取历史股票价格数据（每小时粒度），用于与Reddit数据对齐。
时间范围：2021年全年（2021-01-01 至 2021-12-31）
"""

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timezone
import argparse
from typing import List, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 根据计划定义的股票列表
PRIORITY_STOCKS = {
    'high': ['GME', 'AMC', 'BB', 'PLTR', 'TSLA'],
    'medium': ['NIO', 'SOFI', 'HOOD', 'SNDL', 'WISH'],
    'low': ['AAPL', 'AMD', 'NOK']
}

# 所有股票代码
ALL_STOCKS = PRIORITY_STOCKS['high'] + PRIORITY_STOCKS['medium'] + PRIORITY_STOCKS['low']


def expand_daily_to_hourly(daily_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    将每日数据扩展为每小时数据
    
    对于每个交易日，将价格数据复制到交易时间内的每个小时（9:30 AM - 4:00 PM EST）
    然后转换为UTC时间
    
    Args:
        daily_df: 每日数据DataFrame，索引为日期
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        每小时数据的DataFrame
    """
    # 重置索引，将日期作为列
    daily_df = daily_df.reset_index()
    
    # 找到日期列
    date_col = None
    for col in ['Date', 'Datetime', 'index']:
        if col in daily_df.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.warning("Could not find date column, using index")
        daily_df['date'] = daily_df.index
        date_col = 'date'
    
    # 确保日期列是datetime类型
    daily_df[date_col] = pd.to_datetime(daily_df[date_col])
    
    # 创建每小时的数据
    hourly_data = []
    
    for _, row in daily_df.iterrows():
        date = row[date_col]
        
        # 如果日期没有时区信息，假设是市场时间（EST）
        if date.tz is None:
            # 本地化为EST时区
            date_est = pd.Timestamp(date).tz_localize('US/Eastern')
        else:
            date_est = date.tz_convert('US/Eastern')
        
        # 交易时间：9:30 AM - 4:00 PM EST
        # 创建该交易日的每小时时间点
        trading_hours_est = []
        
        # 9:30 AM (开盘)
        trading_hours_est.append(date_est.replace(hour=9, minute=30, second=0, microsecond=0))
        
        # 10:00 AM - 3:00 PM (整点)
        for hour in range(10, 16):
            trading_hours_est.append(date_est.replace(hour=hour, minute=0, second=0, microsecond=0))
        
        # 4:00 PM (收盘)
        trading_hours_est.append(date_est.replace(hour=16, minute=0, second=0, microsecond=0))
        
        # 为每个交易小时创建一行数据
        for hour_ts_est in trading_hours_est:
            # 转换为UTC
            utc_ts = hour_ts_est.tz_convert('UTC')
            
            # 创建新行
            new_row = row.copy()
            new_row['timestamp'] = utc_ts
            hourly_data.append(new_row)
    
    # 创建新的DataFrame
    hourly_df = pd.DataFrame(hourly_data)
    
    # 删除原来的日期列（如果存在）
    if date_col in hourly_df.columns and date_col != 'timestamp':
        hourly_df = hourly_df.drop(columns=[date_col])
    
    # 确保timestamp列存在
    if 'timestamp' not in hourly_df.columns:
        logger.error("Failed to create timestamp column")
        return pd.DataFrame()
    
    # 按时间戳排序
    hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)
    
    return hourly_df


def download_stock_data(
    symbol: str,
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31',
    interval: str = '1h',
    retry_count: int = 3,
    retry_delay: int = 5
) -> Optional[pd.DataFrame]:
    """
    下载单个股票的历史价格数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        interval: 数据间隔 ('1h' for hourly)
        retry_count: 重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    for attempt in range(retry_count):
        try:
            logger.info(f"Downloading {symbol} data (attempt {attempt + 1}/{retry_count})...")
            
            # 创建ticker对象
            ticker = yf.Ticker(symbol)
            
            # 下载历史数据
            # 注意：yfinance的1h数据通常只支持最近60天，对于2021年的历史数据
            # 我们需要使用1d数据，然后通过扩展生成每小时数据
            # 对于历史数据，使用1d间隔更可靠
            
            # 对于2021年的历史数据，使用1d间隔
            if interval == '1h':
                logger.info(f"Note: 1h interval may not be available for historical data, using 1d instead")
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    auto_adjust=True,
                    prepost=False
                )
                # 将1d数据扩展为每小时数据
                if not hist.empty:
                    hist = expand_daily_to_hourly(hist, start_date, end_date)
            else:
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=False
                )
            
            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # 重置索引，将DatetimeIndex转换为列
            hist = hist.reset_index()
            
            # 重命名列
            hist.columns = [col.lower().replace(' ', '_') for col in hist.columns]
            
            # 确保时间戳列存在
            if 'datetime' in hist.columns:
                hist.rename(columns={'datetime': 'timestamp'}, inplace=True)
            elif 'date' in hist.columns:
                hist.rename(columns={'date': 'timestamp'}, inplace=True)
            
            # 选择需要的列
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in hist.columns]
            
            if 'timestamp' not in hist.columns:
                logger.error(f"Timestamp column not found for {symbol}")
                return None
            
            # 确保时间戳是datetime类型
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])
            
            # 转换为UTC时区（如果还没有）
            if hist['timestamp'].dt.tz is None:
                # 假设是市场时间（EST/EDT），转换为UTC
                # 美国股市交易时间：9:30 AM - 4:00 PM EST (UTC-5) 或 EDT (UTC-4)
                hist['timestamp'] = hist['timestamp'].dt.tz_localize('US/Eastern').dt.tz_convert('UTC')
            else:
                hist['timestamp'] = hist['timestamp'].dt.tz_convert('UTC')
            
            # 选择并重新排列列
            hist = hist[available_cols]
            
            # 按时间戳排序
            hist = hist.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully downloaded {len(hist)} records for {symbol}")
            logger.info(f"Date range: {hist['timestamp'].min()} to {hist['timestamp'].max()}")
            
            return hist
            
        except Exception as e:
            logger.error(f"Error downloading {symbol} (attempt {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download {symbol} after {retry_count} attempts")
                return None
    
    return None


def save_stock_data(df: pd.DataFrame, symbol: str, output_dir: str = 'data/stock_prices'):
    """
    保存股票数据到CSV文件
    
    Args:
        df: 股票数据DataFrame
        symbol: 股票代码
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{symbol}_2021.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 保存为CSV
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {symbol} data to {filepath}")
    
    return filepath


def validate_data(df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> bool:
    """
    验证数据的完整性
    
    Args:
        df: 股票数据DataFrame
        symbol: 股票代码
        start_date: 期望的开始日期
        end_date: 期望的结束日期
    
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"Validation failed for {symbol}: Empty dataframe")
        return False
    
    # 检查时间范围
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    data_start = df['timestamp'].min()
    data_end = df['timestamp'].max()
    
    logger.info(f"{symbol} data range: {data_start} to {data_end}")
    
    # 检查是否有缺失值
    missing = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"{symbol} has missing values:\n{missing[missing > 0]}")
    
    # 检查数据量（至少应该有交易日的数据）
    # 2021年大约有252个交易日，每小时数据应该有更多
    if len(df) < 100:
        logger.warning(f"{symbol} has very few data points: {len(df)}")
    
    return True


def download_multiple_stocks(
    symbols: List[str],
    start_date: str = '2021-01-01',
    end_date: str = '2021-12-31',
    output_dir: str = 'data/stock_prices',
    delay_between_downloads: float = 1.0
):
    """
    批量下载多个股票的数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        delay_between_downloads: 每次下载之间的延迟（秒），避免API限制
    """
    logger.info(f"Starting batch download for {len(symbols)} stocks")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    successful = []
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        # 下载数据
        df = download_stock_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # 验证数据
            if validate_data(df, symbol, start_date, end_date):
                # 保存数据
                save_stock_data(df, symbol, output_dir)
                successful.append(symbol)
            else:
                logger.warning(f"Skipping {symbol} due to validation failure")
                failed.append(symbol)
        else:
            failed.append(symbol)
        
        # 延迟以避免API限制
        if i < len(symbols):
            time.sleep(delay_between_downloads)
    
    # 总结
    logger.info("\n" + "="*50)
    logger.info("Download Summary:")
    logger.info(f"Successful: {len(successful)} stocks")
    logger.info(f"Failed: {len(failed)} stocks")
    
    if successful:
        logger.info(f"\nSuccessfully downloaded: {', '.join(successful)}")
    if failed:
        logger.warning(f"\nFailed to download: {', '.join(failed)}")
    
    return successful, failed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载股票历史价格数据')
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='股票代码列表（例如：GME AMC TSLA）。如果不指定，将下载所有计划中的股票'
    )
    parser.add_argument(
        '--priority',
        choices=['high', 'medium', 'low', 'all'],
        default='all',
        help='下载优先级：high（高优先级）、medium（中优先级）、low（低优先级）、all（全部）'
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
        '--output-dir',
        default='data/stock_prices',
        help='输出目录，默认：data/stock_prices'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='每次下载之间的延迟（秒），默认：1.0'
    )
    
    args = parser.parse_args()
    
    # 确定要下载的股票列表
    if args.symbols:
        symbols = args.symbols
    else:
        if args.priority == 'all':
            symbols = ALL_STOCKS
        else:
            symbols = PRIORITY_STOCKS.get(args.priority, ALL_STOCKS)
    
    logger.info(f"Will download data for {len(symbols)} stocks: {', '.join(symbols)}")
    
    # 批量下载
    download_multiple_stocks(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        delay_between_downloads=args.delay
    )


if __name__ == '__main__':
    main()

