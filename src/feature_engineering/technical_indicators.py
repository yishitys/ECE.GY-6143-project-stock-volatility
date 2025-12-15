"""
Technical Indicators Calculation Module

Calculate stock technical indicators: moving averages, RSI, MACD, volatility, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_returns(df: pd.DataFrame, close_col: str, method: str = 'simple') -> pd.Series:
    """
    Calculate returns
    
    Args:
        df: DataFrame containing price data
        close_col: Close price column name
        method: Calculation method ('simple' or 'log')
    
    Returns:
        Returns series
    """
    if method == 'simple':
        return df[close_col].pct_change()
    elif method == 'log':
        return np.log(df[close_col] / df[close_col].shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")


def calculate_sma(df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        df: DataFrame containing price data
        price_col: Price column name
        window: Window size (hours)
    
    Returns:
        Moving average series
    """
    return df[price_col].rolling(window=window, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, price_col: str, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        df: DataFrame containing price data
        price_col: Price column name
        window: Window size (hours)
    
    Returns:
        Exponential moving average series
    """
    return df[price_col].ewm(span=window, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, price_col: str, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df: DataFrame containing price data
        price_col: Price column name
        window: Window size, default 14
    
    Returns:
        RSI series (0-100)
    """
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    price_col: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD indicator (Moving Average Convergence Divergence)
    
    Args:
        df: DataFrame containing price data
        price_col: Price column name
        fast: Fast EMA period, default 12
        slow: Slow EMA period, default 26
        signal: Signal line EMA period, default 9
    
    Returns:
        DataFrame with MACD, signal line, and histogram
    """
    ema_fast = calculate_ema(df, price_col, fast)
    ema_slow = calculate_ema(df, price_col, slow)
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    result = pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    })
    
    return result


def calculate_atr(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    window: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame containing price data
        high_col: High price column name
        low_col: Low price column name
        close_col: Close price column name
        window: Window size, default 14
    
    Returns:
        ATR series
    """
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift(1))
    low_close = np.abs(df[low_col] - df[close_col].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    
    return atr


def calculate_volatility(
    df: pd.DataFrame,
    returns_col: str,
    window: int = 24
) -> pd.Series:
    """
    计算历史波动率（滚动标准差）
    
    Args:
        df: 包含收益率数据的DataFrame
        returns_col: 收益率列名
        window: 窗口大小（小时数），默认24小时
    
    Returns:
        波动率序列
    """
    return df[returns_col].rolling(window=window, min_periods=1).std()


def calculate_technical_indicators(
    df: pd.DataFrame,
    close_col: str = None,
    high_col: str = None,
    low_col: str = None,
    volume_col: str = None,
    windows: Dict[str, int] = None
) -> pd.DataFrame:
    """
    计算所有技术指标
    
    Args:
        df: 包含股票价格数据的DataFrame
        close_col: 收盘价列名（如果为None则自动检测）
        high_col: 最高价列名（如果为None则自动检测）
        low_col: 最低价列名（如果为None则自动检测）
        volume_col: 成交量列名（如果为None则自动检测）
        windows: 窗口大小字典，例如 {'sma': [5, 10, 20], 'rsi': 14}
    
    Returns:
        添加了技术指标的DataFrame
    """
    df = df.copy()
    
    # 自动检测列名（支持多种命名格式）
    if close_col is None:
        # 尝试多种格式：'close', '_close', 'GME_close'等
        close_cols = [col for col in df.columns if 'close' in col.lower() and col.lower().endswith('close')]
        if not close_cols:
            close_cols = [col for col in df.columns if col.endswith('_close')]
        if close_cols:
            close_col = close_cols[0]
        else:
            raise ValueError(f"无法找到收盘价列，请指定close_col参数。可用列: {list(df.columns)}")
    
    if high_col is None:
        high_cols = [col for col in df.columns if 'high' in col.lower() and col.lower().endswith('high')]
        if not high_cols:
            high_cols = [col for col in df.columns if col.endswith('_high')]
        high_col = high_cols[0] if high_cols else None
    
    if low_col is None:
        low_cols = [col for col in df.columns if 'low' in col.lower() and col.lower().endswith('low')]
        if not low_cols:
            low_cols = [col for col in df.columns if col.endswith('_low')]
        low_col = low_cols[0] if low_cols else None
    
    if volume_col is None:
        volume_cols = [col for col in df.columns if 'volume' in col.lower() and col.lower().endswith('volume')]
        if not volume_cols:
            volume_cols = [col for col in df.columns if col.endswith('_volume')]
        volume_col = volume_cols[0] if volume_cols else None
    
    # Default window sizes
    if windows is None:
        windows = {
            'sma': [5, 10, 20],
            'ema': [5, 10, 20],
            'rsi': 14,
            'atr': 14,
            'volatility': 24
        }
    
    logger.info(f"计算技术指标，使用列: close={close_col}, high={high_col}, low={low_col}")
    
    # 1. 收益率
    df['returns'] = calculate_returns(df, close_col, method='simple')
    df['log_returns'] = calculate_returns(df, close_col, method='log')
    df['price_change_pct'] = df['returns'] * 100
    
    # 2. 移动平均
    if 'sma' in windows:
        for window in windows['sma']:
            df[f'sma_{window}'] = calculate_sma(df, close_col, window)
            df[f'price_vs_sma_{window}'] = (df[close_col] - df[f'sma_{window}']) / df[close_col] * 100
    
    if 'ema' in windows:
        for window in windows['ema']:
            df[f'ema_{window}'] = calculate_ema(df, close_col, window)
            df[f'price_vs_ema_{window}'] = (df[close_col] - df[f'ema_{window}']) / df[close_col] * 100
    
    # 3. RSI
    if 'rsi' in windows:
        rsi_window = windows['rsi'] if isinstance(windows['rsi'], int) else 14
        df['rsi'] = calculate_rsi(df, close_col, window=rsi_window)
    
    # 4. MACD
    df_macd = calculate_macd(df, close_col)
    df['macd'] = df_macd['macd']
    df['macd_signal'] = df_macd['macd_signal']
    df['macd_histogram'] = df_macd['macd_histogram']
    
    # 5. ATR
    if high_col and low_col and 'atr' in windows:
        atr_window = windows['atr'] if isinstance(windows['atr'], int) else 14
        df['atr'] = calculate_atr(df, high_col, low_col, close_col, window=atr_window)
        df['atr_pct'] = df['atr'] / df[close_col] * 100  # ATR百分比
    
    # 6. 波动率
    if 'volatility' in windows:
        vol_window = windows['volatility'] if isinstance(windows['volatility'], int) else 24
        df['volatility'] = calculate_volatility(df, 'returns', window=vol_window)
        df['log_volatility'] = calculate_volatility(df, 'log_returns', window=vol_window)
    
    # 7. 成交量指标
    if volume_col:
        df['volume_ma_5'] = calculate_sma(df, volume_col, 5)
        df['volume_ma_20'] = calculate_sma(df, volume_col, 20)
        df['volume_change_pct'] = df[volume_col].pct_change() * 100
        df['volume_ratio'] = df[volume_col] / (df['volume_ma_20'] + 1e-8)
    
    # 8. 滞后特征
    for lag in [1, 2, 3]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag) if 'volatility' in df.columns else None
    
    # 9. 滚动统计特征
    for window in [5, 10, 20]:
        df[f'returns_rolling_mean_{window}'] = df['returns'].rolling(window=window, min_periods=1).mean()
        df[f'returns_rolling_std_{window}'] = df['returns'].rolling(window=window, min_periods=1).std()
        df[f'returns_rolling_max_{window}'] = df['returns'].rolling(window=window, min_periods=1).max()
        df[f'returns_rolling_min_{window}'] = df['returns'].rolling(window=window, min_periods=1).min()
    
    logger.info(f"技术指标计算完成，新增 {len([col for col in df.columns if col not in df.columns])} 个特征列")
    
    return df


if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    
    # 添加src目录到路径
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 加载测试数据
    data_path = 'data/processed/merged_data_GME.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"加载数据: {len(df)} 条记录")
        
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 显示新增的特征列
        new_cols = [col for col in df.columns if col not in ['timestamp', 'post_count', 'total_comments', 
                                                               'total_score', 'unique_authors', 'stock_symbol',
                                                               'has_reddit_data', 'has_stock_data']]
        logger.info(f"新增特征列数量: {len(new_cols)}")
        logger.info(f"示例特征列: {new_cols[:10]}")
        
        # 保存结果
        output_path = 'data/processed/merged_data_GME_with_indicators.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"已保存包含技术指标的数据到: {output_path}")
    else:
        logger.warning(f"测试数据文件不存在: {data_path}")


