"""
目标变量计算模块

计算下一小时的波动率作为预测目标。
支持多种波动率定义方式。
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_log_return(close_prices: pd.Series) -> pd.Series:
    """
    Calculate log returns
    
    Args:
        close_prices: Close price series
    
    Returns:
        Log return series
    """
    return np.log(close_prices / close_prices.shift(1))


def calculate_volatility_target(
    df: pd.DataFrame,
    method: str = 'log_return_abs',
    close_col: str = None,
    high_col: str = None,
    low_col: str = None,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Calculate next hour volatility as target variable
    
    Args:
        df: DataFrame containing stock price data
        method: Volatility calculation method
            - 'log_return_abs': Absolute value of next hour log return
            - 'price_range': Next hour price range (high - low) / close
            - 'realized_vol': Next hour realized volatility (standard deviation)
        close_col: Close price column name, if None auto-detect (format: {SYMBOL}_close)
        high_col: High price column name, if None auto-detect
        low_col: Low price column name, if None auto-detect
        timestamp_col: Timestamp column name
    
    Returns:
        DataFrame with added target variable
    """
    df = df.copy()
    
    # Ensure sorted by time
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Auto-detect column names
    if close_col is None:
        close_cols = [col for col in df.columns if col.endswith('_close')]
        if close_cols:
            close_col = close_cols[0]
        else:
            raise ValueError("Cannot find close price column, please specify close_col parameter")
    
    if high_col is None:
        high_cols = [col for col in df.columns if col.endswith('_high')]
        if high_cols:
            high_col = high_cols[0]
        else:
            high_col = None
    
    if low_col is None:
        low_cols = [col for col in df.columns if col.endswith('_low')]
        if low_cols:
            low_col = low_cols[0]
        else:
            low_col = None
    
    logger.info(f"Using columns: close={close_col}, high={high_col}, low={low_col}")
    logger.info(f"Calculation method: {method}")
    
    # Calculate target variables
    if method == 'log_return_abs':
        # Method 1: Absolute value of next hour log return
        next_close = df[close_col].shift(-1)
        current_close = df[close_col]
        target = np.abs(np.log(next_close / current_close))
        target_name = 'target_volatility_log_return_abs'
    
    elif method == 'price_range':
        # Method 2: Next hour price range
        if high_col is None or low_col is None:
            raise ValueError("price_range method requires high and low columns")
        next_high = df[high_col].shift(-1)
        next_low = df[low_col].shift(-1)
        next_close = df[close_col].shift(-1)
        target = (next_high - next_low) / next_close
        target_name = 'target_volatility_price_range'
    
    elif method == 'realized_vol':
        # Method 3: Next hour realized volatility (using rolling window standard deviation)
        # Simplified to next hour log return standard deviation
        # Since we only have hourly data, we use next hour log return
        next_close = df[close_col].shift(-1)
        current_close = df[close_col]
        returns = np.log(next_close / current_close)
        # Use rolling window to calculate standard deviation (window size=1, i.e., single point)
        target = returns.abs()  # Simplified version, should be rolling std in practice
        target_name = 'target_volatility_realized'
    
    else:
        raise ValueError(f"Unknown volatility calculation method: {method}")
    
    # Add target variable to DataFrame
    df[target_name] = target
    
    # Last row has no next hour data, set as NaN
    df.loc[df.index[-1], target_name] = np.nan
    
    logger.info(f"Target variable '{target_name}' calculated successfully")
    logger.info(f"Valid values: {df[target_name].notna().sum()} / {len(df)}")
    
    return df


def calculate_multiple_targets(
    df: pd.DataFrame,
    methods: list = ['log_return_abs', 'price_range'],
    **kwargs
) -> pd.DataFrame:
    """
    Calculate multiple target variables
    
    Args:
        df: DataFrame containing stock price data
        methods: List of volatility calculation methods
        **kwargs: Additional parameters to pass to calculate_volatility_target
    
    Returns:
        DataFrame with multiple target variables added
    """
    df = df.copy()
    
    for method in methods:
        try:
            df = calculate_volatility_target(df, method=method, **kwargs)
            logger.info(f"Successfully calculated target variable: {method}")
        except Exception as e:
            logger.warning(f"Error calculating target variable {method}: {e}")
    
    return df


def analyze_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'target_volatility_log_return_abs'
) -> Dict:
    """
    Analyze target variable distribution
    
    Args:
        df: DataFrame containing target variable
        target_col: Target variable column name
    
    Returns:
        Dictionary with statistical information
    """
    if target_col not in df.columns:
        raise ValueError(f"Target variable column '{target_col}' does not exist")
    
    target = df[target_col].dropna()
    
    if len(target) == 0:
        logger.warning("Target variable has no valid values")
        return {}
    
    stats = {
        'count': len(target),
        'mean': float(target.mean()),
        'median': float(target.median()),
        'std': float(target.std()),
        'min': float(target.min()),
        'max': float(target.max()),
        'q25': float(target.quantile(0.25)),
        'q75': float(target.quantile(0.75)),
        'missing_count': int(df[target_col].isna().sum()),
        'missing_rate': float(df[target_col].isna().sum() / len(df))
    }
    
    logger.info(f"Target variable '{target_col}' statistics:")
    logger.info(f"  Valid count: {stats['count']}")
    logger.info(f"  Mean: {stats['mean']:.6f}")
    logger.info(f"  Median: {stats['median']:.6f}")
    logger.info(f"  Std Dev: {stats['std']:.6f}")
    logger.info(f"  Min: {stats['min']:.6f}")
    logger.info(f"  Max: {stats['max']:.6f}")
    logger.info(f"  Missing count: {stats['missing_count']} ({stats['missing_rate']*100:.2f}%)")
    
    return stats


def save_target_analysis_report(
    stats: Dict,
    output_path: str,
    target_col: str = 'target_volatility_log_return_abs'
):
    """
    Save target variable analysis report
    
    Args:
        stats: Dictionary with statistical information
        output_path: Output file path
        target_col: Target variable column name
    """
    report = f"""# Target Variable Analysis Report

**Target Variable**: {target_col}
**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Statistical Information

| Metric | Value |
|--------|--------|
| Valid Count | {stats.get('count', 0):,} |
| Missing Count | {stats.get('missing_count', 0):,} |
| Missing Rate | {stats.get('missing_rate', 0)*100:.2f}% |

### Descriptive Statistics

| Metric | Value |
|--------|--------|
| Mean | {stats.get('mean', 0):.6f} |
| Median | {stats.get('median', 0):.6f} |
| Std Dev | {stats.get('std', 0):.6f} |
| Min | {stats.get('min', 0):.6f} |
| Max | {stats.get('max', 0):.6f} |
| 25% Quantile | {stats.get('q25', 0):.6f} |
| 75% Quantile | {stats.get('q75', 0):.6f} |

---

## Notes

- Target variable represents next hour volatility
- Last record has no next hour data, so target variable is NaN
- Recommended to remove rows with NaN target when training models
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Target variable analysis report saved to: {output_path}")


if __name__ == '__main__':
    # Test code
    import sys
    import os
    
    # Add src directory to path
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Load test data
    data_path = 'data/processed/merged_data_GME.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"Loaded data: {len(df)} records")
        
        # Calculate target variables
        df = calculate_multiple_targets(df, methods=['log_return_abs', 'price_range'])
        
        # Analyze target variables
        stats = analyze_target_distribution(df, 'target_volatility_log_return_abs')
        
        # Save report
        report_path = 'data/processed/target_analysis_report.md'
        save_target_analysis_report(stats, report_path)
        
        # Save updated data with target variables
        output_path = 'data/processed/merged_data_GME_with_target.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data with target variables to: {output_path}")
    else:
        logger.warning(f"Test data file does not exist: {data_path}")


