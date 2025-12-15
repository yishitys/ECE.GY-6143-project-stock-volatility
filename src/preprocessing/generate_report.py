"""
Generate Preprocessing Report

Creates data preprocessing statistical report including data quality assessment,
time range, missing values, and other information.
"""

import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_preprocessing_report(
    merged_df: pd.DataFrame,
    stock_symbol: str,
    output_dir: str = 'data/processed',
    timestamp_col: str = 'timestamp'
) -> str:
    """
    Generate preprocessing report
    
    Args:
        merged_df: Merged DataFrame
        stock_symbol: Stock symbol
        output_dir: Output directory
        timestamp_col: Timestamp column name
    
    Returns:
        Report file path
    """
    if merged_df is None or merged_df.empty:
        logger.error("Empty DataFrame provided")
        return ""
    
    logger.info("Generating preprocessing report...")
    
    # Create report content
    report_lines = []
    report_lines.append("# Data Preprocessing Report")
    report_lines.append("")
    report_lines.append(f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Stock Symbol**: {stock_symbol}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 1. Data Overview
    report_lines.append("## 1. Data Overview")
    report_lines.append("")
    report_lines.append("### 1.1 Basic Information")
    report_lines.append("")
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|--------|--------|")
    report_lines.append(f"| Total Records | {len(merged_df):,} |")  
    
    if timestamp_col in merged_df.columns:
        min_ts = merged_df[timestamp_col].min()
        max_ts = merged_df[timestamp_col].max()
        report_lines.append(f"| Start Time | {min_ts} |")
        report_lines.append(f"| End Time | {max_ts} |")
        
        # Calculate time span
        time_span = max_ts - min_ts
        days = time_span.days
        hours = time_span.total_seconds() / 3600
        report_lines.append(f"| Time Span | {days} days ({hours:.0f} hours) |")
    
    report_lines.append("")
    
    # 2. Reddit Data Statistics
    report_lines.append("## 2. Reddit Data Statistics")
    report_lines.append("")
    
    if 'post_count' in merged_df.columns:
        report_lines.append("### 2.1 Post Statistics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|--------|")
        report_lines.append(f"| Total Posts | {merged_df['post_count'].sum():,} |")
        report_lines.append(f"| Average Posts per Hour | {merged_df['post_count'].mean():.2f} |")
        report_lines.append(f"| Median Posts per Hour | {merged_df['post_count'].median():.2f} |")
        report_lines.append(f"| Maximum Posts per Hour | {merged_df['post_count'].max():,} |")
        report_lines.append(f"| Hours with Posts | {(merged_df['post_count'] > 0).sum():,} |")
        report_lines.append(f"| Hours without Posts | {(merged_df['post_count'] == 0).sum():,} |")
        report_lines.append("")
        
        # Post distribution
        report_lines.append("### 2.2 Post Distribution")
        report_lines.append("")
        report_lines.append(f"| Post Count Range | Hours | Percentage |")
        report_lines.append(f"|------------------|--------|--------|")
        
        bins = [0, 1, 5, 10, 50, 100, float('inf')]
        labels = ['0', '1', '2-5', '6-10', '11-50', '51-100', '100+']
        
        for i in range(len(bins)-1):
            if i == 0:
                count = (merged_df['post_count'] == 0).sum()
            elif i == len(bins)-2:
                count = (merged_df['post_count'] > bins[i]).sum()
            else:
                count = ((merged_df['post_count'] > bins[i]) & (merged_df['post_count'] <= bins[i+1])).sum()
            
            percentage = count / len(merged_df) * 100
            report_lines.append(f"| {labels[i]} | {count:,} | {percentage:.1f}% |")
        
        report_lines.append("")
    
    if 'total_comments' in merged_df.columns:
        report_lines.append("### 2.3 Comment Statistics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|--------|")
        report_lines.append(f"| Total Comments | {merged_df['total_comments'].sum():,} |")
        report_lines.append(f"| Average Comments per Hour | {merged_df['total_comments'].mean():.2f} |")
        report_lines.append("")
    
    if 'unique_authors' in merged_df.columns:
        report_lines.append("### 2.4 Author Statistics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|--------|")
        report_lines.append(f"| Total Unique Authors | {merged_df['unique_authors'].sum():,} |")
        report_lines.append(f"| Average Unique Authors per Hour | {merged_df['unique_authors'].mean():.2f} |")
        report_lines.append("")
    
    # 3. Stock Data Statistics
    report_lines.append("## 3. Stock Data Statistics")
    report_lines.append("")
    
    # Find stock price column
    price_cols = [col for col in merged_df.columns if f'{stock_symbol}_close' in col or 'close' in col.lower()]
    if price_cols:
        price_col = price_cols[0]
        report_lines.append("### 3.1 Price Statistics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|--------|")
        
        price_data = merged_df[price_col].dropna()
        if len(price_data) > 0:
            report_lines.append(f"| Average Price | ${price_data.mean():.2f} |")
            report_lines.append(f"| Median Price | ${price_data.median():.2f} |")
            report_lines.append(f"| Minimum Price | ${price_data.min():.2f} |")
            report_lines.append(f"| Maximum Price | ${price_data.max():.2f} |")
            report_lines.append(f"| Price Std Dev | ${price_data.std():.2f} |")
        
        report_lines.append("")
    
    # 4. Data Quality Assessment
    report_lines.append("## 4. Data Quality Assessment")
    report_lines.append("")
    
    report_lines.append("### 4.1 Missing Values Statistics")
    report_lines.append("")
    report_lines.append(f"| Column | Missing Values | Missing Rate |")
    report_lines.append(f"|--------|---|---|")
    
    missing = merged_df.isnull().sum()
    for col, count in missing[missing > 0].items():
        percentage = count / len(merged_df) * 100
        report_lines.append(f"| {col} | {count:,} | {percentage:.1f}% |")
    
    if missing.sum() == 0:
        report_lines.append("| - | 0 | 0% |")
    
    report_lines.append("")
    
    # 数据可用性
    if 'has_reddit_data' in merged_df.columns:
        report_lines.append("### 4.2 数据可用性")
        report_lines.append("")
        report_lines.append(f"| 数据类型 | 可用小时数 | 可用率 |")
        report_lines.append(f"|---------|-----------|--------|")
        
        reddit_available = merged_df['has_reddit_data'].sum()
        reddit_rate = reddit_available / len(merged_df) * 100
        report_lines.append(f"| Reddit数据 | {reddit_available:,} | {reddit_rate:.1f}% |")
        
        if 'has_stock_data' in merged_df.columns:
            stock_available = merged_df['has_stock_data'].sum()
            stock_rate = stock_available / len(merged_df) * 100
            report_lines.append(f"| 股票数据 | {stock_available:,} | {stock_rate:.1f}% |")
        
        # 同时有Reddit和股票数据的小时数
        both_available = ((merged_df['has_reddit_data'] == 1) & 
                         (merged_df.get('has_stock_data', pd.Series([0]*len(merged_df))) == 1)).sum()
        both_rate = both_available / len(merged_df) * 100
        report_lines.append(f"| 同时有Reddit和股票数据 | {both_available:,} | {both_rate:.1f}% |")
        
        report_lines.append("")
    
    # 5. Time Distribution
    if timestamp_col in merged_df.columns:
        report_lines.append("## 5. Time Distribution")
        report_lines.append("")
        
        merged_df['date'] = merged_df[timestamp_col].dt.date
        merged_df['hour'] = merged_df[timestamp_col].dt.hour
        merged_df['weekday'] = merged_df[timestamp_col].dt.weekday
        
        report_lines.append("### 5.1 Distribution by Date")
        report_lines.append("")
        date_counts = merged_df['date'].value_counts().sort_index()
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|--------|")
        report_lines.append(f"| Unique Dates | {len(date_counts)} |")
        report_lines.append(f"| Average Records per Day | {date_counts.mean():.2f} |")
        report_lines.append("")
        
        report_lines.append("### 5.2 Distribution by Hour")
        report_lines.append("")
        hour_counts = merged_df['hour'].value_counts().sort_index()
        report_lines.append(f"| Hour | Records |")
        report_lines.append(f"|------|--------|")
        for hour, count in hour_counts.items():
            report_lines.append(f"| {hour:02d}:00 | {count:,} |")
        report_lines.append("")
        
        report_lines.append("### 5.3 Distribution by Weekday")
        report_lines.append("")
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = merged_df['weekday'].value_counts().sort_index()
        report_lines.append(f"| Weekday | Records |")
        report_lines.append(f"|--------|--------|")
        for weekday, count in weekday_counts.items():
            report_lines.append(f"| {weekday_names[weekday]} | {count:,} |")
        report_lines.append("")
    
    # 6. Summary
    report_lines.append("## 6. Summary")
    report_lines.append("")
    report_lines.append("### Data Quality Assessment")
    report_lines.append("")
    
    # Assess data quality
    quality_score = 100
    issues = []
    
    if missing.sum() > 0:
        max_missing_rate = (missing / len(merged_df)).max() * 100
        if max_missing_rate > 50:
            quality_score -= 30
            issues.append(f"High missing rate: {max_missing_rate:.1f}%")
        elif max_missing_rate > 20:
            quality_score -= 15
            issues.append(f"Moderate missing rate: {max_missing_rate:.1f}%")
    
    if 'post_count' in merged_df.columns:
        zero_post_rate = (merged_df['post_count'] == 0).sum() / len(merged_df) * 100
        if zero_post_rate > 80:
            quality_score -= 20
            issues.append(f"High zero post rate: {zero_post_rate:.1f}%")
    
    if quality_score >= 90:
        quality_level = "Excellent"
    elif quality_score >= 70:
        quality_level = "Good"
    elif quality_score >= 50:
        quality_level = "Fair"
    else:
        quality_level = "Needs Improvement"
    
    report_lines.append(f"**Data Quality Score**: {quality_score}/100 ({quality_level})")
    report_lines.append("")
    
    if issues:
        report_lines.append("**Issues Identified**:")
        report_lines.append("")
        for issue in issues:
            report_lines.append(f"- {issue}")
    else:
        report_lines.append("**No major issues found**")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*Report automatically generated by data preprocessing script*")
    
    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'preprocessing_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {report_path}")
    
    return report_path

