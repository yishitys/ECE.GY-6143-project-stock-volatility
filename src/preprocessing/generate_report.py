"""
生成预处理报告

创建数据预处理统计报告，包括数据质量评估、时间范围、缺失值等信息。
"""

import pandas as pd
import os
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
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
    生成预处理报告
    
    Args:
        merged_df: 合并后的DataFrame
        stock_symbol: 股票代码
        output_dir: 输出目录
        timestamp_col: 时间戳列名
    
    Returns:
        报告文件路径
    """
    if merged_df is None or merged_df.empty:
        logger.error("Empty DataFrame provided")
        return ""
    
    logger.info("Generating preprocessing report...")
    
    # 创建报告内容
    report_lines = []
    report_lines.append("# 数据预处理报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**股票代码**: {stock_symbol}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 1. 数据概览
    report_lines.append("## 1. 数据概览")
    report_lines.append("")
    report_lines.append("### 1.1 基本信息")
    report_lines.append("")
    report_lines.append(f"| 指标 | 数值 |")
    report_lines.append(f"|------|------|")
    report_lines.append(f"| 总记录数 | {len(merged_df):,} |")
    
    if timestamp_col in merged_df.columns:
        min_ts = merged_df[timestamp_col].min()
        max_ts = merged_df[timestamp_col].max()
        report_lines.append(f"| 开始时间 | {min_ts} |")
        report_lines.append(f"| 结束时间 | {max_ts} |")
        
        # 计算时间跨度
        time_span = max_ts - min_ts
        days = time_span.days
        hours = time_span.total_seconds() / 3600
        report_lines.append(f"| 时间跨度 | {days} 天 ({hours:.0f} 小时) |")
    
    report_lines.append("")
    
    # 2. Reddit数据统计
    report_lines.append("## 2. Reddit数据统计")
    report_lines.append("")
    
    if 'post_count' in merged_df.columns:
        report_lines.append("### 2.1 帖子统计")
        report_lines.append("")
        report_lines.append(f"| 指标 | 数值 |")
        report_lines.append(f"|------|------|")
        report_lines.append(f"| 总帖子数 | {merged_df['post_count'].sum():,} |")
        report_lines.append(f"| 平均每小时帖子数 | {merged_df['post_count'].mean():.2f} |")
        report_lines.append(f"| 中位数每小时帖子数 | {merged_df['post_count'].median():.2f} |")
        report_lines.append(f"| 最大每小时帖子数 | {merged_df['post_count'].max():,} |")
        report_lines.append(f"| 有帖子的小时数 | {(merged_df['post_count'] > 0).sum():,} |")
        report_lines.append(f"| 无帖子的小时数 | {(merged_df['post_count'] == 0).sum():,} |")
        report_lines.append("")
        
        # 帖子分布
        report_lines.append("### 2.2 帖子分布")
        report_lines.append("")
        report_lines.append(f"| 帖子数范围 | 小时数 | 占比 |")
        report_lines.append(f"|-----------|--------|------|")
        
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
        report_lines.append("### 2.3 评论统计")
        report_lines.append("")
        report_lines.append(f"| 指标 | 数值 |")
        report_lines.append(f"|------|------|")
        report_lines.append(f"| 总评论数 | {merged_df['total_comments'].sum():,} |")
        report_lines.append(f"| 平均每小时评论数 | {merged_df['total_comments'].mean():.2f} |")
        report_lines.append("")
    
    if 'unique_authors' in merged_df.columns:
        report_lines.append("### 2.4 作者统计")
        report_lines.append("")
        report_lines.append(f"| 指标 | 数值 |")
        report_lines.append(f"|------|------|")
        report_lines.append(f"| 总唯一作者数 | {merged_df['unique_authors'].sum():,} |")
        report_lines.append(f"| 平均每小时唯一作者数 | {merged_df['unique_authors'].mean():.2f} |")
        report_lines.append("")
    
    # 3. 股票数据统计
    report_lines.append("## 3. 股票数据统计")
    report_lines.append("")
    
    # 查找股票价格列
    price_cols = [col for col in merged_df.columns if f'{stock_symbol}_close' in col or 'close' in col.lower()]
    if price_cols:
        price_col = price_cols[0]
        report_lines.append("### 3.1 价格统计")
        report_lines.append("")
        report_lines.append(f"| 指标 | 数值 |")
        report_lines.append(f"|------|------|")
        
        price_data = merged_df[price_col].dropna()
        if len(price_data) > 0:
            report_lines.append(f"| 平均价格 | ${price_data.mean():.2f} |")
            report_lines.append(f"| 中位数价格 | ${price_data.median():.2f} |")
            report_lines.append(f"| 最低价格 | ${price_data.min():.2f} |")
            report_lines.append(f"| 最高价格 | ${price_data.max():.2f} |")
            report_lines.append(f"| 价格标准差 | ${price_data.std():.2f} |")
        
        report_lines.append("")
    
    # 4. 数据质量评估
    report_lines.append("## 4. 数据质量评估")
    report_lines.append("")
    
    report_lines.append("### 4.1 缺失值统计")
    report_lines.append("")
    report_lines.append(f"| 列名 | 缺失值数量 | 缺失率 |")
    report_lines.append(f"|------|-----------|--------|")
    
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
    
    # 5. 时间分布
    if timestamp_col in merged_df.columns:
        report_lines.append("## 5. 时间分布")
        report_lines.append("")
        
        merged_df['date'] = merged_df[timestamp_col].dt.date
        merged_df['hour'] = merged_df[timestamp_col].dt.hour
        merged_df['weekday'] = merged_df[timestamp_col].dt.weekday
        
        report_lines.append("### 5.1 按日期分布")
        report_lines.append("")
        date_counts = merged_df['date'].value_counts().sort_index()
        report_lines.append(f"| 指标 | 数值 |")
        report_lines.append(f"|------|------|")
        report_lines.append(f"| 唯一日期数 | {len(date_counts)} |")
        report_lines.append(f"| 平均每天记录数 | {date_counts.mean():.2f} |")
        report_lines.append("")
        
        report_lines.append("### 5.2 按小时分布")
        report_lines.append("")
        hour_counts = merged_df['hour'].value_counts().sort_index()
        report_lines.append(f"| 小时 | 记录数 |")
        report_lines.append(f"|------|--------|")
        for hour, count in hour_counts.items():
            report_lines.append(f"| {hour:02d}:00 | {count:,} |")
        report_lines.append("")
        
        report_lines.append("### 5.3 按星期分布")
        report_lines.append("")
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_counts = merged_df['weekday'].value_counts().sort_index()
        report_lines.append(f"| 星期 | 记录数 |")
        report_lines.append(f"|------|--------|")
        for weekday, count in weekday_counts.items():
            report_lines.append(f"| {weekday_names[weekday]} | {count:,} |")
        report_lines.append("")
    
    # 6. 总结
    report_lines.append("## 6. 总结")
    report_lines.append("")
    report_lines.append("### 数据质量评估")
    report_lines.append("")
    
    # 评估数据质量
    quality_score = 100
    issues = []
    
    if missing.sum() > 0:
        max_missing_rate = (missing / len(merged_df)).max() * 100
        if max_missing_rate > 50:
            quality_score -= 30
            issues.append(f"高缺失率：{max_missing_rate:.1f}%")
        elif max_missing_rate > 20:
            quality_score -= 15
            issues.append(f"中等缺失率：{max_missing_rate:.1f}%")
    
    if 'post_count' in merged_df.columns:
        zero_post_rate = (merged_df['post_count'] == 0).sum() / len(merged_df) * 100
        if zero_post_rate > 80:
            quality_score -= 20
            issues.append(f"高零帖子率：{zero_post_rate:.1f}%")
    
    if quality_score >= 90:
        quality_level = "优秀"
    elif quality_score >= 70:
        quality_level = "良好"
    elif quality_score >= 50:
        quality_level = "一般"
    else:
        quality_level = "需要改进"
    
    report_lines.append(f"**数据质量评分**: {quality_score}/100 ({quality_level})")
    report_lines.append("")
    
    if issues:
        report_lines.append("**发现的问题**:")
        report_lines.append("")
        for issue in issues:
            report_lines.append(f"- {issue}")
    else:
        report_lines.append("**未发现重大问题**")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("*报告由数据预处理脚本自动生成*")
    
    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'preprocessing_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {report_path}")
    
    return report_path

