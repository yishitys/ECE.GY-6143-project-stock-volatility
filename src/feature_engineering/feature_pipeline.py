"""
特征工程主流程

整合所有特征工程步骤：
1. 加载合并数据
2. 计算目标变量
3. 生成文本嵌入（如果尚未生成）
4. 聚合文本特征
5. 计算技术指标
6. 合并所有特征
7. 处理缺失值
8. 保存最终特征数据集
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List
import sys

# 添加src目录到路径
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_merged_data(merged_data_path: str) -> pd.DataFrame:
    """
    加载合并后的数据
    
    Args:
        merged_data_path: 合并数据文件路径
    
    Returns:
        合并数据DataFrame
    """
    if not os.path.exists(merged_data_path):
        raise FileNotFoundError(f"合并数据文件不存在: {merged_data_path}")
    
    logger.info(f"加载合并数据: {merged_data_path}")
    df = pd.read_csv(merged_data_path, parse_dates=['timestamp'])
    logger.info(f"加载了 {len(df)} 条记录")
    
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
    加载Reddit帖子数据用于生成嵌入
    
    Args:
        stock_symbol: 股票代码
        subreddits: 子版块列表
        reddit_data_dir: Reddit数据目录
        cleaned_reddit_path: 清洗后的Reddit数据路径
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        Reddit帖子DataFrame
    """
    # 尝试加载已清洗的数据
    if os.path.exists(cleaned_reddit_path):
        logger.info(f"加载已清洗的Reddit数据: {cleaned_reddit_path}")
        df = pd.read_csv(cleaned_reddit_path, parse_dates=['timestamp'], low_memory=False)
        logger.info(f"加载了 {len(df)} 条帖子")
        return df
    
    # 否则从原始数据加载
    logger.info("从原始数据加载Reddit帖子...")
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
        raise ValueError("无法加载Reddit数据")
    
    # 清洗文本
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
    构建完整的特征工程流程
    
    Args:
        stock_symbol: 股票代码
        merged_data_path: 合并数据路径（如果为None则自动生成）
        reddit_data_dir: Reddit数据目录
        output_dir: 输出目录
        embedding_model: 嵌入模型名称
        aggregation_method: 聚合方法 ('mean', 'weighted_mean', 'max')
        generate_embeddings: 是否生成嵌入（如果缓存不存在）
        use_cached_embeddings: 是否使用缓存的嵌入
    
    Returns:
        包含所有特征的DataFrame
    """
    logger.info("="*60)
    logger.info(f"开始特征工程流程: {stock_symbol}")
    logger.info("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: 加载合并数据
    if merged_data_path is None:
        merged_data_path = os.path.join(output_dir, f'merged_data_{stock_symbol}.csv')
    
    df = load_merged_data(merged_data_path)
    
    # Step 2: 计算目标变量
    logger.info("\n" + "="*60)
    logger.info("Step 2: 计算目标变量")
    logger.info("="*60)
    df = calculate_multiple_targets(df, methods=['log_return_abs', 'price_range'])
    
    # 分析目标变量
    target_col = 'target_volatility_log_return_abs'
    if target_col in df.columns:
        stats = analyze_target_distribution(df, target_col)
        report_path = os.path.join(output_dir, f'target_analysis_{stock_symbol}.md')
        save_target_analysis_report(stats, report_path, target_col)
    
    # Step 3: 生成文本嵌入（如果需要）
    embedding_df = None
    if generate_embeddings or not use_cached_embeddings:
        logger.info("\n" + "="*60)
        logger.info("Step 3: 生成文本嵌入")
        logger.info("="*60)
        
        # 加载Reddit帖子
        reddit_posts = load_reddit_posts_for_embeddings(
            stock_symbol=stock_symbol,
            reddit_data_dir=reddit_data_dir
        )
        
        # 生成嵌入
        cache_file = os.path.join(output_dir, 'embeddings', 
                                 f'embeddings_{embedding_model.replace("/", "_")}.pkl')
        
        # 在开始生成嵌入前，检查是否需要生成并提前告知
        if not (use_cached_embeddings and os.path.exists(cache_file)):
            logger.info("\n" + "!"*60)
            logger.info("⚠️  重要提示：即将开始生成文本嵌入")
            logger.info("!"*60)
            logger.info(f"📊 需要处理的帖子数量: {len(reddit_posts):,}")
            logger.info(f"🤖 使用的模型: {embedding_model}")
            logger.info(f"⏱️  预计耗时: 30分钟到数小时（取决于硬件性能）")
            logger.info(f"💾 嵌入将缓存到: {cache_file}")
            logger.info("📝 开始生成嵌入...")
            logger.info("!"*60 + "\n")
        
        if use_cached_embeddings and os.path.exists(cache_file):
            logger.info(f"使用缓存的嵌入: {cache_file}")
            try:
                embedding_df = pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"无法加载缓存，重新生成: {e}")
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
    
    # Step 4: 聚合文本特征
    if embedding_df is not None and not embedding_df.empty:
        logger.info("\n" + "="*60)
        logger.info("Step 4: 聚合文本特征")
        logger.info("="*60)
        
        aggregated_embeddings = aggregate_embeddings_by_hour(
            posts_df=embedding_df,
            timestamp_col='timestamp',
            embedding_prefix='embedding_',
            aggregation_method=aggregation_method,
            weight_col='score' if aggregation_method == 'weighted_mean' else None
        )
        
        # 合并聚合后的嵌入特征到原始DataFrame
        # 只提取嵌入列（不包括Reddit统计，因为原始df已有）
        embedding_cols = [col for col in aggregated_embeddings.columns if col.startswith('embedding_') or col == 'timestamp']
        embeddings_only = aggregated_embeddings[embedding_cols].copy()
        
        # 合并嵌入特征到原始DataFrame，保留所有原始列（包括股票价格数据）
        df = pd.merge(
            df,
            embeddings_only,
            on='timestamp',
            how='left'
        )
        
        # 处理缺失小时
        df = handle_missing_hours(
            df,
            timestamp_col='timestamp',
            embedding_prefix='embedding_',
            fill_method='zero'
        )
    
    # Step 5: 计算技术指标
    logger.info("\n" + "="*60)
    logger.info("Step 5: 计算技术指标")
    logger.info("="*60)
    df = calculate_technical_indicators(df)
    
    # Step 6: 处理缺失值
    logger.info("\n" + "="*60)
    logger.info("Step 6: 处理缺失值")
    logger.info("="*60)
    
    # 删除目标变量为NaN的行（最后一行）
    initial_len = len(df)
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if target_cols:
        df = df.dropna(subset=target_cols)
        logger.info(f"删除了 {initial_len - len(df)} 行（目标变量为NaN）")
    
    # 填充其他缺失值（使用前向填充）
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        # 对数值列使用前向填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
        missing_after = df.isnull().sum().sum()
        logger.info(f"缺失值: {missing_before} -> {missing_after}")
    
    # Step 7: 保存最终特征数据集
    logger.info("\n" + "="*60)
    logger.info("Step 7: 保存特征数据集")
    logger.info("="*60)
    
    output_path = os.path.join(output_dir, f'features_{stock_symbol}.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"特征数据集已保存到: {output_path}")
    logger.info(f"最终特征数量: {len(df.columns)}")
    logger.info(f"最终记录数量: {len(df)}")
    
    # 生成特征报告
    generate_feature_report(df, stock_symbol, output_dir)
    
    return df


def generate_feature_report(df: pd.DataFrame, stock_symbol: str, output_dir: str):
    """
    生成特征工程报告
    
    Args:
        df: 特征DataFrame
        stock_symbol: 股票代码
        output_dir: 输出目录
    """
    report_path = os.path.join(output_dir, f'feature_report_{stock_symbol}.md')
    
    # 统计特征类型
    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    reddit_cols = ['post_count', 'total_comments', 'total_score', 'unique_authors']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    technical_cols = [col for col in df.columns if col not in 
                      embedding_cols + reddit_cols + target_cols + ['timestamp', 'stock_symbol', 
                                                                     'has_reddit_data', 'has_stock_data']]
    
    report = f"""# 特征工程报告

**股票代码**: {stock_symbol}
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 数据概览

| 指标 | 数值 |
|------|------|
| 总记录数 | {len(df):,} |
| 总特征数 | {len(df.columns)} |
| 时间范围 | {df['timestamp'].min()} 至 {df['timestamp'].max()} |

---

## 特征分类

### 1. 文本嵌入特征
- **数量**: {len(embedding_cols)}
- **维度**: {len(embedding_cols)} 维
- **说明**: 使用sentence-transformers生成的文本嵌入向量

### 2. Reddit统计特征
- **数量**: {len([col for col in reddit_cols if col in df.columns])}
- **特征**: {', '.join([col for col in reddit_cols if col in df.columns])}

### 3. 技术指标特征
- **数量**: {len(technical_cols)}
- **主要类型**: 
  - 收益率特征
  - 移动平均特征
  - RSI、MACD指标
  - 波动率特征
  - 成交量特征
  - 滞后特征
  - 滚动统计特征

### 4. 目标变量
- **数量**: {len(target_cols)}
- **变量**: {', '.join(target_cols)}

---

## 特征列表

### 文本嵌入特征
{chr(10).join([f'- {col}' for col in embedding_cols[:10]])}
... (共 {len(embedding_cols)} 个)

### Reddit统计特征
{chr(10).join([f'- {col}' for col in reddit_cols if col in df.columns])}

### 技术指标特征（示例）
{chr(10).join([f'- {col}' for col in technical_cols[:20]])}
... (共 {len(technical_cols)} 个)

---

## 数据质量

| 检查项 | 结果 |
|--------|------|
| 缺失值总数 | {df.isnull().sum().sum()} |
| 目标变量缺失 | {df[target_cols].isnull().sum().sum() if target_cols else 0} |
| 重复记录 | {df.duplicated().sum()} |

---

## 说明

- 所有特征已按时间排序
- 目标变量为NaN的记录已删除
- 其他缺失值已使用前向填充处理
- 文本嵌入使用 {embedding_model if 'embedding_model' in locals() else 'all-MiniLM-L6-v2'} 模型生成
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"特征工程报告已保存到: {report_path}")


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='特征工程主流程')
    parser.add_argument('--symbol', type=str, default='GME', help='股票代码')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2', 
                       help='嵌入模型名称')
    parser.add_argument('--aggregation', type=str, default='mean', 
                       choices=['mean', 'weighted_mean', 'max'],
                       help='聚合方法')
    parser.add_argument('--no-embeddings', action='store_true', 
                       help='跳过嵌入生成（仅使用已有特征）')
    
    args = parser.parse_args()
    
    df = build_feature_pipeline(
        stock_symbol=args.symbol,
        embedding_model=args.embedding_model,
        aggregation_method=args.aggregation,
        generate_embeddings=not args.no_embeddings
    )
    
    logger.info("特征工程流程完成！")


