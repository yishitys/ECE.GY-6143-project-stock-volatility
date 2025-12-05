"""
数据预处理模块

包含以下功能：
- clean_text: Reddit文本清洗
- align_timestamps: 时间戳对齐
- merge_data: 数据合并
- generate_report: 生成预处理报告
- preprocess: 主预处理脚本
"""

from .clean_text import clean_text_content, clean_reddit_data
from .align_timestamps import (
    group_reddit_by_hour,
    align_stock_to_hourly,
    create_hourly_index
)
from .merge_data import (
    merge_reddit_and_stock,
    handle_missing_values,
    add_data_availability_flags,
    validate_merged_data
)
from .generate_report import generate_preprocessing_report

__all__ = [
    'clean_text_content',
    'clean_reddit_data',
    'group_reddit_by_hour',
    'align_stock_to_hourly',
    'create_hourly_index',
    'merge_reddit_and_stock',
    'handle_missing_values',
    'add_data_availability_flags',
    'validate_merged_data',
    'generate_preprocessing_report',
]

