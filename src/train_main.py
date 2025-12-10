"""
统一训练入口

整合所有组件，提供命令行接口进行模型训练和评估。
"""

import pandas as pd
import numpy as np
import logging
import os
import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加src目录到路径
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from feature_engineering.feature_pipeline import build_feature_pipeline
from models.train import train_model as train_xgboost
from models.lstm_model import LSTMVolatilityModel
from evaluation.evaluate import evaluate_model, print_metrics, save_evaluation_report
from evaluation.visualize_results import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_error_distribution,
    plot_feature_importance
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_lstm_model(
    stock_symbol: str,
    features_path: str = None,
    target_col: str = 'target_volatility_log_return_abs',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sequence_length: int = 24,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    save_model: bool = True,
    model_output_dir: str = 'models'
):
    """
    训练LSTM模型
    
    Args:
        stock_symbol: 股票代码
        features_path: 特征数据路径
        target_col: 目标变量列名
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        sequence_length: 序列长度
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        epochs: 训练轮数
        batch_size: 批处理大小
        save_model: 是否保存模型
        model_output_dir: 模型输出目录
    
    Returns:
        (训练好的模型, 评估结果字典)
    """
    logger.info("="*60)
    logger.info(f"开始训练LSTM模型: {stock_symbol}")
    logger.info("="*60)
    
    # 加载特征数据
    if features_path is None:
        features_path = f'data/processed/features_{stock_symbol}.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特征数据文件不存在: {features_path}")
    
    logger.info(f"加载特征数据: {features_path}")
    df = pd.read_csv(features_path, parse_dates=['timestamp'])
    logger.info(f"加载了 {len(df)} 条记录")
    
    # 创建模型
    model = LSTMVolatilityModel(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size
    )
    
    # 准备特征
    X, y = model.prepare_features(df, target_col=target_col)
    
    # 划分数据集
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    logger.info(f"数据划分: 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
    
    # 训练
    model.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # 评估
    logger.info("\n训练集评估:")
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_model(y_train, y_train_pred, task_type='regression')
    print_metrics(train_metrics, task_type='regression')
    
    logger.info("\n验证集评估:")
    y_val_pred = model.predict(X_val)
    val_metrics = evaluate_model(y_val, y_val_pred, task_type='regression')
    print_metrics(val_metrics, task_type='regression')
    
    logger.info("\n测试集评估:")
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred, task_type='regression')
    print_metrics(test_metrics, task_type='regression')
    
    # 保存模型
    if save_model:
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(model_output_dir, f'lstm_{stock_symbol}.pth')
        model.save(model_path)
    
    # 汇总结果
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_params': {
            'sequence_length': sequence_length,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }
    }
    
    return model, results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='股票波动率预测模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练XGBoost模型
  python train_main.py --symbol GME --model xgboost
  
  # 训练LSTM模型
  python train_main.py --symbol GME --model lstm
  
  # 先进行特征工程，再训练模型
  python train_main.py --symbol GME --model xgboost --build-features
  
  # 训练两个模型并对比
  python train_main.py --symbol GME --model both
        """
    )
    
    parser.add_argument('--symbol', type=str, default='GME', help='股票代码')
    parser.add_argument('--model', type=str, choices=['xgboost', 'lstm', 'both'], 
                       default='xgboost', help='模型类型')
    parser.add_argument('--build-features', action='store_true',
                       help='是否先进行特征工程（如果特征数据不存在）')
    parser.add_argument('--features', type=str, default=None,
                       help='特征数据路径（如果为None则自动生成）')
    parser.add_argument('--target', type=str, default='target_volatility_log_return_abs',
                       help='目标变量列名')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='结果输出目录')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='模型保存目录')
    
    # XGBoost参数
    parser.add_argument('--xgboost-params', type=str, default=None,
                       help='XGBoost参数字典（JSON格式）')
    parser.add_argument('--cv', action='store_true',
                       help='进行交叉验证（仅XGBoost）')
    parser.add_argument('--tune', action='store_true',
                       help='进行超参数调优（仅XGBoost）')
    
    # LSTM参数
    parser.add_argument('--sequence-length', type=int, default=24,
                       help='LSTM序列长度（小时数）')
    parser.add_argument('--hidden-size', type=int, default=64,
                       help='LSTM隐藏层大小')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数（LSTM）')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批处理大小（LSTM）')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("股票波动率预测模型训练")
    logger.info("="*60)
    logger.info(f"股票代码: {args.symbol}")
    logger.info(f"模型类型: {args.model}")
    
    # 检查特征数据
    if args.features is None:
        features_path = f'data/processed/features_{args.symbol}.csv'
    else:
        features_path = args.features
    
    if not os.path.exists(features_path) or args.build_features:
        logger.info("特征数据不存在或需要重建，开始特征工程...")
        build_feature_pipeline(
            stock_symbol=args.symbol,
            output_dir='data/processed',
            embedding_model='all-MiniLM-L6-v2',
            aggregation_method='mean'
        )
        features_path = f'data/processed/features_{args.symbol}.csv'
    
    # 训练模型
    all_results = {}
    
    if args.model in ['xgboost', 'both']:
        logger.info("\n" + "="*60)
        logger.info("训练XGBoost模型")
        logger.info("="*60)
        
        model_params = {}
        if args.xgboost_params:
            import json
            model_params = json.loads(args.xgboost_params)
        
        xgboost_model, xgboost_results = train_xgboost(
            stock_symbol=args.symbol,
            features_path=features_path,
            target_col=args.target,
            model_params=model_params,
            do_cross_validation=args.cv,
            do_hyperparameter_tuning=args.tune,
            save_model=True,
            model_output_dir=args.model_dir
        )
        
        all_results['xgboost'] = xgboost_results
        
        # 生成可视化
        logger.info("\n生成XGBoost模型可视化...")
        df = pd.read_csv(features_path, parse_dates=['timestamp'])
        X, y = xgboost_model.prepare_features(df, target_col=args.target)
        
        # 划分测试集
        split_idx = int(len(X) * 0.85)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        y_pred = xgboost_model.predict(X_test)
        timestamps = df['timestamp'].iloc[split_idx:split_idx+len(y_pred)]
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        plot_predictions_vs_actual(
            y_test.values, y_pred,
            timestamps=timestamps,
            output_path=f'{args.output_dir}/figures/xgboost_predictions_{args.symbol}.png',
            title=f'XGBoost预测结果 - {args.symbol}'
        )
        plot_residuals(
            y_test.values, y_pred,
            output_path=f'{args.output_dir}/figures/xgboost_residuals_{args.symbol}.png'
        )
        plot_error_distribution(
            y_test.values, y_pred,
            output_path=f'{args.output_dir}/figures/xgboost_errors_{args.symbol}.png'
        )
        
        # 特征重要性
        importance = xgboost_model.get_feature_importance(top_n=20)
        plot_feature_importance(
            importance,
            output_path=f'{args.output_dir}/figures/xgboost_importance_{args.symbol}.png'
        )
        
        # 保存评估报告
        save_evaluation_report(
            xgboost_results['test_metrics'],
            f'{args.output_dir}/evaluation_report_xgboost_{args.symbol}.md',
            model_name='XGBoost',
            stock_symbol=args.symbol
        )
    
    if args.model in ['lstm', 'both']:
        logger.info("\n" + "="*60)
        logger.info("训练LSTM模型")
        logger.info("="*60)
        
        lstm_model, lstm_results = train_lstm_model(
            stock_symbol=args.symbol,
            features_path=features_path,
            target_col=args.target,
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_model=True,
            model_output_dir=args.model_dir
        )
        
        all_results['lstm'] = lstm_results
        
        # 生成可视化
        logger.info("\n生成LSTM模型可视化...")
        df = pd.read_csv(features_path, parse_dates=['timestamp'])
        X, y = lstm_model.prepare_features(df, target_col=args.target)
        
        # 划分测试集
        split_idx = int(len(X) * 0.85)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        y_pred = lstm_model.predict(X_test)
        timestamps = df['timestamp'].iloc[split_idx:split_idx+len(y_pred)]
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        plot_predictions_vs_actual(
            y_test, y_pred,
            timestamps=timestamps,
            output_path=f'{args.output_dir}/figures/lstm_predictions_{args.symbol}.png',
            title=f'LSTM预测结果 - {args.symbol}'
        )
        plot_residuals(
            y_test, y_pred,
            output_path=f'{args.output_dir}/figures/lstm_residuals_{args.symbol}.png'
        )
        plot_error_distribution(
            y_test, y_pred,
            output_path=f'{args.output_dir}/figures/lstm_errors_{args.symbol}.png'
        )
        
        # 保存评估报告
        save_evaluation_report(
            lstm_results['test_metrics'],
            f'{args.output_dir}/evaluation_report_lstm_{args.symbol}.md',
            model_name='LSTM',
            stock_symbol=args.symbol
        )
    
    # 模型对比（如果训练了两个模型）
    if args.model == 'both' and len(all_results) == 2:
        logger.info("\n生成模型对比图...")
        comparison_results = {
            'XGBoost': all_results['xgboost']['test_metrics'],
            'LSTM': all_results['lstm']['test_metrics']
        }
        plot_model_comparison(
            comparison_results,
            output_path=f'{args.output_dir}/figures/model_comparison_{args.symbol}.png'
        )
    
    logger.info("\n" + "="*60)
    logger.info("训练完成！")
    logger.info("="*60)
    logger.info(f"结果保存在: {args.output_dir}")
    logger.info(f"模型保存在: {args.model_dir}")


if __name__ == '__main__':
    main()


