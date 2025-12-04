"""
股票价格数据分析脚本

分析所有已收集的股票价格数据，生成详细的统计报告和可视化图表。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class StockDataAnalyzer:
    """股票数据分析器"""
    
    def __init__(self, data_dir='data/stock_prices', output_dir='data/stock_prices'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.stock_data = {}
        self.statistics = {}
        
    def load_all_stocks(self):
        """加载所有股票数据"""
        print("正在加载股票数据...")
        csv_files = glob.glob(os.path.join(self.data_dir, '*_2021.csv'))
        
        for file_path in csv_files:
            symbol = os.path.basename(file_path).replace('_2021.csv', '')
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # 计算衍生指标
                df['price_change'] = df['close'].pct_change()
                df['price_change_abs'] = df['close'].diff().abs()
                df['volatility'] = df['price_change'].rolling(window=24).std()  # 24小时滚动波动率
                
                self.stock_data[symbol] = df
                print(f"  ✓ 加载 {symbol}: {len(df)} 行数据")
            except Exception as e:
                print(f"  ✗ 加载 {symbol} 失败: {e}")
        
        print(f"\n成功加载 {len(self.stock_data)} 只股票的数据\n")
        return self.stock_data
    
    def calculate_basic_statistics(self):
        """计算基本统计信息"""
        print("正在计算基本统计信息...")
        
        for symbol, df in self.stock_data.items():
            stats = {
                'symbol': symbol,
                'total_rows': len(df),
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max(),
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days
                },
                'price_stats': {
                    'open': {
                        'mean': df['open'].mean(),
                        'median': df['open'].median(),
                        'std': df['open'].std(),
                        'min': df['open'].min(),
                        'max': df['open'].max(),
                        'q25': df['open'].quantile(0.25),
                        'q75': df['open'].quantile(0.75)
                    },
                    'high': {
                        'mean': df['high'].mean(),
                        'median': df['high'].median(),
                        'std': df['high'].std(),
                        'min': df['high'].min(),
                        'max': df['high'].max(),
                    },
                    'low': {
                        'mean': df['low'].mean(),
                        'median': df['low'].median(),
                        'std': df['low'].std(),
                        'min': df['low'].min(),
                        'max': df['low'].max(),
                    },
                    'close': {
                        'mean': df['close'].mean(),
                        'median': df['close'].median(),
                        'std': df['close'].std(),
                        'min': df['close'].min(),
                        'max': df['close'].max(),
                    }
                },
                'volume_stats': {
                    'mean': df['volume'].mean(),
                    'median': df['volume'].median(),
                    'std': df['volume'].std(),
                    'min': df['volume'].min(),
                    'max': df['volume'].max(),
                    'total': df['volume'].sum()
                },
                'returns': {
                    'mean': df['price_change'].mean(),
                    'std': df['price_change'].std(),
                    'min': df['price_change'].min(),
                    'max': df['price_change'].max(),
                    'annualized_volatility': df['price_change'].std() * np.sqrt(252 * 8)  # 年化波动率
                },
                'data_quality': {
                    'missing_values': df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().to_dict(),
                    'zero_volume_count': (df['volume'] == 0).sum(),
                    'negative_price_count': ((df[['open', 'high', 'low', 'close']] < 0).any(axis=1)).sum()
                },
                'price_extremes': {
                    'highest_price': df.loc[df['high'].idxmax(), ['timestamp', 'high']].to_dict(),
                    'lowest_price': df.loc[df['low'].idxmin(), ['timestamp', 'low']].to_dict(),
                    'max_volume': df.loc[df['volume'].idxmax(), ['timestamp', 'volume']].to_dict()
                }
            }
            
            self.statistics[symbol] = stats
        
        print(f"完成 {len(self.statistics)} 只股票的统计分析\n")
        return self.statistics
    
    def create_price_trend_plots(self):
        """创建价格走势图"""
        print("正在生成价格走势图...")
        
        # 所有股票的价格走势对比
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 收盘价走势
        ax1 = axes[0]
        for symbol, df in self.stock_data.items():
            # 只显示每日收盘价（每小时数据太多，取每日最后一个）
            daily_close = df.groupby(df['timestamp'].dt.date)['close'].last()
            dates = pd.to_datetime(daily_close.index)
            ax1.plot(dates, daily_close.values, label=symbol, alpha=0.7, linewidth=1.5)
        
        ax1.set_title('Stock Price Trends (Daily Close Prices) - 2021', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Close Price (USD)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 成交量对比
        ax2 = axes[1]
        for symbol, df in self.stock_data.items():
            daily_volume = df.groupby(df['timestamp'].dt.date)['volume'].sum()
            dates = pd.to_datetime(daily_volume.index)
            ax2.plot(dates, daily_volume.values, label=symbol, alpha=0.7, linewidth=1.5)
        
        ax2.set_title('Trading Volume Trends (Daily) - 2021', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'price_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 价格走势图已保存")
        
        # 为每个股票创建单独的价格走势图
        for symbol, df in self.stock_data.items():
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 价格走势
            ax1 = axes[0]
            daily_data = df.groupby(df['timestamp'].dt.date).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            dates = pd.to_datetime(daily_data.index)
            
            ax1.plot(dates, daily_data['close'], label='Close', linewidth=2, color='blue')
            ax1.fill_between(dates, daily_data['low'], daily_data['high'], 
                           alpha=0.3, color='gray', label='High-Low Range')
            ax1.set_title(f'{symbol} Price Trend - 2021', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price (USD)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 成交量
            ax2 = axes[1]
            ax2.bar(dates, daily_data['volume'], alpha=0.6, color='green')
            ax2.set_title(f'{symbol} Trading Volume - 2021', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'{symbol}_price_trend.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 已为 {len(self.stock_data)} 只股票生成单独的价格走势图\n")
    
    def create_distribution_plots(self):
        """创建分布图"""
        print("正在生成分布图...")
        
        # 收盘价分布对比
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 收盘价分布
        ax = axes[0, 0]
        for symbol, df in self.stock_data.items():
            ax.hist(df['close'], bins=50, alpha=0.5, label=symbol, density=True)
        ax.set_title('Close Price Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Close Price (USD)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 收益率分布
        ax = axes[0, 1]
        for symbol, df in self.stock_data.items():
            returns = df['price_change'].dropna()
            ax.hist(returns, bins=50, alpha=0.5, label=symbol, density=True)
        ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Returns', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 成交量分布（对数尺度）
        ax = axes[1, 0]
        for symbol, df in self.stock_data.items():
            ax.hist(np.log10(df['volume'] + 1), bins=50, alpha=0.5, label=symbol, density=True)
        ax.set_title('Volume Distribution (Log Scale)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Log10(Volume)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 波动率分布
        ax = axes[1, 1]
        for symbol, df in self.stock_data.items():
            volatility = df['volatility'].dropna()
            ax.hist(volatility, bins=50, alpha=0.5, label=symbol, density=True)
        ax.set_title('Volatility Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Volatility', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 分布图已保存\n")
    
    def create_correlation_heatmap(self):
        """创建相关性热力图"""
        print("正在生成相关性热力图...")
        
        # 计算股票之间的相关性（基于收盘价）
        close_prices = pd.DataFrame()
        for symbol, df in self.stock_data.items():
            # 使用每日收盘价
            daily_close = df.groupby(df['timestamp'].dt.date)['close'].last()
            close_prices[symbol] = daily_close
        
        # 对齐日期
        close_prices = close_prices.dropna()
        
        # 计算相关性矩阵
        correlation_matrix = close_prices.corr()
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Stock Price Correlation Matrix - 2021', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 相关性热力图已保存\n")
    
    def create_volatility_comparison(self):
        """创建波动率对比图"""
        print("正在生成波动率对比图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 年化波动率对比
        ax = axes[0]
        symbols = []
        volatilities = []
        for symbol, stats in self.statistics.items():
            symbols.append(symbol)
            volatilities.append(stats['returns']['annualized_volatility'] * 100)  # 转换为百分比
        
        bars = ax.bar(symbols, volatilities, color='steelblue', alpha=0.7)
        ax.set_title('Annualized Volatility Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stock Symbol', fontsize=12)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 平均收益率对比
        ax = axes[1]
        returns = []
        for symbol, stats in self.statistics.items():
            returns.append(stats['returns']['mean'] * 100)  # 转换为百分比
        
        bars = ax.bar(symbols, returns, color='green', alpha=0.7)
        ax.set_title('Average Returns Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stock Symbol', fontsize=12)
        ax.set_ylabel('Average Returns (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'volatility_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 波动率对比图已保存\n")
    
    def generate_report(self):
        """生成统计报告"""
        print("正在生成统计报告...")
        
        report_lines = []
        report_lines.append("# 股票价格数据统计报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 1. 数据概览
        report_lines.append("## 1. 数据概览")
        report_lines.append("")
        report_lines.append("### 1.1 基本信息")
        report_lines.append("")
        report_lines.append("| 股票代码 | 数据行数 | 开始日期 | 结束日期 | 数据天数 |")
        report_lines.append("|---------|---------|---------|---------|---------|")
        
        for symbol, stats in sorted(self.statistics.items()):
            date_range = stats['date_range']
            report_lines.append(
                f"| {symbol} | {stats['total_rows']:,} | "
                f"{date_range['start'].strftime('%Y-%m-%d')} | "
                f"{date_range['end'].strftime('%Y-%m-%d')} | "
                f"{date_range['days']} |"
            )
        
        report_lines.append("")
        
        # 2. 价格统计
        report_lines.append("## 2. 价格统计分析")
        report_lines.append("")
        
        for symbol, stats in sorted(self.statistics.items()):
            report_lines.append(f"### 2.{list(sorted(self.statistics.keys())).index(symbol) + 1} {symbol}")
            report_lines.append("")
            
            price_stats = stats['price_stats']
            report_lines.append("#### 收盘价统计")
            report_lines.append("")
            report_lines.append("| 指标 | 数值 |")
            report_lines.append("|------|------|")
            report_lines.append(f"| 均值 | ${price_stats['close']['mean']:.2f} |")
            report_lines.append(f"| 中位数 | ${price_stats['close']['median']:.2f} |")
            report_lines.append(f"| 标准差 | ${price_stats['close']['std']:.2f} |")
            report_lines.append(f"| 最小值 | ${price_stats['close']['min']:.2f} |")
            report_lines.append(f"| 最大值 | ${price_stats['close']['max']:.2f} |")
            report_lines.append("")
            
            report_lines.append("#### 价格范围")
            report_lines.append("")
            report_lines.append("| 指标 | 数值 |")
            report_lines.append("|------|------|")
            report_lines.append(f"| 开盘价均值 | ${price_stats['open']['mean']:.2f} |")
            report_lines.append(f"| 最高价均值 | ${price_stats['high']['mean']:.2f} |")
            report_lines.append(f"| 最低价均值 | ${price_stats['low']['mean']:.2f} |")
            report_lines.append("")
            
            # 极值
            extremes = stats['price_extremes']
            report_lines.append("#### 价格极值")
            report_lines.append("")
            report_lines.append("| 类型 | 日期 | 价格 |")
            report_lines.append("|------|------|------|")
            report_lines.append(
                f"| 最高价 | {extremes['highest_price']['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                f"${extremes['highest_price']['high']:.2f} |"
            )
            report_lines.append(
                f"| 最低价 | {extremes['lowest_price']['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                f"${extremes['lowest_price']['low']:.2f} |"
            )
            report_lines.append("")
            
            # 成交量
            volume_stats = stats['volume_stats']
            report_lines.append("#### 成交量统计")
            report_lines.append("")
            report_lines.append("| 指标 | 数值 |")
            report_lines.append("|------|------|")
            report_lines.append(f"| 平均成交量 | {volume_stats['mean']:,.0f} |")
            report_lines.append(f"| 中位数成交量 | {volume_stats['median']:,.0f} |")
            report_lines.append(f"| 总成交量 | {volume_stats['total']:,.0f} |")
            report_lines.append(f"| 最大成交量 | {volume_stats['max']:,.0f} |")
            report_lines.append(
                f"| 最大成交量日期 | {extremes['max_volume']['timestamp'].strftime('%Y-%m-%d %H:%M')} |"
            )
            report_lines.append("")
            
            # 收益率和波动率
            returns = stats['returns']
            report_lines.append("#### 收益率与波动率")
            report_lines.append("")
            report_lines.append("| 指标 | 数值 |")
            report_lines.append("|------|------|")
            report_lines.append(f"| 平均收益率 | {returns['mean']*100:.4f}% |")
            report_lines.append(f"| 收益率标准差 | {returns['std']*100:.4f}% |")
            report_lines.append(f"| 年化波动率 | {returns['annualized_volatility']*100:.2f}% |")
            report_lines.append(f"| 最大单期收益 | {returns['max']*100:.2f}% |")
            report_lines.append(f"| 最大单期亏损 | {returns['min']*100:.2f}% |")
            report_lines.append("")
            
            # 数据质量
            quality = stats['data_quality']
            report_lines.append("#### 数据质量")
            report_lines.append("")
            report_lines.append("| 检查项 | 结果 |")
            report_lines.append("|--------|------|")
            missing = quality['missing_values']
            total_missing = sum(missing.values())
            report_lines.append(f"| 缺失值总数 | {total_missing} |")
            report_lines.append(f"| 零成交量记录 | {quality['zero_volume_count']} |")
            report_lines.append(f"| 负价格记录 | {quality['negative_price_count']} |")
            report_lines.append("")
            
            # 图表引用
            report_lines.append(f"![{symbol} Price Trend](figures/{symbol}_price_trend.png)")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        # 3. 综合对比
        report_lines.append("## 3. 综合对比分析")
        report_lines.append("")
        
        report_lines.append("### 3.1 价格走势对比")
        report_lines.append("")
        report_lines.append("![Price Trends](figures/price_trends.png)")
        report_lines.append("")
        
        report_lines.append("### 3.2 分布分析")
        report_lines.append("")
        report_lines.append("![Distributions](figures/distributions.png)")
        report_lines.append("")
        
        report_lines.append("### 3.3 相关性分析")
        report_lines.append("")
        report_lines.append("![Correlation Heatmap](figures/correlation_heatmap.png)")
        report_lines.append("")
        
        report_lines.append("### 3.4 波动率与收益率对比")
        report_lines.append("")
        report_lines.append("![Volatility Comparison](figures/volatility_comparison.png)")
        report_lines.append("")
        
        # 4. 总结
        report_lines.append("## 4. 数据质量总结")
        report_lines.append("")
        report_lines.append("### 4.1 数据完整性")
        report_lines.append("")
        total_stocks = len(self.statistics)
        total_rows = sum(s['total_rows'] for s in self.statistics.values())
        report_lines.append(f"- 成功分析股票数量: **{total_stocks}** 只")
        report_lines.append(f"- 总数据行数: **{total_rows:,}** 行")
        report_lines.append("")
        
        report_lines.append("### 4.2 数据质量评估")
        report_lines.append("")
        all_missing = sum(sum(s['data_quality']['missing_values'].values()) 
                         for s in self.statistics.values())
        all_zero_volume = sum(s['data_quality']['zero_volume_count'] 
                             for s in self.statistics.values())
        all_negative = sum(s['data_quality']['negative_price_count'] 
                          for s in self.statistics.values())
        
        report_lines.append(f"- 缺失值总数: {all_missing}")
        report_lines.append(f"- 零成交量记录: {all_zero_volume}")
        report_lines.append(f"- 负价格记录: {all_negative}")
        report_lines.append("")
        
        if all_missing == 0 and all_negative == 0:
            report_lines.append("✅ **数据质量优秀**: 无缺失值和异常值")
        elif all_missing < total_rows * 0.01:
            report_lines.append("⚠️ **数据质量良好**: 存在少量缺失值，但不影响分析")
        else:
            report_lines.append("❌ **数据质量需注意**: 存在较多缺失值，建议检查数据源")
        report_lines.append("")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'statistics_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ 统计报告已保存到: {report_path}\n")
        return report_path
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("="*60)
        print("股票价格数据分析")
        print("="*60)
        print()
        
        # 1. 加载数据
        self.load_all_stocks()
        
        # 2. 计算统计信息
        self.calculate_basic_statistics()
        
        # 3. 生成可视化
        self.create_price_trend_plots()
        self.create_distribution_plots()
        self.create_correlation_heatmap()
        self.create_volatility_comparison()
        
        # 4. 生成报告
        report_path = self.generate_report()
        
        print("="*60)
        print("分析完成！")
        print("="*60)
        print(f"统计报告: {report_path}")
        print(f"图表目录: {self.figures_dir}")
        print()


if __name__ == '__main__':
    analyzer = StockDataAnalyzer()
    analyzer.run_full_analysis()

