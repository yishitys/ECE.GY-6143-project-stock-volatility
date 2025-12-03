# 基于社交媒体讨论量的股票波动率预测

## 项目概述

本项目旨在基于社交媒体平台（Reddit 和 Stocktwits）上的讨论量来预测股票是否会在下一小时内经历显著波动。与情感分析方法不同，我们仅关注讨论的**数量**，并使用**文本嵌入**来捕获帖子的语义信息。

**课程**: ECE.GY 6143 - 机器学习  
**团队成员**: Yishi Tang, Yuxin  
**指导老师**: Professor Rangan

## 项目目标

- 使用社交媒体讨论量预测短期股票波动率（下一小时）
- 利用 Reddit 帖子的文本嵌入来捕获语义模式
- 处理不同日期/小时帖子数量变化的挑战
- 构建结合文本和数值特征的稳健机器学习模型

## 数据源

- **Reddit**: 来自相关子版块的股票讨论（如 r/stocks, r/wallstreetbets）
- **Stocktwits**: 通过 API 获取的股票讨论平台数据
- **股票价格数据**: 历史价格和成交量数据（Yahoo Finance, Alpha Vantage）

## 技术方案

### 1. 数据收集与预处理

- 收集每小时聚合的讨论计数（帖子数、评论数、唯一用户数）
- 通过时间戳对齐社交媒体数据和股票价格数据
- 使用预训练模型（如 `sentence-transformers`）为所有帖子生成文本嵌入
- 计算目标变量：下一小时的已实现波动率或价格变化幅度

### 2. 处理变长帖子数量

**主要方法：聚合特征**
- 使用以下方法按小时聚合文本嵌入：
  - 均值池化（Mean pooling）
  - 加权平均（按点赞数/回复数）
  - 最大池化（Max pooling）
  - 基于注意力的聚合
- 提取统计特征：帖子数、评论数、唯一用户数、增长率
- 处理零帖子小时：使用零向量或历史平均值

**替代方法：**
- 固定窗口加填充（用于 RNN/LSTM 模型）
- 变长序列模型（Transformer/LSTM）

### 3. 特征工程

**讨论量特征：**
- 原始计数：每小时帖子数、评论数、唯一用户数
- 归一化特征：相对于历史平均值的比例
- 时间特征：一天中的小时、星期几、交易日指示器

**文本嵌入特征：**
- 预训练嵌入（如 sentence-transformers 的 `all-MiniLM-L6-v2`）
- 使用多种池化策略按小时聚合嵌入
- 嵌入维度：384 或 768（取决于模型）

**股票特征：**
- 当前价格、成交量
- 技术指标：移动平均、RSI、MACD
- 历史波动率指标

### 4. 模型架构

**主要模型：**
1. **时间序列模型**：LSTM/GRU/Transformer 用于序列预测
2. **传统机器学习**：XGBoost/LightGBM 配合滑动窗口特征
3. **混合模型**：结合文本嵌入（CNN/Transformer）和数值特征

**输入格式：**
- 过去 N 小时的时间序列（如 6、12、24 小时）
- 特征：讨论量 + 文本嵌入 + 股票特征

**输出：**
- 回归：连续波动率指标
- 分类：二分类（高/低波动）或多分类

### 5. 评估指标

**回归指标：**
- RMSE（均方根误差）
- MAE（平均绝对误差）
- R² 分数

**分类指标：**
- 准确率、精确率、召回率、F1 分数
- ROC-AUC

**金融指标（如适用）：**
- 夏普比率
- 最大回撤

## 项目结构

```
ECE.GY-6143-project-stock-volatility/
├── data/
│   ├── raw/              # 原始收集的数据
│   ├── processed/         # 处理和清洗后的数据
│   └── embeddings/       # 缓存的文本嵌入
├── src/
│   ├── data_collection/   # 收集 Reddit/Stocktwits 数据的脚本
│   ├── preprocessing/     # 数据清洗和对齐
│   ├── feature_engineering/  # 特征提取和聚合
│   ├── models/           # 模型定义和训练
│   └── evaluation/       # 评估脚本和指标
├── notebooks/            # 用于探索的 Jupyter notebooks
├── config/               # 配置文件
├── results/              # 模型输出和可视化
├── requirements.txt      # Python 依赖项
└── README.md
```

## 实施时间线

1. **数据收集**（1-2 周）
   - 收集 1-2 个月的历史数据
   - 关注 5-10 只热门股票（如 AAPL, TSLA, GME）

2. **数据预处理**（1 周）
   - 讨论数据和价格数据之间的时间对齐
   - 文本清洗和嵌入生成

3. **特征工程**（1 周）
   - 实现讨论量聚合
   - 实现文本嵌入聚合
   - 构建时间序列特征

4. **模型开发**（2-3 周）
   - 基线模型（简单回归/分类）
   - LSTM/Transformer 模型
   - 特征重要性分析

5. **评估与优化**（1 周）
   - 回测和评估
   - 超参数调优
   - 结果可视化

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd ECE.GY-6143-project-stock-volatility

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

*（随着实施进展将更新）*

```bash
# 数据收集
python src/data_collection/collect_reddit_data.py

# 预处理
python src/preprocessing/preprocess_data.py

# 特征工程
python src/feature_engineering/generate_features.py

# 模型训练
python src/models/train_model.py

# 评估
python src/evaluation/evaluate_model.py
```

## 关键挑战与解决方案

### 挑战 1：变长帖子数量
**解决方案**：按时间窗口聚合特征，对嵌入使用多种池化策略

### 挑战 2：零帖子小时
**解决方案**：使用零向量或历史平均值，添加二元"是否有讨论"特征

### 挑战 3：文本嵌入聚合
**解决方案**：实现多种聚合方法（均值、加权、最大、注意力）并比较性能

### 挑战 4：时间对齐
**解决方案**：仔细对齐社交媒体数据和股票价格数据的时间戳

## 依赖项

*（待更新）*
- Python 3.8+
- pandas, numpy
- scikit-learn
- pytorch/tensorflow
- sentence-transformers
- praw (Reddit API)
- yfinance (股票数据)

## 结果

*（模型训练后将更新）*

## 参考资料

- Reddit API: https://www.reddit.com/dev/api/
- Stocktwits API: https://stocktwits.com/developers
- Sentence Transformers: https://www.sbert.net/

## 许可证

*（待确定）*

## 联系方式

- Yishi Tang
- Yuxin

---

**注意**：本项目是 ECE.GY 6143 机器学习课程的一部分。实施正在进行中，此 README 将随着项目进展而更新。

**Test** 测试权限 --五块