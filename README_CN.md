# 基于社交媒体讨论量的股票波动率预测

## 项目概述

本项目旨在基于社交媒体平台（Reddit）上的讨论量来预测股票是否会在下一小时内经历显著波动。与情感分析方法不同，我们仅关注讨论的**数量**，并使用**文本嵌入**来捕获帖子的语义信息。

**课程**: ECE.GY 6143 - 机器学习  
**团队成员**: Yishi Tang, Yuxin  
**指导老师**: Professor Rangan

## 项目目标

- 使用社交媒体讨论量预测短期股票波动率（下一小时）
- 利用 Reddit 帖子的文本嵌入来捕获语义模式
- 处理不同日期/小时帖子数量变化的挑战
- 构建结合文本和数值特征的稳健机器学习模型

## 数据源

### 文本数据（本地）
- **来源**: 来自 Kaggle 的 Reddit 金融数据
  - **数据集**: [Reddit Finance Data](https://www.kaggle.com/datasets/leukipp/reddit-finance-data)
  - **数据集作者**: leukipp
  - **描述**: 预收集的来自金融相关 Reddit 子版块的提交数据
- **存储**: 从 Kaggle 下载后，所有文本数据存储在本地 `data/` 目录中
- **格式**: 包含多个 Reddit 子版块提交数据的 CSV 和 H5 文件：
  - `r/stocks`, `r/wallstreetbets`, `r/investing`, `r/stockmarket`
  - `r/options`, `r/pennystocks`, `r/finance`, `r/forex`
  - `r/personalfinance`, `r/robinhood`, `r/gme` 等
- **数据字段**: 
  - 帖子元数据：`id`, `author`, `created`, `title`, `selftext`
  - 参与度指标：`score`, `num_comments`, `upvote_ratio`
  - 内容标志：`is_self`, `is_video`, `removed`, `deleted`
- **位置**: `data/raw/{subreddit_name}/submissions_reddit.csv` 和 `.h5`
- **注意**: 要使用此数据集，请从 Kaggle 下载并将文件解压到 `data/raw/` 目录

### 股价数据（外部 API）
- **Yahoo Finance** (`yfinance`): 历史价格和成交量数据
- **Alpha Vantage**: 股票市场数据的替代来源
- **实时或历史**: 可通过 API 获取实时和历史数据

## 技术方案

### 1. 数据加载与预处理

**文本数据加载:**
- 从 `data/raw/` 目录加载 Reddit 提交数据（CSV/H5 文件）
- 过滤和清洗帖子：
  - 删除已删除/已移除的帖子
  - 按日期范围过滤（与股价数据可用性对齐）
  - 提取文本内容：合并 `title` + `selftext`
- 解析时间戳并按小时对齐

**股价数据加载:**
- 通过 `yfinance` 或 Alpha Vantage API 获取股价数据
- 提取：OHLCV（开盘、最高、最低、收盘、成交量）数据
- 计算技术指标：移动平均、RSI、MACD
- 与 Reddit 数据对齐时间戳（小时粒度）

**数据对齐:**
- 按时间戳合并文本和价格数据（小时窗口）
- 处理时区差异（Reddit UTC vs. 市场交易时间）
- 创建包含文本和价格特征的统一数据集

### 2. 处理变长帖子数量

**主要方法：聚合特征**
- 使用以下方法按小时聚合文本嵌入：
  - **均值池化**：该小时内所有帖子嵌入的平均值
  - **加权平均**：按 `score` 或 `num_comments` 加权
  - **最大池化**：跨嵌入的元素级最大值
  - **基于注意力的聚合**：每个帖子的可学习权重
- 提取统计特征：帖子数、评论数、唯一用户数、增长率
- 处理零帖子小时：使用零向量或历史平均值

**替代方法:**
- **固定窗口加填充**：填充到固定长度用于 RNN/LSTM 模型
- **变长序列**：使用 Transformer/LSTM 处理可变长度输入
- **分层模型**：单独处理每个帖子，然后在小时级别聚合

### 3. 特征工程

**讨论量特征:**
- 原始计数：每小时帖子数、评论数、唯一用户数
- 归一化特征：相对于历史平均值（滚动均值/中位数）
- 时间特征：一天中的小时、星期几、交易日指示器
- 增长率：与前几小时相比讨论量的变化

**文本嵌入特征:**
- 预训练嵌入（如 sentence-transformers 的 `all-MiniLM-L6-v2`）
- 为每个帖子的文本（`title` + `selftext`）生成嵌入
- 使用多种池化策略按小时聚合嵌入
- 嵌入维度：384 或 768（取决于模型选择）
- 在 `data/processed/embeddings/` 中缓存嵌入以提高效率

**股票特征:**
- 当前价格、成交量、价格变化
- 技术指标：
  - 移动平均（SMA、EMA）
  - RSI（相对强弱指数）
  - MACD（移动平均收敛散度）
  - 布林带
- 历史波动率指标（滚动标准差）
- 价格动量特征

### 4. 模型架构

**主要模型:**

1. **时间序列模型**
   - **LSTM/GRU**: 处理顺序小时特征
   - **Transformer**: 用于时间模式的注意力机制
   - **时间卷积网络 (TCN)**: 高效的序列建模

2. **传统机器学习模型**
   - **XGBoost/LightGBM**: 配合滑动窗口特征
   - **随机森林**: 用于特征重要性分析的基线
   - **支持向量回归**: 用于非线性关系

3. **混合模型**
   - **多模态架构**: 文本嵌入和数值特征的独立编码器
   - **早期融合**: 在模型输入前连接所有特征
   - **后期融合**: 训练独立模型并集成预测

**输入格式:**
- 过去 N 小时的时间序列（如 6、12、24 小时）
- 每小时特征：讨论量 + 聚合文本嵌入 + 股票特征
- 总特征维度：~400-800（取决于嵌入模型和聚合方法）

**输出:**
- **回归**: 连续波动率指标（如已实现波动率、价格变化幅度）
- **分类**: 二分类（高/低波动）或多分类（低/中/高）

### 5. 评估指标

**回归指标:**
- RMSE（均方根误差）
- MAE（平均绝对误差）
- R² 分数
- 方向准确率（正确预测涨跌）

**分类指标:**
- 准确率、精确率、召回率、F1 分数
- ROC-AUC
- 混淆矩阵分析

**金融指标（如适用）:**
- 夏普比率（如果用于交易策略）
- 最大回撤
- 胜率

## 项目结构

```
ECE.GY-6143-project-stock-volatility/
├── data/
│   ├── raw/                    # 本地 Reddit 数据（CSV/H5 文件）
│   │   ├── stocks/
│   │   ├── wallstreetbets/
│   │   ├── investing/
│   │   └── ... (其他子版块)
│   ├── processed/              # 处理和清洗后的数据
│   │   ├── merged_data.csv     # 对齐的文本 + 价格数据
│   │   └── embeddings/         # 缓存的文本嵌入
│   └── stock_prices/           # 下载的股价数据
├── src/
│   ├── data_loading/           # 加载本地文本数据的脚本
│   │   ├── load_reddit_data.py
│   │   └── load_stock_data.py
│   ├── preprocessing/          # 数据清洗和对齐
│   │   ├── clean_text.py
│   │   └── align_timestamps.py
│   ├── feature_engineering/    # 特征提取和聚合
│   │   ├── generate_embeddings.py
│   │   ├── aggregate_features.py
│   │   └── technical_indicators.py
│   ├── models/                 # 模型定义和训练
│   │   ├── lstm_model.py
│   │   ├── transformer_model.py
│   │   ├── xgboost_model.py
│   │   └── train.py
│   └── evaluation/             # 评估脚本和指标
│       ├── evaluate.py
│       └── visualize_results.py
├── notebooks/                  # 用于探索的 Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── config/                     # 配置文件
│   ├── model_config.yaml
│   └── data_config.yaml
├── results/                    # 模型输出和可视化
│   ├── models/                 # 保存的模型检查点
│   └── figures/                # 图表和可视化
├── requirements.txt            # Python 依赖项
└── README.md
```

## 实施方案

### 方案 1: 简单聚合 + 传统机器学习（基线）
**优点**: 实现快速，可解释性强，良好的基线
**缺点**: 可能丢失时间信息
**步骤**:
1. 从 `data/raw/` 加载所有 Reddit 数据
2. 按小时聚合帖子（计数、平均嵌入）
3. 通过 API 获取股价
4. 使用小时特征训练 XGBoost/LightGBM
5. 在测试集上评估

### 方案 2: 时间序列 LSTM/GRU
**优点**: 捕获时间依赖性，适合序列数据
**缺点**: 需要更多数据，训练时间更长
**步骤**:
1. 创建过去 N 小时的序列（如 24 小时）
2. 每个时间步：聚合文本特征 + 股票特征
3. 训练 LSTM/GRU 预测下一小时波动率
4. 使用注意力机制关注重要小时

### 方案 3: 基于 Transformer 的模型
**优点**: 最先进，能很好地处理长序列
**缺点**: 更复杂，需要更多计算资源
**步骤**:
1. 使用 Transformer 编码器处理时间序列
2. 文本嵌入和股票特征的独立编码器
3. 文本和价格模态之间的交叉注意力
4. 使用回归头预测波动率

### 方案 4: 分层模型（帖子级 → 小时级）
**优点**: 保留单个帖子信息，更细粒度
**缺点**: 最复杂，计算成本高
**步骤**:
1. 使用 transformer 单独编码每个帖子
2. 将帖子级表示聚合到小时级
3. 与股票特征结合
4. 预测波动率

### 推荐实施顺序:
1. **从方案 1 开始**（基线）- 让整个流程端到端运行
2. **转向方案 2**（LSTM）- 添加时间建模
3. **尝试方案 3**（Transformer）- 如果时间允许
4. **比较所有方案** - 最终评估和报告

## 实施时间线

1. **数据加载与探索**（1 周）
   - 从 `data/raw/` 目录加载 Reddit 数据
   - 探索数据分布、时间范围、帖子数量
   - 获取样本股价数据
   - 创建数据加载管道

2. **数据预处理**（1 周）
   - 清洗和过滤 Reddit 帖子
   - 按时间戳对齐文本和价格数据
   - 处理缺失数据和边缘情况
   - 创建统一数据集

3. **特征工程**（1-2 周）
   - 为所有帖子生成文本嵌入
   - 实现聚合策略（均值、加权等）
   - 计算股票技术指标
   - 创建时间序列特征

4. **模型开发**（2-3 周）
   - 实现基线模型（XGBoost）
   - 实现 LSTM/GRU 模型
   - 实现 Transformer 模型（可选）
   - 超参数调优

5. **评估与分析**（1 周）
   - 在历史数据上回测
   - 特征重要性分析
   - 错误分析和可视化
   - 最终报告准备

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd ECE.GY-6143-project-stock-volatility

# 安装依赖
pip install -r requirements.txt

# 从 Kaggle 下载 Reddit 数据
# 方法 1: 使用 Kaggle API（推荐）
# 首先，安装 kaggle: pip install kaggle
# 设置 Kaggle API 凭证（参见 https://www.kaggle.com/docs/api）
kaggle datasets download -d leukipp/reddit-finance-data -p data/raw/
cd data/raw/
unzip reddit-finance-data.zip
cd ../..

# 方法 2: 手动下载
# 从 https://www.kaggle.com/datasets/leukipp/reddit-finance-data 下载
# 将文件解压到 data/raw/ 目录
```

## 使用方法

### 数据加载
```bash
# 从 data/raw/ 加载和预处理 Reddit 数据
python src/data_loading/load_reddit_data.py --subreddits stocks wallstreetbets

# 获取股价数据
python src/data_loading/load_stock_data.py --symbols AAPL TSLA GME --start 2021-01-01
```

### 预处理
```bash
# 清洗和对齐数据
python src/preprocessing/clean_text.py
python src/preprocessing/align_timestamps.py
```

### 特征工程
```bash
# 生成文本嵌入
python src/feature_engineering/generate_embeddings.py

# 按小时聚合特征
python src/feature_engineering/aggregate_features.py
```

### 模型训练
```bash
# 训练基线模型
python src/models/train.py --model xgboost --config config/model_config.yaml

# 训练 LSTM 模型
python src/models/train.py --model lstm --config config/model_config.yaml
```

### 评估
```bash
# 评估模型
python src/evaluation/evaluate.py --model_path results/models/best_model.pkl
```

## 关键挑战与解决方案

### 挑战 1: 变长帖子数量
**解决方案**: 按时间窗口聚合特征，对嵌入使用多种池化策略。使用零向量或历史平均值处理零帖子小时。

### 挑战 2: 数据对齐
**解决方案**: 仔细对齐 Reddit 数据（UTC）和股票市场数据（市场交易时间）的时间戳。处理时区转换和非交易时间。

### 挑战 3: 文本嵌入聚合
**解决方案**: 实现多种聚合方法（均值、加权、最大、注意力）并比较性能。在聚合中考虑帖子重要性（分数、评论）。

### 挑战 4: 计算效率
**解决方案**: 在 `data/processed/embeddings/` 中缓存嵌入。对大型数据集使用批处理。考虑使用 H5 格式以提高 I/O 速度。

### 挑战 5: 缺失数据
**解决方案**: 处理缺失的股价数据（周末、节假日）。对缺失值使用前向填充或插值。添加数据可用性的二元指示器。

## 依赖项

- Python 3.8+
- **数据处理**: pandas, numpy, h5py
- **机器学习**: scikit-learn, xgboost, lightgbm
- **深度学习**: pytorch 或 tensorflow
- **文本嵌入**: sentence-transformers
- **股票数据**: yfinance, alpha-vantage
- **可视化**: matplotlib, seaborn, plotly
- **工具**: tqdm, pyyaml

## 结果

*（模型训练后将更新）*

## 参考资料

- **Reddit 数据来源**: [Kaggle 上的 Reddit 金融数据](https://www.kaggle.com/datasets/leukipp/reddit-finance-data) by leukipp
- **股票数据 API**:
  - Yahoo Finance: https://pypi.org/project/yfinance/
  - Alpha Vantage: https://www.alphavantage.co/documentation/
- **Sentence Transformers**: https://www.sbert.net/
- **时间序列预测**: 关于 LSTM/Transformer 用于金融预测的各种论文

## 许可证

*（待确定）*

## 联系方式

- Yishi Tang
- Yuxin

---

**注意**: 本项目是 ECE.GY 6143 机器学习课程的一部分。实施正在进行中，此 README 将随着项目进展而更新。
