# 特征工程报告

**股票代码**: GME
**生成时间**: 2025-12-10 15:27:43

---

## 数据概览

| 指标 | 数值 |
|------|------|
| 总记录数 | 2,001 |
| 总特征数 | 444 |
| 时间范围 | 2021-01-04 14:00:00+00:00 至 2021-12-30 20:00:00+00:00 |

---

## 特征分类

### 1. 文本嵌入特征
- **数量**: 384
- **维度**: 384 维
- **说明**: 使用sentence-transformers生成的文本嵌入向量

### 2. Reddit统计特征
- **数量**: 4
- **特征**: post_count, total_comments, total_score, unique_authors

### 3. 技术指标特征
- **数量**: 50
- **主要类型**: 
  - 收益率特征
  - 移动平均特征
  - RSI、MACD指标
  - 波动率特征
  - 成交量特征
  - 滞后特征
  - 滚动统计特征

### 4. 目标变量
- **数量**: 2
- **变量**: target_volatility_log_return_abs, target_volatility_price_range

---

## 特征列表

### 文本嵌入特征
- embedding_0
- embedding_1
- embedding_2
- embedding_3
- embedding_4
- embedding_5
- embedding_6
- embedding_7
- embedding_8
- embedding_9
... (共 384 个)

### Reddit统计特征
- post_count
- total_comments
- total_score
- unique_authors

### 技术指标特征（示例）
- GME_open
- GME_high
- GME_low
- GME_close
- GME_volume
- returns
- log_returns
- price_change_pct
- sma_5
- price_vs_sma_5
- sma_10
- price_vs_sma_10
- sma_20
- price_vs_sma_20
- ema_5
- price_vs_ema_5
- ema_10
- price_vs_ema_10
- ema_20
- price_vs_ema_20
... (共 50 个)

---

## 数据质量

| 检查项 | 结果 |
|--------|------|
| 缺失值总数 | 0 |
| 目标变量缺失 | 0 |
| 重复记录 | 0 |

---

## 说明

- 所有特征已按时间排序
- 目标变量为NaN的记录已删除
- 其他缺失值已使用前向填充处理
- 文本嵌入使用 all-MiniLM-L6-v2 模型生成
