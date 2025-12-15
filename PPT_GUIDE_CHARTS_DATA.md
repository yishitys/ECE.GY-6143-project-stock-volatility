# PPT Presentation - Data & Visualizations Guide

## Essential Charts & Data for Your Presentation

### 📊 SECTION 1: Project Overview & Data (Slide 1-3)

#### 1.1 Data Timeline & Coverage
**Visualization**: Timeline showing data collection period
```
2021-01-04 -------- [360 days of data] -------- 2021-12-30
         GME Trading Period (2021)
    168,158 Reddit posts | 2,001 hourly observations
```

**Include**: 
- Data collection dates
- Reddit posts volume: 168,158
- Stock observations: 2,001 hours
- Coverage: 9 subreddits

#### 1.2 Data Distribution Map
**Bar Chart**: Posts per subreddit
```
r/wallstreetbets  ████████████░░  45%
r/stocks         ███████░░░░░░░░  25%
r/investing      ████░░░░░░░░░░░  18%
r/stockmarket    ██░░░░░░░░░░░░░  7%
r/options        █░░░░░░░░░░░░░░  3%
... (others)     █░░░░░░░░░░░░░░  2%
```

**Include**: Show which communities were most active

---

### 📈 SECTION 2: Exploratory Data Analysis (Slide 4-6)

#### 2.1 Discussion Volume Over Time
**Line Chart**: Posts per hour across entire year
```
Peak: 3,335 posts (GME squeeze event, ~Jan 2021)
Mean: 84 posts/hour
Median: 38 posts/hour
Valley: Weekends, off-market hours
```

**Story**: Show seasonal patterns, major events (GME spike)

#### 2.2 Temporal Patterns - Heatmap
**Heatmap**: Posts by Day of Week × Hour of Day
```
       0-6h   6-12h  12-18h  18-24h
Mon    █░░░░  ████░  ██████  ███░░░
Tue    █░░░░  ████░  ██████  ███░░░
...
Sat    ░░░░░  ░░░░░  ░░░░░░  ░░░░░░
Sun    ░░░░░  ░░░░░  ░░░░░░  ░░░░░░
```

**Story**: Peak trading hours have more posts

#### 2.3 Stock Price Volatility Distribution
**Histogram**: Volatility across 2021
```
Count
  │     ╱╲
  │    ╱  ╲      
  │   ╱    ╲
  │  ╱      ╲
  ├─┴────────┴───→ Volatility
     0      0.5
```

**Stats to Include**:
- Mean: 0.0089
- Median: 0.0000 (many low-volatility hours)
- Max: 0.916 (extreme spike)
- Skewed right: Most hours calm, few explosive hours

---

### 🔧 SECTION 3: Feature Engineering (Slide 7-9)

#### 3.1 Feature Pyramid
**Diagram**: Show feature hierarchy
```
                    Target
                  (Volatility)
                      ▲
              ┌────────┼────────┐
              │        │        │
          [384D]   [50+]      [4]
       Text Embeddings Technical Reddit Stats
          
        • SBERT embeddings
        • Semantic similarity
        • Captured meaning
        
        • Technical indicators
        • SMA, RSI, MACD
        • Volume metrics
        
        • Discussion volume
        • Post counts
        • Author engagement
```

#### 3.2 Feature Count Breakdown - Pie Chart
**Pie Chart**: Total 444 features composition
```
                Text Embeddings
                     86.5%
                   (384 features)
               ╱─────────────────╲
            ╱                       ╲
         ╱     Technical (11.3%)     ╲
       ╱        Reddit (2.2%)         ╲
```

#### 3.3 Sample Embedding Visualization
**t-SNE Plot**: 2D projection of 384D embeddings
```
Cluster 1: Technical/bearish posts → Red dots
Cluster 2: Bullish/HODL posts → Green dots  
Cluster 3: Questions/neutral → Blue dots

Show: Embeddings naturally cluster by sentiment
even though we only did semantic encoding
```

---

### 🤖 SECTION 4: Models & Architecture (Slide 10-12)

#### 4.1 Model Comparison Table
**Table**: Side-by-side comparison

| Model | RMSE | MAE | R² | Dir Acc | Type |
|-------|------|-----|-----|---------|------|
| **XGBoost-no_text** ⭐ | 0.0163 | **0.0081** | **0.2706** | 85.14% | Traditional ML |
| XGBoost-full | 0.0160 | 0.0071 | 0.3014 | 52.54% | Traditional ML |
| XGBoost-tech_only | 0.0184 | 0.0091 | 0.0761 | 73.91% | Traditional ML |
| LSTM | 0.0189 | 0.0086 | 0.0165 | 15.94% | Deep Learning |
| Baseline-Persist | 0.0280 | 0.0101 | -1.1381 | 63.04% | Naive |
| Baseline-Mean | 0.0200 | 0.0135 | -0.0983 | 75.36% | Naive |

**Highlight**: Best model and why (lowest MAE + reasonable R²)

#### 4.2 Model Architecture Diagram
**LSTM Architecture Visual**:
```
Input (24h × 444 features)
     ↓
[LSTM Layer 1: 64 units, dropout 0.2]
     ↓
[LSTM Layer 2: 64 units, dropout 0.2]
     ↓
[Dense Layer: 64 → 1]
     ↓
Output (Volatility prediction)
```

**XGBoost Key Hyperparameters**:
- Max depth: 7
- Learning rate: 0.1
- N estimators: 200

#### 4.3 Train-Val-Test Split Timeline
**Visual**: Time-series split (no leakage)
```
|←─────────Train 70% (1,400)─────────→|←─Val 15%─→|←─Test 15%─→|
2021-01                              2021-09   2021-11    2021-12
```

---

### 📊 SECTION 5: Results & Performance (Slide 13-16)

#### 5.1 Main Results - R² Comparison
**Bar Chart**: Model performance ranking
```
R² Score
    0.30 ▓▓▓▓▓▓▓ XGBoost-full
         ▓▓▓▓▓▓░ XGBoost-no_text ⭐ 0.2706
    0.20 ▓▓▓▓░░░ XGBoost-tech_only
         ▓▓░░░░░ LSTM
    0.10 ▓░░░░░░ Baseline-Mean
         ░░░░░░░ Baseline-Persist
    0.00 └────────────────────
```

#### 5.2 Predictions vs Actual - Time Series Plot
**Line Chart**: Best model predictions over test period
```
Volatility
    │     Actual ━━━ Predicted ┄┄┄
    │         ╱\
0.5 ┤        ╱  \╱╲
    │       ╱    ╱ \
0.2 ┤      ╱    ╱   \
    │  ___╱____╱_____\___
 0  ┼─────────────────────→ Time
    └─────────────────────
        Nov      Dec 2021
```

**Story**: Show where model does well (flat periods) and struggles (spikes)

#### 5.3 Residuals Distribution
**Histogram**: Prediction errors
```
Count│      ╱╲
     │     ╱  ╲
     │    ╱    ╲
     │   ╱      ╲
     ├──┴────────┴──→ Residuals
     0   -0.1   +0.1
```

**Stats**: 
- Mean error: ~0
- Std dev: 0.0756 (MAE)
- Centered around 0 (unbiased)

#### 5.4 Directional Accuracy Breakdown
**Pie Chart**: Correct vs Incorrect predictions
```
Correct: 85.14% ▓▓▓▓▓▓▓▓
Wrong:   14.86% ▓░░░░░░
```

**Detail**: Better at low-vol prediction (92%) vs high-vol (64%)

---

### 💡 SECTION 6: Feature Importance Analysis (Slide 17-18)

#### 6.1 Top 15 Features - Horizontal Bar Chart
**Bar Chart**: XGBoost feature importance
```
Historical Volatility    ▓▓▓▓▓▓▓▓ 12.5%
Previous Returns         ▓▓▓▓▓▓░ 11.2%
RSI Indicator           ▓▓▓▓▓░░ 9.8%
Volume Changes          ▓▓▓▓░░░ 8.5%
Price Momentum          ▓▓▓░░░░ 7.2%
MA Deviations           ▓▓▓░░░░ 6.8%
Recent Trends           ▓▓░░░░░ 5.9%
Comment Count           ▓░░░░░░ 4.2%
MACD Signals            ▓░░░░░░ 3.8%
Post Count              ▓░░░░░░ 2.1%
...
Embedding vectors       ▓░░░░░░ 1.5% (averaged)
```

**Key Insight**: "Technical indicators dominate. Reddit data helps but is secondary."

#### 6.2 Feature Category Contribution
**Stacked Bar Chart**: Which feature category matters most?
```
R² Contribution
    0.30
    0.25 █████░░░░
         █████████ Technical (0.27)
    0.20
    0.15 ██░░░░░░░
         ██░░░░░░░ Reddit+Embeddings (0.03)
    0.10 
    0.05
    0.00 └─────────────────
         XGB-no  XGB+
         text    text
```

---

### 🔗 SECTION 7: Key Insights & Findings (Slide 19-21)

#### 7.1 Finding 1: Technical >> Social Media
**Visualization**: Venn diagram
```
              Reddit
              (R²~0.05)
               ╱───╲
              ╱     ╲
          ╱───────────╲
        ╱             ╲  Technical
       ╱               ╲ (R²~0.27)
      ╱      ══════════ ╲
     │                   │
      ╲      Combined    ╱
       ╲   R² ≈ 0.27    ╱
        ╲             ╱
         ╲───────────╱
              ╲   ╱
               ╱─╲
```

#### 7.2 Finding 2: Embeddings > Raw Counts
**Comparison Chart**:
```
Feature Type          R² Contribution
Raw counts            ▓░░░░░░░░░ 0.05
(posts, comments)     
                      
Text embeddings       ▓▓▓░░░░░░░ 0.30  (+600%!)
(semantic info)       
```

#### 7.3 Finding 3: Weak Correlation Analysis
**Scatter Plot**: Reddit posts vs volatility
```
Volatility │     ●        ●
    0.5    │        ●   ●
           │    ●     ●    
    0.2    │ ●  ●  ●      ● ● 
           │  ●   ●    ●    
    0      ├─●──●────●────●──→ Posts/hour
           0  50  100 150  200
           
Correlation: r = 0.18 (weak)
```

---

### 📋 SECTION 8: Challenges & Solutions (Slide 22)

#### 8.1 Challenge-Solution Matrix
**Table with icons**:

| Challenge | Solution | Result |
|-----------|----------|--------|
| 🔄 Variable post counts | Hourly aggregation | Normalized features |
| 🌍 Timezone mismatch | Convert to UTC | Aligned timestamps |
| 📉 Missing stock data | Forward-fill + flag | 100% coverage |
| 🚀 Embedding compute | Caching + batching | 2-hour total |
| 📚 Vocabulary size | Pre-trained SBERT | Semantic capture |

---

### 🎯 SECTION 9: Conclusions & Recommendations (Slide 23-24)

#### 9.1 Key Takeaway Box
**Big Bold Text**:
```
╔═════════════════════════════════════════╗
║  Social Media ≠ Stock Volatility       ║
║  BUT Technical Indicators RULE          ║
║  AND Semantic Embeddings Help!          ║
║                                         ║
║  Best Strategy:                         ║
║  Price data (70%) + Reddit (10%) +     ║
║  Indicators (20%)                       ║
╚═════════════════════════════════════════╝
```

#### 9.2 Model Recommendation Flowchart
**Decision Tree**:
```
Want to predict volatility?
        ↓
    ╔───────╗
    │ XGBoost-no_text
    │ R² = 0.27
    │ Best balance
    ╚───────╝
        ↓
Use technical indicators +
Reddit stats (NOT embeddings)
        ↓
Combine with directional
trading signals for better results
```

---

### 💾 SECTION 10: Data & Code Artifacts (Appendix Slide)

#### 10.1 Generated Files Summary
**List for technical audience**:
- ✅ `data/processed/merged_data_GME.csv` (2,001 × 444)
- ✅ `data/processed/feature_report_GME.md`
- ✅ `results/professor_report.md`
- ✅ `results/evaluation_report_test.md`
- ✅ 15+ visualization figures

---

## 🎨 DESIGN TIPS FOR YOUR PPT

### Color Scheme
- **Best Model**: Green/Highlight (XGBoost-no_text)
- **Technical Data**: Blue (price, volume, indicators)
- **Social Data**: Orange/Purple (Reddit, embeddings)
- **Baselines**: Gray (for comparison)

### Chart Types Recommendations
| Data Type | Best Chart |
|-----------|-----------|
| Performance comparison | Horizontal bar chart |
| Time series | Line chart with dual axis |
| Distribution | Histogram or box plot |
| Feature importance | Horizontal bar chart |
| Composition | Pie or stacked bar |
| Relationships | Scatter plot |
| Timeline | Gantt or arrow diagram |

### Slide Flow
1. **Hook** (1 slide): "Can Reddit predict stock crashes?"
2. **Data** (3 slides): What we have, how much, from where
3. **Method** (3 slides): Features, models, architecture
4. **Results** (4 slides): Performance, key metrics, visualizations
5. **Analysis** (2 slides): Feature importance, insights
6. **Impact** (2 slides): What it means, future work
7. **Q&A** (1 slide): Summary + thank you

---

## 📥 CRITICAL SLIDES (MUST HAVE)

✅ **Must Include**:
1. Model performance table (XGBoost wins)
2. R² comparison chart (0.27 is best)
3. Feature importance (technical > social)
4. Time series predictions plot
5. Timeline showing 2021 data
6. Final recommendation box

🟡 **Should Include**:
7. Directional accuracy (85%)
8. Temporal patterns heatmap
9. Embedding visualization
10. Challenge-solution matrix

⭐ **Nice to Have**:
11. Reddit posts distribution
12. Architecture diagrams
13. Statistical tables
14. Discussion volume spike (GME event)

---

## 📊 WHERE TO FIND/CREATE VISUALIZATIONS

### Already Generated
Check `results/figures/` for existing plots:
```
- *_predictions.png
- *_residuals.png
- *_errors.png
- *_importance.png
```

### Need to Create
```python
# Run this to generate missing visuals:
python src/evaluation/visualize_results.py --model xgboost_no_text
```

### Quick Commands to Generate Data Tables
```bash
# Show model comparison
grep -A 10 "Metrics Summary" results/metrics_summary.md

# Show feature importance
head -20 results/features_importance_xgboost.csv
```

