# Stock Volatility Prediction from Social Media Discussion Volume

## Project Overview

This project aims to predict whether a stock will experience significant volatility within the next hour based on discussion volume on social media platforms (Reddit and Stocktwits). Unlike sentiment analysis approaches, we focus solely on the **volume** of discussions and use **text embeddings** to capture semantic information from posts.

**Course**: ECE.GY 6143 - Machine Learning  
**Team Members**: Yishi Tang, Yuxin  
**Instructor**: Professor Rangan

## Project Goals

- Predict short-term stock volatility (next hour) using social media discussion volume
- Leverage text embeddings from Reddit posts to capture semantic patterns
- Handle the challenge of variable post counts across different days/hours
- Build a robust machine learning model that combines text and numerical features

## Data Sources

- **Reddit**: Stock-related discussions from subreddits (e.g., r/stocks, r/wallstreetbets)
- **Stocktwits**: Stock discussion platform data via API
- **Stock Price Data**: Historical price and volume data (Yahoo Finance, Alpha Vantage)

## Technical Approach

### 1. Data Collection & Preprocessing

- Collect hourly aggregated discussion counts (posts, comments, unique users)
- Align social media data with stock price data by timestamp
- Generate text embeddings for all posts using pre-trained models (e.g., `sentence-transformers`)
- Calculate target variable: realized volatility or price change magnitude for the next hour

### 2. Handling Variable Post Counts

**Primary Approach: Aggregated Features**
- Aggregate text embeddings per hour using:
  - Mean pooling
  - Weighted average (by upvotes/replies)
  - Max pooling
  - Attention-based aggregation
- Extract statistical features: post count, comment count, unique users, growth rate
- Handle zero-post hours with zero vectors or historical averages

**Alternative Approaches:**
- Fixed window with padding for RNN/LSTM models
- Variable-length sequence models (Transformer/LSTM)

### 3. Feature Engineering

**Discussion Volume Features:**
- Raw counts: posts/hour, comments/hour, unique users/hour
- Normalized features: relative to historical averages
- Temporal features: hour of day, day of week, trading day indicator

**Text Embedding Features:**
- Pre-trained embeddings (e.g., `all-MiniLM-L6-v2` from sentence-transformers)
- Aggregated embeddings per hour using multiple pooling strategies
- Embedding dimensions: 384 or 768 (depending on model)

**Stock Features:**
- Current price, volume
- Technical indicators: moving averages, RSI, MACD
- Historical volatility measures

### 4. Model Architecture

**Primary Models:**
1. **Time Series Models**: LSTM/GRU/Transformer for sequential prediction
2. **Traditional ML**: XGBoost/LightGBM with sliding window features
3. **Hybrid Models**: Combine text embeddings (CNN/Transformer) with numerical features

**Input Format:**
- Time series of past N hours (e.g., 6, 12, 24 hours)
- Features: discussion volume + text embeddings + stock features

**Output:**
- Regression: Continuous volatility measure
- Classification: Binary (high/low volatility) or multi-class

### 5. Evaluation Metrics

**Regression Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC

**Financial Metrics (if applicable):**
- Sharpe Ratio
- Maximum Drawdown

## Project Structure

```
ECE.GY-6143-project-stock-volatility/
├── data/
│   ├── raw/              # Raw collected data
│   ├── processed/         # Processed and cleaned data
│   └── embeddings/       # Cached text embeddings
├── src/
│   ├── data_collection/   # Scripts for collecting Reddit/Stocktwits data
│   ├── preprocessing/     # Data cleaning and alignment
│   ├── feature_engineering/  # Feature extraction and aggregation
│   ├── models/           # Model definitions and training
│   └── evaluation/       # Evaluation scripts and metrics
├── notebooks/            # Jupyter notebooks for exploration
├── config/               # Configuration files
├── results/              # Model outputs and visualizations
├── requirements.txt      # Python dependencies
└── README.md
```

## Implementation Timeline

1. **Data Collection** (1-2 weeks)
   - Collect 1-2 months of historical data
   - Focus on 5-10 popular stocks (e.g., AAPL, TSLA, GME)

2. **Data Preprocessing** (1 week)
   - Time alignment between discussion and price data
   - Text cleaning and embedding generation

3. **Feature Engineering** (1 week)
   - Implement discussion volume aggregation
   - Implement text embedding aggregation
   - Build time series features

4. **Model Development** (2-3 weeks)
   - Baseline models (simple regression/classification)
   - LSTM/Transformer models
   - Feature importance analysis

5. **Evaluation & Optimization** (1 week)
   - Backtesting and evaluation
   - Hyperparameter tuning
   - Results visualization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ECE.GY-6143-project-stock-volatility

# Install dependencies
pip install -r requirements.txt
```

## Usage

*(To be updated as implementation progresses)*

```bash
# Data collection
python src/data_collection/collect_reddit_data.py

# Preprocessing
python src/preprocessing/preprocess_data.py

# Feature engineering
python src/feature_engineering/generate_features.py

# Model training
python src/models/train_model.py

# Evaluation
python src/evaluation/evaluate_model.py
```

## Key Challenges & Solutions

### Challenge 1: Variable Post Counts
**Solution**: Aggregate features per time window, use multiple pooling strategies for embeddings

### Challenge 2: Zero-Post Hours
**Solution**: Use zero vectors or historical averages, add binary "has_discussion" feature

### Challenge 3: Text Embedding Aggregation
**Solution**: Implement multiple aggregation methods (mean, weighted, max, attention) and compare performance

### Challenge 4: Temporal Alignment
**Solution**: Careful timestamp alignment between social media data and stock price data

## Dependencies

*(To be updated)*
- Python 3.8+
- pandas, numpy
- scikit-learn
- pytorch/tensorflow
- sentence-transformers
- praw (Reddit API)
- yfinance (stock data)

## Results

*(To be updated after model training)*

## References

- Reddit API: https://www.reddit.com/dev/api/
- Stocktwits API: https://stocktwits.com/developers
- Sentence Transformers: https://www.sbert.net/

## License

*(To be determined)*

## Contact

- Yishi Tang
- Yuxin

---

**Note**: This project is part of ECE.GY 6143 Machine Learning course. The implementation is ongoing and this README will be updated as the project progresses.
