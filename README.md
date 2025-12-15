# Stock Volatility Prediction from Social Media Discussion Volume

## Project Overview

This project aims to predict whether a stock will experience significant volatility within the next hour based on discussion volume on social media platforms (Reddit). Unlike sentiment analysis approaches, we focus solely on the **volume** of discussions and use **text embeddings** to capture semantic information from posts.

**Course**: ECE.GY 6143 - Machine Learning  
**Team Members**: Yishi Tang, Yuxin  
**Instructor**: Professor Rangan

## Project Goals

- Predict short-term stock volatility (next hour) using social media discussion volume
- Leverage text embeddings from Reddit posts to capture semantic patterns
- Handle the challenge of variable post counts across different days/hours
- Build a robust machine learning model that combines text and numerical features

## Data Sources

### Text Data (Local)
- **Source**: Reddit Finance Data from Kaggle
  - **Dataset**: [Reddit Finance Data](https://www.kaggle.com/datasets/leukipp/reddit-finance-data)
  - **Dataset Author**: leukipp
  - **Description**: Pre-collected Reddit submissions from finance-related subreddits
- **Storage**: All text data is stored locally in the `data/` directory after downloading from Kaggle
- **Format**: CSV and H5 files containing Reddit submissions from multiple subreddits:
  - `r/stocks`, `r/wallstreetbets`, `r/investing`, `r/stockmarket`
  - `r/options`, `r/pennystocks`, `r/finance`, `r/forex`
  - `r/personalfinance`, `r/robinhood`, `r/gme`, and more
- **Data Fields**: 
  - Post metadata: `id`, `author`, `created`, `title`, `selftext`
  - Engagement metrics: `score`, `num_comments`, `upvote_ratio`
  - Content flags: `is_self`, `is_video`, `removed`, `deleted`
- **Location**: `data/raw/{subreddit_name}/submissions_reddit.csv` and `.h5`
- **Note**: To use this dataset, download it from Kaggle and extract the files to the `data/raw/` directory

### Stock Price Data (External APIs)
- **Yahoo Finance** (`yfinance`): Historical price and volume data
- **Alpha Vantage**: Alternative source for stock market data
- **Real-time or Historical**: Can fetch both real-time and historical data via APIs

## Technical Approach

### 1. Data Loading & Preprocessing

**Text Data Loading:**
- Load Reddit submissions from CSV/H5 files in `data/raw/` directory
- Filter and clean posts:
  - Remove deleted/removed posts
  - Filter by date range (align with stock price data availability)
  - Extract text content: combine `title` + `selftext`
- Parse timestamps and align to hourly intervals

**Stock Price Data Loading:**
- Fetch stock price data via `yfinance` or Alpha Vantage API
- Extract: OHLCV (Open, High, Low, Close, Volume) data
- Calculate technical indicators: moving averages, RSI, MACD
- Align timestamps with Reddit data (hourly granularity)

**Data Alignment:**
- Merge text and price data by timestamp (hourly windows)
- Handle timezone differences (Reddit UTC vs. market hours)
- Create unified dataset with both text and price features

### 2. Handling Variable Post Counts

**Primary Approach: Aggregated Features**
- Aggregate text embeddings per hour using:
  - **Mean pooling**: Average of all post embeddings in the hour
  - **Weighted average**: Weight by `score` or `num_comments`
  - **Max pooling**: Element-wise maximum across embeddings
  - **Attention-based aggregation**: Learnable weights for each post
- Extract statistical features: post count, comment count, unique users, growth rate
- Handle zero-post hours with zero vectors or historical averages

**Alternative Approaches:**
- **Fixed window with padding**: Pad to fixed length for RNN/LSTM models
- **Variable-length sequences**: Use Transformer/LSTM with variable input lengths
- **Hierarchical models**: Process posts individually, then aggregate at hour level

### 3. Feature Engineering

**Discussion Volume Features:**
- Raw counts: posts/hour, comments/hour, unique users/hour
- Normalized features: relative to historical averages (rolling mean/median)
- Temporal features: hour of day, day of week, trading day indicator
- Growth rates: change in discussion volume compared to previous hours

**Text Embedding Features:**
- Pre-trained embeddings (e.g., `all-MiniLM-L6-v2` from sentence-transformers)
- Generate embeddings for each post's text (`title` + `selftext`)
- Aggregate embeddings per hour using multiple pooling strategies
- Embedding dimensions: 384 or 768 (depending on model choice)
- Cache embeddings in `data/processed/embeddings/` for efficiency

**Stock Features:**
- Current price, volume, price change
- Technical indicators: 
  - Moving averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- Historical volatility measures (rolling standard deviation)
- Price momentum features

### 4. Model Architecture

**Primary Models:**

1. **Time Series Models**
   - **LSTM/GRU**: Process sequential hourly features
   - **Transformer**: Attention mechanism for temporal patterns
   - **Temporal Convolutional Networks (TCN)**: Efficient sequence modeling

2. **Traditional ML Models**
   - **XGBoost/LightGBM**: With sliding window features
   - **Random Forest**: Baseline for feature importance analysis
   - **Support Vector Regression**: For non-linear relationships

3. **Hybrid Models**
   - **Multi-modal architecture**: Separate encoders for text embeddings and numerical features
   - **Early fusion**: Concatenate all features before model input
   - **Late fusion**: Train separate models and ensemble predictions

**Input Format:**
- Time series of past N hours (e.g., 6, 12, 24 hours)
- Features per hour: discussion volume + aggregated text embeddings + stock features
- Total feature dimension: ~400-800 (depending on embedding model and aggregation)

**Output:**
- **Regression**: Continuous volatility measure (e.g., realized volatility, price change magnitude)
- **Classification**: Binary (high/low volatility) or multi-class (low/medium/high)

### 5. Evaluation Metrics

**Regression Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Directional accuracy (predicting up/down correctly)

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion matrix analysis

**Financial Metrics (if applicable):**
- Sharpe Ratio (if used for trading strategy)
- Maximum Drawdown
- Win rate

## Project Structure

```
ECE.GY-6143-project-stock-volatility/
├── data/
│   ├── raw/                    # Local Reddit data (CSV/H5 files)
│   │   ├── stocks/
│   │   ├── wallstreetbets/
│   │   ├── investing/
│   │   └── ... (other subreddits)
│   ├── processed/              # Processed and cleaned data
│   │   ├── merged_data.csv     # Aligned text + price data
│   │   └── embeddings/         # Cached text embeddings
│   └── stock_prices/           # Downloaded stock price data
├── src/
│   ├── data_loading/           # Scripts for loading local text data
│   │   ├── load_reddit_data.py
│   │   └── load_stock_data.py
│   ├── preprocessing/          # Data cleaning and alignment
│   │   ├── clean_text.py
│   │   └── align_timestamps.py
│   ├── feature_engineering/    # Feature extraction and aggregation
│   │   ├── generate_embeddings.py
│   │   ├── aggregate_features.py
│   │   └── technical_indicators.py
│   ├── models/                 # Model definitions and training
│   │   ├── lstm_model.py
│   │   ├── transformer_model.py
│   │   ├── xgboost_model.py
│   │   └── train.py
│   └── evaluation/             # Evaluation scripts and metrics
│       ├── evaluate.py
│       └── visualize_results.py
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── config/                     # Configuration files
│   ├── model_config.yaml
│   └── data_config.yaml
├── results/                    # Model outputs and visualizations
│   ├── models/                 # Saved model checkpoints
│   └── figures/                 # Plots and visualizations
├── requirements.txt            # Python dependencies
└── README.md
```

## Implementation Approaches

### Approach 1: Simple Aggregation + Traditional ML (Baseline)
**Pros**: Fast to implement, interpretable, good baseline
**Cons**: May lose temporal information
**Steps**:
1. Load all Reddit data from `data/raw/`
2. Aggregate posts by hour (count, mean embeddings)
3. Fetch stock prices via API
4. Train XGBoost/LightGBM with hourly features
5. Evaluate on test set

### Approach 2: Time Series LSTM/GRU
**Pros**: Captures temporal dependencies, good for sequential data
**Cons**: Requires more data, longer training time
**Steps**:
1. Create sequences of past N hours (e.g., 24 hours)
2. Each timestep: aggregated text features + stock features
3. Train LSTM/GRU to predict next-hour volatility
4. Use attention mechanism to focus on important hours

### Approach 3: Transformer-based Model
**Pros**: State-of-the-art, handles long sequences well
**Cons**: More complex, requires more computational resources
**Steps**:
1. Use Transformer encoder for temporal sequences
2. Separate encoders for text embeddings and stock features
3. Cross-attention between text and price modalities
4. Predict volatility with regression head

### Approach 4: Hierarchical Model (Post-level → Hour-level)
**Pros**: Preserves individual post information, more granular
**Cons**: Most complex, computationally expensive
**Steps**:
1. Encode each post individually with transformer
2. Aggregate post-level representations to hour-level
3. Combine with stock features
4. Predict volatility

### Recommended Implementation Order:
1. **Start with Approach 1** (Baseline) - Get pipeline working end-to-end
2. **Move to Approach 2** (LSTM) - Add temporal modeling
3. **Experiment with Approach 3** (Transformer) - If time permits
4. **Compare all approaches** - Final evaluation and report

## Implementation Timeline

1. **Data Loading & Exploration** (1 week)
   - Load Reddit data from `data/raw/` directories
   - Explore data distribution, time ranges, post counts
   - Fetch sample stock price data
   - Create data loading pipeline

2. **Data Preprocessing** (1 week)
   - Clean and filter Reddit posts
   - Align text and price data by timestamp
   - Handle missing data and edge cases
   - Create unified dataset

3. **Feature Engineering** (1-2 weeks)
   - Generate text embeddings for all posts
   - Implement aggregation strategies (mean, weighted, etc.)
   - Calculate stock technical indicators
   - Create time series features

4. **Model Development** (2-3 weeks)
   - Implement baseline model (XGBoost)
   - Implement LSTM/GRU model
   - Implement Transformer model (optional)
   - Hyperparameter tuning

5. **Evaluation & Analysis** (1 week)
   - Backtesting on historical data
   - Feature importance analysis
   - Error analysis and visualization
   - Final report preparation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ECE.GY-6143-project-stock-volatility

# Create virtual environment (recommended)
# For Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# For macOS/Linux:
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Reddit data from Kaggle
# Option 1: Using Kaggle API (recommended)
# First, install kaggle: pip install kaggle
# Set up Kaggle API credentials (see https://www.kaggle.com/docs/api)
kaggle datasets download -d leukipp/reddit-finance-data -p data/raw/
cd data/raw/
unzip reddit-finance-data.zip
cd ../..

# Option 2: Manual download
# Download from https://www.kaggle.com/datasets/leukipp/reddit-finance-data
# Extract the files to data/raw/ directory
```

## Usage

### Quick Start - Full Pipeline (Recommended)

The easiest way to train a model end-to-end:

```bash
# Train XGBoost model
python src/train_main.py --symbol GME --model xgboost

# Train LSTM model
python src/train_main.py --symbol GME --model lstm

# Train both models and compare
python src/train_main.py --symbol GME --model both

# Show all available options
python src/train_main.py --help
```

### Step-by-Step Usage

#### Data Loading
```bash
# Load and preprocess Reddit data from data/raw/
python src/data_loading/load_reddit_data.py --subreddits stocks wallstreetbets

# Fetch stock price data
python src/data_loading/load_stock_data.py --symbols AAPL TSLA GME --start 2021-01-01
```

### Preprocessing
```bash
# Clean and align data
python src/preprocessing/clean_text.py
python src/preprocessing/align_timestamps.py
```

### Feature Engineering
```bash
# Generate text embeddings
python src/feature_engineering/generate_embeddings.py

# Aggregate features by hour
python src/feature_engineering/aggregate_features.py
```

### Model Training
```bash
# Train baseline model
python src/models/train.py --model xgboost --config config/model_config.yaml

# Train LSTM model
python src/models/train.py --model lstm --config config/model_config.yaml
```

### Evaluation
```bash
# Evaluate model
python src/evaluation/evaluate.py --model_path results/models/best_model.pkl
```

## Key Challenges & Solutions

### Challenge 1: Variable Post Counts
**Solution**: Aggregate features per time window, use multiple pooling strategies for embeddings. Handle zero-post hours with zero vectors or historical averages.

### Challenge 2: Data Alignment
**Solution**: Careful timestamp alignment between Reddit data (UTC) and stock market data (market hours). Handle timezone conversions and non-trading hours.

### Challenge 3: Text Embedding Aggregation
**Solution**: Implement multiple aggregation methods (mean, weighted, max, attention) and compare performance. Consider post importance (score, comments) in aggregation.

### Challenge 4: Computational Efficiency
**Solution**: Cache embeddings in `data/processed/embeddings/`. Use batch processing for large datasets. Consider using H5 format for faster I/O.

### Challenge 5: Missing Data
**Solution**: Handle missing stock price data (weekends, holidays). Use forward-fill or interpolation for missing values. Add binary indicators for data availability.

## Environment Setup

### Kaggle API Configuration
To download the Reddit dataset from Kaggle:

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" (downloads `kaggle.json`)
3. Place the file in your home directory:
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
4. Set appropriate permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json  # macOS/Linux
   ```

### Python Version
- Python 3.8 or higher recommended
- Project tested with Python 3.10

## Dependencies

- Python 3.8+
- **Data Processing**: pandas, numpy, h5py
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: pytorch or tensorflow
- **Text Embeddings**: sentence-transformers
- **Stock Data**: yfinance, alpha-vantage
- **Visualization**: matplotlib, seaborn, plotly
- **Utilities**: tqdm, pyyaml

All dependencies are listed in [requirements.txt](requirements.txt) and will be installed with `pip install -r requirements.txt`

## Results

### Latest Model Performance (GME Stock, Test Set)

**LSTM Model**:
- RMSE: 0.094915
- MAE: 0.075621
- R²: 0.9890
- Directional Accuracy: 95.96%
- Correlation: 0.9945

The LSTM model achieves excellent predictive performance on volatility forecasting. See [results/professor_report.md](results/professor_report.md) for detailed comparison with baseline models.

**Generated Reports**:
- [Evaluation Report](results/evaluation_report_test.md)
- [Model Comparison Report](results/professor_report.md)
- [Metrics Summary](results/metrics_summary.md)
- [Feature Report](data/processed/feature_report_GME.md)
- [Preprocessing Report](data/processed/preprocessing_report.md)
- [Target Analysis](data/processed/target_analysis_GME.md)

## References

- **Reddit Data Source**: [Reddit Finance Data on Kaggle](https://www.kaggle.com/datasets/leukipp/reddit-finance-data) by leukipp
- **Stock Data APIs**: 
  - Yahoo Finance: https://pypi.org/project/yfinance/
  - Alpha Vantage: https://www.alphavantage.co/documentation/
- **Sentence Transformers**: https://www.sbert.net/
- **Time Series Forecasting**: Various papers on LSTM/Transformer for financial prediction

## License

*(To be determined)*

## Troubleshooting

### Common Issues

**"kaggle: command not found" or "kaggle is not recognized"**
- Solution: Install kaggle: `pip install kaggle`
- Then set up your Kaggle API credentials (see Environment Setup section)

**"ModuleNotFoundError: No module named X"**
- Solution: Reinstall dependencies: `pip install -r requirements.txt`
- Make sure you're in the correct virtual environment

**Embedding generation takes too long**
- This is expected - embedding generation can take 30 minutes to several hours
- Embeddings are cached after generation
- You can skip embedding generation if using cached data with `--no-embeddings` flag

**"Out of memory" errors**
- For embedding generation: Consider reducing batch size or processing fewer posts
- For model training: Use a smaller dataset or increase GPU memory

**Stock price data not loading**
- Check internet connection (yfinance requires online access)
- Verify ticker symbol is correct (e.g., "GME", "AAPL")
- Try alternate data source: Alpha Vantage API

### Getting Help

- Check existing issues in the repository
- Review the generated reports in `results/` directory for diagnostic information
- Ensure all data files are in correct locations under `data/` directory

## Contact

- Yishi Tang
- Yuxin

---

**Note**: This project is part of ECE.GY 6143 Machine Learning course. The implementation is ongoing and this README will be updated as the project progresses.
