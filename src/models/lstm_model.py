"""
LSTM Time Series Model

Implements LSTM model for volatility prediction.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Please install it: pip install torch")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Time series dataset"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24):
        """
        Initialize dataset
        
        Args:
            X: Feature array, shape (n_samples, n_features)
            y: Target array, shape (n_samples,)
            sequence_length: Sequence length (time window size)
        """
        self.sequence_length = sequence_length
        self.X = X
        self.y = y
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(X) - sequence_length):
            self.sequences.append(X[i:i+sequence_length])
            self.targets.append(y[i+sequence_length])
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        logger.info(f"Created {len(self.sequences)} sequences (sequence length: {sequence_length})")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])


class LSTMModel(nn.Module):
    """LSTM波动率预测模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout ratio
            output_size: Output dimension (usually 1 for regression)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_size)
        
        Returns:
            输出张量，形状为 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(last_output)
        
        return output


class LSTMVolatilityModel:
    """LSTM波动率预测模型包装类"""
    
    def __init__(
        self,
        sequence_length: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        """
        初始化LSTM模型
        
        Args:
            sequence_length: 时间序列窗口大小（小时数）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
            learning_rate: 学习率
            batch_size: 批处理大小
            device: 计算设备 ('cpu' 或 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install it: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.input_size = None
        
        logger.info(f"LSTM模型初始化完成")
        logger.info(f"  序列长度: {sequence_length}")
        logger.info(f"  隐藏层大小: {hidden_size}")
        logger.info(f"  层数: {num_layers}")
        logger.info(f"  设备: {device}")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_volatility_log_return_abs',
        exclude_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和目标变量
        
        Args:
            df: 特征DataFrame
            target_col: 目标变量列名
            exclude_cols: 要排除的列名列表
        
        Returns:
            (特征数组, 目标数组)
        """
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'stock_symbol', 'has_reddit_data', 'has_stock_data']
        
        # 添加目标变量到排除列表
        exclude_cols = exclude_cols + [col for col in df.columns if col.startswith('target_')]
        
        # 选择特征列
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # 删除包含NaN的行
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"准备特征: {len(feature_cols)} 个特征, {len(X)} 个样本")
        logger.info(f"目标变量统计: mean={y.mean():.6f}, std={y.std():.6f}")
        
        self.feature_names = feature_cols
        self.input_size = len(feature_cols)
        
        return X, y
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        logger.info("开始训练LSTM模型...")
        
        # 训练过程中会保存临时最佳模型，确保目录存在（避免首次运行时崩溃）
        Path('models').mkdir(parents=True, exist_ok=True)
        
        # 标准化特征和目标
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        else:
            X_val_scaled = None
            y_val_scaled = None
        
        # 创建数据集和数据加载器
        train_dataset = TimeSeriesDataset(
            X_train_scaled, y_train_scaled, self.sequence_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        if X_val_scaled is not None:
            val_dataset = TimeSeriesDataset(
                X_val_scaled, y_val_scaled, self.sequence_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        # 创建模型
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'models/lstm_best_temp.pth')
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    # 加载最佳模型
                    self.model.load_state_dict(torch.load('models/lstm_best_temp.pth', map_location=self.device))
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        logger.info("模型训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数组
        
        Returns:
            预测值数组（已反标准化）
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        self.model.eval()
        
        # 标准化
        X_scaled = self.scaler_X.transform(X)
        
        # 创建序列
        sequences = []
        for i in range(len(X_scaled) - self.sequence_length):
            sequences.append(X_scaled[i:i+self.sequence_length])
        
        if len(sequences) == 0:
            # 如果数据不足，使用最后一个序列
            sequences = [X_scaled[-self.sequence_length:]]
        
        sequences = np.array(sequences)
        
        # 预测
        predictions_scaled = []
        with torch.no_grad():
            for seq in sequences:
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                pred = self.model(seq_tensor)
                predictions_scaled.append(pred.cpu().numpy()[0, 0])
        
        predictions_scaled = np.array(predictions_scaled)
        
        # 反标准化
        predictions = self.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        # 如果预测数量少于输入数量，填充前面的值
        if len(predictions) < len(X):
            padding = np.full(len(X) - len(predictions), predictions[0] if len(predictions) > 0 else 0)
            predictions = np.concatenate([padding, predictions])
        
        return predictions
    
    def save(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'input_size': self.input_size,
            'feature_names': self.feature_names
        }, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu') -> 'LSTMVolatilityModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            device: 计算设备
        
        Returns:
            LSTMVolatilityModel实例
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        instance = cls(
            sequence_length=checkpoint['sequence_length'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            device=device
        )
        
        instance.scaler_X = checkpoint['scaler_X']
        instance.scaler_y = checkpoint['scaler_y']
        instance.input_size = checkpoint['input_size']
        instance.feature_names = checkpoint['feature_names']
        
        # 创建并加载模型
        instance.model = LSTMModel(
            input_size=instance.input_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"模型已从 {filepath} 加载")
        
        return instance


if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    
    # 添加src目录到路径
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 加载特征数据
    data_path = 'data/processed/features_GME.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        logger.info(f"加载数据: {len(df)} 条记录")
        
        # 创建模型
        model = LSTMVolatilityModel(
            sequence_length=24,
            hidden_size=64,
            num_layers=2,
            batch_size=32
        )
        
        # 准备特征
        X, y = model.prepare_features(df)
        
        # 划分数据集
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        val_split_idx = int(len(X_train) * 0.8)
        X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        # 训练
        model.train(X_train_final, y_train_final, X_val, y_val, epochs=20)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_test[len(y_test)-len(y_pred):], y_pred))
        mae = mean_absolute_error(y_test[len(y_test)-len(y_pred):], y_pred)
        r2 = r2_score(y_test[len(y_test)-len(y_pred):], y_pred)
        
        logger.info(f"\n测试集评估:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.4f}")
        
        # 保存模型
        model_path = 'models/lstm_GME_test.pth'
        model.save(model_path)
        
        logger.info("测试完成！")
    else:
        logger.warning(f"测试数据文件不存在: {data_path}")


