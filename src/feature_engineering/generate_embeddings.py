"""
文本嵌入生成模块

使用sentence-transformers为Reddit帖子生成文本嵌入向量。
支持批量处理和缓存机制。
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Please install it: pip install sentence-transformers")


class EmbeddingGenerator:
    """文本嵌入生成器"""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
        batch_size: int = 32,
        cache_dir: str = 'data/processed/embeddings'
    ):
        """
        初始化嵌入生成器
        
        Args:
            model_name: sentence-transformers模型名称
                - 'all-MiniLM-L6-v2': 384维，快速，推荐用于大量数据
                - 'all-mpnet-base-v2': 768维，更准确但更慢
            device: 计算设备 ('cpu' 或 'cuda')
            batch_size: 批处理大小
            cache_dir: 缓存目录
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install it: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        为文本列表生成嵌入向量
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
        
        Returns:
            嵌入向量数组，形状为 (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # 处理空文本
        texts_processed = []
        empty_indices = []
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                texts_processed.append("")  # 使用空字符串
                empty_indices.append(i)
            else:
                texts_processed.append(text.strip())
        
        logger.info(f"Generating embeddings for {len(texts_processed)} texts...")
        
        # 批量生成嵌入
        embeddings = self.model.encode(
            texts_processed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2归一化
        )
        
        # 对于空文本，使用零向量
        if empty_indices:
            logger.info(f"Found {len(empty_indices)} empty texts, using zero vectors")
            embeddings[empty_indices] = 0.0
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def generate_embeddings_for_posts(
        self,
        df: pd.DataFrame,
        text_col: str = 'text_cleaned',
        post_id_col: str = 'id',
        cache_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        为Reddit帖子生成嵌入向量
        
        Args:
            df: 包含帖子文本的DataFrame
            text_col: 文本列名
            post_id_col: 帖子ID列名
            cache_file: 缓存文件路径（可选）
        
        Returns:
            包含嵌入向量的DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()
        
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in DataFrame")
        
        # 检查缓存
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading embeddings from cache: {cache_file}")
            try:
                cached_data = pd.read_pickle(cache_file)
                if len(cached_data) == len(df):
                    logger.info("Cache loaded successfully")
                    return cached_data
                else:
                    logger.warning(f"Cache size mismatch ({len(cached_data)} vs {len(df)}), regenerating...")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating...")
        
        # 提取文本
        texts = df[text_col].tolist()
        
        # 生成嵌入
        embeddings = self.generate_embeddings(texts, show_progress=True)
        
        # 创建包含嵌入的DataFrame
        result_df = df.copy()
        
        # 将嵌入向量存储为列（每个维度一列）
        embedding_cols = [f'embedding_{i}' for i in range(self.embedding_dim)]
        result_df[embedding_cols] = embeddings
        
        # 保存缓存
        if cache_file:
            try:
                result_df.to_pickle(cache_file)
                logger.info(f"Embeddings cached to: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        return result_df
    
    def load_embeddings_from_cache(
        self,
        cache_file: str
    ) -> Optional[pd.DataFrame]:
        """
        从缓存加载嵌入向量
        
        Args:
            cache_file: 缓存文件路径
        
        Returns:
            包含嵌入的DataFrame，如果文件不存在则返回None
        """
        if not os.path.exists(cache_file):
            return None
        
        try:
            logger.info(f"Loading embeddings from cache: {cache_file}")
            df = pd.read_pickle(cache_file)
            logger.info(f"Loaded {len(df)} embeddings from cache")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None


def generate_embeddings_for_reddit_data(
    reddit_df: pd.DataFrame,
    model_name: str = 'all-MiniLM-L6-v2',
    text_col: str = 'text_cleaned',
    post_id_col: str = 'id',
    output_dir: str = 'data/processed/embeddings',
    cache_file: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    为Reddit数据生成嵌入向量的便捷函数
    
    Args:
        reddit_df: Reddit数据DataFrame
        model_name: 模型名称
        text_col: 文本列名
        post_id_col: 帖子ID列名
        output_dir: 输出目录
        cache_file: 缓存文件路径（如果为None，则自动生成）
        batch_size: 批处理大小
        device: 计算设备
    
    Returns:
        包含嵌入向量的DataFrame
    """
    if reddit_df is None or reddit_df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
    
    # 自动生成缓存文件名
    if cache_file is None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cache_file = os.path.join(output_dir, f'embeddings_{model_name.replace("/", "_")}.pkl')
    
    # 创建嵌入生成器
    generator = EmbeddingGenerator(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        cache_dir=output_dir
    )
    
    # 生成嵌入
    result_df = generator.generate_embeddings_for_posts(
        df=reddit_df,
        text_col=text_col,
        post_id_col=post_id_col,
        cache_file=cache_file
    )
    
    return result_df


if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    
    # 添加src目录到路径
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # 加载测试数据
    data_path = 'data/processed/reddit_cleaned.csv'
    if os.path.exists(data_path):
        logger.info(f"Loading Reddit data from: {data_path}")
        df = pd.read_csv(data_path, nrows=1000)  # 只加载前1000条用于测试
        
        if 'text_cleaned' in df.columns:
            logger.info(f"Loaded {len(df)} posts")
            
            # 生成嵌入（使用小模型进行测试）
            result_df = generate_embeddings_for_reddit_data(
                reddit_df=df,
                model_name='all-MiniLM-L6-v2',
                text_col='text_cleaned',
                batch_size=32
            )
            
            logger.info(f"Generated embeddings for {len(result_df)} posts")
            logger.info(f"Embedding columns: {[col for col in result_df.columns if col.startswith('embedding_')][:5]}...")
        else:
            logger.warning("'text_cleaned' column not found in DataFrame")
    else:
        logger.warning(f"Test data file not found: {data_path}")


