"""
Model Evaluation Module

Calculates evaluation metrics: RMSE, MAE, R², MAPE, directional accuracy, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path

try:
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Evaluation metrics dictionary
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for evaluation")
    
    # Ensure consistent length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        logger.warning("No valid data for evaluation")
        return {}
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    
    # Directional accuracy
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = (np.sign(y_true_diff) == np.sign(y_pred_diff)).mean() * 100
        metrics['directional_accuracy'] = directional_accuracy
    else:
        metrics['directional_accuracy'] = 0.0
    
    # Correlation coefficient
    if len(y_true) > 1:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['correlation'] = correlation
    else:
        metrics['correlation'] = 0.0
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC calculation)
    
    Returns:
        Evaluation metrics dictionary
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for evaluation")
    
    # Ensure consistent length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    if y_pred_proba is not None:
        y_pred_proba = y_pred_proba[:min_len]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # AUC (if probability predictions available)
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except Exception as e:
            logger.warning(f"Cannot calculate AUC: {e}")
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'regression',
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate model (automatically select regression or classification metrics)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: Task type ('regression' or 'classification')
        y_pred_proba: Predicted probabilities (for classification tasks only)
    
    Returns:
        Dictionary with evaluation metrics
    """
    if task_type == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    elif task_type == 'classification':
        return calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def print_metrics(metrics: Dict[str, float], task_type: str = 'regression'):
    """
    Print evaluation metrics
    
    Args:
        metrics: Dictionary with evaluation metrics
        task_type: Task type
    """
    logger.info("="*60)
    logger.info("Model Evaluation Results")
    logger.info("="*60)
    
    if task_type == 'regression':
        logger.info(f"RMSE: {metrics.get('rmse', 0):.6f}")
        logger.info(f"MAE: {metrics.get('mae', 0):.6f}")
        logger.info(f"R²: {metrics.get('r2', 0):.4f}")
        logger.info(f"MAPE: {metrics.get('mape', 0):.2f}%")
        logger.info(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%")
        logger.info(f"Correlation Coefficient: {metrics.get('correlation', 0):.4f}")
    else:
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1 Score: {metrics.get('f1', 0):.4f}")
        logger.info(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")


def save_evaluation_report(
    metrics: Dict[str, float],
    output_path: str,
    task_type: str = 'regression',
    model_name: str = 'Model',
    stock_symbol: str = 'GME'
):
    """
    Save evaluation report
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Output file path
        task_type: Task type
        model_name: Model name
        stock_symbol: Stock symbol
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if task_type == 'regression':
        report = f"""# Model Evaluation Report

**Model**: {model_name}
**Stock Symbol**: {stock_symbol}
**Task Type**: Regression (Volatility Prediction)
**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Evaluation Metrics

| Metric | Value |
|--------|--------|
| RMSE (Root Mean Square Error) | {metrics.get('rmse', 0):.6f} |
| MAE (Mean Absolute Error) | {metrics.get('mae', 0):.6f} |
| R² (Coefficient of Determination) | {metrics.get('r2', 0):.4f} |
| MAPE (Mean Absolute Percentage Error) | {metrics.get('mape', 0):.2f}% |
| Directional Accuracy | {metrics.get('directional_accuracy', 0):.2f}% |
| Correlation Coefficient | {metrics.get('correlation', 0):.4f} |

---

## Notes

- **RMSE**: Root mean square error between predictions and true values, lower is better
- **MAE**: Mean absolute error between predictions and true values, lower is better
- **R²**: Coefficient of determination, represents proportion of variance explained by model, closer to 1 is better
- **MAPE**: Mean absolute percentage error, lower is better
- **Directional Accuracy**: Accuracy of predicting volatility change direction, higher is better
- **Correlation Coefficient**: Linear correlation coefficient between predictions and true values, closer to 1 is better
"""
    else:
        report = f"""# Model Evaluation Report

**Model**: {model_name}
**Stock Symbol**: {stock_symbol}
**Task Type**: Classification
**Generated Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Evaluation Metrics

| Metric | Value |
|--------|--------|
| Accuracy | {metrics.get('accuracy', 0):.4f} |
| Precision | {metrics.get('precision', 0):.4f} |
| Recall | {metrics.get('recall', 0):.4f} |
| F1 Score | {metrics.get('f1', 0):.4f} |
| ROC-AUC | {metrics.get('roc_auc', 0):.4f} |

---

## Confusion Matrix

{np.array(metrics.get('confusion_matrix', [])).__str__()}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to: {output_path}")


if __name__ == '__main__':
    # Test code
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    # Calculate metrics
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Print results
    print_metrics(metrics, task_type='regression')
    
    # Save report
    save_evaluation_report(metrics, 'results/evaluation_report_test.md', 
                          model_name='Test Model', stock_symbol='GME')

