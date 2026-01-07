from typing import Dict, List, Tuple
"""
Statistical analysis functions for Monte Carlo results
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_percentiles(values: List[float], percentiles: List[int] = [5, 25, 50, 75, 95]) -> Dict:
    """
    Calculate percentiles for a list of values
    
    Args:
        values: List of numeric values
        percentiles: List of percentile values (0-100)
        
    Returns:
        Dictionary mapping percentile names to values
    """
    result = {}
    for p in percentiles:
        result[f'percentile_{p:02d}'] = np.percentile(values, p)
    return result


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics from Monte Carlo results
    
    Args:
        results: List of simulation result dictionaries
        
    Returns:
        Dictionary of statistics for each metric
    """
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    metrics = ['annual_return', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    
    stats = {}
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            
            stats[metric] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std_dev': float(values.std()),
                'min_value': float(values.min()),
                'max_value': float(values.max()),
                **calculate_percentiles(values.tolist())
            }
    
    stats['total_simulations'] = len(results)
    stats['successful_simulations'] = len([r for r in results if r is not None])
    
    return stats


def calculate_confidence_intervals(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a list of values
    
    Args:
        values: List of numeric values
        confidence: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(values, lower_percentile)
    upper = np.percentile(values, upper_percentile)
    
    return (lower, upper)


def calculate_probability(values: List[float], threshold: float, operator: str = '>') -> float:
    """
    Calculate probability of metric exceeding threshold
    
    Args:
        values: List of values
        threshold: Threshold value
        operator: Comparison operator ('>', '<', '>=', '<=')
        
    Returns:
        Probability (0-1)
    """
    values_array = np.array(values)
    
    if operator == '>':
        count = np.sum(values_array > threshold)
    elif operator == '<':
        count = np.sum(values_array < threshold)
    elif operator == '>=':
        count = np.sum(values_array >= threshold)
    elif operator == '<=':
        count = np.sum(values_array <= threshold)
    else:
        raise ValueError(f"Unknown operator: {operator}")
    
    return count / len(values) if len(values) > 0 else 0.0


def calculate_parameter_sensitivity(results_df: pd.DataFrame, parameter: str, metric: str = 'annual_return') -> Dict:
    """
    Calculate how a metric varies with a parameter
    
    Args:
        results_df: DataFrame of Monte Carlo results
        parameter: Parameter name to analyze
        metric: Metric to track
        
    Returns:
        Dictionary with parameter values and metric statistics
    """
    # Extract parameter from JSON
    results_df['param_value'] = results_df['parameters'].apply(
        lambda x: eval(x).get(parameter) if isinstance(x, str) else x.get(parameter)
    )
    
    # Group by parameter value
    grouped = results_df.groupby('param_value')[metric]
    
    sensitivity = {}
    for param_val, group in grouped:
        sensitivity[str(param_val)] = {
            'mean': float(group.mean()),
            'std': float(group.std()),
            'count': len(group)
        }
    
    return sensitivity
