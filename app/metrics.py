"""
Métricas de evaluación IR
Implementa métricas según Clase 1
"""

from typing import List, Dict
import numpy as np

def calculate_precision_at_k(results: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Calcular Precision@k"""
    if not results or not ground_truth:
        return 0.0
    
    top_k_results = results[:k]
    relevant = set(ground_truth)
    retrieved = set(top_k_results)
    
    if not retrieved:
        return 0.0
    
    return len(relevant.intersection(retrieved)) / len(retrieved)

def calculate_recall_at_k(results: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Calcular Recall@k"""
    if not results or not ground_truth:
        return 0.0
    
    top_k_results = results[:k]
    relevant = set(ground_truth)
    retrieved = set(top_k_results)
    
    if not relevant:
        return 0.0
    
    return len(relevant.intersection(retrieved)) / len(relevant)

def calculate_ndcg(results: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Calcular nDCG@k"""
    if not results or not ground_truth:
        return 0.0
    
    top_k_results = results[:k]
    relevant = set(ground_truth)
    
    if not relevant:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, doc in enumerate(top_k_results):
        if doc in relevant:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def calculate_mrr(results: List[str], ground_truth: List[str]) -> float:
    """Calcular MRR (Mean Reciprocal Rank)"""
    if not results or not ground_truth:
        return 0.0
    
    relevant = set(ground_truth)
    
    for i, doc in enumerate(results):
        if doc in relevant:
            return 1.0 / (i + 1)
    
    return 0.0

def calculate_metrics(all_results: List[List[str]], 
                     all_ground_truth: List[List[str]]) -> Dict[str, float]:
    """Calcular métricas completas"""
    metrics = {
        'precision_at_5': [],
        'recall_at_5': [],
        'ndcg_at_5': [],
        'mrr': []
    }
    
    for results, gt in zip(all_results, all_ground_truth):
        metrics['precision_at_5'].append(calculate_precision_at_k(results, gt, k=5))
        metrics['recall_at_5'].append(calculate_recall_at_k(results, gt, k=5))
        metrics['ndcg_at_5'].append(calculate_ndcg(results, gt, k=5))
        metrics['mrr'].append(calculate_mrr(results, gt))
    
    # Promedios
    avg_metrics = {}
    for metric, values in metrics.items():
        avg_metrics[metric] = np.mean(values) if values else 0.0
    
    return avg_metrics



