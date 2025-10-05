# app/evaluation/ir_metrics.py
"""
Métricas de Information Retrieval (IR)
Basado en conceptos de la Clase 1: evaluación de sistemas RAG
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class IRMetrics:
    """Métricas de Information Retrieval"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    map_score: float
    total_queries: int
    relevant_docs: int
    retrieved_docs: int

class IRMetricsCalculator:
    """Calculadora de métricas de IR"""
    
    def __init__(self):
        self.query_results: List[Dict[str, Any]] = []
        self.global_metrics: Dict[str, float] = {}
    
    def add_query_result(self, query_id: str, retrieved_docs: List[str], 
                        relevant_docs: Set[str], relevance_scores: Dict[str, int] = None):
        """Agregar resultado de una consulta"""
        result = {
            "query_id": query_id,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs,
            "relevance_scores": relevance_scores or {},
            "metrics": self._calculate_query_metrics(retrieved_docs, relevant_docs, relevance_scores)
        }
        self.query_results.append(result)
        return result
    
    def _calculate_query_metrics(self, retrieved_docs: List[str], 
                               relevant_docs: Set[str], 
                               relevance_scores: Dict[str, int] = None) -> Dict[str, Any]:
        """Calcular métricas para una consulta específica"""
        metrics = {}
        
        # Precision@k y Recall@k
        for k in [1, 3, 5, 10, 20]:
            precision = self._precision_at_k(retrieved_docs, relevant_docs, k)
            recall = self._recall_at_k(retrieved_docs, relevant_docs, k)
            ndcg = self._ndcg_at_k(retrieved_docs, relevance_scores or {}, k)
            
            metrics[f"precision@{k}"] = precision
            metrics[f"recall@{k}"] = recall
            metrics[f"ndcg@{k}"] = ndcg
        
        # MRR (Mean Reciprocal Rank)
        metrics["mrr"] = self._mrr(retrieved_docs, relevant_docs)
        
        # MAP (Mean Average Precision)
        metrics["map"] = self._map_score(retrieved_docs, relevant_docs)
        
        return metrics
    
    def _precision_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Precision@k"""
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = len(relevant_docs.intersection(set(retrieved_at_k)))
        return relevant_retrieved / k
    
    def _recall_at_k(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Recall@k"""
        if not relevant_docs:
            return 1.0 if k == 0 else 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = len(relevant_docs.intersection(set(retrieved_at_k)))
        return relevant_retrieved / len(relevant_docs)
    
    def _ndcg_at_k(self, retrieved_docs: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        """Normalized Discounted Cumulative Gain@k"""
        if k == 0:
            return 0.0
        
        # DCG@k
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = relevance_scores.get(doc, 0)
            dcg += relevance / np.log2(i + 2)  # i+2 porque log2(1) = 0
        
        # IDCG@k (Ideal DCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _mrr(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def _map_score(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Mean Average Precision"""
        if not relevant_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def calculate_global_metrics(self) -> Dict[str, float]:
        """Calcular métricas globales del sistema"""
        if not self.query_results:
            return {}
        
        global_metrics = {}
        
        # Agregar métricas por k
        for k in [1, 3, 5, 10, 20]:
            precision_values = [r["metrics"][f"precision@{k}"] for r in self.query_results]
            recall_values = [r["metrics"][f"recall@{k}"] for r in self.query_results]
            ndcg_values = [r["metrics"][f"ndcg@{k}"] for r in self.query_results]
            
            global_metrics[f"avg_precision@{k}"] = np.mean(precision_values)
            global_metrics[f"avg_recall@{k}"] = np.mean(recall_values)
            global_metrics[f"avg_ndcg@{k}"] = np.mean(ndcg_values)
        
        # MRR y MAP globales
        mrr_values = [r["metrics"]["mrr"] for r in self.query_results]
        map_values = [r["metrics"]["map"] for r in self.query_results]
        
        global_metrics["avg_mrr"] = np.mean(mrr_values)
        global_metrics["avg_map"] = np.mean(map_values)
        
        # Estadísticas adicionales
        global_metrics["total_queries"] = len(self.query_results)
        global_metrics["avg_relevant_docs"] = np.mean([len(r["relevant_docs"]) for r in self.query_results])
        global_metrics["avg_retrieved_docs"] = np.mean([len(r["retrieved_docs"]) for r in self.query_results])
        
        self.global_metrics = global_metrics
        return global_metrics
    
    def get_query_analysis(self, query_id: str) -> Dict[str, Any]:
        """Obtener análisis detallado de una consulta"""
        for result in self.query_results:
            if result["query_id"] == query_id:
                return result
        return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de rendimiento"""
        if not self.global_metrics:
            self.calculate_global_metrics()
        
        return {
            "total_queries": self.global_metrics.get("total_queries", 0),
            "best_precision": max([v for k, v in self.global_metrics.items() if k.startswith("avg_precision@")]),
            "best_recall": max([v for k, v in self.global_metrics.items() if k.startswith("avg_recall@")]),
            "best_ndcg": max([v for k, v in self.global_metrics.items() if k.startswith("avg_ndcg@")]),
            "mrr": self.global_metrics.get("avg_mrr", 0),
            "map": self.global_metrics.get("avg_map", 0)
        }

# Instancia global del calculador de métricas IR
ir_metrics_calculator = IRMetricsCalculator()
