# app/evaluation/security_metrics.py
"""
Métricas de Seguridad y Guardrails
Basado en conceptos de la Clase 2: evaluación de sistemas de seguridad
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta

@dataclass
class SecurityMetrics:
    """Métricas de seguridad del sistema"""
    total_queries: int
    blocked_queries: int
    sanitized_queries: int
    rate_limited_queries: int
    security_warnings: int
    false_positives: int
    false_negatives: int
    avg_processing_time: float
    security_score: float

class SecurityMetricsCalculator:
    """Calculadora de métricas de seguridad"""
    
    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.security_patterns: Dict[str, int] = defaultdict(int)
        self.time_series: List[Dict[str, Any]] = []
    
    def add_security_event(self, event_type: str, query: str, result: Dict[str, Any], 
                          processing_time: float, metadata: Dict[str, Any] = None):
        """Agregar evento de seguridad"""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "query": query,
            "result": result,
            "processing_time": processing_time,
            "metadata": metadata or {}
        }
        self.security_events.append(event)
        
        # Actualizar métricas de rendimiento
        self.performance_metrics[event_type].append(processing_time)
        
        # Actualizar patrones de seguridad
        if "blocked" in event_type:
            self.security_patterns["blocked"] += 1
        if "sanitized" in event_type:
            self.security_patterns["sanitized"] += 1
        if "rate_limited" in event_type:
            self.security_patterns["rate_limited"] += 1
    
    def calculate_security_metrics(self) -> SecurityMetrics:
        """Calcular métricas de seguridad"""
        if not self.security_events:
            return SecurityMetrics(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0)
        
        total_queries = len(self.security_events)
        blocked_queries = sum(1 for e in self.security_events if "blocked" in e["event_type"])
        sanitized_queries = sum(1 for e in self.security_events if "sanitized" in e["event_type"])
        rate_limited_queries = sum(1 for e in self.security_events if "rate_limited" in e["event_type"])
        security_warnings = sum(1 for e in self.security_events if "warning" in e["event_type"])
        
        # Calcular falsos positivos y negativos (requiere evaluación manual)
        false_positives = self._calculate_false_positives()
        false_negatives = self._calculate_false_negatives()
        
        # Tiempo promedio de procesamiento
        avg_processing_time = np.mean([e["processing_time"] for e in self.security_events])
        
        # Puntuación de seguridad (0-100)
        security_score = self._calculate_security_score(
            total_queries, blocked_queries, sanitized_queries, 
            false_positives, false_negatives
        )
        
        return SecurityMetrics(
            total_queries=total_queries,
            blocked_queries=blocked_queries,
            sanitized_queries=sanitized_queries,
            rate_limited_queries=rate_limited_queries,
            security_warnings=security_warnings,
            false_positives=false_positives,
            false_negatives=false_negatives,
            avg_processing_time=avg_processing_time,
            security_score=security_score
        )
    
    def _calculate_false_positives(self) -> int:
        """Calcular falsos positivos (requiere evaluación manual)"""
        # Por ahora, estimación basada en patrones
        return max(0, self.security_patterns["blocked"] - self.security_patterns["sanitized"])
    
    def _calculate_false_negatives(self) -> int:
        """Calcular falsos negativos (requiere evaluación manual)"""
        # Por ahora, estimación basada en patrones
        return max(0, self.security_patterns["sanitized"] - self.security_patterns["blocked"])
    
    def _calculate_security_score(self, total_queries: int, blocked_queries: int, 
                                sanitized_queries: int, false_positives: int, 
                                false_negatives: int) -> float:
        """Calcular puntuación de seguridad (0-100)"""
        if total_queries == 0:
            return 0.0
        
        # Puntuación base por efectividad
        effectiveness = (blocked_queries + sanitized_queries) / total_queries
        
        # Penalización por falsos positivos/negativos
        false_positive_penalty = (false_positives / total_queries) * 0.3
        false_negative_penalty = (false_negatives / total_queries) * 0.5
        
        # Puntuación final
        score = (effectiveness * 100) - (false_positive_penalty * 100) - (false_negative_penalty * 100)
        return max(0.0, min(100.0, score))
    
    def get_security_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Obtener tendencias de seguridad en las últimas N horas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e["timestamp"] > cutoff_time]
        
        # Agrupar por hora
        hourly_data = defaultdict(list)
        for event in recent_events:
            hour = event["timestamp"].hour
            hourly_data[hour].append(event)
        
        trends = {
            "blocked_queries": [],
            "sanitized_queries": [],
            "rate_limited_queries": [],
            "security_warnings": []
        }
        
        for hour in range(24):
            events_in_hour = hourly_data.get(hour, [])
            trends["blocked_queries"].append(sum(1 for e in events_in_hour if "blocked" in e["event_type"]))
            trends["sanitized_queries"].append(sum(1 for e in events_in_hour if "sanitized" in e["event_type"]))
            trends["rate_limited_queries"].append(sum(1 for e in events_in_hour if "rate_limited" in e["event_type"]))
            trends["security_warnings"].append(sum(1 for e in events_in_hour if "warning" in e["event_type"]))
        
        return trends
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Análisis de rendimiento de seguridad"""
        if not self.performance_metrics:
            return {}
        
        analysis = {}
        for event_type, times in self.performance_metrics.items():
            if times:
                analysis[event_type] = {
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times),
                    "count": len(times)
                }
        
        return analysis
    
    def get_security_patterns(self) -> Dict[str, Any]:
        """Obtener patrones de seguridad detectados"""
        return {
            "total_patterns": len(self.security_patterns),
            "pattern_distribution": dict(self.security_patterns),
            "most_common_pattern": max(self.security_patterns.items(), key=lambda x: x[1]) if self.security_patterns else None,
            "security_events_by_type": Counter([e["event_type"] for e in self.security_events])
        }
    
    def get_security_recommendations(self) -> List[str]:
        """Obtener recomendaciones de seguridad basadas en métricas"""
        recommendations = []
        
        if not self.security_events:
            return ["No hay datos suficientes para generar recomendaciones"]
        
        metrics = self.calculate_security_metrics()
        
        # Recomendaciones basadas en métricas
        if metrics.false_positives > metrics.total_queries * 0.1:
            recommendations.append("Alto número de falsos positivos. Considerar ajustar umbrales de seguridad.")
        
        if metrics.false_negatives > metrics.total_queries * 0.05:
            recommendations.append("Alto número de falsos negativos. Revisar patrones de detección.")
        
        if metrics.avg_processing_time > 2.0:
            recommendations.append("Tiempo de procesamiento alto. Optimizar algoritmos de seguridad.")
        
        if metrics.security_score < 70:
            recommendations.append("Puntuación de seguridad baja. Revisar configuración de guardrails.")
        
        if not recommendations:
            recommendations.append("Sistema de seguridad funcionando correctamente.")
        
        return recommendations

# Instancia global del calculador de métricas de seguridad
security_metrics_calculator = SecurityMetricsCalculator()
