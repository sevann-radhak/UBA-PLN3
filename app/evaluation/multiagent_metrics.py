# app/evaluation/multiagent_metrics.py
"""
Métricas de Sistemas Multiagente
Basado en conceptos de la Clase 3: evaluación de orquestación y coordinación
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta

@dataclass
class MultiAgentMetrics:
    """Métricas del sistema multiagente"""
    total_sessions: int
    successful_sessions: int
    failed_sessions: int
    avg_agents_per_session: float
    avg_execution_time: float
    agent_utilization: Dict[str, float]
    tool_utilization: Dict[str, float]
    coordination_score: float
    efficiency_score: float

class MultiAgentMetricsCalculator:
    """Calculadora de métricas de sistemas multiagente"""
    
    def __init__(self):
        self.sessions: List[Dict[str, Any]] = []
        self.agent_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.tool_performance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.coordination_events: List[Dict[str, Any]] = []
    
    def add_session(self, session_id: str, agents_used: List[str], tools_used: List[str],
                   execution_time: float, success: bool, result: Dict[str, Any]):
        """Agregar sesión del sistema multiagente"""
        session = {
            "session_id": session_id,
            "timestamp": datetime.now(),
            "agents_used": agents_used,
            "tools_used": tools_used,
            "execution_time": execution_time,
            "success": success,
            "result": result
        }
        self.sessions.append(session)
        
        # Actualizar métricas de agentes
        for agent in agents_used:
            self.agent_performance[agent].append({
                "session_id": session_id,
                "execution_time": execution_time,
                "success": success
            })
        
        # Actualizar métricas de herramientas
        for tool in tools_used:
            self.tool_performance[tool].append({
                "session_id": session_id,
                "execution_time": execution_time,
                "success": success
            })
    
    def add_coordination_event(self, event_type: str, from_agent: str, to_agent: str,
                             data_transferred: Dict[str, Any], success: bool):
        """Agregar evento de coordinación entre agentes"""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "data_transferred": data_transferred,
            "success": success
        }
        self.coordination_events.append(event)
    
    def calculate_multiagent_metrics(self) -> MultiAgentMetrics:
        """Calcular métricas del sistema multiagente"""
        if not self.sessions:
            return MultiAgentMetrics(0, 0, 0, 0.0, 0.0, {}, {}, 0.0, 0.0)
        
        total_sessions = len(self.sessions)
        successful_sessions = sum(1 for s in self.sessions if s["success"])
        failed_sessions = total_sessions - successful_sessions
        
        # Promedio de agentes por sesión
        avg_agents_per_session = np.mean([len(s["agents_used"]) for s in self.sessions])
        
        # Tiempo promedio de ejecución
        avg_execution_time = np.mean([s["execution_time"] for s in self.sessions])
        
        # Utilización de agentes
        agent_utilization = self._calculate_agent_utilization()
        
        # Utilización de herramientas
        tool_utilization = self._calculate_tool_utilization()
        
        # Puntuación de coordinación
        coordination_score = self._calculate_coordination_score()
        
        # Puntuación de eficiencia
        efficiency_score = self._calculate_efficiency_score()
        
        return MultiAgentMetrics(
            total_sessions=total_sessions,
            successful_sessions=successful_sessions,
            failed_sessions=failed_sessions,
            avg_agents_per_session=avg_agents_per_session,
            avg_execution_time=avg_execution_time,
            agent_utilization=agent_utilization,
            tool_utilization=tool_utilization,
            coordination_score=coordination_score,
            efficiency_score=efficiency_score
        )
    
    def _calculate_agent_utilization(self) -> Dict[str, float]:
        """Calcular utilización de agentes"""
        if not self.agent_performance:
            return {}
        
        utilization = {}
        for agent, performances in self.agent_performance.items():
            if performances:
                success_rate = sum(1 for p in performances if p["success"]) / len(performances)
                avg_time = np.mean([p["execution_time"] for p in performances])
                utilization[agent] = success_rate * (1.0 / (1.0 + avg_time))  # Penalizar tiempo alto
        
        return utilization
    
    def _calculate_tool_utilization(self) -> Dict[str, float]:
        """Calcular utilización de herramientas"""
        if not self.tool_performance:
            return {}
        
        utilization = {}
        for tool, performances in self.tool_performance.items():
            if performances:
                success_rate = sum(1 for p in performances if p["success"]) / len(performances)
                avg_time = np.mean([p["execution_time"] for p in performances])
                utilization[tool] = success_rate * (1.0 / (1.0 + avg_time))
        
        return utilization
    
    def _calculate_coordination_score(self) -> float:
        """Calcular puntuación de coordinación"""
        if not self.coordination_events:
            return 0.0
        
        successful_coordinations = sum(1 for e in self.coordination_events if e["success"])
        total_coordinations = len(self.coordination_events)
        
        return (successful_coordinations / total_coordinations) * 100 if total_coordinations > 0 else 0.0
    
    def _calculate_efficiency_score(self) -> float:
        """Calcular puntuación de eficiencia"""
        if not self.sessions:
            return 0.0
        
        # Factores de eficiencia
        success_rate = sum(1 for s in self.sessions if s["success"]) / len(self.sessions)
        avg_time = np.mean([s["execution_time"] for s in self.sessions])
        time_efficiency = 1.0 / (1.0 + avg_time)  # Penalizar tiempo alto
        
        # Eficiencia de coordinación
        coordination_efficiency = self._calculate_coordination_score() / 100
        
        # Puntuación combinada
        efficiency_score = (success_rate * 0.4 + time_efficiency * 0.3 + coordination_efficiency * 0.3) * 100
        
        return efficiency_score
    
    def get_agent_analysis(self, agent_name: str) -> Dict[str, Any]:
        """Análisis detallado de un agente"""
        if agent_name not in self.agent_performance:
            return {}
        
        performances = self.agent_performance[agent_name]
        if not performances:
            return {}
        
        return {
            "total_executions": len(performances),
            "success_rate": sum(1 for p in performances if p["success"]) / len(performances),
            "avg_execution_time": np.mean([p["execution_time"] for p in performances]),
            "min_execution_time": np.min([p["execution_time"] for p in performances]),
            "max_execution_time": np.max([p["execution_time"] for p in performances]),
            "std_execution_time": np.std([p["execution_time"] for p in performances])
        }
    
    def get_tool_analysis(self, tool_name: str) -> Dict[str, Any]:
        """Análisis detallado de una herramienta"""
        if tool_name not in self.tool_performance:
            return {}
        
        performances = self.tool_performance[tool_name]
        if not performances:
            return {}
        
        return {
            "total_uses": len(performances),
            "success_rate": sum(1 for p in performances if p["success"]) / len(performances),
            "avg_execution_time": np.mean([p["execution_time"] for p in performances]),
            "min_execution_time": np.min([p["execution_time"] for p in performances]),
            "max_execution_time": np.max([p["execution_time"] for p in performances]),
            "std_execution_time": np.std([p["execution_time"] for p in performances])
        }
    
    def get_coordination_analysis(self) -> Dict[str, Any]:
        """Análisis de coordinación entre agentes"""
        if not self.coordination_events:
            return {}
        
        # Análisis por tipo de evento
        event_types = Counter([e["event_type"] for e in self.coordination_events])
        
        # Análisis por agente
        agent_coordinations = defaultdict(int)
        for event in self.coordination_events:
            agent_coordinations[event["from_agent"]] += 1
            agent_coordinations[event["to_agent"]] += 1
        
        # Análisis de éxito
        successful_events = sum(1 for e in self.coordination_events if e["success"])
        total_events = len(self.coordination_events)
        
        return {
            "total_coordination_events": total_events,
            "successful_coordinations": successful_events,
            "coordination_success_rate": successful_events / total_events if total_events > 0 else 0,
            "event_types": dict(event_types),
            "agent_coordination_counts": dict(agent_coordinations),
            "most_active_agent": max(agent_coordinations.items(), key=lambda x: x[1]) if agent_coordinations else None
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Obtener tendencias de rendimiento"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_sessions = [s for s in self.sessions if s["timestamp"] > cutoff_time]
        
        # Agrupar por hora
        hourly_data = defaultdict(list)
        for session in recent_sessions:
            hour = session["timestamp"].hour
            hourly_data[hour].append(session)
        
        trends = {
            "success_rate": [],
            "avg_execution_time": [],
            "agents_per_session": []
        }
        
        for hour in range(24):
            sessions_in_hour = hourly_data.get(hour, [])
            if sessions_in_hour:
                success_rate = sum(1 for s in sessions_in_hour if s["success"]) / len(sessions_in_hour)
                avg_time = np.mean([s["execution_time"] for s in sessions_in_hour])
                avg_agents = np.mean([len(s["agents_used"]) for s in sessions_in_hour])
            else:
                success_rate = 0.0
                avg_time = 0.0
                avg_agents = 0.0
            
            trends["success_rate"].append(success_rate)
            trends["avg_execution_time"].append(avg_time)
            trends["agents_per_session"].append(avg_agents)
        
        return trends
    
    def get_optimization_recommendations(self) -> List[str]:
        """Obtener recomendaciones de optimización"""
        recommendations = []
        
        if not self.sessions:
            return ["No hay datos suficientes para generar recomendaciones"]
        
        metrics = self.calculate_multiagent_metrics()
        
        # Recomendaciones basadas en métricas
        if metrics.successful_sessions / metrics.total_sessions < 0.8:
            recommendations.append("Tasa de éxito baja. Revisar configuración de agentes.")
        
        if metrics.avg_execution_time > 10.0:
            recommendations.append("Tiempo de ejecución alto. Optimizar coordinación entre agentes.")
        
        if metrics.coordination_score < 70:
            recommendations.append("Puntuación de coordinación baja. Mejorar comunicación entre agentes.")
        
        if metrics.efficiency_score < 70:
            recommendations.append("Puntuación de eficiencia baja. Revisar utilización de recursos.")
        
        # Recomendaciones específicas por agente
        for agent, utilization in metrics.agent_utilization.items():
            if utilization < 0.5:
                recommendations.append(f"Agente {agent} con baja utilización. Considerar optimización.")
        
        # Recomendaciones específicas por herramienta
        for tool, utilization in metrics.tool_utilization.items():
            if utilization < 0.5:
                recommendations.append(f"Herramienta {tool} con baja utilización. Considerar reemplazo.")
        
        if not recommendations:
            recommendations.append("Sistema multiagente funcionando correctamente.")
        
        return recommendations

# Instancia global del calculador de métricas multiagente
multiagent_metrics_calculator = MultiAgentMetricsCalculator()
