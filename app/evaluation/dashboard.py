# app/evaluation/dashboard.py
"""
Dashboard de Evaluación Integral
Integra métricas de todas las clases del curso
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .ir_metrics import ir_metrics_calculator
from .security_metrics import security_metrics_calculator
from .multiagent_metrics import multiagent_metrics_calculator

class EvaluationDashboard:
    """Dashboard de evaluación integral"""
    
    def __init__(self):
        self.ir_calculator = ir_metrics_calculator
        self.security_calculator = security_metrics_calculator
        self.multiagent_calculator = multiagent_metrics_calculator
    
    def render_dashboard(self):
        """Renderizar dashboard completo"""
        st.set_page_config(page_title="Dashboard de Evaluación PLN3", layout="wide")
        
        st.title("📊 Dashboard de Evaluación Integral - PLN3")
        st.markdown("**Sistema de Evaluación Completo: RAG + Guardrails + Multiagente**")
        
        # Sidebar para configuración
        self._render_sidebar()
        
        # Métricas principales
        self._render_main_metrics()
        
        # Tabs para diferentes tipos de métricas
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 IR Metrics", "🛡️ Security", "🤖 Multiagent", "📈 Trends"])
        
        with tab1:
            self._render_ir_metrics()
        
        with tab2:
            self._render_security_metrics()
        
        with tab3:
            self._render_multiagent_metrics()
        
        with tab4:
            self._render_trends_analysis()
    
    def _render_sidebar(self):
        """Renderizar sidebar de configuración"""
        with st.sidebar:
            st.header("⚙️ Configuración")
            
            # Filtros de tiempo
            time_range = st.selectbox(
                "Rango de tiempo",
                ["Últimas 24 horas", "Última semana", "Último mes", "Todo el tiempo"]
            )
            
            # Métricas a mostrar
            st.subheader("Métricas a mostrar")
            show_ir = st.checkbox("IR Metrics", value=True)
            show_security = st.checkbox("Security Metrics", value=True)
            show_multiagent = st.checkbox("Multiagent Metrics", value=True)
            
            # Configuración de visualización
            st.subheader("Visualización")
            chart_theme = st.selectbox("Tema de gráficos", ["plotly", "plotly_white", "plotly_dark"])
            
            # Botón de actualización
            if st.button("🔄 Actualizar Dashboard"):
                st.rerun()
    
    def _render_main_metrics(self):
        """Renderizar métricas principales"""
        st.header("📊 Métricas Principales")
        
        # Obtener métricas de cada sistema
        ir_metrics = self.ir_calculator.calculate_global_metrics()
        security_metrics = self.security_calculator.calculate_security_metrics()
        multiagent_metrics = self.multiagent_calculator.calculate_multiagent_metrics()
        
        # Crear columnas para métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔍 IR Score",
                f"{ir_metrics.get('avg_map', 0):.3f}",
                delta=f"MRR: {ir_metrics.get('avg_mrr', 0):.3f}"
            )
        
        with col2:
            st.metric(
                "🛡️ Security Score",
                f"{security_metrics.security_score:.1f}",
                delta=f"Blocked: {security_metrics.blocked_queries}"
            )
        
        with col3:
            st.metric(
                "🤖 Multiagent Score",
                f"{multiagent_metrics.efficiency_score:.1f}",
                delta=f"Success: {multiagent_metrics.successful_sessions}/{multiagent_metrics.total_sessions}"
            )
        
        with col4:
            st.metric(
                "⚡ Overall Performance",
                f"{self._calculate_overall_score(ir_metrics, security_metrics, multiagent_metrics):.1f}",
                delta="Sistema Integrado"
            )
    
    def _render_ir_metrics(self):
        """Renderizar métricas de IR"""
        st.header("🔍 Métricas de Information Retrieval")
        
        ir_metrics = self.ir_calculator.calculate_global_metrics()
        
        if not ir_metrics:
            st.warning("No hay datos de IR disponibles")
            return
        
        # Gráfico de Precision@k
        k_values = [1, 3, 5, 10, 20]
        precision_values = [ir_metrics.get(f"avg_precision@{k}", 0) for k in k_values]
        recall_values = [ir_metrics.get(f"avg_recall@{k}", 0) for k in k_values]
        ndcg_values = [ir_metrics.get(f"avg_ndcg@{k}", 0) for k in k_values]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Precision@k", "Recall@k", "nDCG@k", "Métricas Globales"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Precision@k
        fig.add_trace(
            go.Scatter(x=k_values, y=precision_values, mode='lines+markers', name='Precision@k'),
            row=1, col=1
        )
        
        # Recall@k
        fig.add_trace(
            go.Scatter(x=k_values, y=recall_values, mode='lines+markers', name='Recall@k'),
            row=1, col=2
        )
        
        # nDCG@k
        fig.add_trace(
            go.Scatter(x=k_values, y=ndcg_values, mode='lines+markers', name='nDCG@k'),
            row=2, col=1
        )
        
        # Métricas globales
        global_metrics = [
            ir_metrics.get('avg_mrr', 0),
            ir_metrics.get('avg_map', 0)
        ]
        metric_names = ['MRR', 'MAP']
        
        fig.add_trace(
            go.Bar(x=metric_names, y=global_metrics, name='Global Metrics'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Métricas de Information Retrieval")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de métricas detalladas
        st.subheader("📋 Métricas Detalladas")
        metrics_df = pd.DataFrame([
            {"Métrica": f"Precision@{k}", "Valor": f"{ir_metrics.get(f'avg_precision@{k}', 0):.3f}"}
            for k in k_values
        ] + [
            {"Métrica": f"Recall@{k}", "Valor": f"{ir_metrics.get(f'avg_recall@{k}', 0):.3f}"}
            for k in k_values
        ] + [
            {"Métrica": f"nDCG@{k}", "Valor": f"{ir_metrics.get(f'avg_ndcg@{k}', 0):.3f}"}
            for k in k_values
        ] + [
            {"Métrica": "MRR", "Valor": f"{ir_metrics.get('avg_mrr', 0):.3f}"},
            {"Métrica": "MAP", "Valor": f"{ir_metrics.get('avg_map', 0):.3f}"}
        ])
        
        st.dataframe(metrics_df, use_container_width=True)
    
    def _render_security_metrics(self):
        """Renderizar métricas de seguridad"""
        st.header("🛡️ Métricas de Seguridad")
        
        security_metrics = self.security_calculator.calculate_security_metrics()
        
        # Métricas principales de seguridad
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", security_metrics.total_queries)
            st.metric("Blocked Queries", security_metrics.blocked_queries)
        
        with col2:
            st.metric("Sanitized Queries", security_metrics.sanitized_queries)
            st.metric("Rate Limited", security_metrics.rate_limited_queries)
        
        with col3:
            st.metric("Security Warnings", security_metrics.security_warnings)
            st.metric("Avg Processing Time", f"{security_metrics.avg_processing_time:.2f}s")
        
        # Gráfico de tendencias de seguridad
        trends = self.security_calculator.get_security_trends(24)
        
        if trends:
            fig = go.Figure()
            
            for event_type, values in trends.items():
                fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=values,
                    mode='lines+markers',
                    name=event_type.replace('_', ' ').title()
                ))
            
            fig.update_layout(
                title="Tendencias de Seguridad (24 horas)",
                xaxis_title="Hora del día",
                yaxis_title="Número de eventos",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de patrones
        patterns = self.security_calculator.get_security_patterns()
        
        if patterns:
            st.subheader("🔍 Análisis de Patrones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribución de Patrones:**")
                pattern_df = pd.DataFrame(list(patterns["pattern_distribution"].items()))
                pattern_df.columns = ["Patrón", "Frecuencia"]
                st.dataframe(pattern_df)
            
            with col2:
                st.write("**Eventos por Tipo:**")
                event_df = pd.DataFrame(list(patterns["security_events_by_type"].items()))
                event_df.columns = ["Tipo de Evento", "Cantidad"]
                st.dataframe(event_df)
        
        # Recomendaciones de seguridad
        recommendations = self.security_calculator.get_security_recommendations()
        
        if recommendations:
            st.subheader("💡 Recomendaciones de Seguridad")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    def _render_multiagent_metrics(self):
        """Renderizar métricas de multiagente"""
        st.header("🤖 Métricas de Sistema Multiagente")
        
        multiagent_metrics = self.multiagent_calculator.calculate_multiagent_metrics()
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Sessions", multiagent_metrics.total_sessions)
            st.metric("Successful Sessions", multiagent_metrics.successful_sessions)
        
        with col2:
            st.metric("Avg Agents/Session", f"{multiagent_metrics.avg_agents_per_session:.1f}")
            st.metric("Avg Execution Time", f"{multiagent_metrics.avg_execution_time:.2f}s")
        
        with col3:
            st.metric("Coordination Score", f"{multiagent_metrics.coordination_score:.1f}")
            st.metric("Efficiency Score", f"{multiagent_metrics.efficiency_score:.1f}")
        
        # Gráfico de utilización de agentes
        if multiagent_metrics.agent_utilization:
            st.subheader("📊 Utilización de Agentes")
            
            agent_df = pd.DataFrame(list(multiagent_metrics.agent_utilization.items()))
            agent_df.columns = ["Agente", "Utilización"]
            
            fig = px.bar(agent_df, x="Agente", y="Utilización", title="Utilización de Agentes")
            st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de utilización de herramientas
        if multiagent_metrics.tool_utilization:
            st.subheader("🔧 Utilización de Herramientas")
            
            tool_df = pd.DataFrame(list(multiagent_metrics.tool_utilization.items()))
            tool_df.columns = ["Herramienta", "Utilización"]
            
            fig = px.bar(tool_df, x="Herramienta", y="Utilización", title="Utilización de Herramientas")
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de coordinación
        coordination_analysis = self.multiagent_calculator.get_coordination_analysis()
        
        if coordination_analysis:
            st.subheader("🔄 Análisis de Coordinación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Estadísticas de Coordinación:**")
                coord_df = pd.DataFrame([
                    {"Métrica": "Total Events", "Valor": coordination_analysis["total_coordination_events"]},
                    {"Métrica": "Successful", "Valor": coordination_analysis["successful_coordinations"]},
                    {"Métrica": "Success Rate", "Valor": f"{coordination_analysis['coordination_success_rate']:.3f}"}
                ])
                st.dataframe(coord_df)
            
            with col2:
                st.write("**Agentes Más Activos:**")
                if coordination_analysis["most_active_agent"]:
                    st.write(f"**{coordination_analysis['most_active_agent'][0]}**: {coordination_analysis['most_active_agent'][1]} eventos")
        
        # Recomendaciones de optimización
        recommendations = self.multiagent_calculator.get_optimization_recommendations()
        
        if recommendations:
            st.subheader("💡 Recomendaciones de Optimización")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    def _render_trends_analysis(self):
        """Renderizar análisis de tendencias"""
        st.header("📈 Análisis de Tendencias")
        
        # Tendencias de rendimiento multiagente
        performance_trends = self.multiagent_calculator.get_performance_trends(24)
        
        if performance_trends:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Success Rate", "Execution Time", "Agents per Session", "Overall Performance"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Success Rate
            fig.add_trace(
                go.Scatter(x=list(range(24)), y=performance_trends["success_rate"], 
                          mode='lines+markers', name='Success Rate'),
                row=1, col=1
            )
            
            # Execution Time
            fig.add_trace(
                go.Scatter(x=list(range(24)), y=performance_trends["avg_execution_time"], 
                          mode='lines+markers', name='Execution Time'),
                row=1, col=2
            )
            
            # Agents per Session
            fig.add_trace(
                go.Scatter(x=list(range(24)), y=performance_trends["agents_per_session"], 
                          mode='lines+markers', name='Agents per Session'),
                row=2, col=1
            )
            
            # Overall Performance (combinado)
            overall_performance = [
                (s + e + a) / 3 for s, e, a in zip(
                    performance_trends["success_rate"],
                    [1.0 / (1.0 + t) for t in performance_trends["avg_execution_time"]],
                    [a / 5.0 for a in performance_trends["agents_per_session"]]  # Normalizar
                )
            ]
            
            fig.add_trace(
                go.Scatter(x=list(range(24)), y=overall_performance, 
                          mode='lines+markers', name='Overall Performance'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="Tendencias de Rendimiento (24 horas)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Resumen de rendimiento
        st.subheader("📊 Resumen de Rendimiento")
        
        # Obtener métricas de todos los sistemas
        ir_metrics = self.ir_calculator.calculate_global_metrics()
        security_metrics = self.security_calculator.calculate_security_metrics()
        multiagent_metrics = self.multiagent_calculator.calculate_multiagent_metrics()
        
        # Crear resumen
        summary_data = {
            "Sistema": ["IR", "Security", "Multiagent", "Overall"],
            "Score": [
                ir_metrics.get('avg_map', 0) * 100,
                security_metrics.security_score,
                multiagent_metrics.efficiency_score,
                self._calculate_overall_score(ir_metrics, security_metrics, multiagent_metrics)
            ],
            "Status": [
                "✅ Excelente" if ir_metrics.get('avg_map', 0) > 0.8 else "⚠️ Mejorable",
                "✅ Excelente" if security_metrics.security_score > 80 else "⚠️ Mejorable",
                "✅ Excelente" if multiagent_metrics.efficiency_score > 80 else "⚠️ Mejorable",
                "✅ Excelente" if self._calculate_overall_score(ir_metrics, security_metrics, multiagent_metrics) > 80 else "⚠️ Mejorable"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    def _calculate_overall_score(self, ir_metrics: Dict, security_metrics: Any, multiagent_metrics: Any) -> float:
        """Calcular puntuación general del sistema"""
        ir_score = ir_metrics.get('avg_map', 0) * 100
        security_score = security_metrics.security_score
        multiagent_score = multiagent_metrics.efficiency_score
        
        # Ponderación: IR 40%, Security 30%, Multiagent 30%
        overall_score = (ir_score * 0.4 + security_score * 0.3 + multiagent_score * 0.3)
        
        return overall_score

# Instancia global del dashboard
evaluation_dashboard = EvaluationDashboard()
