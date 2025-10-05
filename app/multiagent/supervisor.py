# app/multiagent/supervisor.py
"""
Supervisor para el sistema multiagente
Basado en conceptos de la Clase 3: LangGraph y orquestación inteligente
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .state import AgentState, AgentResult, memory_system
from .agents import agent_factory

@dataclass
class SupervisorConfig:
    """Configuración del supervisor"""
    max_iterations: int = 10
    timeout: float = 120.0
    enable_memory: bool = True
    enable_logging: bool = True

class MultiAgentSupervisor:
    """Supervisor principal del sistema multiagente"""
    
    def __init__(self, config: SupervisorConfig = None):
        self.config = config or SupervisorConfig()
        self.agent_history: List[AgentResult] = []
        self.execution_log: List[Dict[str, Any]] = []
    
    def determine_next_agent(self, state: AgentState) -> str:
        """Determina el siguiente agente a ejecutar basado en el estado"""
        
        # Lógica de routing basada en el estado actual
        if not state.get("research_results"):
            return "research"
        elif not state.get("summary"):
            return "summarizer"
        elif not state.get("validation_result"):
            return "validator"
        else:
            return "complete"
    
    def execute_agent(self, agent_type: str, state: AgentState) -> AgentResult:
        """Ejecuta un agente específico"""
        try:
            # Crear agente
            agent = agent_factory.create_agent(agent_type)
            
            # Ejecutar agente
            result = agent.execute(state)
            
            # Log de ejecución
            if self.config.enable_logging:
                self.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent_type": agent_type,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "tools_used": result.tools_used
                })
            
            return result
            
        except Exception as e:
            return AgentResult(
                agent_name=agent_type,
                success=False,
                result=None,
                metadata={"error": str(e)},
                execution_time=0.0,
                tools_used=[],
                error=str(e)
            )
    
    def update_state(self, state: AgentState, result: AgentResult) -> AgentState:
        """Actualiza el estado basado en el resultado del agente"""
        
        # Actualizar estado según el tipo de agente
        if result.agent_name == "research_agent":
            state["research_results"] = [result.result] if result.success else []
            state["current_agent"] = "research"
        
        elif result.agent_name == "summarizer_agent":
            state["summary"] = result.result if result.success else ""
            state["current_agent"] = "summarizer"
        
        elif result.agent_name == "validator_agent":
            state["validation_result"] = result.result if result.success else {}
            state["current_agent"] = "validator"
        
        # Actualizar historial
        state["agent_history"].append({
            "agent": result.agent_name,
            "success": result.success,
            "timestamp": datetime.now().isoformat(),
            "execution_time": result.execution_time
        })
        
        # Actualizar herramientas usadas
        state["tools_used"].extend(result.tools_used)
        
        return state
    
    def should_continue(self, state: AgentState, iteration: int) -> bool:
        """Determina si el sistema debe continuar ejecutándose"""
        
        # Verificar límites
        if iteration >= self.config.max_iterations:
            return False
        
        # Verificar timeout
        if hasattr(state.get("timestamp"), 'timestamp'):
            # Si timestamp es datetime, convertir a float
            start_time = state["timestamp"].timestamp()
        else:
            # Si timestamp es float, usar directamente
            start_time = state.get("timestamp", time.time())
        
        if time.time() - start_time > self.config.timeout:
            return False
        
        # Verificar si hay errores críticos
        if state.get("error_message"):
            return False
        
        # Verificar si el proceso está completo
        if state.get("is_complete"):
            return False
        
        return True
    
    def execute_multiagent_workflow(self, initial_state: AgentState) -> Dict[str, Any]:
        """Ejecuta el flujo completo del sistema multiagente"""
        
        start_time = time.time()
        state = initial_state.copy()
        iteration = 0
        
        # Log inicial
        if self.config.enable_logging:
            self.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "event": "workflow_started",
                "breed_name": state["breed_name"],
                "user_query": state["user_query"]
            })
        
        try:
            while self.should_continue(state, iteration):
                iteration += 1
                
                # Determinar siguiente agente
                next_agent = self.determine_next_agent(state)
                
                if next_agent == "complete":
                    state["is_complete"] = True
                    break
                
                # Ejecutar agente
                result = self.execute_agent(next_agent, state)
                
                # Actualizar estado
                state = self.update_state(state, result)
                
                # Verificar si hay errores
                if not result.success:
                    state["error_message"] = result.error
                    break
                
                # Pequeña pausa entre agentes
                time.sleep(0.1)
            
            # Compilar resultado final
            final_result = {
                "success": state.get("is_complete", False),
                "breed_name": state["breed_name"],
                "user_query": state["user_query"],
                "research_results": state.get("research_results", []),
                "summary": state.get("summary", ""),
                "validation_result": state.get("validation_result", {}),
                "agent_history": state["agent_history"],
                "tools_used": list(set(state["tools_used"])),
                "execution_time": time.time() - start_time,
                "iterations": iteration,
                "error_message": state.get("error_message")
            }
            
            # Log final
            if self.config.enable_logging:
                self.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "event": "workflow_completed",
                    "success": final_result["success"],
                    "execution_time": final_result["execution_time"],
                    "iterations": iteration
                })
            
            return final_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "iterations": iteration
            }
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Obtiene el log de ejecución"""
        return self.execution_log
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los agentes"""
        if not self.execution_log:
            return {"total_executions": 0}
        
        agent_stats = {}
        for log_entry in self.execution_log:
            if "agent_type" in log_entry:
                agent_type = log_entry["agent_type"]
                if agent_type not in agent_stats:
                    agent_stats[agent_type] = {
                        "executions": 0,
                        "successful": 0,
                        "total_time": 0.0
                    }
                
                agent_stats[agent_type]["executions"] += 1
                if log_entry.get("success", False):
                    agent_stats[agent_type]["successful"] += 1
                agent_stats[agent_type]["total_time"] += log_entry.get("execution_time", 0.0)
        
        return {
            "total_executions": len(self.execution_log),
            "agent_statistics": agent_stats
        }

# Instancia global del supervisor
multiagent_supervisor = MultiAgentSupervisor()
