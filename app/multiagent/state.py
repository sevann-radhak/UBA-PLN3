# app/multiagent/state.py
"""
Estado compartido para el sistema multiagente
Basado en conceptos de la Clase 3: LangGraph y orquestación
"""

from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from datetime import datetime

class AgentState(TypedDict):
    """Estado compartido entre agentes usando LangGraph"""
    # Contexto de la consulta
    user_query: str
    breed_name: str
    original_query: str
    
    # Resultados de agentes
    research_results: List[Dict[str, Any]]
    summary: str
    validation_result: Dict[str, Any]
    
    # Metadatos del sistema
    current_agent: str
    agent_history: List[Dict[str, Any]]
    tools_used: List[str]
    
    # Control de flujo
    next_agent: Optional[str]
    is_complete: bool
    error_message: Optional[str]
    
    # Memoria persistente
    session_id: str
    timestamp: datetime
    user_context: Dict[str, Any]

@dataclass
class AgentResult:
    """Resultado de un agente individual"""
    agent_name: str
    success: bool
    result: Any
    metadata: Dict[str, Any]
    execution_time: float
    tools_used: List[str]
    error: Optional[str] = None

@dataclass
class ToolResult:
    """Resultado de una herramienta externa"""
    tool_name: str
    success: bool
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None

class MemorySystem:
    """Sistema de memoria persistente para el multiagente"""
    
    def __init__(self):
        self.session_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_memory: Dict[str, List[AgentResult]] = {}
        self.tool_memory: Dict[str, List[ToolResult]] = {}
    
    def store_agent_result(self, session_id: str, result: AgentResult):
        """Almacena resultado de un agente"""
        if session_id not in self.agent_memory:
            self.agent_memory[session_id] = []
        self.agent_memory[session_id].append(result)
    
    def store_tool_result(self, session_id: str, result: ToolResult):
        """Almacena resultado de una herramienta"""
        if session_id not in self.tool_memory:
            self.tool_memory[session_id] = []
        self.tool_memory[session_id].append(result)
    
    def get_agent_history(self, session_id: str) -> List[AgentResult]:
        """Obtiene historial de agentes para una sesión"""
        return self.agent_memory.get(session_id, [])
    
    def get_tool_history(self, session_id: str) -> List[ToolResult]:
        """Obtiene historial de herramientas para una sesión"""
        return self.tool_memory.get(session_id, [])
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Obtiene contexto completo de la sesión"""
        return {
            "agent_history": self.get_agent_history(session_id),
            "tool_history": self.get_tool_history(session_id),
            "total_agents": len(self.get_agent_history(session_id)),
            "total_tools": len(self.get_tool_history(session_id))
        }

# Instancia global del sistema de memoria
memory_system = MemorySystem()


