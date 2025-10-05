# app/multiagent/agents.py
"""
Agentes especializados para el sistema multiagente
Basado en conceptos de la Clase 3: agentes especializados con LangGraph
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .state import AgentState, AgentResult, memory_system
from .tools import tool_manager

@dataclass
class AgentConfig:
    """Configuración para un agente"""
    name: str
    description: str
    max_iterations: int = 3
    timeout: float = 30.0
    tools: List[str] = None
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []

class ResearchAgent:
    """Agente especializado en búsqueda de información"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig(
            name="research_agent",
            description="Agente especializado en búsqueda de información sobre razas de perros",
            tools=["wikipedia", "arxiv", "breed_database"]
        )
    
    def execute(self, state: AgentState) -> AgentResult:
        """Ejecuta el agente de investigación"""
        start_time = time.time()
        
        try:
            # Obtener información de la base de datos local
            breed_result = tool_manager.execute_tool(
                "breed_database",
                breed_name=state["breed_name"]
            )
            
            # Buscar información adicional en Wikipedia
            wikipedia_result = tool_manager.execute_tool(
                "wikipedia",
                query=f"{state['breed_name']} perro raza",
                max_results=3
            )
            
            # Buscar papers científicos en ArXiv
            arxiv_result = tool_manager.execute_tool(
                "arxiv",
                query=f"dog breed {state['breed_name']} genetics behavior",
                max_results=3
            )
            
            # Compilar resultados
            research_data = {
                "breed_database": breed_result.data if breed_result.success else None,
                "wikipedia": wikipedia_result.data if wikipedia_result.success else None,
                "arxiv": arxiv_result.data if arxiv_result.success else None,
                "tools_used": ["breed_database", "wikipedia", "arxiv"],
                "successful_tools": [
                    tool for tool, result in [
                        ("breed_database", breed_result),
                        ("wikipedia", wikipedia_result),
                        ("arxiv", arxiv_result)
                    ] if result.success
                ],
                "primary_source": "breed_database" if breed_result.success else "wikipedia" if wikipedia_result.success else "arxiv"
            }
            
            # Almacenar en memoria
            result = AgentResult(
                agent_name=self.config.name,
                success=True,
                result=research_data,
                metadata={
                    "breed_name": state["breed_name"],
                    "query": state["user_query"],
                    "tools_executed": 3,
                    "successful_tools": len(research_data["successful_tools"])
                },
                execution_time=time.time() - start_time,
                tools_used=research_data["tools_used"]
            )
            
            memory_system.store_agent_result(state["session_id"], result)
            return result
            
        except Exception as e:
            result = AgentResult(
                agent_name=self.config.name,
                success=False,
                result=None,
                metadata={"error": str(e)},
                execution_time=time.time() - start_time,
                tools_used=[],
                error=str(e)
            )
            memory_system.store_agent_result(state["session_id"], result)
            return result

class SummarizerAgent:
    """Agente especializado en resumir información"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig(
            name="summarizer_agent",
            description="Agente especializado en resumir información sobre razas de perros",
            tools=[]
        )
    
    def execute(self, state: AgentState) -> AgentResult:
        """Ejecuta el agente de resumen"""
        start_time = time.time()
        
        try:
            # Obtener resultados de investigación
            research_results = state.get("research_results", [])
            
            if not research_results:
                # Crear resumen básico con información disponible
                summary = {
                    "breed_name": state["breed_name"],
                    "summary_text": f"Información básica sobre {state['breed_name']}. No se encontraron fuentes externas adicionales.",
                    "sources_used": [],
                    "completeness_score": 0.1
                }
                
                return AgentResult(
                    agent_name=self.config.name,
                    success=True,
                    result=summary,
                    metadata={
                        "breed_name": state["breed_name"],
                        "sources_count": 0,
                        "completeness": 0.1,
                        "fallback": True
                    },
                    execution_time=time.time() - start_time,
                    tools_used=[]
                )
            
            # Procesar información de diferentes fuentes
            summary_parts = []
            
            # Priorizar información de la base de datos local
            if research_results[0].get("breed_database"):
                breed_info = research_results[0]["breed_database"]["breed_info"]
                summary_parts.append(f"**Información detallada de la raza:**\n{breed_info}")
            
            # Información adicional de Wikipedia
            if research_results[0].get("wikipedia"):
                wiki_data = research_results[0]["wikipedia"]
                if wiki_data.get('extract'):
                    summary_parts.append(f"**Información adicional de Wikipedia:**\n{wiki_data.get('extract', 'No disponible')}")
            
            # Información de ArXiv
            if research_results[0].get("arxiv"):
                arxiv_data = research_results[0]["arxiv"]
                if arxiv_data:
                    summary_parts.append(f"**Investigación científica:**\n{len(arxiv_data)} papers encontrados")
            
            # Si no hay información de la base de datos, usar fuentes alternativas
            if not research_results[0].get("breed_database"):
                if research_results[0].get("wikipedia"):
                    wiki_data = research_results[0]["wikipedia"]
                    summary_parts.append(f"**Información de Wikipedia:**\n{wiki_data.get('extract', 'No disponible')}")
                else:
                    summary_parts.append(f"**Información básica:**\nInformación general sobre {state['breed_name']} disponible a través de fuentes externas.")
            
            # Crear resumen estructurado
            summary = {
                "breed_name": state["breed_name"],
                "summary_text": "\n\n".join(summary_parts),
                "sources_used": research_results[0].get("successful_tools", []),
                "completeness_score": len(research_results[0].get("successful_tools", [])) / 3.0
            }
            
            result = AgentResult(
                agent_name=self.config.name,
                success=True,
                result=summary,
                metadata={
                    "breed_name": state["breed_name"],
                    "sources_count": len(summary["sources_used"]),
                    "completeness": summary["completeness_score"]
                },
                execution_time=time.time() - start_time,
                tools_used=[]
            )
            
            memory_system.store_agent_result(state["session_id"], result)
            return result
            
        except Exception as e:
            result = AgentResult(
                agent_name=self.config.name,
                success=False,
                result=None,
                metadata={"error": str(e)},
                execution_time=time.time() - start_time,
                tools_used=[],
                error=str(e)
            )
            memory_system.store_agent_result(state["session_id"], result)
            return result

class ValidatorAgent:
    """Agente especializado en validar información"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig(
            name="validator_agent",
            description="Agente especializado en validar información sobre razas de perros",
            tools=[]
        )
    
    def execute(self, state: AgentState) -> AgentResult:
        """Ejecuta el agente de validación"""
        start_time = time.time()
        
        try:
            # Obtener resumen para validar
            summary = state.get("summary", "")
            
            if not summary:
                # Crear validación básica
                validation_results = {
                    "has_breed_info": bool(state.get("breed_name")),
                    "has_summary_text": False,
                    "has_sources": False,
                    "completeness_score": 0.1,
                    "is_complete": False,
                    "is_sufficient": bool(state.get("breed_name")),
                    "validation_timestamp": datetime.now().isoformat(),
                    "fallback": True
                }
                
                return AgentResult(
                    agent_name=self.config.name,
                    success=True,
                    result=validation_results,
                    metadata={
                        "breed_name": state["breed_name"],
                        "validation_passed": validation_results["is_sufficient"],
                        "fallback": True
                    },
                    execution_time=time.time() - start_time,
                    tools_used=[]
                )
            
            # Validaciones
            validation_results = {
                "has_breed_info": bool(summary.get("breed_name")),
                "has_summary_text": bool(summary.get("summary_text")),
                "has_sources": bool(summary.get("sources_used")),
                "completeness_score": summary.get("completeness_score", 0.0),
                "is_complete": summary.get("completeness_score", 0.0) >= 0.5,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # Determinar si la información es suficiente
            validation_results["is_sufficient"] = (
                validation_results["has_breed_info"] and
                validation_results["has_summary_text"] and
                validation_results["completeness_score"] >= 0.3
            )
            
            result = AgentResult(
                agent_name=self.config.name,
                success=True,
                result=validation_results,
                metadata={
                    "breed_name": state["breed_name"],
                    "validation_passed": validation_results["is_sufficient"]
                },
                execution_time=time.time() - start_time,
                tools_used=[]
            )
            
            memory_system.store_agent_result(state["session_id"], result)
            return result
            
        except Exception as e:
            result = AgentResult(
                agent_name=self.config.name,
                success=False,
                result=None,
                metadata={"error": str(e)},
                execution_time=time.time() - start_time,
                tools_used=[],
                error=str(e)
            )
            memory_system.store_agent_result(state["session_id"], result)
            return result

class AgentFactory:
    """Factory para crear agentes especializados"""
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig = None) -> Any:
        """Crea un agente del tipo especificado"""
        if agent_type == "research":
            return ResearchAgent(config)
        elif agent_type == "summarizer":
            return SummarizerAgent(config)
        elif agent_type == "validator":
            return ValidatorAgent(config)
        else:
            raise ValueError(f"Tipo de agente desconocido: {agent_type}")
    
    @staticmethod
    def get_available_agents() -> List[str]:
        """Obtiene lista de agentes disponibles"""
        return ["research", "summarizer", "validator"]
    
    @staticmethod
    def get_agent_description(agent_type: str) -> str:
        """Obtiene descripción de un agente"""
        descriptions = {
            "research": "Agente especializado en búsqueda de información sobre razas de perros",
            "summarizer": "Agente especializado en resumir información de múltiples fuentes",
            "validator": "Agente especializado en validar la calidad y completitud de la información"
        }
        return descriptions.get(agent_type, "Agente desconocido")

# Instancia global del factory
agent_factory = AgentFactory()
