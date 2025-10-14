# app/multiagent/__init__.py
"""
Sistema Multiagente para PLN3
Integra RAG híbrido + Guardrails + Multiagente
"""

from .state import AgentState, AgentResult, memory_system
from .agents import agent_factory, ResearchAgent, SummarizerAgent, ValidatorAgent
from .tools import tool_manager, WikipediaTool, ArxivTool, BreedDatabaseTool
from .supervisor import MultiAgentSupervisor, multiagent_supervisor

__all__ = [
    "AgentState",
    "AgentResult", 
    "memory_system",
    "agent_factory",
    "ResearchAgent",
    "SummarizerAgent", 
    "ValidatorAgent",
    "tool_manager",
    "WikipediaTool",
    "ArxivTool",
    "BreedDatabaseTool",
    "MultiAgentSupervisor",
    "multiagent_supervisor"
]


