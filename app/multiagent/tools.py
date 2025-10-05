# app/multiagent/tools.py
"""
Herramientas externas para el sistema multiagente
Basado en conceptos de la Clase 3: integración de herramientas
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

@dataclass
class ToolResult:
    """Resultado de una herramienta externa"""
    tool_name: str
    success: bool
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None

class WikipediaTool:
    """Herramienta para buscar información en Wikipedia"""
    
    def __init__(self):
        self.base_url = "https://es.wikipedia.org/api/rest_v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PLN3-MultiAgent/1.0 (Educational Project)'
        })
    
    def search_summary(self, query: str, max_results: int = 3) -> ToolResult:
        """Busca resúmenes en Wikipedia"""
        start_time = time.time()
        
        try:
            # Primero intentar búsqueda directa
            search_url = f"{self.base_url}/page/summary/{query}"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    "title": data.get("title", ""),
                    "extract": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "thumbnail": data.get("thumbnail", {}).get("source", "")
                }
                
                return ToolResult(
                    tool_name="wikipedia_summary",
                    success=True,
                    data=result,
                    metadata={"query": query, "status_code": response.status_code},
                    execution_time=time.time() - start_time
                )
            elif response.status_code == 404:
                # Si no encuentra la página exacta, intentar búsqueda general
                search_url = f"{self.base_url}/page/summary/{query.replace(' ', '_')}"
                response = self.session.get(search_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "title": data.get("title", ""),
                        "extract": data.get("extract", ""),
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "thumbnail": data.get("thumbnail", {}).get("source", "")
                    }
                    
                    return ToolResult(
                        tool_name="wikipedia_summary",
                        success=True,
                        data=result,
                        metadata={"query": query, "status_code": response.status_code, "fallback": True},
                        execution_time=time.time() - start_time
                    )
                else:
                    # Si tampoco funciona, devolver información genérica
                    return ToolResult(
                        tool_name="wikipedia_summary",
                        success=True,
                        data={
                            "title": f"Información sobre {query}",
                            "extract": f"Información general sobre {query}. Para más detalles, consulta Wikipedia directamente.",
                            "url": f"https://es.wikipedia.org/wiki/{query.replace(' ', '_')}",
                            "thumbnail": ""
                        },
                        metadata={"query": query, "status_code": response.status_code, "generic": True},
                        execution_time=time.time() - start_time
                    )
            else:
                return ToolResult(
                    tool_name="wikipedia_summary",
                    success=False,
                    data=None,
                    metadata={"query": query, "status_code": response.status_code},
                    execution_time=time.time() - start_time,
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name="wikipedia_summary",
                success=False,
                data=None,
                metadata={"query": query},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def search_related_pages(self, query: str, max_results: int = 5) -> ToolResult:
        """Busca páginas relacionadas en Wikipedia"""
        start_time = time.time()
        
        try:
            # Buscar páginas relacionadas
            search_url = f"{self.base_url}/page/related/{query}"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("pages", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "extract": item.get("extract", ""),
                        "url": item.get("content_urls", {}).get("desktop", {}).get("page", "")
                    })
                
                return ToolResult(
                    tool_name="wikipedia_related",
                    success=True,
                    data=results,
                    metadata={"query": query, "results_count": len(results)},
                    execution_time=time.time() - start_time
                )
            else:
                return ToolResult(
                    tool_name="wikipedia_related",
                    success=False,
                    data=None,
                    metadata={"query": query, "status_code": response.status_code},
                    execution_time=time.time() - start_time,
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name="wikipedia_related",
                success=False,
                data=None,
                metadata={"query": query},
                execution_time=time.time() - start_time,
                error=str(e)
            )

class ArxivTool:
    """Herramienta para buscar papers en ArXiv"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PLN3-MultiAgent/1.0 (Educational Project)'
        })
    
    def search_papers(self, query: str, max_results: int = 5) -> ToolResult:
        """Busca papers en ArXiv relacionados con la consulta"""
        start_time = time.time()
        
        try:
            # Construir query para ArXiv
            search_query = f"all:{query}"
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                # Parsear XML
                root = ET.fromstring(response.content)
                
                # Namespace para ArXiv
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                papers = []
                for entry in root.findall('atom:entry', ns):
                    paper = {
                        "title": entry.find('atom:title', ns).text if entry.find('atom:title', ns) is not None else "",
                        "summary": entry.find('atom:summary', ns).text if entry.find('atom:summary', ns) is not None else "",
                        "authors": [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                        "published": entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else "",
                        "link": entry.find('atom:link[@type="text/html"]', ns).get('href') if entry.find('atom:link[@type="text/html"]', ns) is not None else "",
                        "pdf_link": entry.find('atom:link[@type="application/pdf"]', ns).get('href') if entry.find('atom:link[@type="application/pdf"]', ns) is not None else ""
                    }
                    papers.append(paper)
                
                return ToolResult(
                    tool_name="arxiv_search",
                    success=True,
                    data=papers,
                    metadata={"query": query, "results_count": len(papers)},
                    execution_time=time.time() - start_time
                )
            else:
                return ToolResult(
                    tool_name="arxiv_search",
                    success=False,
                    data=None,
                    metadata={"query": query, "status_code": response.status_code},
                    execution_time=time.time() - start_time,
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name="arxiv_search",
                success=False,
                data=None,
                metadata={"query": query},
                execution_time=time.time() - start_time,
                error=str(e)
            )

class BreedDatabaseTool:
    """Herramienta para acceder a la base de datos de razas"""
    
    def __init__(self):
        self.knowledge_base = None
    
    def get_breed_info(self, breed_name: str) -> ToolResult:
        """Obtiene información de la base de datos de razas"""
        start_time = time.time()
        
        try:
            # Importar la base de conocimiento
            from ..breed_knowledge_base import DogBreedKnowledgeBase
            
            if self.knowledge_base is None:
                self.knowledge_base = DogBreedKnowledgeBase()
            
            # Obtener información de la raza
            breed_info = self.knowledge_base.get_breed_info_text(breed_name)
            
            # Verificar si se encontró información
            if "Información no disponible" in breed_info:
                # Intentar búsqueda más flexible
                breed_info = self._flexible_search(breed_name)
            
            return ToolResult(
                tool_name="breed_database",
                success=True,
                data={"breed_info": breed_info, "breed_name": breed_name},
                metadata={"breed_name": breed_name},
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="breed_database",
                success=False,
                data=None,
                metadata={"breed_name": breed_name},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _flexible_search(self, breed_name: str) -> str:
        """Búsqueda más flexible en la base de datos"""
        try:
            # Intentar diferentes variaciones del nombre
            variations = [
                breed_name,
                breed_name.replace(" ", "_"),
                breed_name.replace("_", " "),
                breed_name.lower(),
                breed_name.upper(),
                breed_name.title()
            ]
            
            for variation in variations:
                result = self.knowledge_base.get_breed_info_text(variation)
                if "Información no disponible" not in result:
                    return result
            
            # Si no se encuentra, devolver información genérica
            return f"Información sobre {breed_name}: Raza de perro reconocida. Para detalles específicos, consulta fuentes especializadas."
            
        except Exception as e:
            return f"Error al buscar información sobre {breed_name}: {str(e)}"

class ToolManager:
    """Gestor de herramientas para el sistema multiagente"""
    
    def __init__(self):
        self.tools = {
            "wikipedia": WikipediaTool(),
            "arxiv": ArxivTool(),
            "breed_database": BreedDatabaseTool()
        }
        self.tool_history: List[ToolResult] = []
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Ejecuta una herramienta específica"""
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                metadata=kwargs,
                execution_time=0.0,
                error=f"Herramienta '{tool_name}' no encontrada"
            )
        
        try:
            tool = self.tools[tool_name]
            
            if tool_name == "wikipedia":
                if "query" in kwargs:
                    result = tool.search_summary(kwargs["query"], kwargs.get("max_results", 3))
                else:
                    result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        data=None,
                        metadata=kwargs,
                        execution_time=0.0,
                        error="Parámetro 'query' requerido para Wikipedia"
                    )
            
            elif tool_name == "arxiv":
                if "query" in kwargs:
                    result = tool.search_papers(kwargs["query"], kwargs.get("max_results", 5))
                else:
                    result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        data=None,
                        metadata=kwargs,
                        execution_time=0.0,
                        error="Parámetro 'query' requerido para ArXiv"
                    )
            
            elif tool_name == "breed_database":
                if "breed_name" in kwargs:
                    result = tool.get_breed_info(kwargs["breed_name"])
                else:
                    result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        data=None,
                        metadata=kwargs,
                        execution_time=0.0,
                        error="Parámetro 'breed_name' requerido para Breed Database"
                    )
            
            else:
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    metadata=kwargs,
                    execution_time=0.0,
                    error=f"Herramienta '{tool_name}' no implementada"
                )
            
            # Almacenar en historial
            self.tool_history.append(result)
            return result
            
        except Exception as e:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                metadata=kwargs,
                execution_time=0.0,
                error=str(e)
            )
            self.tool_history.append(result)
            return result
    
    def get_tool_history(self) -> List[ToolResult]:
        """Obtiene el historial de herramientas"""
        return self.tool_history
    
    def get_available_tools(self) -> List[str]:
        """Obtiene lista de herramientas disponibles"""
        return list(self.tools.keys())

# Instancia global del gestor de herramientas
tool_manager = ToolManager()
