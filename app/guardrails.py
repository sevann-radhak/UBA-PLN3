# app/guardrails.py
"""
Sistema de Guardrails y Seguridad para el proyecto PLN3
Implementa conceptos de la Clase 2: sanitización, validación y seguridad
"""

import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import streamlit as st

@dataclass
class SecurityConfig:
    """Configuración de seguridad para el sistema"""
    max_query_length: int = 500
    max_response_length: int = 2000
    rate_limit_requests: int = 10  # requests per minute
    rate_limit_window: int = 60  # seconds
    allowed_topics: List[str] = None
    blocked_patterns: List[str] = None
    
    def __post_init__(self):
        if self.allowed_topics is None:
            self.allowed_topics = [
                "perro", "raza", "cuidado", "salud", "entrenamiento", 
                "alimentación", "ejercicio", "comportamiento", "características"
            ]
        
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r"ignore\s+previous\s+instructions",
                r"forget\s+everything",
                r"you\s+are\s+now",
                r"act\s+as\s+if",
                r"pretend\s+to\s+be",
                r"system\s+prompt",
                r"jailbreak",
                r"bypass",
                r"override",
                r"admin\s+access",
                r"root\s+access",
                r"sudo",
                r"password",
                r"api\s+key",
                r"secret",
                r"token"
            ]

class InputSanitizer:
    """Sanitizador de entrada del usuario"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in config.blocked_patterns]
    
    def sanitize_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Sanitiza la consulta del usuario y detecta patrones maliciosos
        
        Returns:
            Tuple[str, List[str]]: (query_sanitizada, warnings)
        """
        if not query or not isinstance(query, str):
            return "", ["Query vacía o inválida"]
        
        warnings = []
        original_query = query
        
        # 1. Verificar longitud
        if len(query) > self.config.max_query_length:
            query = query[:self.config.max_query_length]
            warnings.append(f"Query truncada a {self.config.max_query_length} caracteres")
        
        # 2. Detectar patrones maliciosos
        malicious_patterns = []
        for pattern in self.compiled_patterns:
            if pattern.search(query):
                malicious_patterns.append(pattern.pattern)
        
        if malicious_patterns:
            warnings.append(f"Patrones sospechosos detectados: {malicious_patterns}")
            # Reemplazar patrones maliciosos con texto seguro
            for pattern in self.compiled_patterns:
                query = pattern.sub("[PATRÓN BLOQUEADO]", query)
        
        # 3. Limpiar caracteres especiales peligrosos
        query = re.sub(r'[<>"\']', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # 4. Verificar que la query no esté vacía después de la limpieza
        if not query:
            query = "Consulta sobre razas de perros"
            warnings.append("Query reemplazada por consulta por defecto")
        
        return query, warnings
    
    def validate_topic(self, query: str) -> bool:
        """Verifica si la consulta está relacionada con temas permitidos"""
        query_lower = query.lower()
        return any(topic in query_lower for topic in self.config.allowed_topics)

class OutputValidator:
    """Validador de salida del LLM"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_response(self, response: str) -> Tuple[bool, List[str]]:
        """
        Valida la respuesta del LLM
        
        Returns:
            Tuple[bool, List[str]]: (es_válida, warnings)
        """
        if not response or not isinstance(response, str):
            return False, ["Respuesta vacía o inválida"]
        
        warnings = []
        
        # 1. Verificar longitud
        if len(response) > self.config.max_response_length:
            warnings.append(f"Respuesta muy larga: {len(response)} caracteres")
        
        # 2. Detectar contenido inapropiado
        inappropriate_patterns = [
            r"como\s+hacer\s+bombas",
            r"como\s+matar",
            r"violencia",
            r"drogas",
            r"armas",
            r"hackear",
            r"robar"
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                warnings.append(f"Contenido inapropiado detectado: {pattern}")
        
        # 3. Verificar que la respuesta sea relevante
        if len(response.strip()) < 10:
            warnings.append("Respuesta muy corta o vacía")
        
        # 4. Detectar respuestas de error del LLM
        error_patterns = [
            r"no\s+puedo",
            r"no\s+debo",
            r"no\s+estoy\s+autorizado",
            r"error\s+del\s+sistema",
            r"fallo\s+interno"
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                warnings.append(f"Posible error del LLM: {pattern}")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def format_safe_response(self, response: str, warnings: List[str]) -> str:
        """Formatea una respuesta segura con warnings si es necesario"""
        if warnings:
            warning_text = "\n\n⚠️ **Advertencias de Seguridad:**\n"
            for warning in warnings:
                warning_text += f"- {warning}\n"
            return response + warning_text
        
        return response

class RateLimiter:
    """Limitador de velocidad para prevenir abuso"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests = {}  # {user_id: [timestamps]}
    
    def is_rate_limited(self, user_id: str = "default") -> Tuple[bool, str]:
        """
        Verifica si el usuario ha excedido el límite de velocidad
        
        Returns:
            Tuple[bool, str]: (está_limitado, mensaje)
        """
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        # Limpiar requests antiguos
        if user_id in self.requests:
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id] 
                if req_time > window_start
            ]
        else:
            self.requests[user_id] = []
        
        # Verificar límite
        if len(self.requests[user_id]) >= self.config.rate_limit_requests:
            return True, f"Límite de velocidad excedido. Máximo {self.config.rate_limit_requests} requests por minuto."
        
        # Registrar nueva request
        self.requests[user_id].append(now)
        return False, "OK"
    
    def get_remaining_requests(self, user_id: str = "default") -> int:
        """Obtiene el número de requests restantes"""
        if user_id not in self.requests:
            return self.config.rate_limit_requests
        
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        recent_requests = [
            req_time for req_time in self.requests[user_id] 
            if req_time > window_start
        ]
        
        return max(0, self.config.rate_limit_requests - len(recent_requests))

class SecurityGuardrails:
    """Sistema principal de guardrails que integra todos los componentes"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.sanitizer = InputSanitizer(self.config)
        self.validator = OutputValidator(self.config)
        self.rate_limiter = RateLimiter(self.config)
    
    def process_query(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Procesa una consulta del usuario con todas las validaciones de seguridad
        
        Returns:
            Dict con información de seguridad y query procesada
        """
        result = {
            "original_query": query,
            "sanitized_query": "",
            "is_safe": False,
            "warnings": [],
            "rate_limited": False,
            "rate_limit_message": "",
            "topic_valid": False,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        # 1. Verificar rate limiting
        is_limited, rate_message = self.rate_limiter.is_rate_limited(user_id)
        if is_limited:
            result["rate_limited"] = True
            result["rate_limit_message"] = rate_message
            result["processing_time"] = time.time() - start_time
            return result
        
        # 2. Sanitizar query
        sanitized_query, sanitization_warnings = self.sanitizer.sanitize_query(query)
        result["sanitized_query"] = sanitized_query
        result["warnings"].extend(sanitization_warnings)
        
        # 3. Validar tema
        topic_valid = self.sanitizer.validate_topic(sanitized_query)
        result["topic_valid"] = topic_valid
        
        if not topic_valid:
            result["warnings"].append("Consulta no relacionada con razas de perros")
        
        # 4. Determinar si es segura
        result["is_safe"] = (
            not result["rate_limited"] and 
            len(result["warnings"]) == 0 and 
            topic_valid
        )
        
        result["processing_time"] = time.time() - start_time
        return result
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Valida una respuesta del LLM
        
        Returns:
            Dict con información de validación
        """
        is_valid, warnings = self.validator.validate_response(response)
        
        return {
            "response": response,
            "is_valid": is_valid,
            "warnings": warnings,
            "safe_response": self.validator.format_safe_response(response, warnings)
        }
    
    def get_security_status(self, user_id: str = "default") -> Dict[str, Any]:
        """Obtiene el estado de seguridad del sistema"""
        remaining_requests = self.rate_limiter.get_remaining_requests(user_id)
        
        return {
            "rate_limit_remaining": remaining_requests,
            "rate_limit_total": self.config.rate_limit_requests,
            "rate_limit_window": self.config.rate_limit_window,
            "max_query_length": self.config.max_query_length,
            "max_response_length": self.config.max_response_length,
            "allowed_topics": self.config.allowed_topics,
            "blocked_patterns_count": len(self.config.blocked_patterns)
        }

# Instancia global del sistema de guardrails
guardrails_system = SecurityGuardrails()
