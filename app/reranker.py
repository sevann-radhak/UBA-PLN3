"""
Re-ranqueo con Cross-Encoder
Implementa re-ranqueo neural según Clase 1
"""

from typing import Dict, List, Tuple, Optional
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Re-ranqueo con Cross-Encoder para mejorar precisión"""
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 device: Optional[str] = None):
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(self, 
               query: str, 
               candidates: List[Tuple[str, str, Dict]]) -> List[Tuple[str, str, Dict, float]]:
        """
        Re-ranquear candidatos con Cross-Encoder
        Parámetros:
            query: Consulta del usuario
            candidates: Lista de candidatos [(doc_id, chunk_text, meta)]
        Retorna:
            Lista re-ranqueada [(doc_id, chunk_text, meta, score)]
        """
        if not candidates:
            return []
        
        # Crear pares (query, documento)
        pairs = [(query, c[1]) for c in candidates]
        
        # Obtener scores del Cross-Encoder
        scores = self.model.predict(pairs)
        
        # Combinar resultados con scores
        results = [(c[0], c[1], c[2], float(s)) for c, s in zip(candidates, scores)]
        
        # Ordenar por score descendente
        results.sort(key=lambda x: x[3], reverse=True)
        
        return results

