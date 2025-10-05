"""
Fusión RRF (Reciprocal Rank Fusion)
Implementa fusión de rankings según Clase 1
"""

from typing import List, Dict

def rrf_combine(*ranked_lists: List[str], k: float = 60.0) -> List[str]:
    """
    Fusión RRF de múltiples listas ordenadas
    Parámetros:
        ranked_lists: Listas de resultados ordenados
        k: Parámetro de suavizado (default: 60.0)
    """
    scores: Dict[str, float] = {}
    
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1.0)
    
    # Ordenar por score descendente
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items]

