"""
Búsqueda vectorial con Pinecone
Implementa embeddings y búsqueda semántica según Clase 1
"""

import os
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import pinecone

class PineconeSearcher:
    """Búsqueda vectorial con Pinecone para información de razas"""
    
    def __init__(self, 
                 index_name: str,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cloud: str = "aws",
                 region: str = "us-east-1",
                 api_key: Optional[str] = None,
                 namespace: str = "default"):
        
        self.index_name = index_name
        self.model_name = model_name
        self.namespace = namespace
        
        # Inicializar modelo de embeddings
        self.embedding_model = SentenceTransformer(model_name)
        
        # Inicializar Pinecone
        if api_key:
            pinecone.init(api_key=api_key, environment=region)
        else:
            pinecone.init()
        
        # Obtener índice
        self.index = pinecone.Index(index_name)
        
        # Registry para metadatos
        self.registry = {}
    
    def upsert_chunks(self, 
                     chunks_per_doc: Dict[str, List[str]], 
                     docs_meta: Dict[str, Dict]) -> None:
        """Subir chunks a Pinecone"""
        vectors = []
        
        for doc_id, chunks in chunks_per_doc.items():
            for i, chunk in enumerate(chunks):
                # Generar embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Crear ID único
                chunk_id = f"{doc_id}::chunk_{i}"
                
                # Metadatos
                meta = {
                    "doc_id": doc_id,
                    "chunk_idx": i,
                    "text": chunk[:1000],  # Preview del texto
                    **docs_meta.get(doc_id, {})
                }
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": meta
                })
                
                # Guardar en registry
                self.registry[chunk_id] = meta
        
        # Subir en lotes
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               meta_filter: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Búsqueda vectorial
        Retorna: [(chunk_id, score, metadata)]
        """
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Búsqueda en Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=meta_filter
        )
        
        # Formatear resultados
        formatted_results = []
        for match in results.matches:
            chunk_id = match.id
            score = match.score
            metadata = match.metadata
            
            formatted_results.append((chunk_id, score, metadata))
        
        return formatted_results
    
    def clear_namespace(self) -> None:
        """Limpiar namespace (usar con cuidado)"""
        self.index.delete(delete_all=True, namespace=self.namespace)

def make_chunk_id(doc_id: str, chunk_idx: int) -> str:
    """Crear ID único para chunk"""
    return f"{doc_id}::chunk_{chunk_idx}"

def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    """Parsear ID de chunk"""
    doc_id, chunk_part = chunk_id.split("::")
    chunk_idx = int(chunk_part.split("_")[1])
    return doc_id, chunk_idx
