"""
Pipeline RAG híbrido para razas de perros
Implementa conceptos de Clase 1: BM25 + Vectorial + Cross-Encoder
"""

import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from app.breed_knowledge_base import DogBreedKnowledgeBase
from app.documents import Document, chunk_text
from app.bm25_index import BM25Index
from app.vector_pinecone import PineconeSearcher, make_chunk_id, parse_chunk_id
from app.fusion import rrf_combine
from app.reranker import CrossEncoderReranker

# Cargar variables de entorno
load_dotenv()

class DogBreedRAGPipeline:
    """
    Pipeline RAG híbrido para información de razas de perros
    Implementa: BM25 + Vectorial + Cross-Encoder según Clase 1
    """
    
    def __init__(self, 
                 class_mapping_path: str = "./class_mapping.json",
                 max_tokens_chunk: int = 300,
                 overlap: int = 80,
                 ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        # Configuración
        self.max_tokens_chunk = max_tokens_chunk
        self.overlap = overlap
        self.ce_model = ce_model
        
        # Base de conocimiento
        self.knowledge_base = DogBreedKnowledgeBase(class_mapping_path)
        self.breed_docs = self.knowledge_base.create_breed_documents()
        
        # Chunking de documentos
        self.chunks_per_doc = self._create_chunks()
        
        # Índice BM25
        self.bm25_index = BM25Index(self.breed_docs, self.chunks_per_doc)
        
        # Vector search (Pinecone)
        self.vector_searcher = self._setup_vector_search()
        
        # Cross-Encoder para re-ranqueo
        self.reranker = CrossEncoderReranker(model_name=ce_model)
        
        # Mapeo global de chunks
        self.global_chunks, self.global_map = self._create_global_mapping()
    
    def _create_chunks(self) -> Dict[str, List[str]]:
        """Crear chunks de documentos para RAG"""
        chunks_per_doc = {}
        
        for doc in self.breed_docs:
            chunks = chunk_text(doc.text, self.max_tokens_chunk, self.overlap)
            if not chunks:
                # Fallback para documentos cortos
                chunks = [doc.text] if doc.text else []
            chunks_per_doc[doc.id] = chunks
        
        return chunks_per_doc
    
    def _setup_vector_search(self) -> Optional[PineconeSearcher]:
        """Configurar búsqueda vectorial con Pinecone"""
        try:
            # Verificar si Pinecone está configurado
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                print("⚠️ PINECONE_API_KEY no encontrada. Usando solo BM25.")
                return None
            
            # Configurar Pinecone
            searcher = PineconeSearcher(
                index_name=os.getenv('PINECONE_INDEX', 'dog-breeds-index'),
                model_name=os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                region=os.getenv('PINECONE_REGION', 'us-east-1'),
                api_key=pinecone_api_key,
                namespace="dog-breeds-v1"
            )
            
            # Upsert chunks a Pinecone
            docs_meta = {doc.id: {"source": doc.source, "page": doc.page} for doc in self.breed_docs}
            searcher.upsert_chunks(self.chunks_per_doc, docs_meta)
            
            print("✅ Pinecone configurado correctamente.")
            return searcher
            
        except Exception as e:
            print(f"⚠️ Error configurando Pinecone: {e}")
            print("🔄 Continuando con solo BM25...")
            return None
    
    def _create_global_mapping(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        """Crear mapeo global de chunks para BM25"""
        global_chunks = []
        global_map = []
        
        for doc in self.breed_docs:
            for i, chunk in enumerate(self.chunks_per_doc[doc.id]):
                global_chunks.append(chunk)
                global_map.append((doc.id, i))
        
        return global_chunks, global_map
    
    def retrieve_hybrid(self, 
                       query: str, 
                       top_k: int = 50,
                       meta_filter: Optional[dict] = None) -> List[str]:
        """
        Recuperación híbrida: BM25 + Vectorial + RRF
        """
        # BM25 search
        bm25_hits = []
        for gi, _score in self.bm25_index.search(query, top_k=top_k):
            doc_id, local_i = self.global_map[gi]
            bm25_hits.append(make_chunk_id(doc_id, local_i))
        
        # Vector search
        vec_hits = []
        if self.vector_searcher is not None:
            try:
                vec_results = self.vector_searcher.search(query, top_k=top_k, meta_filter=meta_filter)
                vec_hits = [cid for (cid, _s, _m) in vec_results]
            except Exception as e:
                print(f"⚠️ Error en búsqueda vectorial: {e}")
                vec_hits = []
        
        # RRF Fusion
        if vec_hits:
            combined = rrf_combine(bm25_hits, vec_hits)
        else:
            combined = bm25_hits
        
        return combined[:top_k]
    
    def retrieve_with_metadata(self, 
                              query: str, 
                              top_k: int = 20,
                              per_doc_cap: int = 2,
                              meta_filter: Optional[dict] = None) -> List[Tuple[str, str, Dict]]:
        """
        Recuperación con metadatos y límite por documento
        """
        chunk_ids = self.retrieve_hybrid(query, top_k=top_k * 3, meta_filter=meta_filter)
        results = []
        seen = {}
        
        for cid in chunk_ids:
            doc_id, local_i = parse_chunk_id(cid)
            seen[doc_id] = seen.get(doc_id, 0)
            
            if seen[doc_id] >= per_doc_cap:
                continue
            
            seen[doc_id] += 1
            chunk_text = self.chunks_per_doc[doc_id][local_i]
            
            # Metadatos
            if self.vector_searcher is not None:
                meta = self.vector_searcher.registry.get(cid, {})
            else:
                doc = next((d for d in self.breed_docs if d.id == doc_id), None)
                meta = {
                    "doc_id": doc_id,
                    "local_idx": local_i,
                    "source": doc.source if doc else "unknown",
                    "page": doc.page if doc else 1
                }
            
            results.append((doc_id, chunk_text, meta))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_and_rerank(self, 
                           query: str, 
                           top_retrieve: int = 30, 
                           top_final: int = 5) -> List[Tuple[str, str, Dict, float]]:
        """
        Recuperación y re-ranqueo con Cross-Encoder
        """
        # Recuperar candidatos
        candidates = self.retrieve_with_metadata(query, top_k=top_retrieve)
        
        # Re-ranquear con Cross-Encoder
        reranked = self.reranker.rerank(query, candidates)
        
        return reranked[:top_final]
    
    def get_breed_info(self, breed_name: str, user_query: str) -> str:
        """
        Obtener información de raza usando RAG híbrido
        """
        # Construir query contextualizada
        contextual_query = f"{breed_name} {user_query}"
        
        # Recuperación y re-ranqueo
        results = self.retrieve_and_rerank(contextual_query, top_retrieve=20, top_final=5)
        
        # Construir contexto con citas
        context = self._build_cited_context(results)
        
        return context
    
    def _build_cited_context(self, results: List[Tuple[str, str, Dict, float]]) -> str:
        """Construir contexto con citas para el LLM"""
        if not results:
            return "No se encontró información relevante sobre esta raza."
        
        context_parts = []
        for i, (doc_id, chunk, meta, score) in enumerate(results, 1):
            source = meta.get("source", "breed_database")
            page = meta.get("page", 1)
            citation = f"[{source}, p. {page}]"
            
            context_parts.append(f"{chunk}\n{citation}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def evaluate_retrieval(self, 
                          test_queries: List[str], 
                          ground_truth: List[List[str]]) -> Dict[str, float]:
        """
        Evaluar rendimiento del RAG con métricas IR
        """
        from app.metrics import calculate_metrics
        
        all_results = []
        for query in test_queries:
            results = self.retrieve_hybrid(query, top_k=10)
            all_results.append(results)
        
        metrics = calculate_metrics(all_results, ground_truth)
        return metrics
