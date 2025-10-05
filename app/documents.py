"""
Módulo de documentos para RAG
Implementa chunking inteligente según Clase 1
"""

from dataclasses import dataclass
from typing import List, Optional
import re
import unicodedata

@dataclass
class Document:
    """Documento para el pipeline RAG"""
    id: str
    text: str
    source: str = ""
    page: Optional[int] = None

# Tokenización básica
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")

def simple_tokenize(text: str) -> List[str]:
    """Tokenización simple para BM25"""
    return _TOKEN_RE.findall((text or "").lower())

# Utilidades para chunking
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑ])")
_SOFT_HYPH = re.compile(r"[\u00AD]")
_HARD_HYPH = re.compile(r"(\w)-\n(\w)")

def _normalize(text: str) -> str:
    """Normalizar texto para chunking"""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u00A0", " ")
    t = _SOFT_HYPH.sub("", t)
    t = _HARD_HYPH.sub(r"\1\2", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _split_sentences(text: str) -> List[str]:
    """Dividir texto en oraciones"""
    if not text:
        return []
    
    parts = _SENT_SPLIT.split(text.strip())
    if len(parts) <= 1:
        return [text.strip()]
    
    return [" ".join(p.split()) for p in parts if p.strip()]

def _count_tokens(t: str) -> int:
    """Contar tokens en texto"""
    return len(simple_tokenize(t))

def _slide_merge(sents: List[str], max_tok: int, overlap: int) -> List[str]:
    """Fusión deslizante de oraciones para chunks"""
    chunks = []
    buf = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if buf_tokens >= 40:
            txt = " ".join(buf).split()
            if len(txt) > max_tok + 20:
                txt = txt[:max_tok + 20]
            chunks.append(" ".join(txt))
        buf, buf_tokens = [], 0

    for s in sents:
        st = _count_tokens(s)
        
        if st > max_tok:
            pieces = re.split(r"[;:]\s+", s)
            for p in pieces:
                pt = _count_tokens(p)
                if pt == 0:
                    continue
                if pt > max_tok:
                    p = " ".join(p.split()[:max_tok])
                    pt = _count_tokens(p)
                
                if buf_tokens + pt <= max_tok:
                    buf.append(p)
                    buf_tokens += pt
                else:
                    flush()
                    if chunks:
                        tail = " ".join(chunks[-1].split()[-overlap:])
                        buf, buf_tokens = [tail], _count_tokens(tail)
                    buf.append(p)
                    buf_tokens += pt
            continue

        if buf_tokens + st <= max_tok:
            buf.append(s)
            buf_tokens += st
        else:
            flush()
            if chunks:
                tail = " ".join(chunks[-1].split()[-overlap:])
                buf, buf_tokens = [tail], _count_tokens(tail)
            buf.append(s)
            buf_tokens += st

    if buf:
        flush()
    return chunks

def chunk_text(text: str, max_tokens: int = 200, overlap: int = 60) -> List[str]:
    """
    Chunking inteligente de texto
    Parámetros optimizados para información de razas de perros
    """
    if not text:
        return []
    
    norm = _normalize(text)
    sents = _split_sentences(norm)
    if not sents:
        return []
    
    chunks = _slide_merge(sents, max_tok=max_tokens, overlap=overlap)
    
    # Filtro final de longitud
    out = []
    for ch in chunks:
        nt = _count_tokens(ch)
        if 20 <= nt <= 220:
            out.append(ch)
    
    return out
