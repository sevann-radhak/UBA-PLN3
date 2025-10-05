# 🐕 PLN3 - Sistema Multiagente para Identificación de Razas de Perros

## 🚀 Deploy en Streamlit Cloud

### Variables de Entorno Requeridas:
```
OPENAI_API_KEY=tu_openai_key
GEMINI_API_KEY=tu_gemini_key
PINECONE_API_KEY=tu_pinecone_key
PINECONE_ENVIRONMENT=tu_environment
```

### Características:
- ✅ **Identificación de Razas** con Vision Transformer
- ✅ **RAG Híbrido** (BM25 + Pinecone + CrossEncoder)
- ✅ **Guardrails de Seguridad**
- ✅ **Sistema Multiagente** (Research, Summarizer, Validator)
- ✅ **Herramientas Externas** (Wikipedia, ArXiv)
- ✅ **Dashboard de Evaluación** con métricas IR, seguridad y multiagente

### Uso:
1. Sube una imagen de un perro
2. Haz preguntas sobre la raza identificada
3. El sistema responderá usando RAG + Multiagente + Guardrails

### Tecnologías:
- **Frontend:** Streamlit
- **AI:** OpenAI GPT-4, Gemini, Vision Transformer
- **RAG:** Pinecone, BM25, Sentence Transformers
- **Seguridad:** Guardrails AI
- **Evaluación:** Métricas IR, Seguridad, Multiagente