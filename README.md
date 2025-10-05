# 🐕 PLN3 - Sistema Multiagente para Identificación de Razas de Perros

## 🚀 Deploy en Streamlit Cloud

### Variables de Entorno Requeridas:
```
OPENAI_API_KEY=tu_openai_key
GEMINI_API_KEY=tu_gemini_key
PINECONE_API_KEY=tu_pinecone_key
PINECONE_ENVIRONMENT=tu_environment
```

## 🏠 Instalación Local

### Prerrequisitos:
- Python 3.8+
- Git
- CUDA (opcional, para GPU)

### Pasos de Instalación:

1. **📥 Clonar el repositorio:**
```bash
git clone https://github.com/sevann-radhak/UBA-PLN3.git
cd UBA-PLN3
```

2. **🐍 Crear entorno virtual:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. **📦 Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **🔑 Configurar variables de entorno:**
```bash
# Crear archivo .env
echo "OPENAI_API_KEY=tu_openai_key_aqui" > .env
echo "GEMINI_API_KEY=tu_gemini_key_aqui" >> .env
echo "PINECONE_API_KEY=tu_pinecone_key_aqui" >> .env
echo "PINECONE_ENVIRONMENT=tu_environment_aqui" >> .env
```

5. **🚀 Ejecutar la aplicación:**
```bash
streamlit run streamlit_app.py
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

### Estructura del Proyecto:
```
UBA-PLN3/
├── app/                    # Código principal
│   ├── evaluation/         # Sistema de evaluación
│   ├── guardrails.py      # Sistema de seguridad
│   ├── multiagent.py      # Sistema multiagente
│   └── rag_pipeline.py    # Pipeline RAG híbrido
├── models/                 # Modelos entrenados
│   └── checkpoints/
│       └── best_model.pt   # Modelo ViT entrenado
├── scripts/               # Scripts de entrenamiento
├── requirements.txt       # Dependencias
├── streamlit_app.py      # Aplicación principal
└── README.md             # Este archivo
```

### Notas Importantes:
- **📁 Modelo grande:** El archivo `best_model.pt` pesa ~329MB
- **🔑 API Keys:** Necesarias para funcionamiento completo
- **💾 Memoria:** Requiere al menos 4GB RAM para el modelo
- **⏱️ Primera ejecución:** Puede tardar 2-3 minutos en cargar el modelo