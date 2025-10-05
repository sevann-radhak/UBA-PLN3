import streamlit as st
from PIL import Image
from app.vision_service import VisionService
from app.llm_agent import SimpleAgent
import json
import base64
import torchvision.transforms as transforms
import requests
import os
import time

# Define el tamaño de la imagen, debe coincidir con el tamaño de validación usado en el entrenamiento
img_size = 224

# Define las transformaciones para la inferencia (basadas en las de validación del entrenamiento)
# Es CRUCIAL que estas transformaciones sean las mismas que usaste para el conjunto de validación.
val_tf = transforms.Compose([
    transforms.Resize(int(img_size * 1.1)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Importar las clases y funciones del otro script
from app.vision_service import VisionService

# --- Configuración del LLM y API ---
# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

# Obtener API key con verificación
API_KEY = os.getenv('API_KEY_GEMINI') 

# Verificar que la API key se cargó correctamente
if not API_KEY:
    st.error("❌ API_KEY_GEMINI no encontrada. Verifica el archivo .env")
    st.stop()

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

#@st.cache_resource
#def load_services(model_path):
#    """Carga los servicios de Visión y el LLM una sola vez."""
    # El VisionService es el mismo de antes, cargado con los pesos del modelo
#    vs = VisionService(model_path)
#    return vs

def run_llm_agent_with_rag(breed_name, user_query):
    """
    Ejecuta el agente LLM con RAG híbrido, guardrails de seguridad y sistema multiagente.
    Implementa pipeline completo según Clases 1, 2 y 3.
    """
    try:
        # Importar sistemas
        from app.rag_pipeline import DogBreedRAGPipeline
        from app.guardrails import guardrails_system
        from app.multiagent import multiagent_supervisor, AgentState
        import uuid
        from datetime import datetime
        
        # 1. PROCESAR QUERY CON GUARDRAILS
        security_result = guardrails_system.process_query(user_query)
        
        # Verificar si la query es segura
        if not security_result["is_safe"]:
            warning_messages = []
            if security_result["rate_limited"]:
                warning_messages.append(f"🚫 {security_result['rate_limit_message']}")
            if security_result["warnings"]:
                warning_messages.extend([f"⚠️ {w}" for w in security_result["warnings"]])
            if not security_result["topic_valid"]:
                warning_messages.append("⚠️ Consulta no relacionada con razas de perros")
            
            return "\n".join(warning_messages) + "\n\nPor favor, reformula tu consulta sobre razas de perros."
        
        # Usar query sanitizada
        safe_query = security_result["sanitized_query"]
        
        # 2. CREAR ESTADO INICIAL PARA MULTIAGENTE
        session_id = str(uuid.uuid4())
        initial_state = AgentState(
            user_query=safe_query,
            breed_name=breed_name,
            original_query=user_query,
            research_results=[],
            summary="",
            validation_result={},
            current_agent="",
            agent_history=[],
            tools_used=[],
            next_agent=None,
            is_complete=False,
            error_message=None,
            session_id=session_id,
            timestamp=datetime.now(),
            user_context={"breed_name": breed_name, "query": safe_query}
        )
        
        # 3. EJECUTAR SISTEMA MULTIAGENTE
        start_time = time.time()
        multiagent_result = multiagent_supervisor.execute_multiagent_workflow(initial_state)
        execution_time = time.time() - start_time
        
        if not multiagent_result["success"]:
            return f"⚠️ **Error en el sistema multiagente:** {multiagent_result.get('error', 'Error desconocido')}"
        
        # 3.1. REGISTRAR MÉTRICAS DE EVALUACIÓN
        try:
            from app.evaluation import ir_metrics_calculator, security_metrics_calculator, multiagent_metrics_calculator
            
            # Registrar métricas de seguridad
            security_metrics_calculator.add_security_event(
                event_type="query_processed",
                query=user_query,
                result=security_result,
                processing_time=execution_time,
                metadata={"breed_name": breed_name}
            )
            
            # Registrar métricas multiagente
            agents_used = multiagent_result.get("agents_used", [])
            tools_used = multiagent_result.get("tools_used", [])
            multiagent_metrics_calculator.add_session(
                session_id=session_id,
                agents_used=agents_used,
                tools_used=tools_used,
                execution_time=execution_time,
                success=multiagent_result["success"],
                result=multiagent_result
            )
            
        except Exception as e:
            # No fallar si las métricas no se pueden registrar
            pass
        
        # 4. OBTENER INFORMACIÓN COMPILADA
        research_data = multiagent_result.get("research_results", [])
        summary_data = multiagent_result.get("summary", {})
        validation_data = multiagent_result.get("validation_result", {})
        
        # Guardar papers científicos en el contexto de la sesión
        if research_data and len(research_data) > 0:
            for result in research_data:
                if result.get("arxiv") and isinstance(result["arxiv"], list):
                    st.session_state['scientific_papers'] = result["arxiv"]
        
        # 5. CONSTRUIR RESPUESTA INTEGRADA
        response_parts = []
        
        # Priorizar información de la base de datos
        if research_data and len(research_data) > 0:
            breed_db_info = research_data[0].get("breed_database")
            if breed_db_info and breed_db_info.get("breed_info"):
                response_parts.append(f"**Información detallada sobre {breed_name}:**\n{breed_db_info['breed_info']}")
            else:
                # Si no hay info de la base de datos, usar resumen
                if summary_data and summary_data.get("summary_text"):
                    response_parts.append(f"**Información sobre {breed_name}:**\n{summary_data['summary_text']}")
        else:
            # Fallback al resumen
            if summary_data and summary_data.get("summary_text"):
                response_parts.append(f"**Información sobre {breed_name}:**\n{summary_data['summary_text']}")
        
        # Fuentes utilizadas
        sources_used = summary_data.get("sources_used", [])
        if sources_used:
            response_parts.append(f"**Fuentes consultadas:** {', '.join(sources_used)}")
        
        # Validación de calidad
        if validation_data and validation_data.get("is_sufficient"):
            response_parts.append("✅ **Información validada y completa**")
        
        # Estadísticas del sistema
        execution_time = multiagent_result.get("execution_time", 0)
        tools_used = multiagent_result.get("tools_used", [])
        if tools_used:
            response_parts.append(f"🔧 **Herramientas utilizadas:** {', '.join(tools_used)}")
        
        # 6. MANEJO ESPECIAL PARA CONSULTAS SOBRE PAPERS CIENTÍFICOS
        if "papers" in safe_query.lower() or "científicos" in safe_query.lower() or "artículos" in safe_query.lower():
            if st.session_state.get('scientific_papers'):
                papers_info = []
                for i, paper in enumerate(st.session_state['scientific_papers'][:3], 1):
                    papers_info.append(f"""
                    **Paper {i}:**
                    - **Título:** {paper.get('title', 'N/A')}
                    - **Autores:** {', '.join(paper.get('authors', []))}
                    - **Resumen:** {paper.get('summary', 'N/A')[:200]}...
                    - **Publicado:** {paper.get('published', 'N/A')}
                    - **Enlace:** {paper.get('link', 'N/A')}
                    """)
                
                return "\n\n".join(papers_info) + "\n\n**Fuentes:** ArXiv (Sistema multiagente)"
            else:
                return "No se encontraron papers científicos en esta sesión. Intenta hacer una consulta sobre la raza del perro primero."
        
        # 7. LLAMADA FINAL A GEMINI CON CONTEXTO ENRIQUECIDO
        if response_parts:
            enriched_context = "\n\n".join(response_parts)
            
            system_prompt = f"""
            Eres un experto en la raza de perro {breed_name}.
            Usa SOLO la información proporcionada para responder.
            Responde ÚNICAMENTE sobre temas relacionados con razas de perros.
            
            Información detallada sobre {breed_name}:
            {enriched_context}
            
            Pregunta del usuario: {safe_query}
            
            INSTRUCCIONES IMPORTANTES:
            - Responde de manera concisa, precisa y amigable
            - Usa la información detallada proporcionada para dar respuestas específicas
            - Si tienes información específica sobre la raza, úsala para responder
            - NUNCA digas "No tengo información específica" si la información está disponible
            - Si la pregunta no es sobre perros, redirige amablemente al tema de razas de perros
            - Siempre termina con una sugerencia útil o pregunta de seguimiento
            """
            
            # Payload para Gemini
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt}
                        ]
                    }
                ],
                "systemInstruction": {
                    "parts": [{"text": "Eres un experto en razas de perros. Responde basándote en la información proporcionada. Solo habla sobre perros."}]
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            result = response.json()
            
            # Extraer respuesta
            candidate = result.get('candidates', [{}])[0]
            text_part = candidate.get('content', {}).get('parts', [{}])[0].get('text', "")
            
            # Validar respuesta con guardrails
            validation_result = guardrails_system.validate_response(text_part)
            
            if not validation_result["is_valid"]:
                # Respuesta más amigable para el usuario
                return f"🤔 **Parece que hay un problema técnico con esa consulta específica.**\n\n" + \
                       f"**Sugerencias:**\n" + \
                       f"- ¿Podrías reformular tu pregunta sobre {breed_name}?\n" + \
                       f"- Intenta preguntar sobre características, cuidados, o comportamiento de la raza\n" + \
                       f"- Si tienes una pregunta específica, puedo ayudarte a encontrar información relevante\n\n" + \
                       f"💡 *El sistema está funcionando correctamente, solo necesitamos ajustar la consulta*"
            
            # Formatear respuesta final
            safe_response = validation_result["safe_response"]
            source_info = f"\n\n**Fuentes:** Sistema multiagente (RAG híbrido + Guardrails + Wikipedia + ArXiv)"
            
            return safe_response + source_info
        else:
            return "⚠️ **No se pudo obtener información suficiente sobre la raza.**"
        
    except Exception as e:
        return f"Error al procesar la consulta con sistema multiagente. Por favor, intentá de nuevo. (Error: {e})"

def run_llm_agent(breed_name, user_query):
    """
    Función de compatibilidad - redirige a RAG
    """
    return run_llm_agent_with_rag(breed_name, user_query)

@st.cache_resource
def load_services(model_path, model_name="vit_base_patch16_224"):
    vs = VisionService(model_path, model_name=model_name)
    #agent = SimpleAgent(model_name='mistral')
    return vs#, agent

def main():
    st.set_page_config(page_title="Product ID + LLM Agent", layout='wide')
    st.title("Identificador de razas canina + Asistente (ViT + LLM)")

    # Crear pestañas para la aplicación
    tab1, tab2 = st.tabs(["🐕 Identificador + Chat", "📊 Dashboard de Evaluación"])
    
    with tab1:
        render_main_interface()
    
    with tab2:
        render_evaluation_dashboard()

def render_main_interface():
    """Renderizar la interfaz principal de identificación y chat"""
    left, right = st.columns([1,1])
    with left:
        uploaded = st.file_uploader("Subí una imagen del perro", type=['jpg','jpeg','png'])
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Imagen subida", use_container_width=True)
            vs = load_services('./models/checkpoints/best_model.pt')
            pred_idx, probs = vs.predict(img)
            from app.product_db import get_class_name
            breed_name = get_class_name(pred_idx, './class_mapping.json')
            confidence = float(probs[pred_idx])
            
            # Obtener las 5 especificaciones más probables
            #top5_idx = probs.argsort()[-5:][::-1]
            #top5_specs = [(get_class_name(i, './class_mapping.json'), float(probs[i])) for i in top5_idx]
            #print(f"Probs: {top5_specs}")
                        
            st.success(f"Predicción({pred_idx}): {breed_name} ({confidence:.2f})")            
            # store context in session
            st.session_state['image_pred'] = {'label': breed_name, 'confidence': confidence}

    with right:
        st.header("Chat con el asistente")
        
        # Inicializar el historial de chat si no existe
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Inicializar contexto de papers científicos si no existe
        if 'scientific_papers' not in st.session_state:
            st.session_state['scientific_papers'] = []
        
        # Mostrar estado de seguridad
        with st.expander("🛡️ Estado de Seguridad", expanded=False):
            try:
                from app.guardrails import guardrails_system
                security_status = guardrails_system.get_security_status()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Requests restantes", security_status["rate_limit_remaining"])
                    st.metric("Límite total", security_status["rate_limit_total"])
                
                with col2:
                    st.metric("Ventana de tiempo", f"{security_status['rate_limit_window']}s")
                    st.metric("Patrones bloqueados", security_status["blocked_patterns_count"])
                
                st.info("🔒 Sistema de guardrails activo: sanitización, validación y rate limiting")
            except Exception as e:
                st.warning(f"⚠️ Error cargando estado de seguridad: {e}")
        
        # Mostrar estado del sistema multiagente
        with st.expander("🤖 Sistema Multiagente", expanded=False):
            try:
                from app.multiagent import multiagent_supervisor, agent_factory
                
                # Estadísticas del supervisor
                stats = multiagent_supervisor.get_agent_statistics()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ejecuciones totales", stats.get("total_executions", 0))
                    st.metric("Agentes disponibles", len(agent_factory.get_available_agents()))
                
                with col2:
                    if stats.get("agent_statistics"):
                        agent_stats = stats["agent_statistics"]
                        st.metric("Agentes activos", len(agent_stats))
                    else:
                        st.metric("Agentes activos", 0)
                
                # Lista de agentes disponibles
                available_agents = agent_factory.get_available_agents()
                st.info(f"🤖 **Agentes disponibles:** {', '.join(available_agents)}")
                
                # Herramientas disponibles
                from app.multiagent.tools import tool_manager
                available_tools = tool_manager.get_available_tools()
                st.info(f"🔧 **Herramientas disponibles:** {', '.join(available_tools)}")
                
            except Exception as e:
                st.warning(f"⚠️ Error cargando estado multiagente: {e}")
        
        # Mostrar estado del sistema de evaluación
        with st.expander("📊 Sistema de Evaluación", expanded=False):
            try:
                from app.evaluation import ir_metrics_calculator, security_metrics_calculator, multiagent_metrics_calculator
                
                # Métricas IR
                ir_metrics = ir_metrics_calculator.calculate_global_metrics()
                if ir_metrics:
                    st.metric("IR Score (MAP)", f"{ir_metrics.get('avg_map', 0):.3f}")
                
                # Métricas de seguridad
                security_metrics = security_metrics_calculator.calculate_security_metrics()
                st.metric("Security Score", f"{security_metrics.security_score:.1f}")
                
                # Métricas multiagente
                multiagent_metrics = multiagent_metrics_calculator.calculate_multiagent_metrics()
                st.metric("Multiagent Score", f"{multiagent_metrics.efficiency_score:.1f}")
                
                # Información sobre el dashboard integrado
                st.info("💡 **Dashboard de evaluación disponible en la pestaña '📊 Dashboard de Evaluación'**")
                
            except Exception as e:
                st.warning(f"⚠️ Error cargando sistema de evaluación: {e}")
            
        user_input = st.text_input("Escribí tu pregunta acerca del perro de la imagen", key="user_question")
        
        if st.button("Enviar", key="send_button"):
            if 'image_pred' not in st.session_state:
                st.warning("Primero sube una imagen para identificar la raza del perro.")
            else:
                breed_name = st.session_state['image_pred']['label'] # Acceder al label correctamente
                
                with st.spinner("🤖 Procesando con sistema multiagente (RAG + Guardrails + Wikipedia + ArXiv)..."):
                    # Llamar al agente LLM con sistema multiagente
                    response = run_llm_agent(breed_name, user_input)
                    
                    st.session_state['chat_history'].append(("User", user_input))
                    st.session_state['chat_history'].append(("Assistant", response))
        
        # Mostrar el historial de chat
        for role, msg in st.session_state['chat_history']:
            if role == "User":
                st.markdown(f"**Tú:** {msg}")
            else:
                st.markdown(f"**Asistente:** {msg}")
        
        # Mostrar información del sistema en el chat
        if st.session_state['chat_history']:
            st.info("🤖 **Sistema Multiagente Activo:**\n- ✅ RAG híbrido (BM25 + Pinecone + CrossEncoder)\n- ✅ Guardrails de seguridad\n- ✅ Agentes especializados (Research, Summarizer, Validator)\n- ✅ Herramientas externas (Wikipedia, ArXiv)\n- ✅ Memoria persistente y orquestación inteligente")

def render_evaluation_dashboard():
    """Renderizar el dashboard de evaluación integrado"""
    try:
        from app.evaluation.dashboard import evaluation_dashboard
        
        # Renderizar el dashboard completo
        evaluation_dashboard.render_dashboard()
        
    except Exception as e:
        st.error(f"Error al cargar el dashboard de evaluación: {e}")
        st.write("Por favor, verifica que todas las dependencias estén instaladas correctamente.")
        
        # Mostrar información básica si el dashboard falla
        st.subheader("📊 Métricas Básicas del Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔍 IR Score", "0.000", delta="Sin datos")
        
        with col2:
            st.metric("🛡️ Security Score", "0.0", delta="Sin datos")
        
        with col3:
            st.metric("🤖 Multiagent Score", "0.0", delta="Sin datos")
        
        st.info("💡 **Para ver métricas detalladas, usa el sistema de chat primero para generar datos de evaluación.**")

if __name__ == "__main__":
    main()

