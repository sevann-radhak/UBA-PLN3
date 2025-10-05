import streamlit as st
from PIL import Image
from app.vision_service import VisionService
from app.llm_agent import SimpleAgent
import json
import base64
import torchvision.transforms as transforms
import requests
import os

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
    Ejecuta el agente LLM con RAG híbrido y guardrails de seguridad.
    Implementa pipeline RAG completo según Clase 1 + Guardrails según Clase 2.
    """
    try:
        # Importar sistemas
        from app.rag_pipeline import DogBreedRAGPipeline
        from app.guardrails import guardrails_system
        
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
        
        # 2. OBTENER CONTEXTO CON RAG
        rag_pipeline = DogBreedRAGPipeline()
        context = rag_pipeline.get_breed_info(breed_name, safe_query)
        
        # 3. CONSTRUIR PROMPT SEGURO
        system_prompt = f"""
        Eres un experto en la raza de perro {breed_name}.
        Usa SOLO la información proporcionada para responder.
        Responde ÚNICAMENTE sobre temas relacionados con razas de perros.
        
        Información sobre {breed_name}:
        {context}
        
        Pregunta del usuario: {safe_query}
        
        Responde de manera concisa, precisa y amigable.
        Si la información no está disponible, indícalo claramente.
        NO respondas sobre temas no relacionados con perros.
        """
        
        # 4. LLAMADA SEGURA A GEMINI
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
        
        # 5. VALIDAR RESPUESTA DEL LLM
        candidate = result.get('candidates', [{}])[0]
        text_part = candidate.get('content', {}).get('parts', [{}])[0].get('text', "")
        
        # Validar respuesta con guardrails
        validation_result = guardrails_system.validate_response(text_part)
        
        if not validation_result["is_valid"]:
            return f"⚠️ **Respuesta bloqueada por seguridad:**\n" + "\n".join(validation_result["warnings"])
        
        # 6. FORMATEAR RESPUESTA SEGURA
        safe_response = validation_result["safe_response"]
        source_info = "\n\n**Fuentes:** Base de conocimiento de razas de perros (RAG híbrido + Guardrails)"
        
        return safe_response + source_info
        
    except Exception as e:
        return f"Error al procesar la consulta con RAG y Guardrails. Por favor, intentá de nuevo. (Error: {e})"

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
            
        user_input = st.text_input("Escribí tu pregunta acerca del perro de la imagen", key="user_question")
        
        if st.button("Enviar", key="send_button"):
            if 'image_pred' not in st.session_state:
                st.warning("Primero sube una imagen para identificar la raza del perro.")
            else:
                breed_name = st.session_state['image_pred']['label'] # Acceder al label correctamente
                
                with st.spinner("🔍 Procesando con RAG híbrido y guardrails de seguridad..."):
                    # Llamar al agente LLM con guardrails
                    response = run_llm_agent(breed_name, user_input)
                    
                    st.session_state['chat_history'].append(("User", user_input))
                    st.session_state['chat_history'].append(("Assistant", response))
        
        # Mostrar el historial de chat
        for role, msg in st.session_state['chat_history']:
            if role == "User":
                st.markdown(f"**Tú:** {msg}")
            else:
                st.markdown(f"**Asistente:** {msg}")
        
        # Mostrar información de seguridad en el chat
        if st.session_state['chat_history']:
            st.info("🛡️ **Sistema de Seguridad Activo:**\n- ✅ Sanitización de entrada\n- ✅ Validación de salida\n- ✅ Rate limiting\n- ✅ Prevención de prompt injection")

if __name__ == "__main__":
    main()

