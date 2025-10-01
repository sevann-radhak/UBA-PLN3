import subprocess
import json
import shlex
from typing import List, Dict

# ---------- Helper: call local LLM via subprocess --------------
def call_local_llm_via_ollama(prompt: str, model_name: str="mistral", max_tokens=512, temperature=0.1):
    """
    Llamada simple usando 'ollama run <model>' por stdin/stdout.
    Requiere que 'ollama' esté en PATH y que 'ollama run <model>' funcione.
    Alternativa: si tenés una API HTTP local de Ollama, hacer requests a ella.
    """
    cmd = f"ollama run {shlex.quote(model_name)} --prompt-file=- --json"
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    prompt_payload = prompt
    stdout, stderr = p.communicate(prompt_payload)
    if p.returncode != 0:
        raise RuntimeError(f"Ollama error: {stderr}")
    # Si Ollama devuelve JSON en stdout, parsealo; si no, devolvemos raw
    try:
        return json.loads(stdout)
    except Exception:
        return stdout

# ---------- Simple agent orchestrator -------------------------
class SimpleAgent:
    def __init__(self, model_name='mistral'):
        self.model_name = model_name

    def create_prompt(self, image_pred, top_specs):
        # image_pred: dict {'label': 'Samsung X123', 'confidence': 0.92}
        # top_specs: list of dicts {'title','text','product_id'}
        context = "Contexto: La imagen fue clasificada como: {} (conf: {:.2f}).\n\nFichas técnicas relevantes:\n".format(image_pred['label'], image_pred['confidence'])
        for i, s in enumerate(top_specs[:3]):
            context += f"#{i+1} Title: {s['title']}\n{ s['text'][:800] }...\n\n"  # incluir fragmento para RAG
        prompt = (
            context +
            "Objetivo: Responda preguntas del usuario sobre el producto, usando SOLO la información provista arriba y, si es necesario, solicite explícitamente invocar herramientas (SEARCH_SPEC, OPEN_URL, FETCH_MANUAL). "
            "Si no hay suficiente información, indique que no se sabe y proponga pasos (ej. buscar web, solicitar foto adicional).\n\n"
            "Usuario: {user_query}\n\nAsistente:"
        )
        return prompt

    def run(self, user_query, image_pred, top_specs):
        prompt_template = self.create_prompt(image_pred, top_specs)
        prompt = prompt_template.format(user_query=user_query)
        # seguridad: sanitizar prompt antes de enviarlo
        safe_prompt = self.apply_safety_filters(prompt)
        llm_out = call_local_llm_via_ollama(safe_prompt, model_name=self.model_name)
        return llm_out

    def apply_safety_filters(self, text):
        # ejemplo simple: bloquear instrucciones para cometer delitos o PII
        banned_terms = ['how to make', 'illegal', 'explode', 'bomb']
        for t in banned_terms:
            if t in text.lower():
                return text + "\n\n[WARNING: user prompt contains disallowed phrase. Respond with refusal.]"
        return text

