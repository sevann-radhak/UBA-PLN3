# evaluation_dashboard.py
"""
Dashboard de Evaluación Integral - PLN3
Página separada para el dashboard de evaluación
"""

import streamlit as st
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.evaluation.dashboard import evaluation_dashboard

def main():
    """Función principal del dashboard de evaluación"""
    try:
        # Renderizar el dashboard completo
        evaluation_dashboard.render_dashboard()
        
    except Exception as e:
        st.error(f"Error al cargar el dashboard de evaluación: {e}")
        st.write("Por favor, verifica que todas las dependencias estén instaladas correctamente.")

if __name__ == "__main__":
    main()
