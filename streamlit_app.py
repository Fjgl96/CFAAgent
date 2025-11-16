# streamlit_app.py
"""
Aplicación Streamlit - Agente Financiero con RAG.
Actualizado para LangChain 1.0+ con LangSmith.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Importar el grafo compilado
try:
    from graph.agent_graph import compiled_graph
    print("✅ Grafo importado correctamente en streamlit_app.")
except Exception as import_error:
    st.error(f"Error crítico al importar el agente: {import_error}")
    print(f"❌ Error crítico importando 'compiled_graph': {import_error}")
    st.stop()

# Importar config para mostrar info de LangSmith
from config import LANGSMITH_ENABLED
import os

# --- Configuración de Página Streamlit ---
st.set_page_config(
    page_title="Agente Financiero Pro",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Título y Cabecera ---
st.title("💰 Agente Financiero Profesional")
st.caption("Impulsado por LangGraph, Anthropic Claude y RAG (Elasticsearch)")

# Mostrar info de LangSmith si está habilitado
if LANGSMITH_ENABLED:
    st.info(f"🔍 **LangSmith activo** - Proyecto: `{os.environ.get('LANGCHAIN_PROJECT', 'N/A')}`")

st.markdown("""
Esta es una calculadora financiera inteligente con acceso a documentación CFA. Puedes:

**📊 Realizar cálculos:**
- Valor Actual Neto (VAN)
- Costo Promedio Ponderado de Capital (WACC)
- Valoración de Bonos
- CAPM, Sharpe Ratio, Gordon Growth, Opciones Call

**📚 Consultar documentación CFA:**
- "¿Qué dice el CFA sobre el WACC?"
- "Explica el concepto de Duration"
- "Busca información sobre el modelo Gordon Growth"

**❓ Obtener ayuda:**
- "Ayuda" o "¿Qué puedes hacer?"
""")
st.divider()

# --- Lógica del Chat ---

# Inicializar historial de chat si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! ¿Qué cálculo financiero necesitas realizar hoy? También puedo consultar la documentación CFA si tienes preguntas teóricas."}
    ]

# Inicializar thread_id único por sesión si no existe
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"Nuevo Thread ID generado para la sesión: {st.session_state.thread_id}")

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar input del usuario
if prompt := st.chat_input("Ej: Calcula VAN: inversión 50k, flujos [15k, 20k, 25k], tasa 12%"):
    
    # Añadir mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Preparar entrada para LangGraph
    graph_input = {"messages": [HumanMessage(content=prompt)]}
    
    # Usar el ID de sesión único guardado en st.session_state
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Ejecutar el grafo y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Calculando... 🧠"):
            final_response_content = ""
            
            try:
                # Usar invoke() para obtener solo el estado final
                final_state = compiled_graph.invoke(graph_input, config=config)
                
                # Extraer la respuesta final
                if final_state and "messages" in final_state and final_state["messages"]:
                    for msg in reversed(final_state["messages"]):
                        is_final_ai_msg = isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', [])
                        if is_final_ai_msg:
                            content = msg.content
                            if isinstance(content, str):
                                final_response_content = content
                            elif isinstance(content, list):
                                text_parts = []
                                for part in content:
                                    if isinstance(part, dict) and 'text' in part:
                                        text_parts.append(part['text'])
                                    elif isinstance(part, str):
                                        text_parts.append(part)
                                final_response_content = "\n".join(text_parts).strip()
                            
                            if final_response_content:
                                break
                
                if not final_response_content:
                    final_response_content = "Lo siento, no pude procesar tu solicitud completamente. ¿Podrías intentarlo de nuevo o reformular?"
                    print("⚠️ No se encontró AIMessage final útil en el estado final.")
            
            except Exception as e:
                final_response_content = f"Ocurrió un error inesperado al procesar tu solicitud."
                import traceback
                error_details = traceback.format_exc()
                print(f"❌ ERROR STREAMLIT RUNTIME: {error_details}")
                st.error(f"{final_response_content} Por favor, intenta de nuevo más tarde.")
            
            # Mostrar la respuesta final
            if final_response_content:
                message_placeholder.markdown(final_response_content)
            else:
                message_placeholder.error("No se pudo obtener una respuesta.")
    
    # Añadir respuesta final (o error) al historial de Streamlit
    if final_response_content:
        st.session_state.messages.append({"role": "assistant", "content": final_response_content})