# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time
import uuid # <-- 1. Importa el módulo UUID

# Importar el grafo compilado
try:
    from graph.agent_graph import compiled_graph
    print("✅ Grafo importado correctamente en streamlit_app.")
except Exception as import_error:
     st.error(f"Error crítico al importar el agente: {import_error}")
     print(f"❌ Error crítico importando 'compiled_graph': {import_error}")
     st.stop()


# --- Configuración de Página Streamlit ---
st.set_page_config(
    page_title="Agente Financiero Pro", 
    layout="centered", 
    initial_sidebar_state="auto" 
)

# --- Título y Cabecera ---
st.title("💰 Agente Financiero Profesional")
st.caption("Impulsado por LangGraph y Anthropic Claude-3-Haiku")
st.markdown("""
Esta es una calculadora financiera inteligente. Puedes pedirle cálculos como:
- Valor Actual Neto (VAN)
- Costo Promedio Ponderado de Capital (WACC)
- Valoración de Bonos
- ... (etc.)

**Nota:** Por ahora, por favor haz una solicitud de cálculo a la vez.
""")
st.divider()

# --- Lógica del Chat ---

# ==========================================================
# 2. INICIALIZACIÓN DE MEMORIA Y THREAD_ID (CORREGIDO)
# ==========================================================
# Inicializar historial de chat si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿Qué cálculo financiero necesitas realizar hoy?"}]

# Inicializar thread_id único por sesión si no existe
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"Nuevo Thread ID generado para la sesión: {st.session_state.thread_id}")

# ==========================================================

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar input del usuario
if prompt := st.chat_input("Ej: Calcular VAN para inversión 50k, flujos [15k, 20k, 25k], tasa 12%"):
    
    # Añadir mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparar entrada para LangGraph
    graph_input = {"messages": [HumanMessage(content=prompt)]}
    
    # ==========================================================
    # 3. USAR EL THREAD_ID ÚNICO DE LA SESIÓN (CORREGIDO)
    # ==========================================================
    # Usar el ID de sesión único guardado en st.session_state
    config = {"configurable": {"thread_id": st.session_state.thread_id}} 
    # ==========================================================

    # Ejecutar el grafo y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        with st.spinner("Calculando... 🧠"):
            final_response_content = "" 

            try:
                # Usar invoke() para obtener solo el estado final
                # Pasamos el config con el thread_id único de la sesión
                final_state = compiled_graph.invoke(graph_input, config=config)
                
                # (El resto de tu lógica para procesar 'final_state' es correcta)
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

            # Mostrar la respuesta final fuera del spinner
            if final_response_content:
                 message_placeholder.markdown(final_response_content)
            else:
                 message_placeholder.error("No se pudo obtener una respuesta.")

    # Añadir respuesta final (o error) al historial de Streamlit
    if final_response_content:
        st.session_state.messages.append({"role": "assistant", "content": final_response_content})