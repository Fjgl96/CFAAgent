# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time # Para simular "escritura"

# Importar el grafo compilado desde graph.agent_graph
# El manejo de errores ahora está dentro de agent_graph.py
try:
    from graph.agent_graph import compiled_graph
    print("✅ Grafo importado correctamente en streamlit_app.")
except Exception as import_error:
     # Si la importación falla (ej. error de sintaxis en agent_graph), mostrar y detener
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
# Añadir un separador o algo de contexto
st.markdown("""
Esta es una calculadora financiera inteligente. Puedes pedirle cálculos como:
- Valor Actual Neto (VAN)
- Costo Promedio Ponderado de Capital (WACC)
- Valoración de Bonos
- Costo de Equity (CAPM)
- Ratio de Sharpe
- Valoración de Acciones (Gordon Growth)
- Precio de Opciones Call (Black-Scholes)

**Nota:** Por ahora, por favor haz una solicitud de cálculo a la vez.
""")
st.divider() # Separador visual

# --- Lógica del Chat ---

# Inicializar historial de chat en st.session_state si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿Qué cálculo financiero necesitas realizar hoy?"}]

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
    # Usar un ID de sesión consistente para la memoria entre turnos
    config = {"configurable": {"thread_id": "streamlit-user-session"}} 

    # Ejecutar el grafo y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        # Mostrar feedback mientras procesa
        with st.spinner("Calculando... 🧠"):
            final_response_content = "" # Variable para el texto final

            try:
                # Usar invoke() para obtener solo el estado final
                final_state = compiled_graph.invoke(graph_input, config=config)
                
                # Procesar el estado final para extraer la respuesta más relevante
                # Iterar desde el final para encontrar la última respuesta útil del asistente (AIMessage)
                if final_state and "messages" in final_state and final_state["messages"]:
                    for msg in reversed(final_state["messages"]):
                         # Buscamos AIMessage que no sea implícitamente una llamada a herramienta pendiente
                         # (getattr es más seguro que acceso directo)
                         is_final_ai_msg = isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', []) 
                         if is_final_ai_msg:
                             content = msg.content
                             if isinstance(content, str): 
                                  final_response_content = content
                             elif isinstance(content, list): # Manejar contenido en lista
                                  text_parts = []
                                  for part in content:
                                      if isinstance(part, dict) and 'text' in part:
                                          text_parts.append(part['text'])
                                      elif isinstance(part, str):
                                           text_parts.append(part)
                                  final_response_content = "\n".join(text_parts).strip()
                             
                             # Si encontramos una respuesta con contenido, la usamos y salimos
                             if final_response_content: 
                                  break 
                
                # Fallback si no se encontró respuesta útil
                if not final_response_content:
                    final_response_content = "Lo siento, no pude procesar tu solicitud completamente. ¿Podrías intentarlo de nuevo o reformular?"
                    print("⚠️ No se encontró AIMessage final útil en el estado final.") # Log para depuración

            except Exception as e:
                # Capturar errores durante la ejecución del grafo
                final_response_content = f"Ocurrió un error inesperado al procesar tu solicitud."
                import traceback
                error_details = traceback.format_exc()
                print(f"❌ ERROR STREAMLIT RUNTIME: {error_details}") 
                st.error(f"{final_response_content} Por favor, intenta de nuevo más tarde.") 
                # No mostramos detalles técnicos al usuario final por defecto
                # message_placeholder.error(f"{final_response_content} Detalles: {e}")
                # Salir del 'with st.spinner' si hubo error
                # st.stop() # Esto detendría la app, quizás no sea ideal. 
                           # Mejor solo mostrar el error y permitir continuar.

        # Mostrar la respuesta final fuera del spinner
        # Simular efecto de "escritura" para mejor UX
        if final_response_content:
             message_placeholder.markdown(final_response_content)
        else:
             # Si algo muy raro pasó y no hay contenido, mostrar error genérico
             message_placeholder.error("No se pudo obtener una respuesta.")


    # Añadir respuesta final (o error) al historial de Streamlit
    if final_response_content: # Solo añadir si hubo una respuesta (incluso si fue un error manejado)
        st.session_state.messages.append({"role": "assistant", "content": final_response_content})
        # Opcional: st.rerun() para actualizar la interfaz si fuera necesario, 
        # pero para chat usualmente no hace falta si se actualiza session_state.