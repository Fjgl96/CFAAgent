# graph/agent_graph.py
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Usando memoria simple
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import streamlit as st # Para mostrar errores si es necesario

# Importar de config
from config import CIRCUIT_BREAKER_MAX_RETRIES

# Importar nodos de agente y supervisor
from agents.financial_agents import (
    supervisor_llm, supervisor_system_prompt,
    agent_nodes, RouterSchema # Necesitamos RouterSchema aquí
)

# --- Estado del Grafo ---
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y] # Acumula mensajes
    next_node: str # Nodo a ejecutar a continuación
    error_count: int # Contador para circuit breaker

# --- Nodo Supervisor (con Circuit Breaker v2 - Manejo robusto de contenido) ---
def supervisor_node(state: AgentState) -> dict: # Añadir type hint de retorno
    """Nodo del supervisor que decide el siguiente paso, con circuit breaker."""
    print("\n--- SUPERVISOR ---") # Log claro para inicio de ciclo
    
    # Lógica del Circuit Breaker
    error_count = state.get('error_count', 0)
    possible_error_detected = False # Flag si detectamos un mensaje de error del agente

    # Solo revisar si hay mensajes y el último es del asistente
    if state['messages'] and isinstance(state['messages'][-1], AIMessage):
        last_message = state['messages'][-1]
        
        # Revisar si es una respuesta final (no tool call) indicando fallo
        if not getattr(last_message, 'tool_calls', []): # Usar getattr para seguridad
            full_content = ""
            if isinstance(last_message.content, str):
                full_content = last_message.content.lower()
            elif isinstance(last_message.content, list): 
                for part in last_message.content:
                    if isinstance(part, dict) and 'text' in part:
                        full_content += part['text'].lower()
                    elif isinstance(part, str): 
                        full_content += part.lower()
            
            # Palabras clave que indican fallo o incapacidad
            error_keywords = ["problema técnico", "no puedo calcular", "error", "no es mi especialidad", "faltan parámetros"]
            if any(keyword in full_content for keyword in error_keywords):
                error_count += 1
                possible_error_detected = True 
                print(f"⚠️ Detectada falla/incapacidad de agente. Conteo errores: {error_count}/{CIRCUIT_BREAKER_MAX_RETRIES}")

    # Comprobar límite de reintentos SOLO si detectamos un error en este ciclo
    if possible_error_detected and error_count >= CIRCUIT_BREAKER_MAX_RETRIES:
        print(f"🚨 Límite de reintentos alcanzado ({CIRCUIT_BREAKER_MAX_RETRIES}). Forzando FINISH.")
        error_msg = f"Error: No se pudo completar la tarea después de {CIRCUIT_BREAKER_MAX_RETRIES} intentos fallidos del agente. El proceso se ha detenido."
        # Devolver estado para finalizar y añadir el mensaje de error
        # IMPORTANTE: Añadir el mensaje aquí para que quede en el estado final
        return {
             "messages": [AIMessage(content=error_msg)], 
             "next_node": "FINISH",
             "error_count": error_count 
        }

    # --- Enrutamiento Normal ---
    # Si no se alcanzó el límite o no hubo error detectado, pedir decisión al LLM
    supervisor_messages = [HumanMessage(content=supervisor_system_prompt)] + state['messages']
    
    next_node_decision = "FINISH" # Default a FINISH por seguridad
    try:
        route: RouterSchema = supervisor_llm.invoke(supervisor_messages)
        # Acceder al atributo 'next_agent' del objeto Pydantic devuelto
        if hasattr(route, 'next_agent'):
             next_node_decision = route.next_agent
        else:
             print("⚠️ Advertencia: Respuesta del supervisor LLM no tuvo 'next_agent'. Usando FINISH.")
             next_node_decision = "FINISH"
             
        print(f"🧭 Supervisor decide ruta: {next_node_decision}")
        
    except Exception as e:
         # Manejo de error si el LLM supervisor falla
         print(f"❌ ERROR en LLM Supervisor al invocar: {type(e).__name__} - {e}")
         st.warning(f"Advertencia: El supervisor LLM falló ({e}). Finalizando por seguridad.")
         next_node_decision = "FINISH" 
         # Considerar incrementar error_count aquí o loggear el fallo persistentemente

    # --- Resetear Contador de Errores ---
    # Resetea si la decisión es FINISH o si cambiamos a un agente DIFERENTE
    # Y SOLO si NO detectamos un error en ESTE ciclo
    previous_node = state.get('next_node', None) # Nodo anterior que fue llamado
    if not possible_error_detected and (next_node_decision == "FINISH" or next_node_decision != previous_node):
        if error_count > 0: # Solo imprimir si realmente se resetea
             print("🔄 Reseteando contador de errores a 0.")
        error_count = 0 
    # Si hubo error detectado pero no alcanzamos límite, mantenemos el conteo incrementado
    # Si no hubo error y repetimos agente, mantenemos el conteo (que debería ser 0)

    # Devolver la decisión y el estado del contador
    return {
        "next_node": next_node_decision,
        "error_count": error_count
    }


# --- Construcción del Grafo ---
def build_graph():
    """Construye y compila el grafo LangGraph."""
    workflow = StateGraph(AgentState)

    # Añadir nodo supervisor
    workflow.add_node("Supervisor", supervisor_node)

    # Añadir nodos de agentes desde el diccionario importado
    for name, node in agent_nodes.items():
        workflow.add_node(name, node)

    # Establecer el punto de entrada
    workflow.set_entry_point("Supervisor")

    # Definir la función de enrutamiento condicional
    def conditional_router(state: AgentState) -> str:
        """Función de enrutamiento basada en la decisión del supervisor."""
        node_to_go = state.get("next_node")
        # Validar ruta antes de devolverla
        valid_nodes = list(agent_nodes.keys()) + ["FINISH"]
        if node_to_go not in valid_nodes:
             print(f"⚠️ Alerta Ruteo: Destino inválido '{node_to_go}'. Forzando FINISH.")
             return "FINISH" 
        print(f"🚦 Enrutando a: {node_to_go}") # Log de ruteo
        return node_to_go

    # Crear el mapeo para las aristas condicionales
    conditional_map = {name: name for name in agent_nodes}
    conditional_map["FINISH"] = END # Mapear FINISH al nodo final especial
    workflow.add_conditional_edges(
        "Supervisor",   
        conditional_router, 
        conditional_map 
    )

    # Añadir aristas de retorno desde cada agente al supervisor
    for name in agent_nodes:
        workflow.add_edge(name, "Supervisor")

    # Compilar el grafo con un checkpointer de memoria
    # MemorySaver es bueno para Streamlit ya que cada sesión es independiente
    memory = MemorySaver()
    try:
        compiled_graph = workflow.compile(checkpointer=memory)
        print("✅ Grafo compilado correctamente con MemorySaver.")
        return compiled_graph
    except Exception as e:
         print(f"❌ ERROR CRÍTICO al compilar el grafo: {e}")
         # Propagar el error para que Streamlit lo muestre
         raise e 


# Crear la instancia global del grafo compilado al importar este módulo
# Manejar errores aquí para que la app falle rápido si hay problemas
try:
    compiled_graph = build_graph()
except Exception as build_error:
     # Si la compilación falla, detener la app Streamlit
     st.error(f"Error fatal al construir el agente gráfico: {build_error}")
     print(f"❌ Abortando: Falla en build_graph(): {build_error}")
     st.stop()

print("✅ Módulo agent_graph cargado.")