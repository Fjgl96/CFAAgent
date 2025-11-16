# graph/agent_graph.py
"""
Grafo de agentes financieros.
Actualizado para LangChain 1.0+
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import streamlit as st

# Importar de config
from config import CIRCUIT_BREAKER_MAX_RETRIES

# Importar nodos de agente y supervisor
from agents.financial_agents import (
    supervisor_llm, supervisor_system_prompt,
    agent_nodes, RouterSchema
)

# --- Estado del Grafo ---

class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Acumula mensajes
    next_node: str  # Nodo a ejecutar a continuación
    error_count: int  # Contador para circuit breaker


# --- Nodo Supervisor (con Circuit Breaker) ---

def supervisor_node(state: AgentState) -> dict:
    """Nodo del supervisor que decide el siguiente paso, con circuit breaker."""
    print("\n--- SUPERVISOR ---")
    
    # Lógica del Circuit Breaker
    error_count = state.get('error_count', 0)
    possible_error_detected = False
    
    # Revisar si hay mensajes y el último es del asistente
    if state['messages'] and isinstance(state['messages'][-1], AIMessage):
        last_message = state['messages'][-1]
        
        # Revisar si es una respuesta final (no tool call) indicando fallo
        if not getattr(last_message, 'tool_calls', []):
            full_content = ""
            if isinstance(last_message.content, str):
                full_content = last_message.content.lower()
            elif isinstance(last_message.content, list):
                for part in last_message.content:
                    if isinstance(part, dict) and 'text' in part:
                        full_content += part['text'].lower()
                    elif isinstance(part, str):
                        full_content += part.lower()
            
            # Palabras clave que indican fallo
            error_keywords = ["problema técnico", "no puedo calcular", "error", "no es mi especialidad", "faltan parámetros"]
            if any(keyword in full_content for keyword in error_keywords):
                error_count += 1
                possible_error_detected = True
                print(f"⚠️ Detectada falla/incapacidad de agente. Conteo errores: {error_count}/{CIRCUIT_BREAKER_MAX_RETRIES}")
    
    # Comprobar límite de reintentos
    if possible_error_detected and error_count >= CIRCUIT_BREAKER_MAX_RETRIES:
        print(f"🚨 Límite de reintentos alcanzado ({CIRCUIT_BREAKER_MAX_RETRIES}). Forzando FINISH.")
        error_msg = f"Error: No se pudo completar la tarea después de {CIRCUIT_BREAKER_MAX_RETRIES} intentos fallidos del agente. El proceso se ha detenido."
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_node": "FINISH",
            "error_count": error_count
        }
    
    # --- Enrutamiento Normal ---
    supervisor_messages = [HumanMessage(content=supervisor_system_prompt)] + state['messages']
    
    next_node_decision = "FINISH"  # Default
    try:
        route: RouterSchema = supervisor_llm.invoke(supervisor_messages)
        if hasattr(route, 'next_agent'):
            next_node_decision = route.next_agent
        else:
            print("⚠️ Advertencia: Respuesta del supervisor LLM no tuvo 'next_agent'. Usando FINISH.")
            next_node_decision = "FINISH"
        
        print(f"🧭 Supervisor decide ruta: {next_node_decision}")
        
    except Exception as e:
        print(f"❌ ERROR en LLM Supervisor al invocar: {type(e).__name__} - {e}")
        st.warning(f"Advertencia: El supervisor LLM falló ({e}). Finalizando por seguridad.")
        next_node_decision = "FINISH"
    
    # --- Resetear Contador de Errores ---
    previous_node = state.get('next_node', None)
    if not possible_error_detected and (next_node_decision == "FINISH" or next_node_decision != previous_node):
        if error_count > 0:
            print("🔄 Reseteando contador de errores a 0.")
        error_count = 0
    
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
        valid_nodes = list(agent_nodes.keys()) + ["FINISH"]
        if node_to_go not in valid_nodes:
            print(f"⚠️ Alerta Ruteo: Destino inválido '{node_to_go}'. Forzando FINISH.")
            return "FINISH"
        print(f"🚦 Enrutando a: {node_to_go}")
        return node_to_go
    
    # Crear el mapeo para las aristas condicionales
    conditional_map = {name: name for name in agent_nodes}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "Supervisor",
        conditional_router,
        conditional_map
    )
    
    # Añadir aristas de retorno desde cada agente al supervisor
    for name in agent_nodes:
        if name in ["Agente_Ayuda", "Agente_RAG"]:
            # Rutas especiales: Ayuda y RAG van directo al final
            workflow.add_edge(name, END)
        else:
            # Los agentes normales vuelven al supervisor
            workflow.add_edge(name, "Supervisor")
    
    # Compilar el grafo con un checkpointer de memoria
    memory = MemorySaver()
    try:
        compiled_graph = workflow.compile(checkpointer=memory)
        print("✅ Grafo compilado correctamente con MemorySaver.")
        return compiled_graph
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al compilar el grafo: {e}")
        raise e


# Crear la instancia global del grafo compilado
try:
    compiled_graph = build_graph()
except Exception as build_error:
    st.error(f"Error fatal al construir el agente gráfico: {build_error}")
    print(f"❌ Abortando: Falla en build_graph(): {build_error}")
    st.stop()

print("✅ Módulo agent_graph cargado (LangChain 1.0).")