# graph/agent_graph.py
"""
Grafo de agentes financieros.
Actualizado: Sincronizado con protocolos de financial_agents.py
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import streamlit as st
from datetime import datetime

# Importar de config
from config import (
    CIRCUIT_BREAKER_MAX_RETRIES,
    CIRCUIT_BREAKER_COOLDOWN,
    ENABLE_POSTGRES_PERSISTENCE,
    get_postgres_uri
)

# Importar nodos de agente y supervisor
from agents.financial_agents import (
    supervisor_llm, supervisor_system_prompt,
    agent_nodes, RouterSchema
)

# Routing eliminado - ahora usamos clasificación LLM simple

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('graph')
except ImportError:
    import logging
    logger = logging.getLogger('graph')

# ========================================
# ESTADO DEL GRAFO
# ========================================

class AgentState(TypedDict):
    """Estado del grafo con tracking de errores mejorado."""
    messages: Annotated[list, lambda x, y: x + y]
    next_node: str
    error_count: int
    error_types: dict
    last_error_time: float
    circuit_open: bool

# ========================================
# HELPERS: DETECCIÓN DE ERRORES (ACTUALIZADO)
# ========================================

def detect_error_type(message: AIMessage) -> str:
    """
    Detecta el tipo de error en un mensaje de agente.
    Sincronizado con las etiquetas de financial_agents.py
    """
    # Extraer contenido del mensaje
    full_content = ""
    if isinstance(message.content, str):
        full_content = message.content
    elif isinstance(message.content, list):
        for part in message.content:
            if isinstance(part, dict) and 'text' in part:
                full_content += part['text']
            elif isinstance(part, str):
                full_content += part
    
    # Normalizar a mayúsculas para buscar etiquetas
    content_upper = full_content.upper()
    
    # ✅ DETECTAR ÉXITO
    if 'TAREA_COMPLETADA' in content_upper:
        return 'success'
    
    # ❌ DETECTAR ERRORES BLOQUEANTES (Técnicos o Lógicos)
    if 'ERROR_BLOQUEANTE' in content_upper:
        return 'tool_failure'  # O 'blocking_error', lo mapeamos a tool_failure para simplificar
    
    # ⚠️ DETECTAR FALTA DE DATOS (Validación)
    if 'FALTAN_DATOS' in content_upper:
        return 'validation'
        
    # Fallback para errores no capturados por protocolo (legacy)
    content_lower = full_content.lower()
    if any(kw in content_lower for kw in ['error calculando', 'problema técnico', 'fallo herramienta']):
        return 'tool_failure'
    
    return 'unknown'


def should_open_circuit(error_types: dict, error_count: int) -> bool:
    """Determina si el circuit breaker debe activarse."""
    if error_types.get('tool_failure', 0) >= 2:
        logger.warning("🚨 Circuit breaker: Múltiples fallos de herramientas")
        return True
    
    if error_types.get('validation', 0) >= 3:
        logger.warning("🚨 Circuit breaker: Múltiples errores de validación")
        return True
    
    if error_count >= CIRCUIT_BREAKER_MAX_RETRIES:
        logger.warning("🚨 Circuit breaker: Límite total de errores alcanzado")
        return True
    
    return False


# ========================================
# NODO SUPERVISOR (HELPERS)
# ========================================

def _check_circuit_breaker_status(state: AgentState) -> dict:
    """Verifica el estado del circuit breaker."""
    circuit_open = state.get('circuit_open', False)
    error_count = state.get('error_count', 0)
    error_types = state.get('error_types', {})

    if circuit_open:
        logger.error("⛔ Circuit breaker ACTIVADO - finalizando ejecución")
        error_msg = (
            "🚨 **Sistema detenido por seguridad**\n\n"
            "El agente ha detectado inconsistencias repetidas.\n"
            f"**Errores:** {error_count} | **Tipos:** {error_types}\n\n"
            "Intenta reformular tu pregunta o proporcionar todos los datos necesarios."
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_node": "FINISH",
            "circuit_open": True
        }
    return None


def _analyze_last_message(messages: list) -> tuple:
    """Analiza el último mensaje para detectar errores."""
    possible_error_detected = False
    error_type = None
    error_count_delta = 0
    error_types_update = {}

    if messages and isinstance(messages[-1], AIMessage):
        last_message = messages[-1]
        if not getattr(last_message, 'tool_calls', []):
            error_type = detect_error_type(last_message)

            if error_type == 'success':
                logger.info("✅ Tarea completada exitosamente")
                possible_error_detected = False
            elif error_type in ['tool_failure', 'validation', 'capability']:
                possible_error_detected = True
                error_count_delta = 1
                error_types_update[error_type] = 1
                logger.warning(f"⚠️ Error detectado - Tipo: {error_type}")

    return possible_error_detected, error_type, error_count_delta, error_types_update


def _handle_circuit_breaker_activation(error_types: dict, error_count: int) -> dict:
    """Genera respuesta de activación del circuit breaker."""
    max_error_type = max(error_types, key=error_types.get) if error_types else 'unknown'

    if max_error_type == 'validation':
        error_msg = "⚠️ **Faltan Datos**: Por favor proporciona todos los parámetros requeridos."
    elif max_error_type == 'tool_failure':
        error_msg = "🔧 **Error Técnico**: Las herramientas no están respondiendo correctamente."
    else:
        error_msg = f"❌ **Procesamiento Detenido**: Demasiados reintentos ({error_count})."

    return {
        "messages": [AIMessage(content=error_msg)],
        "next_node": "FINISH",
        "circuit_open": True
    }


def _execute_routing_decision(state: AgentState, messages: list) -> tuple:
    """Ejecuta la lógica de routing usando supervisor LLM directo."""
    next_node_decision = "FINISH"
    routing_method = "supervisor_llm"
    routing_confidence = 0.95

    try:
        from agents.financial_agents import supervisor_llm, supervisor_system_prompt

        supervisor_messages = [HumanMessage(content=supervisor_system_prompt)] + messages
        route = supervisor_llm.invoke(supervisor_messages)

        next_node_decision = route.next_agent if hasattr(route, 'next_agent') else "FINISH"
        logger.info(f"🧭 Supervisor LLM decide: {next_node_decision}")

    except Exception as e:
        logger.error(f"❌ Error en supervisor: {e}", exc_info=True)
        next_node_decision = "FINISH"

    return next_node_decision, routing_method, routing_confidence


# ========================================
# NODO SUPERVISOR (PRINCIPAL)
# ========================================

def supervisor_node(state: AgentState) -> dict:
    """Supervisor con clasificación simple teoría/práctica/ayuda."""
    logger.info("--- SUPERVISOR (CLASIFICACIÓN SIMPLE) ---")

    messages = state.get('messages', [])
    error_count = state.get('error_count', 0)
    error_types = state.get('error_types', {})

    # 1. Chequeo Circuit Breaker (mantener lógica actual)
    cb_status = _check_circuit_breaker_status(state)
    if cb_status:
        return cb_status

    # 2. Si último mensaje no es del usuario, analizar errores
    if not messages or not isinstance(messages[-1], HumanMessage):
        is_error, error_type, delta_count, delta_types = _analyze_last_message(messages)

        if is_error:
            error_count += delta_count
            for k, v in delta_types.items():
                error_types[k] = error_types.get(k, 0) + v

            if should_open_circuit(error_types, error_count):
                activation = _handle_circuit_breaker_activation(error_types, error_count)
                activation.update({"error_count": error_count, "error_types": error_types})
                return activation

        return {"next_node": "FINISH", "error_count": error_count, "error_types": error_types}

    # 3. CLASIFICACIÓN SIMPLE (NUEVA LÓGICA)
    user_query = messages[-1].content

    prompt_clasificacion = """Clasifica esta consulta financiera en UNA categoría:

**TEORICA**: Si pregunta conceptos, definiciones, explicaciones
- Palabras clave: "qué es", "explica", "define", "concepto", "significado", "what is", "explain", "define"
- Ejemplo: "¿Qué es el WACC?", "Explica duration modificada"

**PRACTICA**: Si solicita cálculos, tiene números, pide resultados específicos
- Palabras clave: "calcula", "determina", "obtén", "encuentra", contiene números
- Ejemplo: "Calcula VAN: inversión 100k, flujos [30k,40k], tasa 10%"

**AYUDA**: Si pregunta qué puede hacer el sistema o pide ayuda
- Palabras clave: "ayuda", "qué puedes hacer", "ejemplos", "help"
- Ejemplo: "¿Qué puedes calcular?", "Ayuda"

Consulta: "{query}"

Responde SOLO UNA PALABRA en mayúsculas: TEORICA, PRACTICA o AYUDA
No des explicaciones, solo la categoría."""

    clasificacion_msg = prompt_clasificacion.format(query=user_query)

    try:
        # Usar LLM con temperatura 0 para determinismo
        from config import get_llm
        llm_clasificador = get_llm()
        clasificacion = llm_clasificador.invoke(clasificacion_msg).content.strip().upper()
        logger.info(f"🏷️ Clasificación: {clasificacion}")
    except Exception as e:
        logger.error(f"❌ Error en clasificación: {e}")
        clasificacion = "PRACTICA"  # Fallback seguro

    # 4. ROUTING BASADO EN CLASIFICACIÓN
    if "TEORICA" in clasificacion or "TEÓRICA" in clasificacion:
        logger.info("📚 Ruta: TEORICA → Agente_RAG")
        return {
            "next_node": "Agente_RAG",
            "error_count": 0,  # Reset en nuevo intent
            "error_types": {},
            "routing_method": "clasificacion_llm",
            "routing_confidence": 0.95
        }

    elif "AYUDA" in clasificacion:
        logger.info("ℹ️ Ruta: AYUDA → Agente_Ayuda")
        return {
            "next_node": "Agente_Ayuda",
            "error_count": 0,
            "error_types": {},
            "routing_method": "clasificacion_llm",
            "routing_confidence": 0.95
        }

    else:  # PRACTICA (default)
        logger.info("🔢 Ruta: PRACTICA → Supervisor decide agente especialista")

        # Usar lógica supervisor original para decidir agente especialista
        next_node, method, confidence = _execute_routing_decision(state, messages)

        # Reset errores si routing cambió
        prev_node = state.get('next_node')
        if next_node == "FINISH" or next_node != prev_node:
            if error_count > 0:
                error_count = 0
                error_types = {}

        return {
            "next_node": next_node,
            "error_count": error_count,
            "error_types": error_types,
            "routing_method": "clasificacion_practica",
            "routing_confidence": confidence
        }


# ========================================
# CONSTRUCCIÓN DEL GRAFO
# ========================================

def build_graph():
    """Construye el grafo con persistencia."""
    logger.info("🏗️ Construyendo grafo...")
    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("Supervisor", supervisor_node)
    for name, node in agent_nodes.items():
        workflow.add_node(name, node)

    # Edges
    workflow.set_entry_point("Supervisor")
    
    def conditional_router(state):
        dest = state.get("next_node")
        return dest if dest in agent_nodes or dest == "FINISH" else "FINISH"

    conditional_map = {name: name for name in agent_nodes}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges("Supervisor", conditional_router, conditional_map)

    # Retornos
    for name in agent_nodes:
        if name in ["Agente_Ayuda", "Agente_RAG"]: 
            workflow.add_edge(name, END) # RAG y Ayuda terminan directo
        elif name == "Agente_Sintesis_RAG":
            workflow.add_edge(name, END)
        else:
            workflow.add_edge(name, "Supervisor")

    # Persistencia
    checkpointer = MemorySaver()
    if ENABLE_POSTGRES_PERSISTENCE:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg_pool
            pool = psycopg_pool.ConnectionPool(conninfo=get_postgres_uri(), min_size=1, max_size=10)
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            logger.info("✅ PostgreSQL Persistence ON")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL falló ({e}), usando MemorySaver")

    return workflow.compile(checkpointer=checkpointer)


# ========================================
# INICIALIZACIÓN DEL GRAFO
# ========================================

# Inicialización Global
try:
    compiled_graph = build_graph()
    logger.info("✅ Grafo compilado (routing simplificado con clasificación LLM)")
except Exception as e:
    logger.error(f"🔥 Error Fatal en Graph Init: {e}")
    st.error("Error crítico del sistema.")
    st.stop()