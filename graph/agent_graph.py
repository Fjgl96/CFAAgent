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
# graph/agent_graph.py

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage  # <--- Agregar SystemMessage
from pydantic import BaseModel, Field  # <--- Nuevo
from typing import Literal             # <--- Nuevo
from config import get_llm             # <--- Asegurar que esto esté importado



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
# graph/agent_graph.py

# === CLASE PARA SALIDA ESTRUCTURADA (SUPERVISOR v2) ===
class DecisionSupervisor(BaseModel):
    """Estructura de decisión del supervisor para clasificación y optimización."""
    categoria: Literal["TEORICA", "PRACTICA", "AYUDA"] = Field(
        description="Categoría de la intención del usuario: TEORICA (conceptos), PRACTICA (cálculos), AYUDA (soporte)."
    )
    query_optimizada: str = Field(
        description="La consulta del usuario reescrita y optimizada para búsqueda vectorial (traducida al inglés si es necesario, con términos técnicos CFA y sin ruido)."
    )
    razonamiento: str = Field(
        description="Breve justificación de la clasificación y optimización."
    )
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

# graph/agent_graph.py

def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor Inteligente v2: Clasifica y Optimiza en un solo paso (Single-Shot).
    """
    logger.info("--- SUPERVISOR (CLASIFICACIÓN + OPTIMIZACIÓN) ---")

    messages = state.get('messages', [])
    error_count = state.get('error_count', 0)
    error_types = state.get('error_types', {})

    # 1. Chequeo Circuit Breaker (Lógica existente)
    cb_status = _check_circuit_breaker_status(state)
    if cb_status:
        return cb_status

    # 2. Análisis de errores previos (Lógica existente)
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

    # === 3. CLASIFICACIÓN Y OPTIMIZACIÓN UNIFICADA ===
    
    user_query = messages[-1].content
    
    # Prompt optimizado para RAG y Clasificación simultánea
    prompt_sistema = """Eres el Supervisor Senior de un sistema de Agentes Financieros CFA.
    Tu misión es doble:
    1. CLASIFICAR la intención:
       - **TEORICA**: Conceptos, definiciones, "qué es", "explica". (Requiere RAG)
       - **PRACTICA**: Cálculos numéricos, "calcula", "determina". (Requiere Especialista)
       - **AYUDA**: "¿Qué puedes hacer?", "Ayuda".

    2. OPTIMIZAR la consulta para búsqueda vectorial (Elasticsearch):
       - Si es TEORICA: Traduce al INGLÉS (el material CFA está en inglés), elimina palabras vacías, añade sinónimos técnicos.
         Ej: "¿Qué es el WACC?" -> "WACC definition weighted average cost of capital formula components"
       - Si es PRACTICA: Extrae y limpia los parámetros numéricos y el objetivo.
       - Si es AYUDA: Déjala simple.
    """
    
    try:
        # Usamos structured output para garantizar el formato JSON y la optimización
        llm_supervisor = get_llm().with_structured_output(DecisionSupervisor)
        
        decision = llm_supervisor.invoke([
            SystemMessage(content=prompt_sistema),
            HumanMessage(content=user_query)
        ])
        
        logger.info(f"🏷️  Categoría: {decision.categoria}")
        logger.info(f"🔍 Query Original: {user_query}")
        logger.info(f"🚀 Query Optimizada: {decision.query_optimizada}")
        
    except Exception as e:
        logger.error(f"❌ Error en supervisor estructurado: {e}")
        # Fallback de seguridad
        decision = DecisionSupervisor(
            categoria="PRACTICA", 
            query_optimizada=user_query, 
            razonamiento="Error en clasificación, fallback a práctica."
        )

    # === 4. ENRUTAMIENTO Y GESTIÓN DE ESTADO ===
    
    if decision.categoria == "TEORICA":
        logger.info("📚 Ruta: TEORICA -> Agente_RAG (Inyectando query optimizada)")
        
        # NOTA TÉCNICA CRÍTICA:
        # Tu grafo usa un reducer 'messages: x + y' (concatenación).
        # NO podemos reemplazar la lista entera o duplicaríamos el historial.
        # Solución: Añadimos la query optimizada como un NUEVO mensaje.
        # El Agente_RAG leerá este último mensaje como la instrucción más reciente.
        
        return {
            "next_node": "Agente_RAG",
            "messages": [HumanMessage(content=decision.query_optimizada)], # Se apende al historial
            "error_count": 0,
            "error_types": {}
        }

    elif decision.categoria == "AYUDA":
        logger.info("ℹ️ Ruta: AYUDA -> Agente_Ayuda")
        return {
            "next_node": "Agente_Ayuda",
            "error_count": 0,
            "error_types": {}
        }

    else: # PRACTICA (Default)
        logger.info("🔢 Ruta: PRACTICA -> Supervisor decide especialista")
        
        # Para casos prácticos, usamos la lógica de routing especialista existente.
        # Nota: Pasamos el estado original, los especialistas suelen preferir
        # ver la query original con los números tal cual los escribió el usuario.
        next_node, method, confidence = _execute_routing_decision(state, messages)
        
        # Reset de errores si cambiamos de nodo
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
            
            # 🔧 CORRECCIÓN: Configurar autocommit=True para permitir operaciones DDL (como crear índices)
            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
            }

            pool = psycopg_pool.ConnectionPool(
                conninfo=get_postgres_uri(), 
                min_size=1, 
                max_size=10,
                kwargs=connection_kwargs  # <-- Esto soluciona el error de transacción
            )
            
            checkpointer = PostgresSaver(pool)
            checkpointer.setup() # Crea las tablas si no existen
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