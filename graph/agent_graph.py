# graph/agent_graph.py
"""
Grafo de agentes financieros.
Actualizado para LangChain 1.0+ con:
- Circuit breaker inteligente (rastrea tipos de errores)
- Logging estructurado
- Mejor manejo de estados de error
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import streamlit as st
from datetime import datetime

# Importar de config
from config import CIRCUIT_BREAKER_MAX_RETRIES, CIRCUIT_BREAKER_COOLDOWN

# Importar nodos de agente y supervisor
from agents.financial_agents import (
    supervisor_llm, supervisor_system_prompt,
    agent_nodes, RouterSchema
)

# Importar sistema de routing (LangChain-native con Runnables)
from routing.langchain_routing import create_routing_node
from pathlib import Path

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
    messages: Annotated[list, lambda x, y: x + y]  # Acumula mensajes
    next_node: str  # Nodo a ejecutar a continuación
    error_count: int  # Contador total de errores
    error_types: dict  # Rastrea tipos específicos de errores
    last_error_time: float  # Timestamp del último error
    circuit_open: bool  # Si el circuit breaker está activado

# ========================================
# HELPERS: DETECCIÓN DE ERRORES
# ========================================

def detect_error_type(message: AIMessage) -> str:
    """
    Detecta el tipo de error en un mensaje de agente.
    
    Args:
        message: Mensaje del agente a analizar
    
    Returns:
        Tipo de error: 'tool_failure', 'validation', 'capability', 'unknown'
    """
    # Extraer contenido del mensaje
    full_content = ""
    if isinstance(message.content, str):
        full_content = message.content.lower()
    elif isinstance(message.content, list):
        for part in message.content:
            if isinstance(part, dict) and 'text' in part:
                full_content += part['text'].lower()
            elif isinstance(part, str):
                full_content += part.lower()
    
    # Clasificar por keywords
    if any(kw in full_content for kw in ['error calculando', 'problema técnico', 'fallo herramienta']):
        return 'tool_failure'
    
    if any(kw in full_content for kw in ['faltan parámetros', 'inválido', 'debe ser mayor']):
        return 'validation'
    
    if any(kw in full_content for kw in ['no es mi especialidad', 'no puedo hacer', 'devuelvo al supervisor']):
        return 'capability'
    
    return 'unknown'


def should_open_circuit(error_types: dict, error_count: int) -> bool:
    """
    Determina si el circuit breaker debe activarse.
    
    Args:
        error_types: Diccionario con contadores por tipo de error
        error_count: Contador total de errores
    
    Returns:
        True si debe activarse el circuit breaker
    """
    # Regla 1: Muchos fallos de herramientas (probablemente infraestructura)
    if error_types.get('tool_failure', 0) >= 2:
        logger.warning("🚨 Circuit breaker: Múltiples fallos de herramientas")
        return True
    
    # Regla 2: Muchos errores de validación (usuario no da info correcta)
    if error_types.get('validation', 0) >= 3:
        logger.warning("🚨 Circuit breaker: Múltiples errores de validación")
        return True
    
    # Regla 3: Total de errores excede límite
    if error_count >= CIRCUIT_BREAKER_MAX_RETRIES:
        logger.warning("🚨 Circuit breaker: Límite total de errores alcanzado")
        return True
    
    return False


# ========================================
# NODO SUPERVISOR (CON CIRCUIT BREAKER INTELIGENTE)
# ========================================

def supervisor_node(state: AgentState) -> dict:
    """
    Nodo del supervisor que decide el siguiente paso.
    Implementa circuit breaker inteligente con tracking de tipos de error.
    """
    logger.info("--- SUPERVISOR ---")
    
    # Extraer estado actual
    error_count = state.get('error_count', 0)
    error_types = state.get('error_types', {})
    circuit_open = state.get('circuit_open', False)
    messages = state['messages']
    
    # Si el circuito está abierto, no continuar
    if circuit_open:
        logger.error("⛔ Circuit breaker ACTIVADO - finalizando ejecución")
        error_msg = (
            "🚨 **Sistema detenido por seguridad**\n\n"
            "El agente ha encontrado múltiples errores y se ha detenido para evitar bucles infinitos.\n\n"
            f"**Errores detectados:** {error_count}\n"
            f"**Tipos de error:** {error_types}\n\n"
            "**Acciones sugeridas:**\n"
            "1. Verifica que tu consulta esté completa y bien formulada\n"
            "2. Si es un cálculo, asegúrate de proporcionar todos los parámetros\n"
            "3. Intenta reformular tu pregunta\n"
            "4. Si el problema persiste, contacta al administrador"
        )
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_node": "FINISH",
            "circuit_open": True
        }
    
    # ========================================
    # ANÁLISIS DEL ÚLTIMO MENSAJE
    # ========================================
    
    possible_error_detected = False
    error_type = None
    
    if messages and isinstance(messages[-1], AIMessage):
        last_message = messages[-1]
        
        # Solo revisar mensajes finales (no tool calls intermedios)
        if not getattr(last_message, 'tool_calls', []):
            error_type = detect_error_type(last_message)
            
            # Si detectamos un error conocido
            if error_type in ['tool_failure', 'validation', 'capability']:
                possible_error_detected = True
                error_count += 1
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                logger.warning(
                    f"⚠️ Error detectado - Tipo: {error_type} | "
                    f"Total: {error_count} | Por tipo: {error_types}"
                )
    
    # ========================================
    # VERIFICAR SI ABRIR CIRCUIT BREAKER
    # ========================================
    
    if possible_error_detected and should_open_circuit(error_types, error_count):
        circuit_open = True
        
        # Mensaje personalizado según tipo de error dominante
        max_error_type = max(error_types, key=error_types.get) if error_types else 'unknown'
        
        if max_error_type == 'validation':
            error_msg = (
                "⚠️ **Información Incompleta**\n\n"
                "He intentado procesar tu solicitud varias veces, pero faltan parámetros necesarios.\n\n"
                "**Por favor, proporciona:**\n"
                "- Todos los valores numéricos requeridos\n"
                "- Especifica claramente qué quieres calcular\n"
                "- Revisa que los valores estén en el formato correcto\n\n"
                "Ejemplo válido: *'Calcula VAN: inversión 100k, flujos [30k, 40k, 50k], tasa 10%'*"
            )
        elif max_error_type == 'tool_failure':
            error_msg = (
                "🔧 **Error de Sistema**\n\n"
                "Las herramientas de cálculo están experimentando problemas técnicos.\n\n"
                "**Acciones sugeridas:**\n"
                "1. Intenta de nuevo en unos momentos\n"
                "2. Verifica tu conexión a internet\n"
                "3. Si el problema persiste, contacta al administrador\n\n"
                f"Errores registrados: {error_types}"
            )
        else:
            error_msg = (
                "❌ **Procesamiento Detenido**\n\n"
                f"No pude completar tu solicitud después de {error_count} intentos.\n\n"
                "**Intenta:**\n"
                "1. Reformular tu pregunta de manera más específica\n"
                "2. Dividir tu consulta en pasos más simples\n"
                "3. Usar el comando 'Ayuda' para ver ejemplos\n\n"
                f"Tipos de error: {error_types}"
            )
        
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_node": "FINISH",
            "error_count": error_count,
            "error_types": error_types,
            "circuit_open": True
        }
    
    # ========================================
    # ENRUTAMIENTO (LANGCHAIN-NATIVE)
    # ========================================

    next_node_decision = "FINISH"  # Default
    routing_method = "unknown"
    routing_confidence = 0.0

    try:
        # Usar nodo de routing (LangChain Runnables)
        global ROUTING_NODE

        if ROUTING_NODE:
            # Ejecutar nodo de routing (usa RunnableBranch internamente)
            result = ROUTING_NODE(state)

            next_node_decision = result.get('next_node', 'FINISH')
            routing_method = result.get('routing_method', 'unknown')
            routing_confidence = result.get('routing_confidence', 0.0)

            logger.info(
                f"🧭 Routing decision: {next_node_decision} "
                f"(method={routing_method}, conf={routing_confidence:.2f})"
            )
        else:
            # Fallback a supervisor directo (si routing no está inicializado)
            logger.warning("⚠️ ROUTING_NODE no inicializado, usando supervisor directo")

            supervisor_messages = [HumanMessage(content=supervisor_system_prompt)] + messages
            route: RouterSchema = supervisor_llm.invoke(supervisor_messages)

            if hasattr(route, 'next_agent'):
                next_node_decision = route.next_agent
            else:
                logger.warning("⚠️ Respuesta del supervisor sin 'next_agent'. Usando FINISH.")
                next_node_decision = "FINISH"

            routing_method = "llm_direct"
            routing_confidence = 0.95

            logger.info(f"🧭 Supervisor decide: {next_node_decision}")

    except Exception as e:
        logger.error(f"❌ Error en routing: {e}", exc_info=True)
        st.warning(f"Advertencia: El routing falló ({e}). Finalizando.")
        next_node_decision = "FINISH"
        routing_method = "error_fallback"
        routing_confidence = 0.0
    
    # ========================================
    # RESETEAR CONTADOR SI TIENE ÉXITO
    # ========================================
    
    previous_node = state.get('next_node', None)
    
    # Si no hubo error y cambió de nodo o finalizó, resetear contadores
    if not possible_error_detected:
        if next_node_decision == "FINISH" or next_node_decision != previous_node:
            if error_count > 0:
                logger.info("🔄 Tarea exitosa - reseteando contadores de error")
                error_count = 0
                error_types = {}
    
    return {
        "next_node": next_node_decision,
        "error_count": error_count,
        "error_types": error_types,
        "circuit_open": circuit_open,
        "last_error_time": datetime.now().timestamp() if possible_error_detected else 0,
        # Metadata del sistema de routing
        "routing_method": routing_method,
        "routing_confidence": routing_confidence
    }


# ========================================
# CONSTRUCCIÓN DEL GRAFO
# ========================================

def build_graph():
    """Construye y compila el grafo LangGraph."""
    logger.info("🏗️ Construyendo grafo de agentes...")
    
    workflow = StateGraph(AgentState)
    
    # Añadir nodo supervisor
    workflow.add_node("Supervisor", supervisor_node)
    logger.debug("   Nodo 'Supervisor' agregado")
    
    # Añadir nodos de agentes
    for name, node in agent_nodes.items():
        workflow.add_node(name, node)
        logger.debug(f"   Nodo '{name}' agregado")
    
    # Establecer punto de entrada
    workflow.set_entry_point("Supervisor")
    
    # Función de enrutamiento condicional
    def conditional_router(state: AgentState) -> str:
        """Enruta basado en la decisión del supervisor."""
        node_to_go = state.get("next_node")
        valid_nodes = list(agent_nodes.keys()) + ["FINISH"]
        
        if node_to_go not in valid_nodes:
            logger.warning(f"⚠️ Destino inválido '{node_to_go}'. Forzando FINISH.")
            return "FINISH"
        
        logger.debug(f"🚦 Enrutando a: {node_to_go}")
        return node_to_go
    
    # Crear mapeo para aristas condicionales
    conditional_map = {name: name for name in agent_nodes}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "Supervisor",
        conditional_router,
        conditional_map
    )
    
    # Aristas de retorno: agentes → supervisor
    for name in agent_nodes:
        if name in ["Agente_Ayuda", "Agente_Sintesis_RAG"]:
            # Ayuda y Síntesis van directo al final (no vuelven al supervisor)
            workflow.add_edge(name, END)
            logger.debug(f"   {name} → END")
        elif name == "Agente_RAG":
            # RAG va DIRECTO a SÍNTESIS
            workflow.add_edge(name, "Agente_Sintesis_RAG")
            logger.debug(f"   {name} → Agente_Sintesis_RAG (Directo)")
        else:
            # Agentes normales vuelven al supervisor
            workflow.add_edge(name, "Supervisor")
            logger.debug(f"   {name} → Supervisor")
    
    # Compilar con checkpointer
    memory = MemorySaver()
    try:
        compiled_graph = workflow.compile(checkpointer=memory)
        logger.info("✅ Grafo compilado correctamente con MemorySaver")
        return compiled_graph
    except Exception as e:
        logger.error(f"❌ Error crítico al compilar grafo: {e}", exc_info=True)
        raise e


# ========================================
# SISTEMA DE ROUTING (LANGCHAIN-NATIVE)
# ========================================

ROUTING_NODE = None

def initialize_routing_system():
    """
    Inicializa el sistema de routing usando herramientas nativas de LangChain.

    ENFOQUE LANGCHAIN-NATIVE:
    - Usa RunnableBranch para routing condicional (idiomático de LangChain)
    - Usa RunnableLambda para wrappear lógica custom
    - Compatible 100% con LCEL (LangChain Expression Language)
    - No usa clases custom - todo son Runnables nativos

    Returns:
        Nodo de routing configurado (función compatible con LangGraph)
    """
    global ROUTING_NODE

    logger.info("🔧 Inicializando sistema de routing (LangChain-native)...")

    try:
        # Ruta al archivo de configuración YAML
        config_path = Path(__file__).parent.parent / "config" / "routing_patterns.yaml"

        # Crear nodo de routing usando RunnableBranch
        # Patrón idiomático de LangChain: composición de Runnables
        ROUTING_NODE = create_routing_node(
            supervisor_llm=supervisor_llm,
            supervisor_prompt=supervisor_system_prompt,  # ← NO SE MODIFICA
            threshold=0.8,  # Umbral ajustable
            config_path=str(config_path) if config_path.exists() else None
        )

        logger.info("  ✅ Routing node creado (RunnableBranch + RunnableLambda)")
        logger.info("🚀 Sistema de routing LangChain-native ACTIVO")

        return ROUTING_NODE

    except Exception as e:
        logger.error(f"❌ Error inicializando routing system: {e}", exc_info=True)
        logger.warning("⚠️ Continuando con supervisor directo (sin optimización)")
        ROUTING_NODE = None
        return None


# ========================================
# INSTANCIA GLOBAL
# ========================================

try:
    compiled_graph = build_graph()
    logger.info("✅ Grafo global inicializado")
except Exception as build_error:
    logger.error(f"❌ Error fatal en build_graph: {build_error}", exc_info=True)
    st.error(f"Error fatal al construir el agente gráfico: {build_error}")
    st.stop()

# Inicializar sistema de routing
try:
    initialize_routing_system()
except Exception as routing_error:
    logger.error(f"❌ Error fatal en routing system: {routing_error}", exc_info=True)
    logger.warning("⚠️ Sistema continuará con routing básico")

logger.info("✅ Módulo agent_graph cargado (LangChain 1.0 + Circuit Breaker + Routing LangChain-native)")