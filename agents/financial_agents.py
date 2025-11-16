# agents/financial_agents.py
"""
Agentes especializados financieros.
Actualizado para LangChain 1.0+ con RAG integrado y logging estructurado.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from typing import Literal
from pydantic import BaseModel, Field

# Importar LLM de config
from config import get_llm

# Importar herramientas individuales
from tools.financial_tools import (
    _calcular_valor_presente_bono, _calcular_van, _calcular_wacc,
    _calcular_gordon_growth, _calcular_capm, _calcular_sharpe_ratio,
    _calcular_opcion_call
)
from tools.help_tools import obtener_ejemplos_de_uso

# Importar RAG
from rag.financial_rag_elasticsearch import buscar_documentacion_financiera

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('agents')
except ImportError:
    import logging
    logger = logging.getLogger('agents')

llm = get_llm()

# ========================================
# PLACEHOLDER DE MENSAJES
# ========================================

messages_placeholder = MessagesPlaceholder(variable_name="messages")

# ========================================
# NODOS ESPECIALES
# ========================================

def nodo_ayuda_directo(state: dict) -> dict:
    """Nodo simple que llama a la herramienta de ayuda directamente."""
    logger.info("📖 Nodo Ayuda invocado")
    try:
        guia_de_preguntas = obtener_ejemplos_de_uso.invoke({})
        logger.debug("✅ Guía de ayuda generada")
        return {
            "messages": [AIMessage(content=guia_de_preguntas)]
        }
    except Exception as e:
        logger.error(f"❌ Error en nodo_ayuda: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error al obtener la guía de ayuda: {e}")]
        }


def nodo_rag(state: dict) -> dict:
    """Nodo que consulta la documentación CFA usando RAG."""
    logger.info("📚 Agente RAG invocado")
    
    # Extraer última pregunta del usuario
    messages = state.get("messages", [])
    if not messages:
        logger.error("❌ Estado sin mensajes en nodo RAG")
        return {
            "messages": [AIMessage(content="Error: No hay mensajes en el estado.")]
        }
    
    last_message = messages[-1]
    
    # Extraer contenido
    if hasattr(last_message, 'content'):
        consulta = last_message.content
    else:
        consulta = str(last_message)
    
    logger.info(f"🔍 Consulta CFA: {consulta[:100]}...")
    
    # Buscar en documentación usando RAG
    try:
        resultado = buscar_documentacion_financiera.invoke({"consulta": consulta})
        logger.info("✅ Respuesta RAG generada")
        
        return {
            "messages": [AIMessage(content=resultado)]
        }
    
    except Exception as e:
        logger.error(f"❌ Error en RAG: {e}", exc_info=True)
        return {
            "messages": [AIMessage(
                content=f"Error al buscar en la documentación: {e}"
            )]
        }


# ========================================
# HELPER: CREAR AGENTE ESPECIALISTA
# ========================================

def crear_agente_especialista(llm_instance, tools_list, system_prompt_text):
    """
    Función helper para crear un agente reactivo con prompt de sistema.
    
    Args:
        llm_instance: Instancia del LLM
        tools_list: Lista de herramientas disponibles
        system_prompt_text: Prompt del sistema para el agente
    
    Returns:
        Agente compilado
    """
    if not tools_list or not all(hasattr(t, 'invoke') for t in tools_list):
        raise ValueError("tools_list debe contener al menos una herramienta válida (Runnable).")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        messages_placeholder,
    ])
    
    # LangChain 1.0: create_react_agent de langgraph.prebuilt
    agent = create_react_agent(llm_instance, tools_list, state_modifier=prompt)
    
    logger.debug(f"✅ Agente creado con {len(tools_list)} herramientas")
    
    return agent


# ========================================
# PROMPTS DE AGENTES ESPECIALISTAS
# ========================================

PROMPT_RENTA_FIJA = """Eres un especialista en Renta Fija.
Tu único trabajo es usar SÓLO tu herramienta 'calcular_valor_bono'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa.
Extrae los parámetros necesarios de la solicitud o del historial y llama a tu herramienta.
Si te piden algo que no puedes hacer con tu herramienta, di "No es mi especialidad, devuelvo al supervisor."."""

PROMPT_FIN_CORP = """Eres un especialista en Finanzas Corporativas.
Tu trabajo es usar SÓLO tus herramientas 'calcular_van' y 'calcular_wacc'.

**PROCESO A SEGUIR:**
1. Revisa el historial para encontrar los parámetros necesarios para tu herramienta.
2. Llama a la herramienta adecuada ('calcular_van' o 'calcular_wacc').
3. **NUNCA respondas usando tu conocimiento general.**
4. Una vez que la herramienta te devuelva un JSON con el resultado, formula tu respuesta.
5. **IMPORTANTE: En tu respuesta, NO repitas los inputs del usuario**. Simplemente reporta el resultado y la interpretación.
6. **Al final de tu respuesta, DEBES escribir: "Tarea completada, devuelvo al supervisor."**

Si te piden algo que no puedes hacer con tus herramientas, di "No es mi especialidad, devuelvo al supervisor."."""

PROMPT_EQUITY = """Eres un especialista en valoración de acciones (Equity).
Tu único trabajo es usar SÓLO tu herramienta 'calcular_gordon_growth'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes. Si una tarea anterior calculó un valor necesario (como Ke), usa ESE valor.
Extrae el 'dividendo_prox_periodo' (D1), la 'tasa_descuento_equity' (Ke) y la 'tasa_crecimiento_dividendos' (g).
Llama a tu herramienta con estos 3 parámetros.
Si no puedes encontrar los 3 parámetros, di "Faltan parámetros, devuelvo al supervisor."."""

PROMPT_PORTAFOLIO = """Eres un especialista en Gestión de Portafolios.
Tu trabajo es usar SÓLO tus herramientas 'calcular_capm' y 'calcular_sharpe_ratio'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa.
Extrae los parámetros necesarios de la solicitud o del historial y llama a la herramienta adecuada.
Si te piden una tarea para la que no tienes herramienta, **NO respondas a esa parte**.
Responde SÓLO la parte que SÍ puedes hacer con tus herramientas.
Luego, di "Tarea parcial completada, devuelvo al supervisor."."""

PROMPT_DERIVADOS = """Eres un especialista en instrumentos derivados.
Tu único trabajo es usar SÓLO tu herramienta 'calcular_opcion_call'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa.
Extrae los parámetros necesarios (S, K, T, r, sigma) de la solicitud o del historial y llama a tu herramienta.
Si te piden algo que no puedes hacer con tu herramienta, di "No es mi especialidad, devuelvo al supervisor."."""

# ========================================
# CREACIÓN DE AGENTES
# ========================================

logger.info("🏗️ Inicializando agentes especialistas...")

try:
    agent_renta_fija = crear_agente_especialista(
        llm, [_calcular_valor_presente_bono], PROMPT_RENTA_FIJA
    )
    logger.debug("✅ Agente Renta Fija creado")
    
    agent_fin_corp = crear_agente_especialista(
        llm, [_calcular_van, _calcular_wacc], PROMPT_FIN_CORP
    )
    logger.debug("✅ Agente Finanzas Corporativas creado")
    
    agent_equity = crear_agente_especialista(
        llm, [_calcular_gordon_growth], PROMPT_EQUITY
    )
    logger.debug("✅ Agente Equity creado")
    
    agent_portafolio = crear_agente_especialista(
        llm, [_calcular_capm, _calcular_sharpe_ratio], PROMPT_PORTAFOLIO
    )
    logger.debug("✅ Agente Portafolio creado")
    
    agent_derivados = crear_agente_especialista(
        llm, [_calcular_opcion_call], PROMPT_DERIVADOS
    )
    logger.debug("✅ Agente Derivados creado")
    
    logger.info("✅ Todos los agentes creados exitosamente")

except Exception as e:
    logger.error(f"❌ ERROR CRÍTICO al crear agentes: {e}", exc_info=True)
    import streamlit as st
    st.error(f"Error inicializando los agentes: {e}")
    st.stop()

# ========================================
# DICCIONARIO DE NODOS
# ========================================

agent_nodes = {
    "Agente_Renta_Fija": agent_renta_fija,
    "Agente_Finanzas_Corp": agent_fin_corp,
    "Agente_Equity": agent_equity,
    "Agente_Portafolio": agent_portafolio,
    "Agente_Derivados": agent_derivados,
    "Agente_Ayuda": nodo_ayuda_directo,
    "Agente_RAG": nodo_rag,
}

logger.info(f"📋 {len(agent_nodes)} agentes registrados")

# ========================================
# SUPERVISOR
# ========================================

class RouterSchema(BaseModel):
    """Elige el siguiente agente a llamar o finaliza."""
    next_agent: Literal[tuple(list(agent_nodes.keys()) + ["FINISH"])] = Field(
        description="El nombre del agente especialista para la tarea. Elige 'FINISH' si la solicitud fue completamente respondida."
    )

# Configurar el LLM supervisor
try:
    supervisor_llm = llm.with_structured_output(RouterSchema)
    logger.info("✅ Supervisor LLM configurado")
except Exception as e:
    logger.error(f"❌ ERROR configurando supervisor: {e}", exc_info=True)
    import streamlit as st
    st.error(f"Error configurando el supervisor: {e}")
    st.stop()

# ========================================
# PROMPT DEL SUPERVISOR
# ========================================

supervisor_system_prompt = """Eres un supervisor MUY eficiente de un equipo de analistas financieros. Tu única función es leer el último mensaje del usuario Y el historial de la conversación para decidir qué especialista debe actuar A CONTINUACIÓN. No respondas tú mismo. SOLO elige el siguiente paso.

Especialistas y sus ÚNICAS herramientas:
- Agente_Renta_Fija: `calcular_valor_bono`
- Agente_Finanzas_Corp: `calcular_van`, `calcular_wacc`
- Agente_Equity: `calcular_gordon_growth`
- Agente_Portafolio: `calcular_capm`, `calcular_sharpe_ratio`
- Agente_Derivados: `calcular_opcion_call`
- Agente_Ayuda: `obtener_ejemplos_de_uso`
- Agente_RAG: `buscar_documentacion_financiera`

PROCESO DE DECISIÓN:
**1. PRIORIDAD MÁXIMA: Revisa el último mensaje del usuario.**

**2. DETECCIÓN DE CONSULTAS RAG:**
Si el usuario hace preguntas teóricas o de conceptos como:
- "qué dice el material CFA sobre..."
- "según el CFA..."
- "explica el concepto de..."
- "busca en la documentación..."
- "qué es [concepto] según CFA..."
→ Elige 'Agente_RAG'

**3. DETECCIÓN DE CONSULTAS DE AYUDA:**
Si el usuario usa palabras como "ayuda", "ejemplos", "qué puedes hacer", "cómo funciona":
→ Elige 'Agente_Ayuda'

**4. PARA CÁLCULOS NUMÉRICOS:**
Elige el agente especialista apropiado según la herramienta necesaria.

**5. Si el último agente completó su parte Y no quedan tareas pendientes:**
→ Elige 'FINISH'

**6. Si el último agente indicó un error y no hay forma de continuar:**
→ Elige 'FINISH'

SOLO devuelve el nombre del agente o "FINISH".
"""

logger.info("✅ Módulo financial_agents cargado (LangChain 1.0 + RAG + Logging)")