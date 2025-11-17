# agents/financial_agents.py
"""
Agentes especializados financieros.
Actualizado para LangGraph 1.0+ (versión moderna).
"""

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
# HELPER: CREAR AGENTE ESPECIALISTA (LANGGRAPH 1.0+)
# ========================================
def nodo_sintesis_rag(state: dict) -> dict:
    """
    Nodo que toma el contexto (del historial) y genera una síntesis.
    """
    logger.info("🧠 Nodo Síntesis RAG invocado")
    messages = state.get("messages", [])
    if not messages:
        logger.error("❌ Estado sin mensajes en nodo Síntesis")
        return {"messages": [AIMessage(content="Error: No hay mensajes en el estado.")]}
    
    try:
        # 1. Bindea el LLM con el prompt de síntesis
        llm_sintesis = llm.bind(system=PROMPT_SINTESIS_RAG)
        
        # 2. Pasa el historial de mensajes (que incluye la pregunta Y el contexto del RAG)
        #    al LLM bindeado con el prompt de síntesis.
        respuesta_sintetizada = llm_sintesis.invoke(messages)
        
        logger.info("✅ Respuesta RAG sintetizada")
        return {
            "messages": [respuesta_sintetizada] # La salida de invoke es una AIMessage
        }
    except Exception as e:
        logger.error(f"❌ Error en nodo_sintesis_rag: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"Error al sintetizar la respuesta: {e}")]}

def crear_agente_especialista(llm_instance, tools_list, system_prompt_text):
    """
    Función helper para crear un agente reactivo con prompt de sistema.
    COMPATIBLE CON LANGGRAPH 1.0.1+ (USA BIND)
    
    Args:
        llm_instance: Instancia del LLM
        tools_list: Lista de herramientas disponibles
        system_prompt_text: Prompt del sistema para el agente
    
    Returns:
        Agente compilado
    """
    if not tools_list or not all(hasattr(t, 'invoke') for t in tools_list):
        raise ValueError("tools_list debe contener al menos una herramienta válida (Runnable).")
    
    # LangGraph 1.0+: Bindear system prompt al LLM
    # Esta es la única forma que funciona en LangGraph 1.0.1+
    llm_with_system = llm_instance.bind(
        system=system_prompt_text
    )
    
    # Crear agente SIN modificadores (solo model + tools)
    agent = create_react_agent(
        llm_with_system,
        tools_list
    )
    
    logger.debug(f"✅ Agente creado con {len(tools_list)} herramientas (LangGraph 1.0.1)")
    
    return agent


# ========================================
# PROMPTS DE AGENTES ESPECIALISTAS
# ========================================

PROMPT_SINTESIS_RAG = """
Eres un asistente financiero experto y un tutor de nivel CFA. Tu tono es profesional, servicial y analítico.

TAREA:
Has recibido una pregunta de un usuario y el contexto relevante de los libros CFA.
Tu trabajo es SINTETIZAR el contexto para generar una respuesta clara y concisa.

REGLAS ABSOLUTAS:
1. NO copies y pegues el contexto. Debes leerlo y generar una respuesta con tus propias palabras (las del rol de experto).
2. Basa tu respuesta ESTRICTAMENTE en el contexto proporcionado. No inventes información.
3. Si el contexto no es suficiente, indica que la información no se encontró en los documentos.
4. Al final de tu respuesta, DEBES citar tus fuentes. El contexto incluirá metadatos (ej. "source", "page_number").

EJEMPLO DE RESPUESTA:
[Tu párrafo de SÍNTESIS aquí...]

---
Fuentes:
- CFA Level 1 2025 - Vol 2, Página 42
- CFA Level 1 2025 - Vol 3, Página 108
""" 


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
    "Agente_Sintesis_RAG": nodo_sintesis_rag
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

# En: agents/financial_agents.py

supervisor_system_prompt = """Eres un supervisor MUY eficiente de un equipo de analistas financieros. Tu única función es leer el historial COMPLETO de la conversación y decidir el siguiente paso.

Especialistas:
- Agente_Renta_Fija: `calcular_valor_bono`
- Agente_Finanzas_Corp: `calcular_van`, `calcular_wacc`
- Agente_Equity: `calcular_gordon_growth`
- Agente_Portafolio: `calcular_capm`, `calcular_sharpe_ratio`
- Agente_Derivados: `calcular_opcion_call`
- Agente_Ayuda: `obtener_ejemplos_de_uso`
- Agente_RAG: `buscar_documentacion_financiera` (SOLO BUSCA)
- Agente_Sintesis_RAG: Sintetiza el contexto de Agente_RAG.

PROCESO DE DECISIÓN (SIGUE ESTAS REGLAS EN ORDEN ESTRICTO):

**1. REGLA DE FINALIZACIÓN (MÁXIMA PRIORIDAD):**
¿Es el último mensaje en el historial una respuesta FINAL y SINTETIZADA de 'Agente_Sintesis_RAG' o una respuesta de un agente de cálculo (como 'Agente_Finanzas_Corp')?
SI ES SÍ: La tarea está 100% completada. No llames a ningún otro agente.
→ Elige 'FINISH'

**2. REGLA DE AYUDA (SEGUNDA PRIORIDAD):**
¿Es el último mensaje del usuario Y pide "ayuda", "ejemplos", o "qué puedes hacer"?
SI ES SÍ:
→ Elige 'Agente_Ayuda'

**3. REGLA DE BÚSQUEDA RAG (TERCERA PRIORIDAD):**
¿Es el último mensaje del usuario Y es una pregunta teórica (ej. "qué es...", "explica...", "busca en la documentación...")?
SI ES SÍ: (y la regla 1 no se aplicó)
→ Elige 'Agente_RAG'

**4. REGLA DE CÁLCULO (CUARTA PRIORIDAD):**
¿Es el último mensaje del usuario Y pide un cálculo numérico (VAN, WACC, etc.)?
SI ES SÍ: (y las reglas 1 y 2 no se aplicaron)
→ Elige el agente especialista apropiado (ej. 'Agente_Finanzas_Corp').

Si ninguna regla aplica, o si la tarea parece completada, elige 'FINISH'.
SOLO devuelve el nombre del agente o "FINISH".
"""

logger.info("✅ Módulo financial_agents cargado (LangGraph 1.0.1+ usando bind)")