# agents/financial_agents.py
"""
Agentes especializados financieros.
Actualizado: 
1. Usa Microservicio RAG externo (Cliente HTTP).
2. Prompts con Máquina de Estados para evitar bucles y alucinaciones.
"""

import os
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Literal
from pydantic import BaseModel, Field

# Importar configuración
from config import get_llm, RAG_API_URL

# Importar herramientas financieras (locales)
from tools.financial_tools import (
    # Herramientas originales
    _calcular_valor_presente_bono, _calcular_van, _calcular_wacc,
    _calcular_gordon_growth, _calcular_capm, _calcular_sharpe_ratio,
    _calcular_opcion_call,
    # Nuevas herramientas CFA Level I
    _calcular_tir, _calcular_payback_period, _calcular_profitability_index,
    _calcular_duration_macaulay, _calcular_duration_modificada, _calcular_convexity,
    _calcular_current_yield, _calcular_bono_cupon_cero,
    _calcular_opcion_put, _calcular_put_call_parity,
    _calcular_treynor_ratio, _calcular_jensen_alpha, _calcular_beta_portafolio,
    _calcular_retorno_portafolio, _calcular_std_dev_portafolio
)
from tools.help_tools import obtener_ejemplos_de_uso

# NOTA: Se eliminó la importación local de rag.financial_rag_elasticsearch
# Ahora usamos la herramienta definida abajo que conecta al microservicio.

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('agents')
except ImportError:
    import logging
    logger = logging.getLogger('agents')

llm = get_llm()

# ========================================
# HERRAMIENTA RAG (CLIENTE MICROSERVICIO)
# ========================================

@tool
def buscar_documentacion_financiera(consulta: str) -> str:
    """
    Busca información en material financiero consultando el Microservicio RAG externo.
    """
    logger.info(f"🔍 Consultando Microservicio RAG: '{consulta[:50]}...'")
    
    if not RAG_API_URL:
        msg = "❌ Error de configuración: RAG_API_URL no definida en secrets/env."
        logger.error(msg)
        return msg

    # Asegurar endpoint correcto
    endpoint = f"{RAG_API_URL.rstrip('/')}/search"

    try:
        response = requests.post(
            endpoint,
            json={"consulta": consulta},
            timeout=45  # Timeout generoso
        )
        
        if response.status_code == 200:
            data = response.json()
            # La API devuelve {"resultado": "texto..."}
            resultado = data.get("resultado", "No se encontró información relevante.")
            logger.info("✅ Respuesta recibida del Microservicio")
            return resultado
        else:
            error_msg = f"Error del Servicio RAG ({response.status_code}): {response.text}"
            logger.error(f"❌ {error_msg}")
            return error_msg

    except Exception as e:
        error_msg = f"Error de Conexión con RAG: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return error_msg

# ========================================
# NODOS ESPECIALES
# ========================================

def nodo_ayuda_directo(state: dict) -> dict:
    """Nodo simple que llama a la herramienta de ayuda directamente."""
    logger.info("📖 Nodo Ayuda invocado")
    try:
        guia_de_preguntas = obtener_ejemplos_de_uso.invoke({})
        logger.debug("✅ Guía de ayuda generada")
        # Etiqueta explícita para el Supervisor
        return {
            "messages": [AIMessage(content=guia_de_preguntas + "\n\nTAREA_COMPLETADA")]
        }
    except Exception as e:
        logger.error(f"❌ Error en nodo_ayuda: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Error al obtener la guía de ayuda: {e}\nERROR_BLOQUEANTE")]
        }


def nodo_rag(state: dict) -> dict:
    """
    Nodo ReAct Autónomo para RAG (Patrón S30).
    Usa la herramienta 'buscar_documentacion_financiera' conectada al microservicio.
    """
    logger.info("📚 Agente RAG ReAct invocado (S30 Pattern)")

    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [AIMessage(content="Error: No hay mensajes en el estado.\nERROR_BLOQUEANTE")]
        }

    last_message = messages[-1]
    consulta = last_message.content if hasattr(last_message, 'content') else str(last_message)

    logger.info(f"🔍 Consulta financiera: {consulta[:100]}...")

    try:
        system_prompt_react = """Eres un Analista Financiero CFA.
        Tu trabajo es buscar en la documentación y sintetizar una respuesta en ESPAÑOL.
        
        FORMATO DE SALIDA OBLIGATORIO:
        1. Tu explicación detallada parafraseada.
        2. Al final, escribe en una línea nueva: TAREA_COMPLETADA
        
        Si NO encuentras información tras buscar:
        "No encontré información relevante en los documentos. TAREA_COMPLETADA"
        """

        llm_react = llm.bind(system=system_prompt_react)
        
        # Crear agente ReAct con la herramienta HTTP
        agent_react = create_react_agent(
            llm_react,
            tools=[buscar_documentacion_financiera]
        )

        # Invocar
        result = agent_react.invoke({"messages": [HumanMessage(content=consulta)]})
        agent_messages = result.get("messages", [])

        if agent_messages:
            final_response = agent_messages[-1].content
            # Asegurar etiqueta de cierre si el LLM la olvidó
            if "TAREA_COMPLETADA" not in final_response and "ERROR" not in final_response:
                final_response += "\n\nTAREA_COMPLETADA"
            
            return {"messages": [AIMessage(content=final_response)]}

        return {
            "messages": [AIMessage(content="No pude procesar la solicitud.\nERROR_BLOQUEANTE")]
        }

    except Exception as e:
        logger.error(f"❌ Error en RAG ReAct: {e}", exc_info=True)
        return {
            "messages": [AIMessage(
                content=f"Error al buscar en el material de estudio: {e}\nERROR_BLOQUEANTE"
            )]
        }


def nodo_sintesis_rag(state: dict) -> dict:
    """
    Nodo passthrough para compatibilidad.
    El RAG ReAct ya hace la síntesis, este nodo solo confirma el cierre.
    """
    return {
        "messages": [AIMessage(content="Síntesis finalizada.\nTAREA_COMPLETADA")]
    }

def crear_agente_especialista(llm_instance, tools_list, system_prompt_text):
    """Crea un agente reactivo con el prompt de sistema binded."""
    if not tools_list:
        raise ValueError("tools_list debe contener al menos una herramienta.")
    
    llm_with_system = llm_instance.bind(system=system_prompt_text)
    
    return create_react_agent(llm_with_system, tools_list)


# ========================================
# PROMPTS DE AGENTES ESPECIALISTAS (PROTOCOLOS)
# ========================================

# Este bloque se inyecta en todos los agentes para estandarizar su comportamiento
PROTOCOLO_SEGURIDAD = """
**PROTOCOLO DE SEGURIDAD Y CIERRE (OBLIGATORIO):**

1. **ANTI-ALUCINACIÓN (ZERO-SHOT SAFETY):**
   - Si una herramienta requiere un parámetro (ej: 'inversion_inicial', 'tasa', 'flujos') y NO está explícitamente en el historial:
   - **ESTÁ PROHIBIDO INVENTARLO**. No asumas 0, 1, ni promedios.
   - TU ÚNICA ACCIÓN es reportar que falta ese dato usando la etiqueta FALTAN_DATOS.

2. **ETIQUETAS DE CIERRE (CRÍTICO):**
   Tu mensaje FINAL debe terminar OBLIGATORIAMENTE con una de estas etiquetas para que el Supervisor sepa qué hacer:

   - **Caso Éxito (Cálculo realizado):**
     "[Respuesta numérica e interpretación].
     TAREA_COMPLETADA"
   
   - **Caso Faltan Datos (No puedes calcular):**
     "Necesito los siguientes datos para proceder: [lista].
     FALTAN_DATOS"
   
   - **Caso Error Técnico o Validación (Inputs inválidos):**
     "No pude realizar el cálculo porque: [razón del error].
     ERROR_BLOQUEANTE"

   - **Caso Fuera de Dominio (No es tu tema):**
     "Esto no es mi especialidad.
     FALTAN_DATOS"
"""

PROMPT_RENTA_FIJA = f"""Eres un especialista en Renta Fija con 6 herramientas de CFA Level I:
1. 'calcular_valor_bono' - Valor presente de bonos
2. 'calcular_duration_macaulay' - Duration Macaulay
3. 'calcular_duration_modificada' - Duration Modificada
4. 'calcular_convexity' - Convexity
5. 'calcular_current_yield' - Current Yield
6. 'calcular_bono_cupon_cero' - Bonos cupón cero

**NOTA ESPECÍFICA:** Si piden Duration Modificada y falta la Macaulay, calcúlala primero si tienes datos, o pide los datos.

{PROTOCOLO_SEGURIDAD}
"""

PROMPT_FIN_CORP = f"""Eres un especialista en Finanzas Corporativas con 5 herramientas de CFA Level I:
1. 'calcular_van' - Valor Actual Neto (NPV)
2. 'calcular_wacc' - Costo Promedio Ponderado de Capital
3. 'calcular_tir' - Tasa Interna de Retorno (IRR)
4. 'calcular_payback_period' - Periodo de Recuperación
5. 'calcular_profitability_index' - Índice de Rentabilidad (PI)

**REGLA CRÍTICA:** Si 'inversion_inicial' es 0, es un error lógico. Retorna ERROR_BLOQUEANTE reportando que la inversión debe ser mayor a 0.

{PROTOCOLO_SEGURIDAD}
"""

PROMPT_EQUITY = f"""Eres un especialista en Equity con 1 herramienta: 'calcular_gordon_growth'.

**REGLA CRÍTICA:** Revisa el historial por si el 'Ke' (costo equity) ya fue calculado por CAPM previamente. Si existe, úsalo.

{PROTOCOLO_SEGURIDAD}
"""

PROMPT_PORTAFOLIO = f"""Eres un especialista en Gestión de Portafolios con 7 herramientas de CFA Level I:
1. 'calcular_capm' - Capital Asset Pricing Model
2. 'calcular_sharpe_ratio' - Sharpe Ratio
3. 'calcular_treynor_ratio' - Treynor Ratio
4. 'calcular_jensen_alpha' - Jensen's Alpha
5. 'calcular_beta_portafolio' - Beta de Portafolio (2 activos)
6. 'calcular_retorno_portafolio' - Retorno Esperado (2 activos)
7. 'calcular_std_dev_portafolio' - Desviación Estándar (2 activos)

**REGLA CRÍTICA:** Los pesos de portafolio deben sumar 1.0. Si no, ERROR_BLOQUEANTE.

{PROTOCOLO_SEGURIDAD}
"""

PROMPT_DERIVADOS = f"""Eres un especialista en Derivados con 3 herramientas de CFA Level I:
1. 'calcular_opcion_call' - Opción Call Europea (Black-Scholes)
2. 'calcular_opcion_put' - Opción Put Europea (Black-Scholes)
3. 'calcular_put_call_parity' - Verificación Put-Call Parity

**REGLA CRÍTICA:** Solo opciones EUROPEAS. Si piden Americanas -> ERROR_BLOQUEANTE explicando que no soportas americanas.

{PROTOCOLO_SEGURIDAD}
"""


# ========================================
# CREACIÓN DE AGENTES
# ========================================

logger.info("🏗️ Inicializando agentes especialistas...")

try:
    agent_renta_fija = crear_agente_especialista(llm, [
        _calcular_valor_presente_bono, _calcular_duration_macaulay, _calcular_duration_modificada,
        _calcular_convexity, _calcular_current_yield, _calcular_bono_cupon_cero
    ], PROMPT_RENTA_FIJA)

    agent_fin_corp = crear_agente_especialista(llm, [
        _calcular_van, _calcular_wacc, _calcular_tir,
        _calcular_payback_period, _calcular_profitability_index
    ], PROMPT_FIN_CORP)

    agent_equity = crear_agente_especialista(llm, [_calcular_gordon_growth], PROMPT_EQUITY)

    agent_portafolio = crear_agente_especialista(llm, [
        _calcular_capm, _calcular_sharpe_ratio, _calcular_treynor_ratio,
        _calcular_jensen_alpha, _calcular_beta_portafolio,
        _calcular_retorno_portafolio, _calcular_std_dev_portafolio
    ], PROMPT_PORTAFOLIO)

    agent_derivados = crear_agente_especialista(llm, [
        _calcular_opcion_call, _calcular_opcion_put, _calcular_put_call_parity
    ], PROMPT_DERIVADOS)
    
    logger.info("✅ Todos los agentes creados exitosamente")

except Exception as e:
    logger.error(f"❌ ERROR CRÍTICO al crear agentes: {e}", exc_info=True)
    import streamlit as st
    st.error(f"Error inicializando los agentes: {e}")
    st.stop()

# Diccionario de Nodos
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

# ========================================
# SUPERVISOR (MÁQUINA DE ESTADOS)
# ========================================

class RouterSchema(BaseModel):
    next_agent: Literal["Agente_Renta_Fija", "Agente_Finanzas_Corp", "Agente_Equity", 
                       "Agente_Portafolio", "Agente_Derivados", "Agente_Ayuda", 
                       "Agente_RAG", "FINISH"] = Field(description="Próximo nodo o FINISH")

supervisor_llm = llm.with_structured_output(RouterSchema)

supervisor_system_prompt = """Eres el Supervisor del sistema financiero.
TU ÚNICA TAREA es decidir el siguiente paso basándote en el ÚLTIMO mensaje del historial.

**MÁQUINA DE ESTADOS (REGLAS DE ORO):**

1. **SI EL ÚLTIMO MENSAJE ES DE UN AGENTE (AI):**
   - ¿Dice "TAREA_COMPLETADA"? -> RESPONDE: `FINISH`
   - ¿Dice "FALTAN_DATOS"? -> RESPONDE: `FINISH` (Devolver al usuario para que responda)
   - ¿Dice "ERROR_BLOQUEANTE"? -> RESPONDE: `FINISH` (No se puede seguir)
   - ¿No dice ninguna etiqueta clara? -> RESPONDE: `FINISH` (Por seguridad ante respuestas ambiguas)

   **EXCEPCIÓN:** Nunca envíes de vuelta al MISMO agente que acaba de hablar si no hay input nuevo del usuario.

2. **SI EL ÚLTIMO MENSAJE ES DEL USUARIO (Human):**
   Enruta según la intención clara:
   - Cálculos de Bonos/Yield -> `Agente_Renta_Fija`
   - VAN, WACC, TIR, Proyectos -> `Agente_Finanzas_Corp`
   - Acciones, Gordon -> `Agente_Equity`
   - Portafolios, CAPM, Betas -> `Agente_Portafolio`
   - Opciones, Black-Scholes -> `Agente_Derivados`
   - Teoría, Conceptos, "¿Qué es...?" -> `Agente_RAG`
   - Ayuda -> `Agente_Ayuda`

**NOTA:** Tu respuesta es SOLO el nombre del nodo (o FINISH).
"""

logger.info("✅ Agentes financieros cargados con protocolos Anti-Hopping")