# agents/financial_agents.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
# Importar create_react_agent desde langchain.agents (preferido en versiones > 0.2.1)
try:
    from langchain.agents import create_react_agent
    print("✅ create_react_agent importado desde langchain.agents.")
except ImportError:
    # Fallback para versiones anteriores
    from langgraph.prebuilt import create_react_agent
    import warnings
    warnings.warn("create_react_agent importado desde langgraph.prebuilt. Considera actualizar langchain.", DeprecationWarning)
    print("⚠️ create_react_agent importado desde langgraph.prebuilt (fallback).")

from typing import Literal # Asegúrate que Literal esté importado
from pydantic import BaseModel, Field

# Importar LLM de config y herramientas individuales de tools
from config import get_llm
from tools.financial_tools import ( # Importa las funciones tool directamente
    _calcular_valor_presente_bono, _calcular_van, _calcular_wacc,
    _calcular_gordon_growth, _calcular_capm, _calcular_sharpe_ratio,
    _calcular_opcion_call
)

from tools.help_tools import obtener_ejemplos_de_uso # O desde 'help_tools' si lo separaste

llm = get_llm() # Obtener la instancia singleton del LLM configurado

# --- Creación de Agentes Especialistas (con Prompts Detallados) ---

messages_placeholder = MessagesPlaceholder(variable_name="messages")

def nodo_ayuda_directo(state: dict) -> dict:
    """
    Un nodo simple que NO usa un LLM.
    Simplemente llama a la herramienta de ayuda y devuelve su contenido
    directamente como un AIMessage.
    """
    print("\n--- NODO AYUDA (DIRECTO) ---")
    try:
        # Llama a la herramienta de ayuda directamente. 
        # .invoke({}) es necesario si la herramienta no toma argumentos.
        guia_de_preguntas = obtener_ejemplos_de_uso.invoke({})
        
        # Devuelve el resultado en el formato correcto para el estado
        return {
            "messages": [AIMessage(content=guia_de_preguntas)]
        }
    except Exception as e:
        print(f"❌ ERROR en nodo_ayuda_directo: {e}")
        return {
            "messages": [AIMessage(content=f"Error al obtener la guía de ayuda: {e}")]
        }


def crear_agente_especialista(llm_instance, tools_list, system_prompt_text):
    """Función helper para crear un agente reactivo con prompt de sistema."""
    # Verificar que tools_list no esté vacía y contenga Runnables (tools)
    if not tools_list or not all(hasattr(t, 'invoke') for t in tools_list):
         raise ValueError("tools_list debe contener al menos una herramienta válida (Runnable).")
         
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        messages_placeholder,
    ])
    # Usar el factory importado
    return create_react_agent(llm_instance, tools_list, prompt=prompt)

# Prompts Detallados (los que preferiste para intentar manejar historial)
PROMPT_RENTA_FIJA = """Eres un especialista en Renta Fija.
Tu único trabajo es usar SÓLO tu herramienta 'calcular_valor_bono'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa.
Extrae los parámetros necesarios de la solicitud o del historial y llama a tu herramienta.
Si te piden algo que no puedes hacer con tu herramienta, di "No es mi especialidad, devuelvo al supervisor."."""

PROMPT_FIN_CORP = """Eres un especialista en Finanzas Corporativas.
Tu trabajo es usar SÓLO tus herramientas 'calcular_van' y 'calcular_wacc'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa (ej. WACC calculado).
Extrae los parámetros necesarios de la solicitud o del historial y llama a la herramienta adecuada.
Si te piden algo que no puedes hacer con tus herramientas, di "No es mi especialidad, devuelvo al supervisor."."""

PROMPT_EQUITY = """Eres un especialista en valoración de acciones (Equity).
Tu único trabajo es usar SÓLO tu herramienta 'calcular_gordon_growth'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes. Si una tarea anterior calculó un valor necesario (como Ke o tasa de descuento), usa ESE valor.
Extrae el 'dividendo_prox_periodo' (D1), la 'tasa_descuento_equity' (Ke) y la 'tasa_crecimiento_dividendos' (g) de la solicitud del usuario o del historial.
Llama a tu herramienta con estos 3 parámetros.
Si no puedes encontrar los 3 parámetros, di "Faltan parámetros, devuelvo al supervisor."."""

PROMPT_PORTAFOLIO = """Eres un especialista en Gestión de Portafolios.
Tu trabajo es usar SÓLO tus herramientas 'calcular_capm' y 'calcular_sharpe_ratio'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa.
Extrae los parámetros necesarios de la solicitud o del historial y llama a la herramienta adecuada.
Si te piden una tarea para la que no tienes herramienta (como 'calcular_gordon_growth'), **NO respondas a esa parte**.
Responde SÓLO la parte que SÍ puedes hacer con tus herramientas.
Luego, di "Tarea parcial completada, devuelvo al supervisor."."""

PROMPT_DERIVADOS = """Eres un especialista en instrumentos derivados.
Tu único trabajo es usar SÓLO tu herramienta 'calcular_opcion_call'.
**NUNCA respondas usando tu conocimiento general.**
Revisa cuidadosamente el historial de mensajes por si necesitas información previa.
Extrae los parámetros necesarios (S, K, T, r, sigma) de la solicitud o del historial y llama a tu herramienta.
Si te piden algo que no puedes hacer con tu herramienta, di "No es mi especialidad, devuelvo al supervisor."."""


# Crear agentes (manejar posibles errores en la creación)
try:
    agent_renta_fija = crear_agente_especialista(llm, [_calcular_valor_presente_bono], PROMPT_RENTA_FIJA)
    agent_fin_corp = crear_agente_especialista(llm, [_calcular_van, _calcular_wacc], PROMPT_FIN_CORP)
    agent_equity = crear_agente_especialista(llm, [_calcular_gordon_growth], PROMPT_EQUITY)
    agent_portafolio = crear_agente_especialista(llm, [_calcular_capm, _calcular_sharpe_ratio], PROMPT_PORTAFOLIO)
    agent_derivados = crear_agente_especialista(llm, [_calcular_opcion_call], PROMPT_DERIVADOS)
except Exception as e:
     print(f"❌ ERROR CRÍTICO al crear agentes especialistas: {e}")
     # En Streamlit, esto debería detener la app si ocurre al inicio
     import streamlit as st
     st.error(f"Error inicializando los agentes: {e}")
     st.stop()


# Diccionario de nodos de agente para el grafo
agent_nodes = {
    "Agente_Renta_Fija": agent_renta_fija,
    "Agente_Finanzas_Corp": agent_fin_corp,
    "Agente_Equity": agent_equity,
    "Agente_Portafolio": agent_portafolio,
    "Agente_Derivados": agent_derivados,
    "Agente_Ayuda": nodo_ayuda_directo,
}

# --- Supervisor ---

class RouterSchema(BaseModel):
    """Elige el siguiente agente a llamar o finaliza."""
    next_agent: Literal[ tuple(list(agent_nodes.keys()) + ["FINISH"]) ] # Usar tuple para Literal
    # next_agent: Literal[
    #     "Agente_Renta_Fija", "Agente_Finanzas_Corp", "Agente_Equity",
    #     "Agente_Portafolio", "Agente_Derivados", "FINISH"
    # ] = Field(description="El nombre del agente especialista para la tarea. Elige 'FINISH' si la solicitud fue completamente respondida.")

# Configurar el LLM supervisor para que use el schema de ruteo
try:
    supervisor_llm = llm.with_structured_output(RouterSchema)
except Exception as e:
     print(f"❌ ERROR configurando supervisor LLM con structured_output: {e}")
     import streamlit as st
     st.error(f"Error configurando el supervisor: {e}")
     st.stop()


# Prompt del supervisor (más robusto, incluye historial)
supervisor_system_prompt = """Eres un supervisor MUY eficiente de un equipo de analistas financieros. Tu única función es leer el último mensaje del usuario Y el historial de la conversación para decidir qué especialista debe actuar A CONTINUACIÓN. No respondas tú mismo. SOLO elige el siguiente paso.

Especialistas y sus ÚNICAS herramientas:
- Agente_Renta_Fija: `calcular_valor_bono`
- Agente_Finanzas_Corp: `calcular_van`, `calcular_wacc`
- Agente_Equity: `calcular_gordon_growth`
- Agente_Portafolio: `calcular_capm`, `calcular_sharpe_ratio`
- Agente_Derivados: `calcular_opcion_call`
- Agente_Ayuda: `obtener_ejemplos_de_uso`

PROCESO DE DECISIÓN:
**1. PRIORIDAD MÁXIMA: Si el último mensaje del usuario es 'ayuda', 'ejemplos', 'qué puedes hacer', o una pregunta similar sobre tus capacidades, elige 'Agente_Ayuda'.**
2. Si es una solicitud de cálculo: Lee el último mensaje del usuario. ¿Qué cálculo financiero pide?
3. Revisa el historial. ¿Hay tareas pendientes de mensajes anteriores? ¿El último agente completó solo una parte?
4. Basado en la tarea pendiente MÁS INMEDIATA, elige el agente especialista CORRECTO.
5. Si el último agente indicó que completó su parte Y no quedan tareas pendientes claras en la solicitud original o historial, elige 'FINISH'.
6. Si el último agente indicó un error o incapacidad ("No es mi especialidad", "Faltan parámetros"), y TÚ (supervisor) no ves una forma clara de redirigir a otro agente para completar la solicitud, elige 'FINISH' para detener el proceso.
SOLO devuelve el nombre del agente o "FINISH".
"""

print("✅ Módulo financial_agents cargado (con prompts detallados).")