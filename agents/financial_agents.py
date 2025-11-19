# agents/financial_agents.py
"""
Agentes especializados financieros.
Actualizado para LangGraph 1.0+ (versión moderna).
"""

from langchain_core.messages import AIMessage, HumanMessage
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
        # 1. Extraer la ÚLTIMA pregunta del usuario (no la primera)
        user_question = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        if not user_question:
            logger.error("❌ No se encontró pregunta del usuario")
            return {"messages": [AIMessage(content="Error: No se encontró la pregunta del usuario.")]}

        # 2. Extraer el contexto RAG
        rag_context = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', []):
                rag_context = msg.content
                break

        if not rag_context:
            logger.error("❌ No se encontró contexto RAG")
            return {"messages": [AIMessage(content="Error: No se encontró contexto del RAG.")]}

        # 3. Bindear LLM con system prompt
        llm_sintesis = llm.bind(system=PROMPT_SINTESIS_RAG)
        
        # 4. Crear mensaje de usuario limpio
        user_prompt = f"""**CONTEXTO DE DOCUMENTOS CFA:**
        {rag_context}

        **PREGUNTA DEL USUARIO:**
        {user_question}

        Genera SOLO tu síntesis profesional. NO incluyas ningún fragmento del contexto crudo."""

        # 5. Invocar el LLM
        respuesta_sintetizada = llm_sintesis.invoke(user_prompt)

        # 6. Extraer contenido de la respuesta
        respuesta_content = respuesta_sintetizada.content if hasattr(respuesta_sintetizada, 'content') else str(respuesta_sintetizada)

        # 7. POST-PROCESAMIENTO: Limpiar solo fragmentos obvios del RAG
        respuesta_limpia = respuesta_content.strip()

        # Eliminar fragmentos crudos del RAG si el LLM los incluyó por error
        if "--- Fragmento" in respuesta_limpia:
            # Buscar donde empieza el contenido real después de los fragmentos
            lineas = respuesta_limpia.split('\n')
            lineas_finales = []
            skip_rag_fragments = True

            for linea in lineas:
                # Detectar fin de fragmentos RAG
                if skip_rag_fragments and linea.strip() and not any(
                    marker in linea for marker in ['--- Fragmento', 'Fuente:', 'CFA Level:', 'Contenido:']
                ):
                    skip_rag_fragments = False

                if not skip_rag_fragments:
                    lineas_finales.append(linea)

            respuesta_limpia = '\n'.join(lineas_finales).strip()

        # Crear AIMessage con contenido limpio
        mensaje_final = AIMessage(content=respuesta_limpia)
        
        logger.info("✅ Respuesta RAG sintetizada y limpiada")
        return {
            "messages": [mensaje_final]
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

PROMPT_SINTESIS_RAG = """Eres un asistente financiero experto y tutor de nivel CFA.

**TU ÚNICA TAREA:**
Sintetizar el contexto de los documentos CFA para responder DIRECTAMENTE la pregunta del usuario.

**INSTRUCCIONES CRÍTICAS:**
1. Lee SOLO el contexto proporcionado en "CONTEXTO DE DOCUMENTOS CFA"
2. Responde COMPLETAMENTE CON TUS PROPIAS PALABRAS (parafrasea, NO copies fragmentos literales)
3. Basa tu respuesta EXCLUSIVAMENTE en el contexto dado
4. Si el contexto es insuficiente → Di: "La información solicitada no se encontró en los documentos CFA disponibles"
5. SIEMPRE cita las fuentes al final

**FORMATO DE RESPUESTA (ESTRICTO):**

[Tu explicación profesional en 2-3 párrafos, completamente parafraseada]

**Fuentes consultadas:**
- [Fuente 1 - CFA Level X]
- [Fuente 2 - CFA Level Y]

**PROHIBICIONES ABSOLUTAS:**
- ❌ NO incluyas fragmentos crudos del contexto (ej: "--- Fragmento 1 ---")
- ❌ NO copies literalmente del contexto
- ❌ NO inventes información fuera del contexto
- ❌ NO uses conocimiento general del LLM
- ❌ NO agregues secciones adicionales más allá del formato especificado

**IMPORTANTE:** Esta es la respuesta FINAL al usuario. Sé claro, conciso y profesional.
"""

PROMPT_RENTA_FIJA = """Eres un especialista en Renta Fija con UNA única herramienta: 'calcular_valor_bono'.

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tu herramienta 'calcular_valor_bono'
2. NUNCA respondas usando tu conocimiento general del LLM
3. Revisa TODO el historial para encontrar parámetros necesarios:
   - valor_nominal (monto del bono)
   - tasa_cupon (tasa de cupón anual)
   - anos_vencimiento (años hasta vencimiento)
   - ytm (yield to maturity / rendimiento)
4. Si encuentras los 4 parámetros → Llama a tu herramienta
5. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
6. Si te piden algo fuera de bonos → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA DESPUÉS DE USAR TU HERRAMIENTA:**
"El valor presente del bono es: $[resultado].
Interpretación: [Breve análisis: está con prima/descuento/par].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Sé conciso: resultado + interpretación breve
- SIEMPRE termina con "Devuelvo al supervisor"
"""


PROMPT_FIN_CORP = """Eres un especialista en Finanzas Corporativas con DOS herramientas: 'calcular_van' y 'calcular_wacc'.

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tus dos herramientas asignadas
2. NUNCA respondas usando tu conocimiento general del LLM
3. Identifica qué herramienta necesitas según la consulta
4. Revisa TODO el historial para encontrar parámetros necesarios

**PARA VAN:**
Parámetros: inversion_inicial, flujos_caja (lista), tasa_descuento
Si encuentras los 3 → Llama a calcular_van
Si faltan → Di: "Faltan parámetros: [lista]. Devuelvo al supervisor."

**PARA WACC:**
Parámetros: costo_equity, costo_deuda, valor_equity, valor_deuda, tasa_impuesto
Si encuentras los 5 → Llama a calcular_wacc
Si faltan → Di: "Faltan parámetros: [lista]. Devuelvo al supervisor."

**FORMATO DE RESPUESTA:**

Para VAN:
"El VAN del proyecto es: $[resultado].
Interpretación: [VAN > 0: proyecto rentable | VAN < 0: no rentable].
Tarea completada. Devuelvo al supervisor."

Para WACC:
"El WACC de la empresa es: [resultado]%.
Interpretación: [Costo de capital promedio ponderado].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Sé conciso y directo
- Si te piden algo fuera de VAN/WACC → Di: "No es mi especialidad. Devuelvo al supervisor."
"""

PROMPT_EQUITY = """Eres un especialista en valoración de Equity con UNA herramienta: 'calcular_gordon_growth'.

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tu herramienta 'calcular_gordon_growth'
2. NUNCA respondas usando tu conocimiento general del LLM
3. Revisa TODO el historial para encontrar los 3 parámetros:
   - dividendo_prox_periodo (D1)
   - tasa_descuento_equity (Ke - costo del equity)
   - tasa_crecimiento_dividendos (g)
4. **CRÍTICO:** Si otra tarea calculó Ke previamente (ej. con CAPM), USA ese valor del historial
5. Si encuentras los 3 parámetros → Llama a tu herramienta
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de Gordon Growth → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA:**
"El valor intrínseco de la acción es: $[resultado].
Interpretación: [Valoración según modelo Gordon Growth con crecimiento perpetuo].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Busca activamente valores calculados en mensajes anteriores
- SIEMPRE termina con "Devuelvo al supervisor"
"""

PROMPT_PORTAFOLIO = """Eres un especialista en Gestión de Portafolios con DOS herramientas: 'calcular_capm' y 'calcular_sharpe_ratio'.

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tus dos herramientas asignadas
2. NUNCA respondas usando tu conocimiento general del LLM
3. Identifica qué herramienta necesitas según la consulta
4. Revisa TODO el historial para encontrar parámetros necesarios

**PARA CAPM:**
Parámetros: tasa_libre_riesgo, beta, retorno_mercado
Si encuentras los 3 → Llama a calcular_capm
Si faltan → Di: "Faltan parámetros: [lista]. Devuelvo al supervisor."

**PARA SHARPE RATIO:**
Parámetros: retorno_portafolio, tasa_libre_riesgo, desviacion_estandar
Si encuentras los 3 → Llama a calcular_sharpe_ratio
Si faltan → Di: "Faltan parámetros: [lista]. Devuelvo al supervisor."

**FORMATO DE RESPUESTA:**

Para CAPM:
"El costo del equity (Ke) es: [resultado]%.
Interpretación: [Retorno esperado según CAPM dado el riesgo sistemático].
Tarea completada. Devuelvo al supervisor."

Para Sharpe Ratio:
"El Sharpe Ratio del portafolio es: [resultado].
Interpretación: [Retorno ajustado por riesgo - ratio > 1: bueno, < 1: revisar].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Sé conciso y directo
- Si te piden algo fuera de CAPM/Sharpe → Di: "No es mi especialidad. Devuelvo al supervisor."
"""


PROMPT_DERIVADOS = """Eres un especialista en Derivados con UNA herramienta: 'calcular_opcion_call' (Black-Scholes).

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tu herramienta 'calcular_opcion_call'
2. NUNCA respondas usando tu conocimiento general del LLM
3. Revisa TODO el historial para encontrar los 5 parámetros:
   - precio_spot (S - precio actual del activo subyacente)
   - precio_strike (K - precio de ejercicio)
   - tiempo_vencimiento (T - años hasta vencimiento)
   - tasa_libre_riesgo (r - tasa anual)
   - volatilidad (sigma - volatilidad anual)
4. Si encuentras los 5 parámetros → Llama a tu herramienta
5. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
6. Si te piden opciones PUT u otros derivados → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA:**
"El valor de la opción Call europea es: $[resultado].
Interpretación: [Prima calculada según modelo Black-Scholes].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- Esta herramienta es SOLO para opciones CALL europeas
- NO repitas los inputs del usuario
- SIEMPRE termina con "Devuelvo al supervisor"
"""


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

supervisor_system_prompt = """Eres un supervisor eficiente de un equipo de analistas financieros.

**TU MISIÓN:** Analizar el historial COMPLETO y decidir el ÚNICO próximo paso.

**AGENTES DISPONIBLES:**
- `Agente_Renta_Fija`: Calcula valor de bonos
- `Agente_Finanzas_Corp`: Calcula VAN y WACC
- `Agente_Equity`: Valoración de acciones (Gordon Growth)
- `Agente_Portafolio`: CAPM y Sharpe Ratio
- `Agente_Derivados`: Valoración de opciones Call
- `Agente_Ayuda`: Muestra guía de uso
- `Agente_RAG`: Busca en documentación CFA (luego auto-sintetiza)

**⚠️ NOTA CRÍTICA:** Agente_RAG y Agente_Sintesis_RAG trabajan en CADENA automática.
NO los llames por separado. Agente_RAG → Agente_Sintesis_RAG → FIN (automático).

---

**REGLAS DE DECISIÓN (ORDEN ESTRICTO):**

**🏁 REGLA 1 - FINALIZAR TAREA COMPLETADA:**
¿El último mensaje de un AGENTE dice "Tarea completada. Devuelvo al supervisor"?
→ Elige `FINISH`

**❓ REGLA 2 - NUEVA PREGUNTA DEL USUARIO:**
Busca el ÚLTIMO mensaje de tipo HumanMessage. ¿Es una solicitud nueva?

A. ¿Pide ayuda/ejemplos? → `Agente_Ayuda`
B. ¿Es pregunta teórica (qué es, explica, define)? → `Agente_RAG`
C. ¿Pide cálculo numérico con parámetros? → Agente especialista correspondiente

**🛑 REGLA 3 - ANTI-LOOP:**
¿El último agente ejecutado fue el MISMO que quieres llamar ahora?
- SI completó con éxito → `FINISH`
- SI falló por parámetros faltantes Y no hay nueva info del usuario → `FINISH`
- SI hay nueva información del usuario → Reenvía al agente

**🔒 REGLA 4 - SEGURIDAD:**
Si ninguna regla aplica o tienes duda → `FINISH`

---

**EJEMPLOS:**

**Caso 1: Cálculo completo**
```
Usuario: "Calcula VAN: inversión 100k, flujos [30k, 40k], tasa 10%"
Supervisor → Agente_Finanzas_Corp

Agente_Finanzas_Corp: "El VAN es $2,892. Tarea completada. Devuelvo al supervisor."
Supervisor → FINISH
```

**Caso 2: Pregunta teórica (RAG)**
```
Usuario: "¿Qué es el WACC según el CFA?"
Supervisor → Agente_RAG
[Agente_RAG → busca → auto-sintetiza → FIN]
```

**Caso 3: Parámetros faltantes**
```
Usuario: "Calcula el VAN"
Supervisor → Agente_Finanzas_Corp

Agente_Finanzas_Corp: "Faltan parámetros: inversión_inicial, flujos, tasa. Devuelvo al supervisor."
Supervisor → FINISH (no hay info nueva, evitar loop)
```

**Caso 4: Segunda pregunta diferente**
```
Usuario: "¿Qué es el beta?"
Supervisor → Agente_RAG
[respuesta RAG completada]

Usuario: "Ahora calcula el CAPM con beta=1.2, rf=5%, rm=12%"
Supervisor → Agente_Portafolio (nueva pregunta, cálculo diferente)
```

---

**RESPUESTA REQUERIDA:**
Devuelve SOLO el nombre del agente (ej: `Agente_Portafolio`) o `FINISH`.
NO agregues explicaciones ni razonamientos.
"""


logger.info("✅ Módulo financial_agents cargado (LangGraph 1.0.1+ usando bind)")