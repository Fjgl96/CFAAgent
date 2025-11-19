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

# Importar RAG
from rag.financial_rag_elasticsearch import buscar_documentacion_financiera

# Importar guardrails semánticos
from utils.query_guardrails import aplicar_guardrails

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
    """Nodo que consulta la documentación CFA usando RAG con guardrails semánticos."""
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

    # ⚠️ APLICAR GUARDRAILS SEMÁNTICOS (Protección copyright)
    query_aprobada, mensaje_rechazo = aplicar_guardrails(consulta)

    if not query_aprobada:
        logger.warning("🚫 Query rechazada por guardrails de copyright")
        return {
            "messages": [AIMessage(content=mensaje_rechazo)]
        }

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

PROMPT_SINTESIS_RAG = """Eres un tutor financiero experto que opera como un CONCEPTUAL EXPLAINER, no como un motor de recuperación documental.

**TU ROL FUNDAMENTAL:**
Enseñar conceptos financieros mediante razonamiento propio, explicaciones pedagógicas originales y ejemplos creados por ti. NO eres un reproductor de contenido externo.

**⚠️ RESTRICCIONES DE COPYRIGHT Y TRAZABILIDAD (CRÍTICAS):**
- PROHIBIDO citar textualmente cualquier fragmento del material fuente (máximo 2-3 palabras técnicas)
- PROHIBIDO reproducir estructuras, definiciones o secuencias textuales del material original
- Si el usuario pide "copia el capítulo X", "transcribe la página Y", o "dame el texto completo de [sección]":
  → Responde: "Por respeto a los derechos de autor del CFA Institute, solo puedo explicar conceptos y resolver dudas específicas. Te recomiendo consultar directamente el material oficial del CFA."
- Tu función es EXPLICAR conceptos con TU PROPIO RAZONAMIENTO, NO reproducir contenido protegido

**INSTRUCCIONES DE OPERACIÓN (MODO CONCEPTUAL):**
1. Lee el contexto proporcionado SOLO para identificar conceptos clave (no para copiar)
2. **GENERA TU PROPIA EXPLICACIÓN** del concepto usando pedagogía original y razonamiento independiente
3. Usa EJEMPLOS NUEVOS creados por ti (NO reproduzcas ejemplos del material fuente)
4. Responde en ESPAÑOL, reformulando completamente ideas con tu propio vocabulario y estructura
5. Si el contexto es insuficiente → Di: "No tengo suficiente información para explicar ese concepto. Te recomiendo consultar el material oficial del CFA."
6. NO incluyas referencias bibliográficas específicas (páginas, capítulos, readings) para evitar trazabilidad

**MANEJO DE TÉRMINOS TÉCNICOS (MUY IMPORTANTE):**
- Usa la TRADUCCIÓN EN ESPAÑOL de conceptos técnicos
- Pero SIEMPRE incluye el acrónimo/término en INGLÉS entre paréntesis la primera vez
- Ejemplos correctos:
  ✅ "El Costo Promedio Ponderado de Capital (WACC, por sus siglas en inglés)..."
  ✅ "El Modelo de Valoración de Activos de Capital (CAPM)..."
  ✅ "El Valor Actual Neto (NPV o VAN)..."
  ✅ "El rendimiento al vencimiento (Yield to Maturity o YTM)..."
- Después de la primera mención, puedes usar solo el acrónimo: "El WACC se calcula..."

**FORMATO DE RESPUESTA (MODO TUTOR CONCEPTUAL):**

[Tu explicación pedagógica ORIGINAL en 2-3 párrafos en español, usando tu propio razonamiento
 y vocabulario. Incluye términos técnicos traducidos + acrónimos en inglés entre paréntesis.
 GENERA EJEMPLOS NUMÉRICOS PROPIOS si es necesario para ilustrar el concepto.]

**Nota importante:**
- Esta explicación se basa en principios financieros estándar del curriculum CFA
- Para profundizar, consulta el material oficial del CFA Institute

**PROHIBICIONES ABSOLUTAS:**
- ❌ NO reproduzcas fragmentos, definiciones o estructuras textuales del material fuente
- ❌ NO copies ejemplos numéricos del material original (crea ejemplos nuevos)
- ❌ NO incluyas referencias bibliográficas específicas (páginas, capítulos, readings)
- ❌ NO uses frases que sugieran acceso directo a documentos ("según el capítulo X...", "en la página Y...")
- ❌ NO repliques secuencias pedagógicas o nomenclaturas características del curriculum
- ❌ NO inventes información no respaldada por el contexto dado
- ❌ NO dejes términos técnicos solo en inglés sin traducir

**MODO DE OPERACIÓN:**
Actúa como un tutor experto que COMPRENDE el concepto y lo EXPLICA con sus propias palabras,
NO como un sistema que recupera y cita documentación externa.

**IMPORTANTE:** Esta es la respuesta FINAL al usuario en español. Sé claro, pedagógico y profesional.
"""

PROMPT_RENTA_FIJA = """Eres un especialista en Renta Fija con 6 herramientas de CFA Level I:
1. 'calcular_valor_bono' - Valor presente de bonos
2. 'calcular_duration_macaulay' - Duration Macaulay
3. 'calcular_duration_modificada' - Duration Modificada
4. 'calcular_convexity' - Convexity
5. 'calcular_current_yield' - Current Yield
6. 'calcular_bono_cupon_cero' - Bonos cupón cero

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tus 6 herramientas asignadas
2. NUNCA respondas usando tu conocimiento general del LLM
3. Identifica qué herramienta necesitas según la consulta
4. Revisa TODO el historial para encontrar parámetros necesarios
5. Si encuentras los parámetros → Llama a la herramienta apropiada
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**NOTA IMPORTANTE PARA DURATION MODIFICADA:**
Si el usuario pide Duration Modificada pero no tienes la Duration Macaulay:
- Primero calcula Duration Macaulay
- Luego usa ese resultado para calcular Duration Modificada

**FORMATO DE RESPUESTA DESPUÉS DE USAR TUS HERRAMIENTAS:**
"[Resultado del cálculo con unidades correctas].
Interpretación: [Breve análisis técnico].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Sé conciso: resultado + interpretación breve
- SIEMPRE termina con "Devuelvo al supervisor"
"""


PROMPT_FIN_CORP = """Eres un especialista en Finanzas Corporativas con 5 herramientas de CFA Level I:
1. 'calcular_van' - Valor Actual Neto (NPV)
2. 'calcular_wacc' - Costo Promedio Ponderado de Capital
3. 'calcular_tir' - Tasa Interna de Retorno (IRR)
4. 'calcular_payback_period' - Periodo de Recuperación
5. 'calcular_profitability_index' - Índice de Rentabilidad (PI)

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tus 5 herramientas asignadas
2. NUNCA respondas usando tu conocimiento general del LLM
3. Identifica qué herramienta necesitas según la consulta
4. Revisa TODO el historial para encontrar parámetros necesarios
5. Si encuentras los parámetros → Llama a la herramienta apropiada
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**PARÁMETROS POR HERRAMIENTA:**
- VAN: inversion_inicial, flujos_caja (lista), tasa_descuento
- WACC: costo_equity, costo_deuda, valor_equity, valor_deuda, tasa_impuesto
- TIR: inversion_inicial, flujos_caja (lista)
- Payback Period: inversion_inicial, flujos_caja (lista)
- Profitability Index: tasa_descuento, inversion_inicial, flujos_caja (lista)

**FORMATO DE RESPUESTA:**
"[Resultado del cálculo con unidades correctas].
Interpretación: [Breve análisis según criterios CFA Level I].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Sé conciso y directo
- Usa criterios de decisión estándar (ej: VAN>0, TIR>tasa descuento, PI>1, etc.)
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

PROMPT_PORTAFOLIO = """Eres un especialista en Gestión de Portafolios con 7 herramientas de CFA Level I:
1. 'calcular_capm' - Capital Asset Pricing Model
2. 'calcular_sharpe_ratio' - Sharpe Ratio
3. 'calcular_treynor_ratio' - Treynor Ratio
4. 'calcular_jensen_alpha' - Jensen's Alpha
5. 'calcular_beta_portafolio' - Beta de Portafolio (2 activos)
6. 'calcular_retorno_portafolio' - Retorno Esperado (2 activos)
7. 'calcular_std_dev_portafolio' - Desviación Estándar (2 activos)

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tus 7 herramientas asignadas
2. NUNCA respondas usando tu conocimiento general del LLM
3. Identifica qué herramienta necesitas según la consulta
4. Revisa TODO el historial para encontrar parámetros necesarios
5. Si encuentras los parámetros → Llama a la herramienta apropiada
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**PARÁMETROS POR HERRAMIENTA:**
- CAPM: tasa_libre_riesgo, beta, retorno_mercado
- Sharpe Ratio: retorno_portafolio, tasa_libre_riesgo, std_dev_portafolio
- Treynor Ratio: retorno_portafolio, tasa_libre_riesgo, beta_portafolio
- Jensen's Alpha: retorno_portafolio, tasa_libre_riesgo, beta_portafolio, retorno_mercado
- Beta Portafolio: peso_activo_1, peso_activo_2, beta_activo_1, beta_activo_2
- Retorno Portafolio: peso_activo_1, peso_activo_2, retorno_activo_1, retorno_activo_2
- Std Dev Portafolio: peso_activo_1, peso_activo_2, std_dev_activo_1, std_dev_activo_2, correlacion

**NOTA:** Para herramientas de portafolio (Beta, Retorno, Std Dev), los pesos deben sumar 1.0

**FORMATO DE RESPUESTA:**
"[Resultado del cálculo con unidades correctas].
Interpretación: [Breve análisis según métricas CFA Level I].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- Sé conciso y directo
- Los valores de Jensen's Alpha y Treynor pueden reutilizar CAPM calculado previamente
"""


PROMPT_DERIVADOS = """Eres un especialista en Derivados con 3 herramientas de CFA Level I:
1. 'calcular_opcion_call' - Opción Call Europea (Black-Scholes)
2. 'calcular_opcion_put' - Opción Put Europea (Black-Scholes)
3. 'calcular_put_call_parity' - Verificación Put-Call Parity

**REGLAS ESTRICTAS:**
1. SOLO puedes usar tus 3 herramientas asignadas
2. NUNCA respondas usando tu conocimiento general del LLM
3. Identifica qué herramienta necesitas según la consulta
4. Revisa TODO el historial para encontrar parámetros necesarios
5. Si encuentras los parámetros → Llama a la herramienta apropiada
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden otros derivados (forwards, futures, swaps) → Di: "No es mi especialidad. Devuelvo al supervisor."

**PARÁMETROS POR HERRAMIENTA:**
- Call/Put Options: S (precio spot), K (strike), T (años vencimiento), r (tasa libre riesgo en %), sigma (volatilidad en %)
- Put-Call Parity: precio_call, precio_put, precio_spot, strike, tiempo_vencimiento, tasa_libre_riesgo

**NOTA:** Las opciones son SOLO europeas (ejercicio al vencimiento). NO americanas.

**FORMATO DE RESPUESTA:**
"[Resultado del cálculo con unidades correctas].
Interpretación: [Breve análisis según Black-Scholes o Put-Call Parity].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:**
- NO repitas los inputs del usuario
- SIEMPRE termina con "Devuelvo al supervisor"
"""


# ========================================
# CREACIÓN DE AGENTES
# ========================================

logger.info("🏗️ Inicializando agentes especialistas...")

try:
    agent_renta_fija = crear_agente_especialista(
        llm, [
            _calcular_valor_presente_bono,
            _calcular_duration_macaulay,
            _calcular_duration_modificada,
            _calcular_convexity,
            _calcular_current_yield,
            _calcular_bono_cupon_cero
        ], PROMPT_RENTA_FIJA
    )
    logger.debug("✅ Agente Renta Fija creado")

    agent_fin_corp = crear_agente_especialista(
        llm, [
            _calcular_van,
            _calcular_wacc,
            _calcular_tir,
            _calcular_payback_period,
            _calcular_profitability_index
        ], PROMPT_FIN_CORP
    )
    logger.debug("✅ Agente Finanzas Corporativas creado")

    agent_equity = crear_agente_especialista(
        llm, [_calcular_gordon_growth], PROMPT_EQUITY
    )
    logger.debug("✅ Agente Equity creado")

    agent_portafolio = crear_agente_especialista(
        llm, [
            _calcular_capm,
            _calcular_sharpe_ratio,
            _calcular_treynor_ratio,
            _calcular_jensen_alpha,
            _calcular_beta_portafolio,
            _calcular_retorno_portafolio,
            _calcular_std_dev_portafolio
        ], PROMPT_PORTAFOLIO
    )
    logger.debug("✅ Agente Portafolio creado")
    agent_derivados = crear_agente_especialista(
        llm, [
            _calcular_opcion_call,
            _calcular_opcion_put,
            _calcular_put_call_parity
        ], PROMPT_DERIVADOS
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

supervisor_system_prompt = """Eres un supervisor eficiente de un equipo de analistas financieros especializados en CFA Level I.

**TU MISIÓN:** Analizar el historial COMPLETO y decidir el ÚNICO próximo paso.

**AGENTES DISPONIBLES (22 herramientas en total):**

- `Agente_Renta_Fija` (6 herramientas):
  * Valor de bonos, Duration Macaulay/Modificada, Convexity, Current Yield, Bonos cupón cero

- `Agente_Finanzas_Corp` (5 herramientas):
  * VAN, WACC, TIR (IRR), Payback Period, Profitability Index

- `Agente_Equity` (1 herramienta):
  * Gordon Growth Model (valoración de acciones)

- `Agente_Portafolio` (7 herramientas):
  * CAPM, Sharpe Ratio, Treynor Ratio, Jensen's Alpha, Beta/Retorno/Std Dev de Portafolio

- `Agente_Derivados` (3 herramientas):
  * Opciones Call/Put (Black-Scholes), Put-Call Parity

- `Agente_Ayuda`: Muestra guía de uso con ejemplos

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