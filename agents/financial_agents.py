# agents/financial_agents.py
"""
Agentes especializados financieros - VERSIÓN MEJORADA.
Actualizado para LangChain 1.0+ con RAG integrado.
Prompts optimizados para control de recursión.
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

llm = get_llm()

# --- Creación de Agentes Especialistas ---

messages_placeholder = MessagesPlaceholder(variable_name="messages")


def nodo_ayuda_directo(state: dict) -> dict:
    """Nodo simple que llama a la herramienta de ayuda directamente."""
    print("\n--- NODO AYUDA (DIRECTO) ---")
    try:
        guia_de_preguntas = obtener_ejemplos_de_uso.invoke({})
        return {
            "messages": [AIMessage(content=guia_de_preguntas)]
        }
    except Exception as e:
        print(f"❌ ERROR en nodo_ayuda_directo: {e}")
        return {
            "messages": [AIMessage(content=f"Error al obtener la guía de ayuda: {e}")]
        }


def nodo_rag(state: dict) -> dict:
    """Nodo que consulta la documentación CFA usando RAG."""
    print("\n--- AGENTE RAG ---")
    
    # Extraer última pregunta del usuario
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [AIMessage(
                content="Error: No hay mensajes en el estado."
            )]
        }
    
    last_message = messages[-1]
    
    # Extraer contenido
    if hasattr(last_message, 'content'):
        consulta = last_message.content
    else:
        consulta = str(last_message)
    
    print(f"📚 Consulta CFA: {consulta}")
    
    # Buscar en documentación usando RAG
    try:
        resultado = buscar_documentacion_financiera.invoke({"consulta": consulta})
        print(f"📄 Respuesta RAG generada")
        
        return {
            "messages": [AIMessage(content=resultado)]
        }
    
    except Exception as e:
        print(f"❌ Error en RAG: {e}")
        return {
            "messages": [AIMessage(
                content=f"Error al buscar en la documentación: {e}"
            )]
        }


def crear_agente_especialista(llm_instance, tools_list, system_prompt_text):
    """Función helper para crear un agente reactivo con prompt de sistema."""
    if not tools_list or not all(hasattr(t, 'invoke') for t in tools_list):
        raise ValueError("tools_list debe contener al menos una herramienta válida (Runnable).")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        messages_placeholder,
    ])
    
    # LangChain 1.0: create_react_agent de langgraph.prebuilt
    return create_react_agent(llm_instance, tools_list, state_modifier=prompt)


# ========================================
# PROMPTS MEJORADOS DE AGENTES ESPECIALISTAS
# ========================================

PROMPT_RENTA_FIJA = """Eres un especialista en Renta Fija con una única responsabilidad: usar la herramienta 'calcular_valor_bono'.

**REGLAS ESTRICTAS:**
1. NUNCA respondas usando tu conocimiento general del LLM
2. SOLO puedes usar tu herramienta asignada
3. Revisa TODO el historial de mensajes para encontrar parámetros necesarios
4. Si encuentras todos los parámetros → Llama a tu herramienta
5. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
6. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA DESPUÉS DE USAR TU HERRAMIENTA:**
"El valor presente del bono es: [resultado]. 
Interpretación: [breve explicación del resultado].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:** 
- NO repitas los inputs del usuario en tu respuesta
- Sé conciso: resultado + interpretación breve
- SIEMPRE termina con "Devuelvo al supervisor"
"""

PROMPT_FIN_CORP = """Eres un especialista en Finanzas Corporativas con acceso a dos herramientas: 'calcular_van' y 'calcular_wacc'.

**REGLAS ESTRICTAS:**
1. NUNCA respondas usando tu conocimiento general del LLM
2. SOLO puedes usar tus dos herramientas asignadas
3. Revisa TODO el historial para encontrar parámetros necesarios
4. Identifica qué herramienta necesitas según la consulta
5. Si encuentras todos los parámetros → Llama a la herramienta apropiada
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA DESPUÉS DE USAR TU HERRAMIENTA:**

Para VAN:
"El VAN del proyecto es: [resultado].
Interpretación: [Si VAN > 0: proyecto rentable | Si VAN < 0: proyecto no rentable].
Tarea completada. Devuelvo al supervisor."

Para WACC:
"El WACC de la empresa es: [resultado]%.
Interpretación: [Breve explicación del costo de capital calculado].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:** 
- NO repitas los inputs del usuario
- Sé conciso y directo
- SIEMPRE termina con "Devuelvo al supervisor"
"""

PROMPT_EQUITY = """Eres un especialista en valoración de Equity con una única responsabilidad: usar la herramienta 'calcular_gordon_growth'.

**REGLAS ESTRICTAS:**
1. NUNCA respondas usando tu conocimiento general del LLM
2. SOLO puedes usar tu herramienta asignada
3. Revisa TODO el historial para encontrar los 3 parámetros necesarios:
   - D1 (dividendo próximo periodo)
   - Ke (costo del equity / tasa de descuento)
   - g (tasa de crecimiento)
4. IMPORTANTE: Si otra tarea calculó Ke previamente, USA ese valor del historial
5. Si encuentras los 3 parámetros → Llama a tu herramienta
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA DESPUÉS DE USAR TU HERRAMIENTA:**
"El valor intrínseco de la acción es: [resultado].
Interpretación: [Breve explicación del resultado].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:** 
- NO repitas los inputs del usuario
- Busca activamente en el historial valores calculados previamente
- SIEMPRE termina con "Devuelvo al supervisor"
"""

PROMPT_PORTAFOLIO = """Eres un especialista en Gestión de Portafolios con acceso a dos herramientas: 'calcular_capm' y 'calcular_sharpe_ratio'.

**REGLAS ESTRICTAS:**
1. NUNCA respondas usando tu conocimiento general del LLM
2. SOLO puedes usar tus dos herramientas asignadas
3. Revisa TODO el historial para encontrar parámetros necesarios
4. Identifica qué herramienta necesitas según la consulta
5. Si encuentras todos los parámetros → Llama a la herramienta apropiada
6. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
7. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA DESPUÉS DE USAR TU HERRAMIENTA:**

Para CAPM:
"El costo del equity (Ke) es: [resultado]%.
Interpretación: [Breve explicación del resultado].
Tarea completada. Devuelvo al supervisor."

Para Sharpe Ratio:
"El Sharpe Ratio del portafolio es: [resultado].
Interpretación: [Breve explicación de la calidad del retorno ajustado por riesgo].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:** 
- NO repitas los inputs del usuario
- Sé conciso y directo
- SIEMPRE termina con "Devuelvo al supervisor"
"""

PROMPT_DERIVADOS = """Eres un especialista en Instrumentos Derivados con una única responsabilidad: usar la herramienta 'calcular_opcion_call'.

**REGLAS ESTRICTAS:**
1. NUNCA respondas usando tu conocimiento general del LLM
2. SOLO puedes usar tu herramienta asignada (Black-Scholes para opciones Call europeas)
3. Revisa TODO el historial para encontrar los 5 parámetros necesarios:
   - S (precio actual del activo)
   - K (precio de ejercicio)
   - T (tiempo hasta vencimiento en años)
   - r (tasa libre de riesgo)
   - sigma (volatilidad)
4. Si encuentras los 5 parámetros → Llama a tu herramienta
5. Si faltan parámetros → Di: "Faltan parámetros: [lista específica]. Devuelvo al supervisor."
6. Si te piden algo fuera de tu especialidad → Di: "No es mi especialidad. Devuelvo al supervisor."

**FORMATO DE RESPUESTA DESPUÉS DE USAR TU HERRAMIENTA:**
"El valor de la opción Call es: [resultado].
Interpretación: [Breve explicación del resultado según Black-Scholes].
Tarea completada. Devuelvo al supervisor."

**IMPORTANTE:** 
- NO repitas los inputs del usuario
- Sé conciso y directo
- SIEMPRE termina con "Devuelvo al supervisor"
"""


# Crear agentes con prompts mejorados
try:
    agent_renta_fija = crear_agente_especialista(llm, [_calcular_valor_presente_bono], PROMPT_RENTA_FIJA)
    agent_fin_corp = crear_agente_especialista(llm, [_calcular_van, _calcular_wacc], PROMPT_FIN_CORP)
    agent_equity = crear_agente_especialista(llm, [_calcular_gordon_growth], PROMPT_EQUITY)
    agent_portafolio = crear_agente_especialista(llm, [_calcular_capm, _calcular_sharpe_ratio], PROMPT_PORTAFOLIO)
    agent_derivados = crear_agente_especialista(llm, [_calcular_opcion_call], PROMPT_DERIVADOS)
except Exception as e:
    print(f"❌ ERROR CRÍTICO al crear agentes especialistas: {e}")
    import streamlit as st
    st.error(f"Error inicializando los agentes: {e}")
    st.stop()


# Diccionario de nodos
agent_nodes = {
    "Agente_Renta_Fija": agent_renta_fija,
    "Agente_Finanzas_Corp": agent_fin_corp,
    "Agente_Equity": agent_equity,
    "Agente_Portafolio": agent_portafolio,
    "Agente_Derivados": agent_derivados,
    "Agente_Ayuda": nodo_ayuda_directo,
    "Agente_RAG": nodo_rag,
}

# --- Supervisor ---

class RouterSchema(BaseModel):
    """Elige el siguiente agente a llamar o finaliza."""
    next_agent: Literal[tuple(list(agent_nodes.keys()) + ["FINISH"])] = Field(
        description="El nombre del agente especialista para la tarea. Elige 'FINISH' si la solicitud fue completamente respondida."
    )


# Configurar el LLM supervisor
try:
    supervisor_llm = llm.with_structured_output(RouterSchema)
except Exception as e:
    print(f"❌ ERROR configurando supervisor LLM con structured_output: {e}")
    import streamlit as st
    st.error(f"Error configurando el supervisor: {e}")
    st.stop()


# ========================================
# PROMPT MEJORADO DEL SUPERVISOR
# ========================================

supervisor_system_prompt = """Eres un supervisor eficiente de un equipo de analistas financieros especializados.

**TU ÚNICA FUNCIÓN:** 
Analizar el historial completo de la conversación y decidir:
- ¿Qué especialista debe actuar AHORA?
- ¿O ya terminamos la tarea? (FINISH)

**ESPECIALISTAS DISPONIBLES Y SUS HERRAMIENTAS:**
- Agente_Renta_Fija: calcular_valor_bono
- Agente_Finanzas_Corp: calcular_van, calcular_wacc
- Agente_Equity: calcular_gordon_growth
- Agente_Portafolio: calcular_capm, calcular_sharpe_ratio
- Agente_Derivados: calcular_opcion_call
- Agente_Ayuda: obtener_ejemplos_de_uso
- Agente_RAG: buscar_documentacion_financiera

**PROCESO DE DECISIÓN (aplicar en este orden exacto):**

1️⃣ **VERIFICAR SI YA TERMINAMOS:**
   Elige 'FINISH' si:
   - El último mensaje del asistente contiene un resultado numérico completo
   - O el último mensaje dice "Tarea completada" o "Devuelvo al supervisor"
   - Y NO hay una nueva pregunta del usuario después
   - Y NO hay tareas pendientes sin resolver
   
   🚨 IMPORTANTE: Si el último agente completó su trabajo y reportó resultado, elige FINISH.

2️⃣ **DETECTAR TIPO DE CONSULTA DEL USUARIO:**
   
   A) **Consulta de Ayuda:**
      Palabras clave: "ayuda", "ejemplos", "qué puedes hacer", "cómo funciona", "guía"
      → Elige 'Agente_Ayuda'
   
   B) **Consulta Teórica/Conceptual:**
      Patrones: "qué dice el CFA", "explica el concepto", "según CFA", "qué es", "busca en la documentación"
      → Elige 'Agente_RAG'
   
   C) **Cálculo Financiero:**
      Identifica la herramienta necesaria:
      - VAN, NPV, TIR, flujos de caja, valor actual neto → Agente_Finanzas_Corp (usa calcular_van)
      - WACC, costo de capital, estructura de capital → Agente_Finanzas_Corp (usa calcular_wacc)
      - Bono, bond, cupón, YTM, yield → Agente_Renta_Fija
      - Gordon, dividendos, valoración de acciones, DDM → Agente_Equity
      - CAPM, beta, costo equity → Agente_Portafolio (usa calcular_capm)
      - Sharpe, ratio, riesgo ajustado → Agente_Portafolio (usa calcular_sharpe_ratio)
      - Opción, call, put, Black-Scholes → Agente_Derivados

3️⃣ **EVITAR BUCLES INFINITOS:**
   - Revisa el historial: ¿el agente que vas a elegir ya fue llamado recientemente?
   - Si sí, y no hay nueva información del usuario → Elige 'FINISH'
   - NUNCA envíes al mismo agente dos veces consecutivas sin que haya nueva info del usuario

4️⃣ **MANEJO DE ERRORES:**
   Si el último mensaje contiene:
   - "No es mi especialidad" → Elige el agente apropiado
   - "Faltan parámetros" → Si el usuario NO proporcionó nueva info → Elige 'FINISH'
   - "Error" o "No puedo" → Intenta otro agente apropiado O elige 'FINISH'

5️⃣ **REGLA DE SEGURIDAD:**
   Si NO estás seguro qué hacer → Elige 'FINISH'
   (Es mejor terminar que crear un bucle infinito)

**RESPUESTA REQUERIDA:**
SOLO devuelve el nombre exacto del agente (ej: "Agente_Finanzas_Corp") o "FINISH".
NO agregues explicaciones, razonamientos ni texto adicional.

**EJEMPLOS DE DECISIÓN CORRECTA:**

Ejemplo 1:
Usuario: "Calcula el VAN: inversión 50k, flujos [15k, 20k, 25k], tasa 10%"
Historial: Solo ese mensaje
→ Decisión: Agente_Finanzas_Corp

Ejemplo 2:
Usuario: "Calcula el VAN: inversión 50k, flujos [15k, 20k, 25k], tasa 10%"
Agente_Finanzas_Corp: "El VAN es 3,542.10. Proyecto rentable. Tarea completada."
Historial: Solo esos 2 mensajes
→ Decisión: FINISH

Ejemplo 3:
Usuario: "¿Qué es el WACC según el CFA?"
Historial: Solo ese mensaje
→ Decisión: Agente_RAG

Ejemplo 4:
Usuario: "Ayuda con ejemplos"
Historial: Solo ese mensaje
→ Decisión: Agente_Ayuda

Ejemplo 5:
Usuario: "Calcula WACC: Ke=12%, Kd=8%, E=60M, D=40M, impuestos=25%"
Agente_Finanzas_Corp: "El WACC es 10.4%. Tarea completada."
Usuario: "Ahora calcula el VAN con WACC de 10.4%, inversión 100k, flujos [30k, 40k, 50k]"
→ Decisión: Agente_Finanzas_Corp

Ejemplo 6:
Usuario: "Calcula el VAN"
Agente_Finanzas_Corp: "Faltan parámetros: inversión_inicial, flujos_caja, tasa_descuento."
Historial: Solo esos 2 mensajes (usuario NO dio nueva info)
→ Decisión: FINISH

**RECUERDA:**
- Analiza TODO el historial antes de decidir
- Prioriza FINISH cuando la tarea esté completa
- NO repitas agentes sin progreso
- Sé conservador: ante duda, elige FINISH
"""

print("✅ Módulo financial_agents cargado con prompts mejorados (LangChain 1.0 + RAG + control de recursión optimizado).")