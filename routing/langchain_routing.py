"""
Sistema de Routing usando herramientas nativas de LangChain.
Refactorizado para usar Runnables, LCEL y patrones idiomáticos de LangChain.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.messages import HumanMessage

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('routing_langchain')
except ImportError:
    import logging
    logger = logging.getLogger('routing_langchain')


# ========================================
# HELPER: CARGAR PATRONES DESDE YAML
# ========================================

def load_routing_patterns(config_path: Optional[str] = None) -> Dict:
    """
    Carga patrones de routing desde archivo YAML.

    Args:
        config_path: Ruta al archivo YAML de configuración

    Returns:
        Diccionario con patrones de routing
    """
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                patterns = yaml.safe_load(f)
                logger.info(f"📄 Patrones cargados desde {config_path}")
                return patterns
        except Exception as e:
            logger.error(f"❌ Error cargando {config_path}: {e}")

    # Patrones por defecto (fallback)
    logger.warning("⚠️ Usando patrones por defecto hardcoded")
    return get_default_patterns()


def get_default_patterns() -> Dict:
    """Retorna configuración por defecto."""
    return {
        'settings': {
            'confidence_threshold': 0.8,
            'min_params_for_bypass': 2
        },
        'calc_intent_patterns': {
            'spanish': [r'\bcalcula(?:r)?\b', r'\bobt[eé]n(?:er)?\b'],
            'english': [r'\bcalculate\b', r'\bcompute\b'],
        },
        'agent_mappings': [
            # Finanzas Corporativas
            {'agent': 'Agente_Finanzas_Corp', 'priority': 10, 'keywords': {'spanish': [r'\bvan\b', 'npv'], 'english': ['npv']}, 'required_params': 3},
            {'agent': 'Agente_Finanzas_Corp', 'priority': 9, 'keywords': {'spanish': [r'\bwacc\b'], 'english': ['wacc']}, 'required_params': 5},
            {'agent': 'Agente_Finanzas_Corp', 'priority': 9, 'keywords': {'spanish': [r'\btir\b', 'irr'], 'english': ['irr']}, 'required_params': 2},
            # Portafolio
            {'agent': 'Agente_Portafolio', 'priority': 10, 'keywords': {'spanish': [r'\bcapm\b'], 'english': ['capm']}, 'required_params': 3},
            {'agent': 'Agente_Portafolio', 'priority': 9, 'keywords': {'spanish': ['sharpe'], 'english': ['sharpe']}, 'required_params': 3},
            # Renta Fija
            {'agent': 'Agente_Renta_Fija', 'priority': 10, 'keywords': {'spanish': ['valor.*bono'], 'english': ['bond.*value']}, 'required_params': 5},
            {'agent': 'Agente_Renta_Fija', 'priority': 9, 'keywords': {'spanish': ['duration.*macaulay'], 'english': ['macaulay duration']}, 'required_params': 5},
            # Equity
            {'agent': 'Agente_Equity', 'priority': 10, 'keywords': {'spanish': ['gordon'], 'english': ['gordon growth']}, 'required_params': 3},
            # Derivados
            {'agent': 'Agente_Derivados', 'priority': 10, 'keywords': {'spanish': ['opci[oó]n.*call'], 'english': ['call option']}, 'required_params': 5},
        ],
        'param_patterns': [
            {'name': 'cantidad_k', 'regex': r'\d+(?:\.\d+)?k\b'},
            {'name': 'porcentaje', 'regex': r'\d+(?:\.\d+)?%'},
            {'name': 'lista', 'regex': r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]'},
            {'name': 'numero', 'regex': r'\d+(?:\.\d+)?'},
        ]
    }


# ========================================
# LÓGICA DE FAST PATTERN (Función pura)
# ========================================

def analyze_query_fast_pattern(state: Dict[str, Any], patterns: Dict) -> Dict[str, Any]:
    """
    Analiza query con pattern matching (lógica pura).

    Esta es la lógica core del FastPatternRouter, pero como función pura
    que puede ser wrapeada en RunnableLambda.

    Args:
        state: Estado del grafo con messages
        patterns: Diccionario de patrones cargados

    Returns:
        Dict con análisis: {
            'target_agent': str,
            'confidence': float,
            'method': str,
            'metadata': dict
        }
    """
    messages = state.get('messages', [])

    if not messages or not isinstance(messages[-1], HumanMessage):
        return {
            'target_agent': 'Supervisor',
            'confidence': 0.0,
            'method': 'fast_pattern',
            'metadata': {'reason': 'no_human_message'}
        }

    texto = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
    texto_lower = texto.lower()

    # 1. Detectar intención de cálculo
    has_intent = False
    for pattern in patterns['calc_intent_patterns'].get('spanish', []) + patterns['calc_intent_patterns'].get('english', []):
        if re.search(pattern, texto_lower, re.IGNORECASE):
            has_intent = True
            break

    # 2. Extraer parámetros numéricos
    params = []
    for param_config in patterns.get('param_patterns', []):
        matches = re.findall(param_config['regex'], texto)
        params.extend(matches)

    # 3. Identificar agente por keywords
    mappings = sorted(patterns.get('agent_mappings', []), key=lambda x: x.get('priority', 0), reverse=True)
    agent_mapping = None

    for mapping in mappings:
        for keyword in mapping['keywords'].get('spanish', []) + mapping['keywords'].get('english', []):
            if re.search(keyword, texto_lower, re.IGNORECASE):
                agent_mapping = mapping
                break
        if agent_mapping:
            break

    # 4. Calcular confianza
    confidence = 0.0
    if has_intent:
        confidence += 0.4
    if agent_mapping:
        confidence += 0.4
        required_params = agent_mapping.get('required_params', 3)
        if len(params) >= required_params:
            confidence += 0.2
        elif len(params) >= required_params - 1:
            confidence += 0.1

    target_agent = agent_mapping['agent'] if agent_mapping else 'Supervisor'

    logger.info(
        f"📊 Fast Pattern: {target_agent} "
        f"(conf={confidence:.2f}, intent={has_intent}, params={len(params)})"
    )

    return {
        'target_agent': target_agent,
        'confidence': confidence,
        'method': 'fast_pattern',
        'metadata': {
            'has_intent': has_intent,
            'params_detected': len(params),
            'params_sample': params[:3] if len(params) > 3 else params,
            'agent_priority': agent_mapping.get('priority', 0) if agent_mapping else 0
        }
    }


# ========================================
# CREACIÓN DE RUNNABLES (ENFOQUE LANGCHAIN)
# ========================================

def create_fast_pattern_runnable(config_path: Optional[str] = None) -> RunnableLambda:
    """
    Crea un Runnable que ejecuta fast pattern matching.

    Este es el enfoque idiomático de LangChain - convertir lógica en Runnable.

    Args:
        config_path: Ruta al archivo YAML de configuración

    Returns:
        RunnableLambda que analiza queries con pattern matching
    """
    # Cargar patrones una vez al crear el Runnable
    patterns = load_routing_patterns(config_path)

    # Crear RunnableLambda con la lógica de análisis
    # Nota: usamos lambda para capturar 'patterns' en el closure
    fast_pattern = RunnableLambda(
        lambda state: analyze_query_fast_pattern(state, patterns),
        name="fast_pattern_router"
    )

    logger.info("✅ FastPatternRunnable creado")
    return fast_pattern


def create_hybrid_routing_branch(
    supervisor_llm,
    supervisor_prompt: str,
    threshold: float = 0.8,
    config_path: Optional[str] = None
) -> RunnableBranch:
    """
    Crea un RunnableBranch que implementa routing híbrido.

    Este es el patrón correcto de LangChain para routing condicional.
    Usa RunnableBranch para decidir entre fast pattern y LLM.

    Args:
        supervisor_llm: LLM configurado para supervisor
        supervisor_prompt: Prompt del supervisor
        threshold: Umbral de confianza para bypass
        config_path: Ruta a configuración YAML

    Returns:
        RunnableBranch con lógica de routing híbrido
    """
    # Crear fast pattern runnable
    fast_pattern = create_fast_pattern_runnable(config_path)

    # Función auxiliar: evalúa si usar fast pattern
    def should_use_fast_pattern(state: Dict[str, Any]) -> bool:
        """Condición: ¿confidence >= threshold?"""
        # Ejecutar análisis fast
        analysis = analyze_query_fast_pattern(
            state,
            load_routing_patterns(config_path)
        )
        confidence = analysis.get('confidence', 0.0)

        # Guardar análisis en state para uso posterior
        state['_fast_analysis'] = analysis

        return confidence >= threshold

    # Función auxiliar: extrae decisión de fast pattern
    def extract_fast_decision(state: Dict[str, Any]) -> str:
        """Extrae target_agent del análisis fast guardado."""
        analysis = state.get('_fast_analysis', {})
        target = analysis.get('target_agent', 'Supervisor')

        logger.info(f"🚀 FAST BYPASS: {target} (conf={analysis.get('confidence', 0):.2f})")
        return target

    # Función auxiliar: usa supervisor LLM
    def use_supervisor_llm(state: Dict[str, Any]) -> str:
        """Fallback a supervisor LLM."""
        logger.info("⚠️ FALLBACK A SUPERVISOR LLM")

        messages = state.get('messages', [])
        supervisor_messages = [HumanMessage(content=supervisor_prompt)] + messages

        try:
            route = supervisor_llm.invoke(supervisor_messages)
            target = route.next_agent if hasattr(route, 'next_agent') else 'FINISH'

            logger.info(f"🧠 Supervisor LLM decide: {target}")
            return target
        except Exception as e:
            logger.error(f"❌ Error en supervisor: {e}")
            return 'FINISH'

    # Crear RunnableBranch (patrón idiomático de LangChain)
    hybrid_branch = RunnableBranch(
        # (condición, runnable_si_verdadero)
        (should_use_fast_pattern, RunnableLambda(extract_fast_decision, name="fast_decision")),
        # default: runnable_si_falso
        RunnableLambda(use_supervisor_llm, name="llm_decision")
    )

    logger.info(f"✅ HybridRoutingBranch creado (threshold={threshold})")
    return hybrid_branch


# ========================================
# INTEGRACIÓN CON LANGGRAPH
# ========================================

def create_routing_node(
    supervisor_llm,
    supervisor_prompt: str,
    threshold: float = 0.8,
    config_path: Optional[str] = None
):
    """
    Crea un nodo de routing compatible con LangGraph.

    Este nodo puede reemplazar directamente al supervisor_node original
    o usarse como pre-procesador.

    Args:
        supervisor_llm: LLM del supervisor
        supervisor_prompt: Prompt del supervisor
        threshold: Umbral para bypass
        config_path: Ruta a config YAML

    Returns:
        Función nodo compatible con LangGraph
    """
    # Crear routing branch
    routing_branch = create_hybrid_routing_branch(
        supervisor_llm=supervisor_llm,
        supervisor_prompt=supervisor_prompt,
        threshold=threshold,
        config_path=config_path
    )

    def routing_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nodo de routing que usa RunnableBranch.

        Compatible con LangGraph - retorna dict con 'next_node'.
        """
        logger.info("🔀 Routing node (LangChain-native) ejecutándose...")

        try:
            # Ejecutar routing branch (devuelve target_agent)
            next_agent = routing_branch.invoke(state)

            # Obtener metadata si existe
            analysis = state.get('_fast_analysis', {})

            return {
                'next_node': next_agent,
                'routing_method': analysis.get('method', 'llm'),
                'routing_confidence': analysis.get('confidence', 0.95),
                'routing_metadata': analysis.get('metadata', {})
            }

        except Exception as e:
            logger.error(f"❌ Error en routing node: {e}", exc_info=True)
            return {
                'next_node': 'FINISH',
                'routing_method': 'error_fallback',
                'routing_confidence': 0.0
            }

    return routing_node


# ========================================
# EJEMPLO DE USO
# ========================================

"""
Ejemplo de cómo usar esto en agent_graph.py:

from routing.langchain_routing import create_routing_node

# En vez de:
# def supervisor_node(state):
#     route = supervisor_llm.invoke(...)
#     return {'next_node': route.next_agent}

# Usar:
supervisor_node = create_routing_node(
    supervisor_llm=supervisor_llm,
    supervisor_prompt=supervisor_system_prompt,
    threshold=0.8,
    config_path="config/routing_patterns.yaml"
)

# El nodo es 100% compatible con LangGraph
workflow.add_node("Supervisor", supervisor_node)
"""
