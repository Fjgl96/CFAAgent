"""
Tests actualizados para el sistema de routing (LangChain-Native).
Valida la lógica de detección de patrones incluyendo las nuevas intenciones RAG.
"""

import os
import sys
from pathlib import Path

# Asegurar que el directorio raíz está en el path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from routing.langchain_routing import analyze_query_fast_pattern, load_routing_patterns

# ========================================
# TEST CASES
# ========================================

TEST_CASES = [
    # --- CÁLCULOS (Debe mantenerse igual) ---
    (
        "Calcula el VAN de un proyecto con inversión inicial de 100000, flujos [30000, 40000, 50000] y tasa 10%",
        "Agente_Finanzas_Corp",
        True,
        "Cálculo VAN completo con todos los parámetros"
    ),
    
    # --- RAG / TEORÍA (NUEVO COMPORTAMIENTO) ---
    (
        "¿Qué es el VAN?",
        "Agente_RAG",
        True, # Bypass activado
        "Pregunta teórica directa -> Bypass a RAG"
    ),
    (
        "Explica el concepto de WACC y sus componentes",
        "Agente_RAG",
        True,
        "Solicitud de explicación -> Bypass a RAG"
    ),
    (
        "What is the internal rate of return?",
        "Agente_RAG",
        True,
        "Pregunta teórica en inglés -> Bypass a RAG"
    ),

    # --- CASOS MIXTOS (Prioridad Cálculo) ---
    (
        "Calcula el WACC y explica qué es",
        "Agente_Finanzas_Corp", 
        False, # Baja confianza por params, pero NO va a RAG
        "Intención mixta: 'Calcula' gana sobre 'qué es'"
    ),

    # --- FALLBACK ---
    (
        "Necesito ayuda",
        "Supervisor",
        False,
        "Solicitud ambigua -> Supervisor"
    ),
]

def test_fast_pattern_logic():
    print("\n" + "="*60)
    print("TESTING FAST PATTERN LOGIC (LangChain-Native)")
    print("="*60 + "\n")

    config_path = Path(__file__).parent.parent / "config" / "routing_patterns.yaml"
    if not config_path.exists():
        print(f"⚠️ Advertencia: No se encontró {config_path}, usando defaults.")
        config_path = None
    
    patterns = load_routing_patterns(str(config_path) if config_path else None)

    results = {'passed': 0, 'failed': 0, 'total': len(TEST_CASES)}

    for query, expected_agent, expected_bypass, description in TEST_CASES:
        print(f"\n📝 Test: {description}\n   Query: '{query}'")
        state = {"messages": [HumanMessage(content=query)]}
        
        decision = analyze_query_fast_pattern(state, patterns)
        
        target_agent = decision.get('target_agent')
        confidence = decision.get('confidence', 0.0)
        bypass_actual = confidence >= 0.8

        agent_match = target_agent == expected_agent
        bypass_match = bypass_actual == expected_bypass

        print(f"   {'✅' if agent_match else '❌'} Agente: {target_agent} (Esp: {expected_agent})")
        print(f"   {'✅' if bypass_match else '⚠️'} Bypass: {bypass_actual} (Esp: {expected_bypass})")

        if agent_match and bypass_match:
            results['passed'] += 1
        else:
            results['failed'] += 1

    print("\n" + "="*60)
    print(f"✅ Pasaron: {results['passed']}/{results['total']}")
    print(f"❌ Fallaron: {results['failed']}/{results['total']}")

if __name__ == "__main__":
    test_fast_pattern_logic()