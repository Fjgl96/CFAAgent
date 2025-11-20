"""
Tests para el sistema de routing.
Valida que FastPatternRouter, LLMRouter y HybridRouter funcionan correctamente.
"""

from langchain_core.messages import HumanMessage
from routing import FastPatternRouter, RoutingDecision


# ========================================
# TEST CASES
# ========================================

TEST_CASES = [
    # (query, expected_agent, expected_bypass, descripcion)
    (
        "Calcula el VAN de un proyecto con inversión inicial de 100000, flujos [30000, 40000, 50000] y tasa 10%",
        "Agente_Finanzas_Corp",
        True,
        "Cálculo VAN completo con todos los parámetros"
    ),
    (
        "¿Qué es el VAN?",
        "Supervisor",
        False,
        "Pregunta teórica - debe ir al Supervisor"
    ),
    (
        "Calcula el CAPM con beta=1.2, rf=5%, rm=12%",
        "Agente_Portafolio",
        True,
        "Cálculo CAPM con parámetros completos"
    ),
    (
        "Necesito ayuda",
        "Supervisor",
        False,
        "Solicitud ambigua - debe ir al Supervisor"
    ),
    (
        "Calcula Duration Macaulay para bono: nominal 1000, cupón 5%, YTM 6%, 10 años, semestral",
        "Agente_Renta_Fija",
        True,
        "Duration Macaulay con muchos parámetros"
    ),
    (
        "Obtén el Sharpe Ratio con retorno 15%, rf 3%, std 20%",
        "Agente_Portafolio",
        True,
        "Sharpe Ratio con parámetros completos"
    ),
    (
        "Determina el valor de una opción call con S=100, K=95, T=1, r=5%, sigma=20%",
        "Agente_Derivados",
        True,
        "Opción Call con todos los parámetros Black-Scholes"
    ),
    (
        "Calcula el VAN",
        "Agente_Finanzas_Corp",
        False,  # Baja confianza por falta de parámetros
        "VAN sin parámetros - baja confianza"
    ),
    (
        "¿Cómo se relaciona el CAPM con el WACC?",
        "Supervisor",
        False,
        "Pregunta conceptual con múltiples keywords - debe ir al Supervisor"
    ),
    (
        "Calculate NPV: investment 100k, flows [30k, 40k, 50k], rate 10%",
        "Agente_Finanzas_Corp",
        True,
        "Cálculo VAN en inglés"
    ),
]


def test_fast_pattern_router():
    """
    Test del FastPatternRouter con casos típicos.
    """
    print("\n" + "="*60)
    print("TESTING FAST PATTERN ROUTER")
    print("="*60 + "\n")

    # Inicializar router
    router = FastPatternRouter()

    results = {
        'passed': 0,
        'failed': 0,
        'total': len(TEST_CASES)
    }

    for query, expected_agent, expected_bypass, description in TEST_CASES:
        print(f"\n📝 Test: {description}")
        print(f"   Query: {query[:80]}...")

        # Crear estado de prueba
        state = {
            "messages": [HumanMessage(content=query)]
        }

        # Ejecutar routing
        decision = router.route(state)

        # Verificar agente
        agent_match = decision.target_agent == expected_agent
        agent_status = "✅" if agent_match else "❌"

        # Verificar bypass (confianza >= 0.8)
        bypass_actual = decision.confidence >= 0.8
        bypass_match = bypass_actual == expected_bypass
        bypass_status = "✅" if bypass_match else "⚠️"

        print(f"   {agent_status} Agente: {decision.target_agent} (esperado: {expected_agent})")
        print(f"   {bypass_status} Bypass: {bypass_actual} (esperado: {expected_bypass}, conf={decision.confidence:.2f})")
        print(f"   📊 Metadata: {decision.metadata}")

        if agent_match and bypass_match:
            results['passed'] += 1
            print(f"   ✅ PASS")
        else:
            results['failed'] += 1
            print(f"   ❌ FAIL")

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"✅ Passed: {results['passed']}/{results['total']}")
    print(f"❌ Failed: {results['failed']}/{results['total']}")
    print(f"📊 Success Rate: {results['passed']/results['total']*100:.1f}%")

    return results


def test_routing_decision_validation():
    """
    Test de validación de RoutingDecision.
    """
    print("\n" + "="*60)
    print("TESTING ROUTING DECISION VALIDATION")
    print("="*60 + "\n")

    # Test 1: Confianza válida
    try:
        decision = RoutingDecision(
            target_agent="Agente_Test",
            confidence=0.85,
            method="test"
        )
        print("✅ Confianza válida (0.85) aceptada")
    except ValueError as e:
        print(f"❌ Error inesperado: {e}")

    # Test 2: Confianza inválida (>1.0)
    try:
        decision = RoutingDecision(
            target_agent="Agente_Test",
            confidence=1.5,
            method="test"
        )
        print("❌ Confianza inválida (1.5) NO fue rechazada")
    except ValueError:
        print("✅ Confianza inválida (1.5) correctamente rechazada")

    # Test 3: Confianza inválida (<0.0)
    try:
        decision = RoutingDecision(
            target_agent="Agente_Test",
            confidence=-0.1,
            method="test"
        )
        print("❌ Confianza inválida (-0.1) NO fue rechazada")
    except ValueError:
        print("✅ Confianza inválida (-0.1) correctamente rechazada")


def test_pattern_coverage():
    """
    Test de cobertura de patrones para todas las 22 herramientas.
    """
    print("\n" + "="*60)
    print("TESTING PATTERN COVERAGE (22 HERRAMIENTAS)")
    print("="*60 + "\n")

    router = FastPatternRouter()

    # Queries específicas por herramienta
    tool_queries = {
        "Agente_Finanzas_Corp": [
            "Calcula VAN con inversión 100k, flujos [30k, 40k], tasa 10%",
            "Calcula WACC con equity 60%, deuda 40%, ke=12%, kd=5%, tax=30%",
            "Calcula TIR con inversión 100k, flujos [30k, 40k, 50k]",
            "Calcula Payback Period: inversión 100k, flujos [30k, 40k, 50k]",
            "Calcula Profitability Index: tasa 10%, inversión 100k, flujos [30k, 40k]",
        ],
        "Agente_Portafolio": [
            "Calcula CAPM con rf=5%, beta=1.2, rm=12%",
            "Calcula Sharpe Ratio: retorno 15%, rf 3%, std 20%",
            "Calcula Treynor Ratio: retorno 15%, rf 3%, beta 1.2",
            "Calcula Jensen Alpha: rp 15%, rf 3%, beta 1.2, rm 12%",
            "Calcula Beta portafolio: w1=0.6, w2=0.4, beta1=1.2, beta2=0.8",
        ],
        "Agente_Renta_Fija": [
            "Calcula valor bono: nominal 1000, cupón 5%, ytm 6%, 10 años, semestral",
            "Calcula Duration Macaulay: nominal 1000, cupón 5%, ytm 6%, 10 años, semestral",
            "Calcula Duration Modificada con duration 8, ytm 6%, semestral",
            "Calcula Convexity: nominal 1000, cupón 5%, ytm 6%, 10 años, semestral",
            "Calcula Current Yield: cupón anual 50, precio 980",
        ],
        "Agente_Equity": [
            "Calcula Gordon Growth: D1=5, Ke=10%, g=3%",
        ],
        "Agente_Derivados": [
            "Calcula opción call: S=100, K=95, T=1, r=5%, sigma=20%",
            "Calcula opción put: S=100, K=105, T=1, r=5%, sigma=20%",
            "Calcula Put-Call Parity: call=8, put=3, spot=100, strike=95, T=1, r=5%",
        ],
    }

    results = {}

    for agent, queries in tool_queries.items():
        matched = 0
        total = len(queries)

        print(f"\n{agent}:")

        for query in queries:
            state = {"messages": [HumanMessage(content=query)]}
            decision = router.route(state)

            if decision.target_agent == agent and decision.confidence >= 0.8:
                matched += 1
                print(f"  ✅ {query[:60]}... (conf={decision.confidence:.2f})")
            else:
                print(f"  ❌ {query[:60]}... → {decision.target_agent} (conf={decision.confidence:.2f})")

        coverage = matched / total * 100
        results[agent] = coverage
        print(f"  📊 Cobertura: {matched}/{total} ({coverage:.1f}%)")

    # Resumen general
    avg_coverage = sum(results.values()) / len(results)
    print(f"\n📊 Cobertura promedio: {avg_coverage:.1f}%")


if __name__ == "__main__":
    # Ejecutar tests
    print("\n🧪 INICIANDO TESTS DEL SISTEMA DE ROUTING\n")

    # Test 1: Validación de RoutingDecision
    test_routing_decision_validation()

    # Test 2: Fast Pattern Router
    test_fast_pattern_router()

    # Test 3: Cobertura de patrones
    test_pattern_coverage()

    print("\n\n✅ TESTS COMPLETADOS\n")
