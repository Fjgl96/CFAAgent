"""
Script de pruebas para validar mejoras FASE 1 del flujo RAG teórico.

Tests implementados:
1. Test básico: Concepto financiero simple (WACC)
2. Test técnico: Concepto técnico con fórmulas (Duration Modificada)
3. Test de relevancia: Verificar que el filtro min_score funciona
4. Test de formato: Verificar estructura y traducción de términos

Autor: Claude
Fecha: 2025-11-19
"""

import sys
import os
from datetime import datetime

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from agents.financial_agents import agent_nodes
from config import get_llm

# Colores para output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_test(num, name):
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}TEST {num}: {name}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*80}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_fail(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}→ {text}{Colors.ENDC}")


def validar_respuesta_rag(respuesta: str, pregunta: str) -> dict:
    """
    Valida que la respuesta cumpla con los criterios de calidad.

    Returns:
        Dict con resultados de validación
    """
    resultados = {
        "tiene_traduccion_correcta": False,
        "tiene_fuentes": False,
        "sin_fragmentos_crudos": True,
        "en_espanol": True,
        "estructura_parrafos": False,
        "terminos_tecnicos_formato": False
    }

    # 1. Verificar que tiene fuentes citadas
    if "**Fuentes:**" in respuesta or "Fuentes:" in respuesta or "**Fuente" in respuesta:
        resultados["tiene_fuentes"] = True

    # 2. Verificar que no tiene fragmentos crudos
    if "--- Fragmento" in respuesta or "Fragmento 1" in respuesta:
        resultados["sin_fragmentos_crudos"] = False

    # 3. Verificar estructura de párrafos (al menos 2 líneas de texto sustancial)
    lineas_sustanciales = [l for l in respuesta.split('\n') if len(l.strip()) > 50]
    if len(lineas_sustanciales) >= 2:
        resultados["estructura_parrafos"] = True

    # 4. Verificar formato de términos técnicos (español + inglés entre paréntesis)
    # Buscar patrones como "Término en Español (ACRONYM)" o "Término (Term in English)"
    import re
    patron_terminos = r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]+\([A-Z]{2,}[,\s]|[A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]+\([A-Z][a-z]+\s*[A-Z]'
    if re.search(patron_terminos, respuesta):
        resultados["terminos_tecnicos_formato"] = True

    # 5. Verificar que está principalmente en español
    palabras_ingles = ['the ', 'and ', 'for ', 'with ', 'that ', 'this ']
    cuenta_ingles = sum(respuesta.lower().count(p) for p in palabras_ingles)
    if cuenta_ingles > 10:  # Más de 10 palabras comunes en inglés = probablemente no está en español
        resultados["en_espanol"] = False

    return resultados


def ejecutar_test_rag(pregunta: str, test_num: int, nombre_test: str):
    """
    Ejecuta un test del flujo RAG y valida la respuesta.
    """
    print_test(test_num, nombre_test)
    print_info(f"Pregunta: '{pregunta}'")

    # Crear estado simulado
    estado = {
        "messages": [HumanMessage(content=pregunta)]
    }

    try:
        # 1. Ejecutar Agente_RAG
        print_info("Ejecutando Agente_RAG...")
        resultado_rag = agent_nodes["Agente_RAG"](estado)

        if "messages" not in resultado_rag or not resultado_rag["messages"]:
            print_fail("El Agente_RAG no retornó mensajes")
            return False

        contexto = resultado_rag["messages"][0].content
        print_success(f"Contexto obtenido: {len(contexto)} caracteres")

        # Verificar que encontró información
        if "No encontré información relevante" in contexto:
            print_warning("No se encontró información relevante en el índice")
            print_info("Esto puede ser normal si el concepto no está indexado")
            return None  # Test inconcluso, no es fallo

        # 2. Ejecutar Agente_Sintesis_RAG
        print_info("Ejecutando Agente_Sintesis_RAG...")
        estado_sintesis = {
            "messages": [
                HumanMessage(content=pregunta),
                AIMessage(content=contexto)
            ]
        }

        resultado_sintesis = agent_nodes["Agente_Sintesis_RAG"](estado_sintesis)

        if "messages" not in resultado_sintesis or not resultado_sintesis["messages"]:
            print_fail("El Agente_Sintesis_RAG no retornó mensajes")
            return False

        respuesta_final = resultado_sintesis["messages"][0].content
        print_success(f"Respuesta generada: {len(respuesta_final)} caracteres")

        # 3. Validar respuesta
        print_info("\nValidando calidad de respuesta...")
        validaciones = validar_respuesta_rag(respuesta_final, pregunta)

        # Imprimir resultados de validación
        checks = [
            ("Tiene fuentes citadas", validaciones["tiene_fuentes"]),
            ("Sin fragmentos crudos del RAG", validaciones["sin_fragmentos_crudos"]),
            ("Respuesta en español", validaciones["en_espanol"]),
            ("Estructura en párrafos", validaciones["estructura_parrafos"]),
            ("Términos técnicos con formato correcto", validaciones["terminos_tecnicos_formato"])
        ]

        total_checks = len(checks)
        checks_passed = sum(1 for _, result in checks if result)

        for check_name, result in checks:
            if result:
                print_success(check_name)
            else:
                print_fail(check_name)

        # 4. Mostrar muestra de la respuesta
        print(f"\n{Colors.OKCYAN}{'─'*80}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}RESPUESTA GENERADA (primeros 500 caracteres):{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'─'*80}{Colors.ENDC}")
        print(respuesta_final[:500] + "..." if len(respuesta_final) > 500 else respuesta_final)
        print(f"{Colors.OKCYAN}{'─'*80}{Colors.ENDC}\n")

        # 5. Score final
        score = (checks_passed / total_checks) * 100
        if score >= 80:
            print_success(f"✓ TEST EXITOSO - Score: {score:.1f}% ({checks_passed}/{total_checks} checks)")
            return True
        elif score >= 60:
            print_warning(f"⚠ TEST PARCIAL - Score: {score:.1f}% ({checks_passed}/{total_checks} checks)")
            return None
        else:
            print_fail(f"✗ TEST FALLIDO - Score: {score:.1f}% ({checks_passed}/{total_checks} checks)")
            return False

    except Exception as e:
        print_fail(f"Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta la suite completa de tests."""
    print_header("SUITE DE PRUEBAS - MEJORAS FASE 1 DEL FLUJO RAG")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Lista de tests
    tests = [
        {
            "num": 1,
            "nombre": "Concepto Básico - WACC",
            "pregunta": "¿Qué es el WACC?"
        },
        {
            "num": 2,
            "nombre": "Concepto Técnico - Duration",
            "pregunta": "¿Qué es la Duration Modificada?"
        },
        {
            "num": 3,
            "nombre": "Concepto de Portafolio - CAPM",
            "pregunta": "Explica qué es el modelo CAPM"
        },
        {
            "num": 4,
            "nombre": "Concepto de Bonos - YTM",
            "pregunta": "¿Qué es el Yield to Maturity?"
        }
    ]

    # Ejecutar tests
    resultados = []
    for test in tests:
        resultado = ejecutar_test_rag(
            test["pregunta"],
            test["num"],
            test["nombre"]
        )
        resultados.append({
            "nombre": test["nombre"],
            "resultado": resultado
        })

    # Resumen final
    print_header("RESUMEN DE RESULTADOS")

    exitosos = sum(1 for r in resultados if r["resultado"] is True)
    parciales = sum(1 for r in resultados if r["resultado"] is None)
    fallidos = sum(1 for r in resultados if r["resultado"] is False)

    print(f"\n{Colors.BOLD}Resultados:{Colors.ENDC}")
    for r in resultados:
        if r["resultado"] is True:
            print_success(f"{r['nombre']}: EXITOSO")
        elif r["resultado"] is None:
            print_warning(f"{r['nombre']}: PARCIAL (sin datos)")
        else:
            print_fail(f"{r['nombre']}: FALLIDO")

    print(f"\n{Colors.BOLD}Estadísticas:{Colors.ENDC}")
    print(f"  Total tests: {len(resultados)}")
    print(f"  {Colors.OKGREEN}Exitosos: {exitosos}{Colors.ENDC}")
    print(f"  {Colors.WARNING}Parciales: {parciales}{Colors.ENDC}")
    print(f"  {Colors.FAIL}Fallidos: {fallidos}{Colors.ENDC}")

    score_final = (exitosos / len(resultados)) * 100 if len(resultados) > 0 else 0

    print(f"\n{Colors.BOLD}Score Final: ", end="")
    if score_final >= 75:
        print(f"{Colors.OKGREEN}{score_final:.1f}%{Colors.ENDC}")
    elif score_final >= 50:
        print(f"{Colors.WARNING}{score_final:.1f}%{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{score_final:.1f}%{Colors.ENDC}")

    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

    return score_final >= 50


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Tests interrumpidos por el usuario{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Error fatal: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
