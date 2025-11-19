# utils/query_guardrails.py
"""
Guardrails semánticos para interceptar consultas riesgosas.
Implementa filtros para prevenir extracción literal de contenido protegido.
"""

import re
from typing import Tuple

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('guardrails')
except ImportError:
    import logging
    logger = logging.getLogger('guardrails')


# ========================================
# PATRONES DE RIESGO
# ========================================

# Patrones que indican solicitud de copia literal
PATRONES_COPIA_LITERAL = [
    # Solicitudes directas de transcripción
    r'\b(transcribe|transcribir|transcripción)\b',
    r'\b(copia|copiar|copíame)\b',
    r'\b(texto\s+(completo|exacto|literal|original))\b',
    r'\b(dame\s+el\s+texto)\b',
    r'\b(muestra\s+el\s+(contenido|texto)\s+completo)\b',

    # Solicitudes de capítulos/secciones completas
    r'\b(capítulo\s+\d+\s+completo)\b',
    r'\b(sección\s+completa)\b',
    r'\b(reading\s+\d+\s+completo)\b',
    r'\b(todo\s+el\s+(capítulo|reading|material))\b',

    # Solicitudes de páginas específicas
    r'\b(página\s+\d+)\b',
    r'\b(páginas\s+\d+\s*-\s*\d+)\b',

    # Solicitudes de citas textuales extensas
    r'\b(cita\s+textual(mente)?)\b',
    r'\b(cítame\s+(textual|literal))\b',
    r'\b(extracto\s+completo)\b',
    r'\b(párrafo\s+exacto)\b',

    # Solicitudes de reproducción PDF/documento
    r'\b(pdf\s+completo)\b',
    r'\b(documento\s+original)\b',
    r'\b(material\s+original)\b',
]

# Patrones que indican solicitud de estructura/outline
PATRONES_ESTRUCTURA = [
    r'\b(outline\s+completo)\b',
    r'\b(índice\s+completo)\b',
    r'\b(tabla\s+de\s+contenidos?\s+completa)\b',
    r'\b(estructura\s+del\s+(libro|material|curriculum))\b',
]

# Mensajes de respuesta
MENSAJE_RECHAZO_COPIA_LITERAL = """Por respeto a los derechos de autor del CFA Institute, no puedo proporcionar copias literales, transcripciones o reproducciones del material original.

**¿Cómo puedo ayudarte?**
- Explicar conceptos financieros con mis propias palabras
- Resolver ejercicios específicos de cálculo
- Aclarar dudas sobre fórmulas y metodologías
- Comparar diferentes modelos de valoración

Por favor, reformula tu pregunta enfocándote en el concepto o cálculo que quieres entender."""

MENSAJE_RECHAZO_ESTRUCTURA = """Por respeto a los derechos de autor, no puedo proporcionar la estructura completa, índice o outline del material CFA.

**Alternativas disponibles:**
- Explicarte temas específicos (ej: "Explica el WACC", "¿Cómo funciona Duration?")
- Resolver cálculos financieros concretos
- Comparar conceptos relacionados

¿Sobre qué tema específico quieres aprender?"""


# ========================================
# FUNCIONES DE DETECCIÓN
# ========================================

def detectar_copia_literal(query: str) -> bool:
    """
    Detecta si la query solicita copia literal de contenido.

    Args:
        query: Consulta del usuario

    Returns:
        True si detecta patrón de copia literal
    """
    query_lower = query.lower()

    for patron in PATRONES_COPIA_LITERAL:
        if re.search(patron, query_lower, re.IGNORECASE):
            logger.warning(f"⚠️ Patrón de copia literal detectado: {patron}")
            return True

    return False


def detectar_solicitud_estructura(query: str) -> bool:
    """
    Detecta si la query solicita estructura completa del material.

    Args:
        query: Consulta del usuario

    Returns:
        True si detecta patrón de solicitud de estructura
    """
    query_lower = query.lower()

    for patron in PATRONES_ESTRUCTURA:
        if re.search(patron, query_lower, re.IGNORECASE):
            logger.warning(f"⚠️ Patrón de estructura completa detectado: {patron}")
            return True

    return False


def validar_query_segura(query: str) -> Tuple[bool, str]:
    """
    Valida si una query es segura desde el punto de vista de copyright.

    Args:
        query: Consulta del usuario

    Returns:
        Tuple[bool, str]: (es_segura, mensaje_rechazo_o_vacio)
        - Si es_segura = True: mensaje = ""
        - Si es_segura = False: mensaje = explicación del rechazo
    """
    logger.info(f"🔍 Validando query: {query[:100]}...")

    # Check 1: Copia literal
    if detectar_copia_literal(query):
        logger.warning("❌ Query rechazada: solicitud de copia literal")
        return (False, MENSAJE_RECHAZO_COPIA_LITERAL)

    # Check 2: Estructura completa
    if detectar_solicitud_estructura(query):
        logger.warning("❌ Query rechazada: solicitud de estructura completa")
        return (False, MENSAJE_RECHAZO_ESTRUCTURA)

    # Query es segura
    logger.info("✅ Query validada como segura")
    return (True, "")


# ========================================
# FUNCIÓN PÚBLICA
# ========================================

def aplicar_guardrails(query: str) -> Tuple[bool, str]:
    """
    Aplica guardrails semánticos a una consulta.

    Esta es la función pública que debe usarse en el flujo del agente.

    Args:
        query: Consulta del usuario

    Returns:
        Tuple[bool, str]:
        - (True, ""): Query aprobada, puede procesarse
        - (False, mensaje): Query rechazada, retornar mensaje al usuario

    Ejemplo:
        >>> aprobada, mensaje = aplicar_guardrails("Explica el WACC")
        >>> if not aprobada:
        >>>     return mensaje
    """
    return validar_query_segura(query)


# ========================================
# TESTS (solo para debugging)
# ========================================

if __name__ == "__main__":
    # Test casos riesgosos
    queries_riesgosas = [
        "Transcribe el capítulo 5 completo",
        "Dame el texto completo sobre WACC",
        "Copia literal de la página 45",
        "Muéstrame el outline completo del CFA Level I",
    ]

    # Test casos seguros
    queries_seguras = [
        "Explica qué es el WACC",
        "¿Cómo se calcula Duration?",
        "Dame un ejemplo de Gordon Growth Model",
        "¿Cuál es la diferencia entre Call y Put?",
    ]

    print("=" * 60)
    print("TESTING GUARDRAILS SEMÁNTICOS")
    print("=" * 60)

    print("\n🔴 QUERIES RIESGOSAS (deben rechazarse):")
    for q in queries_riesgosas:
        aprobada, msg = aplicar_guardrails(q)
        status = "❌ RECHAZADA" if not aprobada else "⚠️ APROBADA (ERROR!)"
        print(f"{status}: {q}")

    print("\n🟢 QUERIES SEGURAS (deben aprobarse):")
    for q in queries_seguras:
        aprobada, msg = aplicar_guardrails(q)
        status = "✅ APROBADA" if aprobada else "⚠️ RECHAZADA (ERROR!)"
        print(f"{status}: {q}")

print("✅ Módulo query_guardrails cargado.")
