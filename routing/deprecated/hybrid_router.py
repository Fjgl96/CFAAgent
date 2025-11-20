"""
Hybrid Router - Combina FastPatternRouter + LLMRouter.
Estrategia: Intenta fast primero, fallback a LLM si confianza baja.
"""

from typing import Dict, Any

from .interfaces import IRouter, RoutingDecision

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('hybrid_router')
except ImportError:
    import logging
    logger = logging.getLogger('hybrid_router')


class HybridRouter(IRouter):
    """
    Router híbrido que combina velocidad (Fast) con precisión (LLM).

    Estrategia de 2 niveles:
    1. Intenta FastPatternRouter (rápido, <10ms)
    2. Si confianza >= threshold → Usa resultado fast
    3. Si confianza < threshold → Fallback a LLM (preciso, ~1-2s)
    """

    def __init__(
        self,
        fast_router: IRouter,
        llm_router: IRouter,
        threshold: float = 0.8
    ):
        """
        Inicializa el router híbrido.

        Args:
            fast_router: Router rápido (FastPatternRouter)
            llm_router: Router LLM (LLMRouter)
            threshold: Umbral de confianza para usar fast router
        """
        self.fast = fast_router
        self.llm = llm_router
        self.threshold = threshold

        logger.info(
            f"✅ HybridRouter inicializado "
            f"(fast={fast_router.__class__.__name__}, "
            f"llm={llm_router.__class__.__name__}, "
            f"threshold={threshold})"
        )

    def can_handle(self, state: Dict[str, Any]) -> float:
        """
        El híbrido siempre puede manejar (tiene LLM fallback).

        Args:
            state: Estado del grafo

        Returns:
            Siempre 1.0
        """
        return 1.0

    def route(self, state: Dict[str, Any]) -> RoutingDecision:
        """
        Rutea usando estrategia híbrida.

        Proceso:
        1. Evalúa confianza del fast router
        2. Si >= threshold → usa fast
        3. Si < threshold → fallback a LLM

        Args:
            state: Estado del grafo

        Returns:
            RoutingDecision del router seleccionado
        """
        logger.info("🔀 HybridRouter: Iniciando análisis en 2 niveles...")

        # ========================================
        # NIVEL 1: FAST ROUTER (Intento)
        # ========================================

        fast_decision = self.fast.route(state)

        logger.info(
            f"📊 Fast Router: {fast_decision.target_agent} "
            f"(confianza={fast_decision.confidence:.2f})"
        )

        # ========================================
        # DECISIÓN: ¿Usar fast o fallback?
        # ========================================

        if fast_decision.confidence >= self.threshold:
            # ✅ CONFIANZA ALTA → Usar fast router
            logger.info(
                f"🚀 FAST BYPASS: {fast_decision.target_agent} "
                f"(confianza {fast_decision.confidence:.2f} >= {self.threshold})"
            )

            # Añadir metadata adicional
            fast_decision.method = "hybrid_fast"
            fast_decision.metadata['fast_bypass'] = True
            fast_decision.metadata['threshold_used'] = self.threshold

            return fast_decision

        else:
            # ⚠️ CONFIANZA BAJA → Fallback a LLM
            logger.info(
                f"⚠️ FALLBACK A LLM: confianza fast={fast_decision.confidence:.2f} "
                f"< threshold={self.threshold}"
            )

            llm_decision = self.llm.route(state)

            # Añadir metadata del intento fast
            llm_decision.method = "hybrid_llm_fallback"
            llm_decision.metadata['fast_attempted'] = True
            llm_decision.metadata['fast_confidence'] = fast_decision.confidence
            llm_decision.metadata['fast_agent'] = fast_decision.target_agent
            llm_decision.metadata['threshold_used'] = self.threshold

            logger.info(
                f"🧠 LLM Fallback: {llm_decision.target_agent} "
                f"(confianza={llm_decision.confidence:.2f})"
            )

            return llm_decision
