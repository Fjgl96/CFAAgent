"""
LLM Router - Wrapper del Supervisor actual.
IMPORTANTE: NO modifica el prompt del supervisor - solo wrappea la lógica existente.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage

from .interfaces import IRouter, RoutingDecision

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('llm_router')
except ImportError:
    import logging
    logger = logging.getLogger('llm_router')


class LLMRouter(IRouter):
    """
    Router que usa el LLM del supervisor para tomar decisiones.

    CRÍTICO: Este router NO toca el prompt del supervisor.
    Solo wrappea la lógica existente en una interfaz IRouter.
    """

    def __init__(
        self,
        supervisor_llm,
        supervisor_prompt: str,
        router_schema
    ):
        """
        Inicializa el LLM Router.

        Args:
            supervisor_llm: Instancia del LLM del supervisor (ya configurada)
            supervisor_prompt: Prompt del supervisor (NO SE MODIFICA)
            router_schema: Schema de RouterSchema para structured output
        """
        self.supervisor_llm = supervisor_llm
        self.supervisor_prompt = supervisor_prompt
        self.router_schema = router_schema

        logger.info("✅ LLMRouter inicializado (wrapper del supervisor existente)")

    def can_handle(self, state: Dict[str, Any]) -> float:
        """
        El LLM siempre puede manejar cualquier query.

        Args:
            state: Estado del grafo

        Returns:
            Siempre 1.0 (máxima confianza)
        """
        return 1.0

    def route(self, state: Dict[str, Any]) -> RoutingDecision:
        """
        Usa el supervisor LLM para decidir el siguiente nodo.

        IMPORTANTE: Usa EXACTAMENTE la misma lógica que supervisor_node()
        en agent_graph.py, sin modificaciones.

        Args:
            state: Estado del grafo

        Returns:
            RoutingDecision con la decisión del supervisor
        """
        logger.info("🧠 LLMRouter: Consultando supervisor LLM...")

        messages = state.get('messages', [])

        if not messages:
            logger.error("❌ Estado sin mensajes")
            return RoutingDecision(
                target_agent="FINISH",
                confidence=0.95,
                method="llm_router",
                metadata={'reason': 'no_messages'}
            )

        # ========================================
        # LÓGICA EXACTA DEL SUPERVISOR ACTUAL
        # (Copiada de agent_graph.py línea 229)
        # ========================================

        supervisor_messages = [
            HumanMessage(content=self.supervisor_prompt)
        ] + messages

        next_node_decision = "FINISH"  # Default

        try:
            # Invocar el supervisor LLM (mismo que antes)
            route = self.supervisor_llm.invoke(supervisor_messages)

            # Extraer decisión
            if hasattr(route, 'next_agent'):
                next_node_decision = route.next_agent
            else:
                logger.warning("⚠️ Respuesta del supervisor sin 'next_agent'. Usando FINISH.")
                next_node_decision = "FINISH"

            logger.info(f"🧭 Supervisor LLM decide: {next_node_decision}")

        except Exception as e:
            logger.error(f"❌ Error en supervisor LLM: {e}", exc_info=True)
            next_node_decision = "FINISH"

        # Crear decisión de routing
        return RoutingDecision(
            target_agent=next_node_decision,
            confidence=0.95,  # LLM tiene alta confianza en sus decisiones
            method="llm_router",
            metadata={
                'supervisor_invoked': True,
                'message_count': len(messages)
            }
        )
