"""
Router Orchestrator - Coordina múltiples estrategias de routing.
Implementa Strategy Pattern para selección dinámica de routers.
"""

from typing import List, Tuple, Optional, Dict, Any

from .interfaces import IRouter, RoutingDecision

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('orchestrator')
except ImportError:
    import logging
    logger = logging.getLogger('orchestrator')


class RouterOrchestrator:
    """
    Orquestador que coordina múltiples routers.

    Permite:
    - Registrar múltiples routers con prioridades
    - Seleccionar automáticamente el mejor router para cada query
    - Tener un router de fallback por defecto

    Principio: Open/Closed - Agregar nuevos routers sin modificar código.
    """

    def __init__(self):
        """Inicializa el orquestador vacío."""
        self.routers: List[Tuple[int, IRouter]] = []  # (priority, router)
        self.default_router: Optional[IRouter] = None

        logger.info("🎭 RouterOrchestrator inicializado")

    def register_router(self, router: IRouter, priority: int = 0) -> None:
        """
        Registra un router con una prioridad específica.

        Args:
            router: Instancia de IRouter
            priority: Prioridad (mayor = más preferido)
        """
        self.routers.append((priority, router))

        # Ordenar por prioridad descendente
        self.routers.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            f"✅ Router registrado: {router.__class__.__name__} "
            f"(prioridad={priority})"
        )

    def set_default_router(self, router: IRouter) -> None:
        """
        Establece el router de fallback.

        Args:
            router: Router a usar si ninguno puede manejar la query
        """
        self.default_router = router
        logger.info(
            f"✅ Default router establecido: {router.__class__.__name__}"
        )

    def route(self, state: Dict[str, Any]) -> RoutingDecision:
        """
        Selecciona el mejor router y delega la decisión.

        Proceso:
        1. Evalúa can_handle() de cada router
        2. Selecciona el router con mayor confianza
        3. Si todos tienen confianza 0, usa default

        Args:
            state: Estado del grafo

        Returns:
            RoutingDecision del router seleccionado

        Raises:
            ValueError: Si no hay routers disponibles
        """
        if not self.routers and not self.default_router:
            logger.error("❌ No hay routers registrados")
            raise ValueError("RouterOrchestrator: No hay routers disponibles")

        # ========================================
        # FASE 1: Evaluar todos los routers
        # ========================================

        best_router = None
        best_confidence = 0.0
        best_priority = -1

        for priority, router in self.routers:
            try:
                confidence = router.can_handle(state)

                logger.debug(
                    f"  {router.__class__.__name__}: "
                    f"confidence={confidence:.2f}, priority={priority}"
                )

                # Seleccionar el mejor (confianza + prioridad)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_router = router
                    best_priority = priority
                elif confidence == best_confidence and priority > best_priority:
                    # Mismo confidence, mayor prioridad
                    best_router = router
                    best_priority = priority

            except Exception as e:
                logger.error(
                    f"❌ Error evaluando {router.__class__.__name__}: {e}",
                    exc_info=True
                )
                continue

        # ========================================
        # FASE 2: Usar mejor router o default
        # ========================================

        selected_router = best_router or self.default_router

        if not selected_router:
            logger.error("❌ No se pudo seleccionar ningún router")
            raise ValueError("RouterOrchestrator: Ningún router disponible")

        logger.info(
            f"📍 Router seleccionado: {selected_router.__class__.__name__} "
            f"(confidence={best_confidence:.2f}, priority={best_priority})"
        )

        # ========================================
        # FASE 3: Delegar decisión
        # ========================================

        decision = selected_router.route(state)

        # Añadir metadata del orquestador
        decision.metadata['orchestrator'] = {
            'selected_router': selected_router.__class__.__name__,
            'selection_confidence': best_confidence,
            'selection_priority': best_priority,
            'total_routers_evaluated': len(self.routers)
        }

        return decision
