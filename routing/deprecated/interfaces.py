"""
Interfaces base para el sistema de routing.
Define contratos que deben cumplir todas las estrategias de ruteo.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RoutingDecision:
    """
    Representa una decisión de ruteo con metadata completa.

    Attributes:
        target_agent: Nombre del agente objetivo
        confidence: Nivel de confianza (0.0 - 1.0)
        method: Método usado para tomar la decisión
        metadata: Información adicional sobre la decisión
    """
    target_agent: str
    confidence: float
    method: str  # "fast_pattern", "llm_router", "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Valida que la confianza esté en rango válido."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence debe estar entre 0.0 y 1.0, recibido: {self.confidence}")


class IRouter(ABC):
    """
    Interfaz abstracta para estrategias de ruteo.

    Implementa el Strategy Pattern - cada estrategia concreta debe implementar
    estos dos métodos para ser compatible con el sistema.
    """

    @abstractmethod
    def route(self, state: Dict[str, Any]) -> RoutingDecision:
        """
        Determina el siguiente nodo/agente a ejecutar.

        Args:
            state: Estado actual del grafo (debe contener 'messages')

        Returns:
            RoutingDecision con el agente objetivo y metadata
        """
        pass

    @abstractmethod
    def can_handle(self, state: Dict[str, Any]) -> float:
        """
        Evalúa qué tan confiado está este router para manejar la consulta.

        Args:
            state: Estado actual del grafo

        Returns:
            Score de confianza entre 0.0 (no puede) y 1.0 (totalmente seguro)
        """
        pass
