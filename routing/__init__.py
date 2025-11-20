"""
Sistema de Routing Inteligente - Arquitectura de 3 Capas
Implementa Strategy Pattern para ruteo flexible y extensible.
"""

from .interfaces import IRouter, RoutingDecision
from .fast_router import FastPatternRouter
from .llm_router import LLMRouter
from .hybrid_router import HybridRouter
from .orchestrator import RouterOrchestrator

__all__ = [
    'IRouter',
    'RoutingDecision',
    'FastPatternRouter',
    'LLMRouter',
    'HybridRouter',
    'RouterOrchestrator',
]
