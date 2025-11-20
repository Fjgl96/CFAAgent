"""
Sistema de Routing Inteligente - LangChain-Native
Usa Runnables nativos de LangChain en vez de clases custom.
"""

from .langchain_routing import (
    create_routing_node,
    create_fast_pattern_runnable,
    create_hybrid_routing_branch,
    analyze_query_fast_pattern,
)

__all__ = [
    'create_routing_node',
    'create_fast_pattern_runnable',
    'create_hybrid_routing_branch',
    'analyze_query_fast_pattern',
]

# Nota: Los archivos en routing/deprecated/ son de la implementación original
# con clases custom (IRouter, FastPatternRouter, etc.) y NO se usan.
# Se mantienen solo como referencia arquitectural.
