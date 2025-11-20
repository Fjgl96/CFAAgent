"""
Fast Pattern Router - Ruteo determinístico basado en patrones.
Utiliza regex y keywords para decisiones instantáneas (< 10ms).
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage

from .interfaces import IRouter, RoutingDecision

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('fast_router')
except ImportError:
    import logging
    logger = logging.getLogger('fast_router')


class FastPatternRouter(IRouter):
    """
    Router basado en pattern matching para decisiones rápidas.

    Estrategia:
    1. Detecta intención de cálculo (keywords: "calcula", "obtén", etc.)
    2. Extrae parámetros numéricos (regex)
    3. Identifica categoría por keywords
    4. Retorna decisión con score de confianza
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el router cargando patrones desde archivo YAML.

        Args:
            config_path: Ruta al archivo YAML de configuración.
                        Si es None, usa valores por defecto hardcoded.
        """
        self.config_path = config_path
        self.patterns = self._load_patterns()

        logger.info(f"✅ FastPatternRouter inicializado con {len(self.patterns.get('agent_mappings', []))} mappings")

    def _load_patterns(self) -> Dict:
        """
        Carga patrones desde archivo YAML o usa defaults.

        Returns:
            Diccionario con configuración de patrones
        """
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    patterns = yaml.safe_load(f)
                    logger.info(f"📄 Patrones cargados desde {self.config_path}")
                    return patterns
            except Exception as e:
                logger.error(f"❌ Error cargando {self.config_path}: {e}")
                logger.warning("⚠️ Usando patrones por defecto")

        # Patrones por defecto (fallback)
        return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict:
        """
        Retorna configuración por defecto si no hay archivo YAML.

        Returns:
            Diccionario con patrones default
        """
        return {
            'settings': {
                'confidence_threshold': 0.8,
                'min_params_for_bypass': 2
            },
            'calc_intent_patterns': {
                'spanish': [
                    r'\bcalcula(?:r)?\b',
                    r'\bobt[eé]n(?:er)?\b',
                    r'\bdetermin(?:a|ar)\b',
                    r'\bcomput(?:a|ar)\b',
                ],
                'english': [
                    r'\bcalculate\b',
                    r'\bcompute\b',
                    r'\bdetermine\b',
                ]
            },
            'agent_mappings': [
                # Finanzas Corporativas
                {'agent': 'Agente_Finanzas_Corp', 'priority': 10, 'keywords': {'spanish': [r'\bvan\b', 'valor actual neto', 'npv'], 'english': ['npv', 'net present value']}, 'required_params': 3},
                {'agent': 'Agente_Finanzas_Corp', 'priority': 9, 'keywords': {'spanish': [r'\bwacc\b', 'costo.*capital'], 'english': ['wacc', 'weighted average cost']}, 'required_params': 5},
                {'agent': 'Agente_Finanzas_Corp', 'priority': 9, 'keywords': {'spanish': [r'\btir\b', 'irr', 'tasa.*interna'], 'english': ['irr', 'internal rate']}, 'required_params': 2},
                {'agent': 'Agente_Finanzas_Corp', 'priority': 8, 'keywords': {'spanish': ['payback', 'recuperaci[oó]n'], 'english': ['payback period']}, 'required_params': 2},
                {'agent': 'Agente_Finanzas_Corp', 'priority': 8, 'keywords': {'spanish': ['profitability.*index', '[íi]ndice.*rentabilidad'], 'english': ['profitability index']}, 'required_params': 3},

                # Portafolio
                {'agent': 'Agente_Portafolio', 'priority': 10, 'keywords': {'spanish': [r'\bcapm\b'], 'english': ['capm', 'capital asset pricing']}, 'required_params': 3},
                {'agent': 'Agente_Portafolio', 'priority': 9, 'keywords': {'spanish': ['sharpe'], 'english': ['sharpe ratio']}, 'required_params': 3},
                {'agent': 'Agente_Portafolio', 'priority': 9, 'keywords': {'spanish': ['treynor'], 'english': ['treynor ratio']}, 'required_params': 3},
                {'agent': 'Agente_Portafolio', 'priority': 9, 'keywords': {'spanish': ['jensen', 'alpha'], 'english': ['jensen alpha']}, 'required_params': 4},
                {'agent': 'Agente_Portafolio', 'priority': 8, 'keywords': {'spanish': ['beta.*portafolio'], 'english': ['portfolio beta']}, 'required_params': 4},

                # Renta Fija
                {'agent': 'Agente_Renta_Fija', 'priority': 10, 'keywords': {'spanish': ['valor.*bono'], 'english': ['bond.*value', 'bond.*price']}, 'required_params': 5},
                {'agent': 'Agente_Renta_Fija', 'priority': 9, 'keywords': {'spanish': ['duration.*macaulay'], 'english': ['macaulay duration']}, 'required_params': 5},
                {'agent': 'Agente_Renta_Fija', 'priority': 9, 'keywords': {'spanish': ['duration.*modificada'], 'english': ['modified duration']}, 'required_params': 3},
                {'agent': 'Agente_Renta_Fija', 'priority': 8, 'keywords': {'spanish': ['convexity', 'convexidad'], 'english': ['convexity']}, 'required_params': 5},
                {'agent': 'Agente_Renta_Fija', 'priority': 8, 'keywords': {'spanish': ['current.*yield'], 'english': ['current yield']}, 'required_params': 2},

                # Equity
                {'agent': 'Agente_Equity', 'priority': 10, 'keywords': {'spanish': ['gordon', 'ddm', 'dividend.*discount'], 'english': ['gordon growth', 'ddm']}, 'required_params': 3},

                # Derivados
                {'agent': 'Agente_Derivados', 'priority': 10, 'keywords': {'spanish': ['opci[oó]n.*call', 'call.*option'], 'english': ['call option']}, 'required_params': 5},
                {'agent': 'Agente_Derivados', 'priority': 10, 'keywords': {'spanish': ['opci[oó]n.*put', 'put.*option'], 'english': ['put option']}, 'required_params': 5},
                {'agent': 'Agente_Derivados', 'priority': 9, 'keywords': {'spanish': ['put.*call.*parity', 'paridad'], 'english': ['put call parity']}, 'required_params': 6},
            ],
            'param_patterns': [
                {'name': 'cantidad_k', 'regex': r'\d+(?:\.\d+)?k\b'},
                {'name': 'porcentaje', 'regex': r'\d+(?:\.\d+)?%'},
                {'name': 'numero', 'regex': r'\d+(?:\.\d+)?'},
                {'name': 'lista', 'regex': r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]'},
            ]
        }

    def _detect_calc_intent(self, texto: str) -> bool:
        """
        Detecta si el usuario quiere realizar un cálculo.

        Args:
            texto: Mensaje del usuario

        Returns:
            True si detecta intención de cálculo
        """
        texto_lower = texto.lower()

        # Buscar en patrones español
        for pattern in self.patterns['calc_intent_patterns'].get('spanish', []):
            if re.search(pattern, texto_lower, re.IGNORECASE):
                logger.debug(f"✓ Intención detectada (ES): {pattern}")
                return True

        # Buscar en patrones inglés
        for pattern in self.patterns['calc_intent_patterns'].get('english', []):
            if re.search(pattern, texto_lower, re.IGNORECASE):
                logger.debug(f"✓ Intención detectada (EN): {pattern}")
                return True

        return False

    def _extract_params(self, texto: str) -> List[str]:
        """
        Extrae parámetros numéricos del texto.

        Args:
            texto: Mensaje del usuario

        Returns:
            Lista de parámetros numéricos encontrados
        """
        parametros = []

        for param_config in self.patterns.get('param_patterns', []):
            pattern = param_config['regex']
            matches = re.findall(pattern, texto)
            parametros.extend(matches)

        logger.debug(f"✓ Parámetros detectados ({len(parametros)}): {parametros[:5]}...")
        return parametros

    def _identify_agent(self, texto: str) -> Optional[Dict]:
        """
        Identifica el agente más apropiado basado en keywords.

        Args:
            texto: Mensaje del usuario

        Returns:
            Dict con info del agente o None si no hay match
        """
        texto_lower = texto.lower()

        # Ordenar por prioridad (mayor primero)
        mappings = sorted(
            self.patterns.get('agent_mappings', []),
            key=lambda x: x.get('priority', 0),
            reverse=True
        )

        matched_agents = []

        for mapping in mappings:
            # Buscar en keywords español
            for keyword in mapping['keywords'].get('spanish', []):
                if re.search(keyword, texto_lower, re.IGNORECASE):
                    matched_agents.append(mapping)
                    logger.debug(f"✓ Match (ES): '{keyword}' → {mapping['agent']}")
                    break  # Solo un match por mapping

            # Si ya encontró en español, skip inglés
            if matched_agents and matched_agents[-1] == mapping:
                continue

            # Buscar en keywords inglés
            for keyword in mapping['keywords'].get('english', []):
                if re.search(keyword, texto_lower, re.IGNORECASE):
                    matched_agents.append(mapping)
                    logger.debug(f"✓ Match (EN): '{keyword}' → {mapping['agent']}")
                    break

        # Si múltiples matches, retornar el de mayor prioridad
        if len(matched_agents) > 1:
            logger.warning(f"⚠️ Múltiples agentes detectados: {[m['agent'] for m in matched_agents[:3]]}")
            return matched_agents[0]  # Mayor prioridad

        return matched_agents[0] if matched_agents else None

    def _calculate_confidence(
        self,
        has_intent: bool,
        num_params: int,
        agent_mapping: Optional[Dict]
    ) -> float:
        """
        Calcula score de confianza para decidir bypass.

        Args:
            has_intent: Si detectó intención de cálculo
            num_params: Cantidad de parámetros numéricos
            agent_mapping: Mapping del agente detectado o None

        Returns:
            Score de 0.0 a 1.0
        """
        score = 0.0

        # Componente 1: Intención (40%)
        if has_intent:
            score += 0.4

        # Componente 2: Agente identificado (40%)
        if agent_mapping:
            score += 0.4

        # Componente 3: Parámetros suficientes (20%)
        if agent_mapping:
            required_params = agent_mapping.get('required_params', 3)
            if num_params >= required_params:
                score += 0.2
            elif num_params >= required_params - 1:
                score += 0.1  # Bonus parcial

        return score

    def can_handle(self, state: Dict[str, Any]) -> float:
        """
        Evalúa qué tan confiado está este router.

        Args:
            state: Estado del grafo

        Returns:
            Score de confianza 0.0-1.0
        """
        messages = state.get('messages', [])

        if not messages:
            return 0.0

        last_message = messages[-1]

        if not isinstance(last_message, HumanMessage):
            return 0.0

        texto = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # Análisis rápido
        has_intent = self._detect_calc_intent(texto)
        params = self._extract_params(texto)
        agent_mapping = self._identify_agent(texto)

        confidence = self._calculate_confidence(has_intent, len(params), agent_mapping)

        logger.debug(
            f"can_handle: intent={has_intent}, params={len(params)}, "
            f"agent={agent_mapping['agent'] if agent_mapping else None}, "
            f"conf={confidence:.2f}"
        )

        return confidence

    def route(self, state: Dict[str, Any]) -> RoutingDecision:
        """
        Determina el siguiente agente usando pattern matching.

        Args:
            state: Estado del grafo

        Returns:
            RoutingDecision con target y metadata
        """
        logger.info("⚡ FastPatternRouter: Analizando query...")

        messages = state.get('messages', [])

        if not messages:
            logger.warning("⚠️ Sin mensajes, retornando Supervisor")
            return RoutingDecision(
                target_agent="Supervisor",
                confidence=0.0,
                method="fast_pattern",
                metadata={'reason': 'no_messages'}
            )

        last_message = messages[-1]

        if not isinstance(last_message, HumanMessage):
            logger.info("ℹ️ Último mensaje no es del usuario")
            return RoutingDecision(
                target_agent="Supervisor",
                confidence=0.0,
                method="fast_pattern",
                metadata={'reason': 'not_human_message'}
            )

        texto = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # Análisis completo
        has_intent = self._detect_calc_intent(texto)
        params = self._extract_params(texto)
        agent_mapping = self._identify_agent(texto)

        confidence = self._calculate_confidence(has_intent, len(params), agent_mapping)

        # Decisión
        target_agent = agent_mapping['agent'] if agent_mapping else "Supervisor"

        logger.info(
            f"📊 Fast Pattern Score: {confidence:.2f} "
            f"(intent={has_intent}, params={len(params)}, agent={target_agent})"
        )

        return RoutingDecision(
            target_agent=target_agent,
            confidence=confidence,
            method="fast_pattern",
            metadata={
                'has_intent': has_intent,
                'params_detected': len(params),
                'params_sample': params[:3] if len(params) > 3 else params,
                'agent_priority': agent_mapping.get('priority', 0) if agent_mapping else 0,
                'required_params': agent_mapping.get('required_params', 0) if agent_mapping else 0
            }
        )
