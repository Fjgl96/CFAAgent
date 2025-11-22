# rag/financial_rag_elasticsearch.py
"""
Sistema RAG - VERSIÓN ELASTICSEARCH CON OPENAI EMBEDDINGS
Actualizado para LangChain 1.0+

Los usuarios consultan material financiero indexado en Elasticsearch.
El admin indexa documentos con generate_index.py
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_core.tools import tool

# Importar configuración
from config_elasticsearch import (
    ES_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    get_elasticsearch_client,
    get_es_config
)

# Importar API key de OpenAI y LLM desde config principal
from config import OPENAI_API_KEY, get_llm

# ========================================
# CLASE RAG ELASTICSEARCH
# ========================================

class FinancialRAGElasticsearch:
    """
    Sistema RAG usando Elasticsearch como vector store con OpenAI Embeddings.
    Solo lectura para usuarios.
    Actualizado para LangChain 1.0+
    """
    
    def __init__(
        self,
        index_name: str = ES_INDEX_NAME,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        
        # Verificar que existe API key
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY no encontrada. "
                "Configúrala en .env o Streamlit Secrets."
            )
        
        # Inicializar embeddings de OpenAI
        print(f"🧠 Cargando modelo de embeddings OpenAI: {embedding_model}")
        print(f"   Dimensiones: {EMBEDDING_DIMENSIONS}")
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_API_KEY,
            # Parámetros opcionales para optimización:
            chunk_size=1000,  # Número de textos por batch
            max_retries=3,
            timeout=30
        )
        
        # Vector store (se conecta a Elasticsearch)
        self.vector_store = None
        
        # Número de resultados a retornar
        self.k_results = 4
        
        # Conectar automáticamente
        self._connect()
    
    def _connect(self) -> bool:
        """Conecta al índice de Elasticsearch."""
        try:
            print(f"📥 Conectando a Elasticsearch (índice: {self.index_name})...")
            
            # Verificar que existe el cliente
            es_client = get_elasticsearch_client()
            if not es_client:
                print("❌ No se pudo conectar a Elasticsearch")
                return False
            
            # Verificar que existe el índice
            if not es_client.indices.exists(index=self.index_name):
                print(f"❌ El índice '{self.index_name}' no existe")
                print("   El administrador debe generar el índice primero:")
                print("   python admin/generate_index.py")
                return False
            
            # Obtener configuración
            es_config = get_es_config()
            
            # Crear ElasticsearchStore (LangChain 1.0 syntax)
            self.vector_store = ElasticsearchStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                es_url=es_config["es_url"],
                es_user=es_config["es_user"],
                es_password=es_config["es_password"]
            )
            
            print(f"✅ Conectado a Elasticsearch (índice: {self.index_name})")
            
            # Mostrar info del índice
            count = es_client.count(index=self.index_name)
            print(f"   Documentos indexados: {count['count']}")
            
            return True
        
        except Exception as e:
            print(f"❌ Error conectando a Elasticsearch: {e}")
            return False

    def get_health_status(self) -> dict:
        """
        Retorna el estado de salud del sistema RAG.
        Determina el estado basado en el vector_store existente.
        """
        # Inferir estado actual
        is_connected = (
            self.vector_store is not None and
            self.embeddings is not None
        )
        
        # Inferir último error chequeando si _connect() falló
        error_msg = None
        if not is_connected:
            error_msg = "RAG no inicializado o conexión fallida"
        
        return {
            "connection_status": "connected" if is_connected else "disconnected",
            "last_error": error_msg,
            "retry_count": 0,  # No es crítico, solo para compatibilidad
            "index_name": self.index_name,
            "embeddings_loaded": self.embeddings is not None,
            "vector_store_ready": self.vector_store is not None
        }

    def search_documents(
        self,
        query: str,
        k: int = None,
        filter_dict: dict = None
    ) -> List[Document]:
        """
        Busca documentos similares a la query en Elasticsearch con búsqueda híbrida.
        REFACTORIZADO para Self-Querying: vectorial + filtros estructurales.

        Args:
            query: Consulta de búsqueda
            k: Número de documentos a retornar
            filter_dict: Filtros de metadata (ej: {"cfa_level": "I", "L1_Topic": "Fixed Income"})

        Returns:
            Lista de documentos relevantes
        """
        if k is None:
            k = self.k_results

        # Verificar que esté conectado
        if self.vector_store is None:
            print("⚠️ No conectado a Elasticsearch. Intentando reconectar...")
            if not self._connect():
                return []

        # Construir mensaje de búsqueda
        search_msg = f"🔍 Búsqueda híbrida: '{query}' (top {k})"
        if filter_dict:
            # Convertir filtros a formato Elasticsearch (metadata.campo)
            es_filters = []
            for key, value in filter_dict.items():
                es_filters.append({"term": {f"metadata.{key}": value}})

            search_msg += f"\n   Filtros: {filter_dict}"

        print(search_msg)

        try:
            # Búsqueda semántica con similarity_search
            # LangChain ElasticsearchStore maneja automáticamente los filtros
            if filter_dict:
                # Construir query dict para Elasticsearch
                # ElasticsearchStore espera filtros en formato específico
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict  # LangChain lo convierte a filtros ES internamente
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )

            if filter_dict and len(results) < k:
                print(f"   ⚠️ Solo {len(results)} resultados con filtros (esperados {k})")
                print(f"   💡 Considera ampliar los filtros si no hay suficientes resultados")

            print(f"✅ {len(results)} documentos encontrados")
            return results

        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            # Fallback: si los filtros causan error, intentar sin ellos
            if filter_dict:
                print("   🔄 Reintentando sin filtros...")
                try:
                    results = self.vector_store.similarity_search(query=query, k=k)
                    print(f"✅ {len(results)} documentos encontrados (sin filtros)")
                    return results
                except:
                    return []
            return []


# ========================================
# INSTANCIA GLOBAL
# ========================================

# Instancia única del sistema RAG
rag_system = FinancialRAGElasticsearch()


# ========================================
# DICCIONARIO DE TÉRMINOS TÉCNICOS (ESPAÑOL ↔ INGLÉS)
# ========================================

TERMINOS_TECNICOS = {
    # ===== FINANZAS CORPORATIVAS =====
    "wacc": ["WACC", "Weighted Average Cost of Capital", "costo promedio ponderado", "costo de capital"],
    "van": ["NPV", "VAN", "Net Present Value", "Valor Actual Neto", "valor presente neto"],
    "tir": ["IRR", "TIR", "Internal Rate of Return", "tasa interna de retorno"],
    "payback": ["Payback Period", "periodo de recuperación", "payback"],
    "profitability_index": ["Profitability Index", "PI", "índice de rentabilidad", "índice de beneficio"],

    # ===== RENTA FIJA =====
    "bono": ["bond", "bono", "fixed income", "renta fija"],
    "cupón": ["coupon", "cupón"],
    "ytm": ["YTM", "yield to maturity", "rendimiento al vencimiento"],
    "duration": ["duration", "duración", "Macaulay duration", "modified duration", "duration modificada"],
    "convexity": ["convexity", "convexidad"],
    "current_yield": ["current yield", "rendimiento corriente", "yield"],
    "zero_coupon": ["zero-coupon bond", "bono cupón cero", "strip bond"],

    # ===== EQUITY =====
    "equity": ["equity", "acciones", "stock", "patrimonio"],
    "dividend": ["dividend", "dividendo"],
    "gordon": ["Gordon Growth", "modelo de Gordon", "dividend discount model", "DDM"],

    # ===== DERIVADOS =====
    "derivado": ["derivative", "derivado", "option", "opción"],
    "call": ["call option", "opción call"],
    "put": ["put option", "opción put"],
    "black-scholes": ["Black-Scholes", "Black Scholes"],
    "volatilidad": ["volatility", "volatilidad", "sigma"],
    "put_call_parity": ["put-call parity", "paridad put-call"],

    # ===== PORTAFOLIO =====
    "capm": ["CAPM", "Capital Asset Pricing Model", "modelo de valoración de activos"],
    "beta": ["beta", "systematic risk", "riesgo sistemático"],
    "sharpe": ["Sharpe ratio", "ratio de Sharpe", "rendimiento ajustado por riesgo"],
    "treynor": ["Treynor ratio", "ratio de Treynor", "índice de Treynor"],
    "jensen": ["Jensen's alpha", "Jensen alpha", "alfa de Jensen"],
    "portfolio": ["portfolio", "portafolio", "cartera"],
    "diversificación": ["diversification", "diversificación"],
    "correlación": ["correlation", "correlación", "covariance", "covarianza"],
    "riesgo": ["risk", "riesgo", "standard deviation", "desviación estándar"],
    "retorno": ["return", "retorno", "rendimiento", "expected return"],
}


# ========================================
# SELF-QUERYING: EXTRACCIÓN DE FILTROS
# ========================================

def extraer_filtros_de_consulta(consulta: str) -> dict:
    """
    Usa el LLM para analizar la pregunta del usuario y extraer filtros estructurales.

    Args:
        consulta: Pregunta del usuario

    Returns:
        Diccionario con filtros detectados. Ejemplo:
        {
            "L1_Topic": "Quantitative Methods",
            "L2_Reading": None,
            "cfa_level": "I"
        }
    """
    from pydantic import BaseModel, Field
    from typing import Optional

    # Esquema para structured output
    class QueryFilters(BaseModel):
        """Filtros extraídos de la consulta del usuario."""
        L1_Topic: Optional[str] = Field(
            None,
            description="Tema principal del CFA (ej: 'Quantitative Methods', 'Fixed Income', 'Equity', 'Derivatives', 'Portfolio Management'). SOLO si el usuario lo menciona explícitamente."
        )
        L2_Reading: Optional[str] = Field(
            None,
            description="Lectura específica mencionada (ej: 'Time Value of Money', 'Duration and Convexity'). SOLO si el usuario la menciona explícitamente."
        )
        cfa_level: Optional[str] = Field(
            None,
            description="Nivel CFA mencionado: 'I', 'II' o 'III'. SOLO si el usuario lo especifica."
        )

    # Mapeo español → inglés para temas comunes
    TOPIC_MAPPING = {
        "métodos cuantitativos": "Quantitative Methods",
        "quantitative methods": "Quantitative Methods",
        "renta fija": "Fixed Income",
        "fixed income": "Fixed Income",
        "bonos": "Fixed Income",
        "equity": "Equity",
        "acciones": "Equity",
        "derivados": "Derivatives",
        "derivatives": "Derivatives",
        "opciones": "Derivatives",
        "portafolio": "Portfolio Management",
        "portfolio": "Portfolio Management",
        "gestión de portafolios": "Portfolio Management",
        "finanzas corporativas": "Corporate Finance",
        "corporate finance": "Corporate Finance"
    }

    try:
        llm = get_llm()
        llm_with_structure = llm.with_structured_output(QueryFilters)

        prompt = f"""Analiza esta pregunta del usuario y extrae SOLO los filtros explícitamente mencionados:

Pregunta: "{consulta}"

INSTRUCCIONES CRÍTICAS:
1. SOLO extrae un filtro si el usuario lo menciona EXPLÍCITAMENTE
2. Si no hay mención explícita, devuelve None para ese campo
3. Para L1_Topic, usa estos valores estándar si se mencionan:
   - Quantitative Methods
   - Fixed Income
   - Equity
   - Derivatives
   - Portfolio Management
   - Corporate Finance

4. Para cfa_level, usa: "I", "II" o "III"

EJEMPLOS:
- "Explica el WACC" → Todos None (no menciona tema específico)
- "Explica el WACC en Finanzas Corporativas" → L1_Topic="Corporate Finance"
- "¿Qué es duration en renta fija?" → L1_Topic="Fixed Income"
- "Bonos de nivel 1" → L1_Topic="Fixed Income", cfa_level="I"
- "CAPM en nivel II" → L1_Topic="Portfolio Management", cfa_level="II"

Extrae los filtros:"""

        response = llm_with_structure.invoke(prompt)

        # Convertir a diccionario
        filters = {}
        if response.L1_Topic:
            # Normalizar con mapeo
            normalized_topic = TOPIC_MAPPING.get(response.L1_Topic.lower(), response.L1_Topic)
            filters["L1_Topic"] = normalized_topic
        if response.L2_Reading:
            filters["L2_Reading"] = response.L2_Reading
        if response.cfa_level:
            filters["cfa_level"] = response.cfa_level.upper()

        if filters:
            print(f"🔍 Filtros detectados: {filters}")
        else:
            print("🔍 No se detectaron filtros específicos (búsqueda abierta)")

        return filters

    except Exception as e:
        print(f"⚠️ Error extrayendo filtros: {e}")
        return {}


def enriquecer_query_bilingue(consulta: str) -> str:
    """
    Enriquece la consulta agregando términos técnicos en inglés si se detectan en español.

    Args:
        consulta: Query original del usuario (probablemente en español)

    Returns:
        Query enriquecida con términos bilingües
    """
    consulta_lower = consulta.lower()
    terminos_agregados = []

    # Buscar términos técnicos en la query
    for key, synonyms in TERMINOS_TECNICOS.items():
        # Si encontramos algún término relacionado en la query
        if any(term.lower() in consulta_lower for term in synonyms):
            # Agregar todos los sinónimos para mejorar la búsqueda
            terminos_agregados.extend(synonyms)

    # Si encontramos términos técnicos, enriquecer la query
    if terminos_agregados:
        # Eliminar duplicados manteniendo orden
        terminos_unicos = list(dict.fromkeys(terminos_agregados))
        terminos_str = " ".join(terminos_unicos)
        query_enriquecida = f"{consulta} {terminos_str}"
        print(f"🔄 Query enriquecida: '{consulta}' → agregados {len(terminos_unicos)} términos")
        return query_enriquecida

    return consulta


# ========================================
# TOOL PARA EL AGENTE
# ========================================

@tool
def buscar_documentacion_financiera(consulta: str) -> str:
    """
    Busca información en material financiero indexado en Elasticsearch.
    REFACTORIZADO con Self-Querying: extrae filtros de la pregunta automáticamente.

    Args:
        consulta: La pregunta o tema a buscar.

    Returns:
        Contexto relevante del material de estudio.
    """
    print(f"\n🔍 RAG Tool invocado con consulta: '{consulta}'")

    # PASO 1: Extraer filtros estructurales de la consulta (Self-Querying)
    filtros = extraer_filtros_de_consulta(consulta)

    # PASO 2: Enriquecer query con términos bilingües
    consulta_enriquecida = enriquecer_query_bilingue(consulta)

    # PASO 3: Buscar documentos con búsqueda híbrida (vectorial + filtros)
    docs = rag_system.search_documents(
        consulta_enriquecida,
        k=3,
        filter_dict=filtros if filtros else None
    )

    if not docs:
        mensaje_error = (
            "No encontré información relevante en el material de estudio indexado. "
            "Esto puede deberse a:\n"
            "1. El tema no está en el material indexado\n"
            "2. El índice no se ha generado aún en Elasticsearch\n"
            "3. Problema de conexión con Elasticsearch\n"
        )

        if filtros:
            mensaje_error += (
                f"4. Los filtros aplicados ({filtros}) son demasiado restrictivos\n\n"
                "💡 Intenta reformular tu pregunta sin mencionar un tema/nivel específico."
            )
        else:
            mensaje_error += (
                "4. La consulta necesita reformularse\n\n"
                "Intenta reformular tu pregunta o consulta directamente al "
                "agente especializado correspondiente."
            )

        return mensaje_error

    # PASO 4: Formatear resultado con metadatos estructurales
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Desconocido')
        content = doc.page_content.strip()

        # Extraer nombre del archivo
        if source != 'Desconocido':
            from pathlib import Path
            source_name = Path(source).name
        else:
            source_name = source

        # Metadata estructural
        cfa_level = doc.metadata.get('cfa_level', 'N/A')
        l1_topic = doc.metadata.get('L1_Topic', 'N/A')
        l2_reading = doc.metadata.get('L2_Reading', 'N/A')
        page_num = doc.metadata.get('page_number', 'N/A')

        context_parts.append(
            f"--- Fragmento {i} ---\n"
            f"Fuente: {source_name}\n"
            f"CFA Level: {cfa_level}\n"
            f"Tema: {l1_topic}\n"
            f"Lectura: {l2_reading}\n"
            f"Página: {page_num}\n"
            f"Contenido:\n{content}"
        )

    full_context = "\n\n".join(context_parts)

    # Mostrar filtros aplicados en la respuesta
    filtros_msg = ""
    if filtros:
        filtros_msg = f"\n\n🔍 Filtros aplicados: {filtros}"

    return f"📚 Información encontrada en el material de estudio:{filtros_msg}\n\n{full_context}"


print("✅ Módulo financial_rag_elasticsearch cargado (LangChain 1.0, OpenAI Embeddings).")