# rag/financial_rag_elasticsearch.py
"""
Sistema RAG - VERSIÓN ELASTICSEARCH CON OPENAI EMBEDDINGS
Actualizado para LangChain 1.0+

Los usuarios consultan el índice de Elasticsearch.
El admin indexa con generate_index.py
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

# Importar API key de OpenAI desde config principal
from config import OPENAI_API_KEY

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
        Busca documentos similares a la query en Elasticsearch.
        
        Args:
            query: Consulta de búsqueda
            k: Número de documentos a retornar
            filter_dict: Filtros de metadata (ej: {"cfa_level": "I"})
        
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
        
        print(f"🔍 Buscando en Elasticsearch con OpenAI: '{query}' (top {k})")
        
        try:
            # Búsqueda semántica con similarity_search
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            print(f"✅ {len(results)} documentos encontrados")
            return results
        
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
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
    Busca información en la documentación financiera CFA indexada en Elasticsearch.
    
    Args:
        consulta: La pregunta o tema a buscar.
    
    Returns:
        Contexto relevante de la documentación.
    """
    print(f"\n🔍 RAG Tool invocado con consulta: '{consulta}'")

    # MEJORA: Enriquecer query con términos bilingües
    consulta_enriquecida = enriquecer_query_bilingue(consulta)

    # Buscar documentos relevantes con query enriquecida
    docs = rag_system.search_documents(consulta_enriquecida, k=3)
    
    if not docs:
        return (
            "No encontré información relevante en la documentación indexada. "
            "Esto puede deberse a:\n"
            "1. El tema no está en el material indexado\n"
            "2. El índice no se ha generado aún en Elasticsearch\n"
            "3. Problema de conexión con Elasticsearch\n"
            "4. La consulta necesita reformularse\n\n"
            "Intenta reformular tu pregunta o consulta directamente al "
            "agente especializado correspondiente."
        )
    
    # Formatear resultado
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
        
        # Metadata adicional
        cfa_level = doc.metadata.get('cfa_level', 'N/A')
        
        context_parts.append(
            f"--- Fragmento {i} ---\n"
            f"Fuente: {source_name}\n"
            f"CFA Level: {cfa_level}\n"
            f"Contenido:\n{content}"
        )
    
    full_context = "\n\n".join(context_parts)
    
    return f"📚 Información encontrada en la documentación CFA:\n\n{full_context}"


print("✅ Módulo financial_rag_elasticsearch cargado (LangChain 1.0, OpenAI Embeddings).")