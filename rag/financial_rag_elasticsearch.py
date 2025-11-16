# rag/financial_rag_elasticsearch.py
"""
Sistema RAG - VERSIÓN ELASTICSEARCH (SOLO LECTURA)
Actualizado para LangChain 1.0+ con:
- Retry logic con exponential backoff
- Fallback cuando Elasticsearch no está disponible
- Logging estructurado
- Manejo robusto de errores
"""

import time
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_core.tools import tool

# Importar configuración consolidada
from config import (
    ES_INDEX_NAME,
    EMBEDDING_MODEL,
    get_elasticsearch_client,
    get_es_config
)

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('rag')
except ImportError:
    import logging
    logger = logging.getLogger('rag')

# ========================================
# CLASE RAG ELASTICSEARCH
# ========================================

class FinancialRAGElasticsearch:
    """
    Sistema RAG usando Elasticsearch como vector store.
    Solo lectura para usuarios.
    
    Características:
    - Conexión con retry automático
    - Fallback cuando ES no está disponible
    - Logging detallado de operaciones
    - Cache de embeddings para performance
    """
    
    def __init__(
        self,
        index_name: str = ES_INDEX_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        max_retries: int = 3
    ):
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        self.max_retries = max_retries
        
        # Estado de conexión
        self.connection_status = "disconnected"
        self.last_error = None
        self.retry_count = 0
        
        # Vector store (se conecta a Elasticsearch)
        self.vector_store = None
        self.embeddings = None
        
        # Número de resultados a retornar
        self.k_results = 4
        
        # Inicializar embeddings
        self._init_embeddings()
        
        # Conectar con retry automático
        self._connect_with_retry()
    
    def _init_embeddings(self):
        """Inicializa el modelo de embeddings."""
        try:
            logger.info(f"🧠 Cargando modelo de embeddings: {self.embedding_model_name}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info("✅ Modelo de embeddings cargado correctamente")
        
        except Exception as e:
            logger.error(f"❌ Error cargando embeddings: {e}", exc_info=True)
            self.embeddings = None
    
    def _connect_with_retry(self) -> bool:
        """
        Conecta al índice de Elasticsearch con retry automático.
        Implementa exponential backoff.
        
        Returns:
            True si conecta exitosamente, False en caso contrario
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📥 Intento {attempt + 1}/{self.max_retries} de conexión a Elasticsearch...")
                
                if self._connect():
                    self.connection_status = "connected"
                    self.retry_count = 0
                    logger.info(f"✅ Conectado a Elasticsearch (índice: {self.index_name})")
                    return True
                
            except Exception as e:
                self.last_error = str(e)
                self.retry_count = attempt + 1
                
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"⚠️ Intento {attempt + 1} falló: {e}. "
                    f"Esperando {wait_time}s antes de reintentar..."
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
        
        # Falló después de todos los reintentos
        self.connection_status = "failed"
        logger.error(
            f"❌ No se pudo conectar a Elasticsearch después de {self.max_retries} intentos. "
            f"Último error: {self.last_error}"
        )
        return False
    
    def _connect(self) -> bool:
        """Intenta conectar al índice de Elasticsearch."""
        # Verificar embeddings
        if not self.embeddings:
            raise Exception("Embeddings no inicializados")
        
        # Verificar que existe el cliente
        es_client = get_elasticsearch_client()
        if not es_client:
            raise Exception("No se pudo crear cliente de Elasticsearch")
        
        # Verificar que existe el índice
        if not es_client.indices.exists(index=self.index_name):
            raise Exception(
                f"El índice '{self.index_name}' no existe. "
                f"El administrador debe ejecutar: python admin/generate_index.py"
            )
        
        # Obtener info del índice
        count = es_client.count(index=self.index_name)
        doc_count = count['count']
        
        if doc_count == 0:
            logger.warning(f"⚠️ El índice '{self.index_name}' está vacío (0 documentos)")
        else:
            logger.info(f"📊 Índice tiene {doc_count} documentos")
        
        # Obtener configuración
        es_config = get_es_config()
        
        # Crear ElasticsearchStore (LangChain 1.0 syntax)
        self.vector_store = ElasticsearchStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            es_url=es_config["es_url"],           # <-- Parámetro correcto
            es_user=es_config["es_user"],         # <-- Parámetro correcto
            es_password=es_config["es_password"]  # <-- Parámetro correcto
        )
        
        return True
    
    def search_documents(
        self,
        query: str,
        k: int = None,
        filter_dict: dict = None
    ) -> List[Document]:
        """
        Busca documentos similares a la query en Elasticsearch.
        Con fallback automático si hay problemas de conexión.
        
        Args:
            query: Consulta de búsqueda
            k: Número de documentos a retornar
            filter_dict: Filtros de metadata (ej: {"cfa_level": "I"})
        
        Returns:
            Lista de documentos relevantes
        """
        if k is None:
            k = self.k_results
        
        logger.info(f"🔍 Búsqueda: '{query}' (top {k})")
        
        # Verificar estado de conexión
        if self.connection_status != "connected":
            logger.warning("⚠️ Elasticsearch no conectado. Intentando reconectar...")
            
            if not self._connect_with_retry():
                # Fallback si no puede conectar
                return self._fallback_response(query)
        
        # Intentar búsqueda
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
            
            logger.info(f"✅ {len(results)} documentos encontrados")
            
            # Log de las fuentes encontradas
            sources = set(doc.metadata.get('source', 'N/A') for doc in results)
            logger.debug(f"   Fuentes: {sources}")
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}", exc_info=True)
            self.connection_status = "error"
            return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> List[Document]:
        """
        Respuesta de fallback cuando Elasticsearch no está disponible.
        
        Args:
            query: La consulta original del usuario
        
        Returns:
            Lista con un documento explicando el problema
        """
        fallback_msg = (
            "⚠️ **Sistema de Documentación Temporalmente No Disponible**\n\n"
            f"Lo siento, no puedo acceder a la documentación en este momento.\n\n"
            f"**Estado de conexión:** {self.connection_status}\n"
        )
        
        if self.last_error:
            fallback_msg += f"**Último error:** {self.last_error}\n\n"
        
        fallback_msg += (
            "**¿Qué puedo hacer?**\n"
            "✅ Puedo ayudarte con cálculos financieros usando las herramientas:\n"
            "   - VAN, WACC, Bonos, CAPM, Sharpe Ratio, Gordon Growth, Opciones Call\n\n"
            "⚠️ El administrador debe verificar la conexión a Elasticsearch.\n"
        )
        
        logger.warning(f"🔄 Retornando respuesta de fallback para query: '{query}'")
        
        return [Document(
            page_content=fallback_msg,
            metadata={
                "source": "system_fallback",
                "error": self.last_error,
                "status": self.connection_status
            }
        )]
    
    def get_health_status(self) -> dict:
        """
        Retorna el estado de salud del sistema RAG.
        
        Returns:
            Diccionario con métricas de estado
        """
        return {
            "connection_status": self.connection_status,
            "last_error": self.last_error,
            "retry_count": self.retry_count,
            "index_name": self.index_name,
            "embeddings_loaded": self.embeddings is not None,
            "vector_store_ready": self.vector_store is not None
        }


# ========================================
# INSTANCIA GLOBAL
# ========================================

# Instancia única del sistema RAG
try:
    rag_system = FinancialRAGElasticsearch()
    logger.info("✅ Sistema RAG inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando sistema RAG: {e}", exc_info=True)
    rag_system = None


# ========================================
# TOOL PARA EL AGENTE
# ========================================

@tool
def buscar_documentacion_financiera(consulta: str) -> str:
    """
    Busca información en la documentación financiera CFA indexada en Elasticsearch.
    Retorna contexto relevante con citas de las fuentes.
    
    Args:
        consulta: La pregunta o tema a buscar.
    
    Returns:
        Contexto relevante de la documentación con citas.
    """
    logger.info(f"🔍 RAG Tool invocado con consulta: '{consulta}'")
    
    # Verificar que el sistema esté inicializado
    if not rag_system:
        error_msg = (
            "❌ El sistema RAG no está disponible. "
            "Por favor contacta al administrador para verificar la configuración de Elasticsearch."
        )
        logger.error("Sistema RAG no inicializado al invocar tool")
        return error_msg
    
    # Buscar documentos relevantes
    docs = rag_system.search_documents(consulta, k=3)
    
    if not docs:
        logger.warning(f"No se encontraron documentos para: '{consulta}'")
        return (
            "No encontré información relevante en la documentación indexada. "
            "Esto puede deberse a:\n"
            "1. El tema no está en el material indexado\n"
            "2. El índice está vacío o desactualizado\n"
            "3. La consulta necesita reformularse\n\n"
            "Intenta reformular tu pregunta o consulta directamente al "
            "agente especializado correspondiente para cálculos."
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
            f"📚 Fuente: {source_name}\n"
            f"📖 CFA Level: {cfa_level}\n"
            f"📄 Contenido:\n{content}"
        )
    
    full_context = "\n\n".join(context_parts)
    
    logger.info(f"✅ RAG retornó {len(docs)} fragmentos")
    
    return f"📚 **Información encontrada en la documentación CFA:**\n\n{full_context}"


logger.info("✅ Módulo financial_rag_elasticsearch cargado (LangChain 1.0, con retry & fallback)")