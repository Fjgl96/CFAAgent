#!/usr/bin/env python3
"""
generate_index_semantic_local.py
Script de ADMINISTRADOR para indexar libros CFA usando SEMANTIC CHUNKING con EMBEDDINGS LOCALES.

🎯 OBJETIVO: $0 costos de indexación usando sentence-transformers (CPU local)
⚡ TRADE-OFF: Indexación más lenta pero GRATIS (vs $50-100 con OpenAI)

DIFERENCIAS vs generate_index_semantic.py:
1. ✅ Embeddings locales (sentence-transformers) en vez de OpenAI
2. ✅ Costo: $0 (vs $50-100)
3. ⚠️  Tiempo: 30-60 minutos (vs 5-10 minutos con OpenAI)
4. ⚠️  Dimensiones: 384 (all-MiniLM-L6-v2) vs 1536 (OpenAI)

USO:
1. Instala dependencias: pip install sentence-transformers llama-index-embeddings-huggingface
2. Coloca tus libros CFA en: ./data/cfa_books/
3. Ejecuta: python admin/generate_index_semantic_local.py
4. Los documentos se indexan en Elasticsearch con índice semántico local

SOLO el administrador ejecuta este script.
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

# Suprimir warnings de LlamaIndex
warnings.filterwarnings('ignore')

# Añadir el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar configuración de Elasticsearch
from config_elasticsearch import (
    get_elasticsearch_client,
    ES_INDEX_NAME,
    ES_URL,
    ES_USERNAME,
    ES_PASSWORD
)

# ========================================
# CONFIGURACIÓN LOCAL EMBEDDINGS
# ========================================

# Modelos recomendados de sentence-transformers:
# 1. all-MiniLM-L6-v2: 384 dims, RÁPIDO, buena calidad (RECOMENDADO para tu laptop)
# 2. all-mpnet-base-v2: 768 dims, mejor calidad pero más lento
# 3. bge-large-en-v1.5: 1024 dims, mejor calidad pero MUCHO más lento en CPU

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Cambiar si quieres más precisión
LOCAL_EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2

# Donde están los libros CFA (relativo al proyecto)
BOOKS_DIR = Path("./data/cfa_books")

# Nombre del índice semántico LOCAL (diferente a OpenAI)
SEMANTIC_LOCAL_INDEX_NAME = ES_INDEX_NAME + "_semantic_local"

# ========================================
# FUNCIONES
# ========================================

def print_header(text):
    """Imprime un header bonito."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_prerequisites():
    """Verifica que todo esté listo."""
    print_header("Verificando Prerrequisitos")

    # 1. Verificar carpeta de libros
    if not BOOKS_DIR.exists():
        print(f"❌ ERROR: No existe la carpeta: {BOOKS_DIR}")
        print(f"   Créala y coloca tus PDFs ahí:")
        print(f"   mkdir -p {BOOKS_DIR}")
        sys.exit(1)

    # 2. Contar archivos
    pdf_count = len(list(BOOKS_DIR.rglob("*.pdf")))
    txt_count = len(list(BOOKS_DIR.rglob("*.txt")))
    md_count = len(list(BOOKS_DIR.rglob("*.md")))
    total = pdf_count + txt_count + md_count

    print(f"📚 Libros encontrados:")
    print(f"   PDFs: {pdf_count}")
    print(f"   TXTs: {txt_count}")
    print(f"   Markdowns: {md_count}")
    print(f"   TOTAL: {total}")

    if total == 0:
        print(f"\n❌ ERROR: No hay archivos en {BOOKS_DIR}")
        sys.exit(1)

    # 3. Verificar dependencias de LlamaIndex + HuggingFace
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.elasticsearch import ElasticsearchStore
        from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
        print("✅ Dependencias de LlamaIndex + HuggingFace instaladas")
    except ImportError as e:
        print(f"❌ ERROR: Falta instalar dependencias")
        print(f"   {e}")
        print(f"\n   Ejecuta:")
        print(f"   pip install sentence-transformers llama-index-embeddings-huggingface")
        sys.exit(1)

    # 4. Verificar conexión a Elasticsearch
    client = get_elasticsearch_client()
    if not client:
        print("❌ ERROR: No se pudo conectar a Elasticsearch")
        sys.exit(1)

    print("\n✅ Todos los prerrequisitos cumplidos\n")
    return True


def load_documents_llamaindex():
    """
    Carga documentos usando SimpleDirectoryReader de LlamaIndex.
    """
    print_header("Cargando Documentos con LlamaIndex")

    from llama_index.core import SimpleDirectoryReader

    print(f"📂 Directorio: {BOOKS_DIR}")

    try:
        reader = SimpleDirectoryReader(
            input_dir=str(BOOKS_DIR),
            recursive=True,
            required_exts=[".pdf", ".txt", ".md"]
        )

        documents = reader.load_data()

        print(f"✅ {len(documents)} documentos cargados\n")

        # Añadir metadata adicional
        for doc in documents:
            source = doc.metadata.get('file_name', '')

            # Detectar Level CFA
            if 'Level_I' in source or 'Level_1' in source:
                doc.metadata['cfa_level'] = 'I'
            elif 'Level_II' in source or 'Level_2' in source:
                doc.metadata['cfa_level'] = 'II'
            elif 'Level_III' in source or 'Level_3' in source:
                doc.metadata['cfa_level'] = 'III'

            doc.metadata['indexed_at'] = datetime.now().isoformat()

        return documents

    except Exception as e:
        print(f"❌ ERROR cargando documentos: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def split_documents_semantic_local(documents):
    """
    Divide documentos usando SEMANTIC CHUNKING con EMBEDDINGS LOCALES (sentence-transformers).

    🎯 CAMBIOS vs OpenAI:
    1. ✅ HuggingFaceEmbedding en vez de OpenAIEmbedding
    2. ✅ Modelo local: all-MiniLM-L6-v2 (384 dims)
    3. ✅ Costo: $0 (vs $50-100)
    4. ⚠️  Tiempo: ~30-60 min en tu laptop i5 (vs 5-10 min con OpenAI)

    Args:
        documents: Lista de documentos de LlamaIndex

    Returns:
        Lista de nodos semánticos
    """
    print_header("Fragmentación Semántica LOCAL (S29 Pattern + CPU)")

    from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    print(f"🧠 Modelo de embeddings LOCAL: {LOCAL_EMBEDDING_MODEL}")
    print(f"📊 Dimensiones: {LOCAL_EMBEDDING_DIMENSIONS}")
    print(f"💰 Costo: $0 (100% local)")
    print(f"⏱️  Tiempo estimado: 30-60 minutos en CPU i5")
    print(f"📊 Método: Semantic Chunking (percentil 95)")
    print(f"   - Corta solo en cambios drásticos de tema")
    print(f"   - Preserva contexto financiero completo\n")

    try:
        # 0. PRE-PROCESAMIENTO DE SEGURIDAD
        print("🛡️  Ejecutando pre-split de seguridad (max 4000 tokens)...")

        pre_splitter = SentenceSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        safe_nodes = pre_splitter.get_nodes_from_documents(documents, show_progress=True)
        print(f"✅ Pre-split completado: {len(documents)} docs → {len(safe_nodes)} bloques seguros\n")

        # 1. Inicializar modelo de embeddings LOCAL (sentence-transformers)
        print(f"🔄 Descargando modelo {LOCAL_EMBEDDING_MODEL} (primera vez solamente)...")
        embed_model = HuggingFaceEmbedding(
            model_name=LOCAL_EMBEDDING_MODEL,
            # Optimizaciones para CPU (sin GPU)
            device="cpu",
            embed_batch_size=32  # Reducir si tienes poca RAM
        )
        print("✅ Modelo local cargado\n")

        # 2. Crear Semantic Splitter
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model
        )

        print("🔍 Ejecutando análisis semántico LOCAL (esto tomará tiempo)...")
        print("   ⏱️  Estimación: ~2-3 segundos por bloque en CPU i5")
        print(f"   📦 Total bloques: {len(safe_nodes)}\n")

        # 3. Fragmentar los nodos seguros
        nodes = splitter.get_nodes_from_documents(safe_nodes, show_progress=True)

        print(f"\n✅ {len(nodes)} nodos semánticos creados")
        print(f"   Promedio: {len(nodes) / max(len(documents), 1):.1f} nodos por documento original\n")

        # Estadísticas
        chunk_sizes = [len(node.text) for node in nodes]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_size = min(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0

        print(f"📏 Estadísticas de tamaño:")
        print(f"   Promedio: {avg_size:.0f} caracteres")
        print(f"   Mínimo: {min_size} caracteres")
        print(f"   Máximo: {max_size} caracteres\n")

        return nodes

    except Exception as e:
        print(f"❌ ERROR en fragmentación semántica: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_or_recreate_index(es_client):
    """Crea o recrea el índice semántico LOCAL en Elasticsearch."""
    print_header("Configurando Índice en Elasticsearch")

    if es_client.indices.exists(index=SEMANTIC_LOCAL_INDEX_NAME):
        print(f"⚠️  El índice '{SEMANTIC_LOCAL_INDEX_NAME}' ya existe.")
        response = input("¿Deseas eliminarlo y recrearlo? (s/n): ")

        if response.lower() == 's':
            print(f"🗑️  Eliminando índice '{SEMANTIC_LOCAL_INDEX_NAME}'...")
            es_client.indices.delete(index=SEMANTIC_LOCAL_INDEX_NAME)
            print("✅ Índice eliminado")
        else:
            print("ℹ️  Los documentos se añadirán al índice existente")
            return

    # Crear índice con dimensiones LOCALES (384 vs 1536 de OpenAI)
    print(f"🔨 Creando índice '{SEMANTIC_LOCAL_INDEX_NAME}'...")

    index_mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": LOCAL_EMBEDDING_DIMENSIONS,  # 384 para all-MiniLM-L6-v2
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {"type": "object"}
            }
        }
    }

    es_client.indices.create(index=SEMANTIC_LOCAL_INDEX_NAME, body=index_mapping)
    print(f"✅ Índice '{SEMANTIC_LOCAL_INDEX_NAME}' creado\n")


def index_nodes_to_elasticsearch(nodes):
    """
    Indexa nodos semánticos en Elasticsearch usando embeddings LOCALES.
    """
    print_header("Indexando Nodos Semánticos en Elasticsearch (LOCAL)")

    from llama_index.vector_stores.elasticsearch import ElasticsearchStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    print(f"🧠 Modelo de embeddings LOCAL: {LOCAL_EMBEDDING_MODEL}")
    print(f"📊 Dimensiones: {LOCAL_EMBEDDING_DIMENSIONS}")
    print(f"💰 Costo: $0")
    print(f"📦 Total de nodos: {len(nodes)}")
    print(f"🎯 Índice destino: {SEMANTIC_LOCAL_INDEX_NAME}\n")

    try:
        # 1. Inicializar embeddings LOCAL
        print("🔄 Cargando modelo local...")
        embed_model = HuggingFaceEmbedding(
            model_name=LOCAL_EMBEDDING_MODEL,
            device="cpu",
            embed_batch_size=32
        )
        print("✅ Modelo local cargado\n")

        # 2. Crear ElasticsearchStore
        vector_store = ElasticsearchStore(
            index_name=SEMANTIC_LOCAL_INDEX_NAME,
            es_url=ES_URL,
            es_user=ES_USERNAME,
            es_password=ES_PASSWORD,
            request_timeout=300,
            retry_on_timeout=True
        )

        # 3. Crear StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 4. Indexación por LOTES
        batch_size = 200
        total_nodes = len(nodes)
        total_batches = (total_nodes + batch_size - 1) // batch_size

        print(f"📤 Iniciando indexación por lotes (Total: {total_batches} batches)...")
        print(f"   ⏱️  Estimación: ~5-10 segundos por batch en CPU i5\n")

        index = None

        for i in range(0, total_nodes, batch_size):
            batch_nodes = nodes[i : i + batch_size]
            current_batch = (i // batch_size) + 1

            print(f"   Processing batch {current_batch}/{total_batches} ({len(batch_nodes)} nodos)...")

            try:
                if index is None:
                    index = VectorStoreIndex(
                        batch_nodes,
                        storage_context=storage_context,
                        embed_model=embed_model,
                        show_progress=False
                    )
                else:
                    index.insert_nodes(batch_nodes)

                print(f"   ✅ Batch {current_batch} completado.")

            except Exception as e:
                print(f"   ❌ Error en batch {current_batch}: {e}")
                raise e

        print(f"\n✅ Todos los {total_nodes} nodos indexados exitosamente.\n")
        return True

    except Exception as e:
        print(f"❌ ERROR indexando nodos: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_index():
    """Verifica que el índice semántico LOCAL se haya creado correctamente."""
    print_header("Verificando Índice")

    es_client = get_elasticsearch_client()

    try:
        count = es_client.count(index=SEMANTIC_LOCAL_INDEX_NAME)
        doc_count = count['count']

        print(f"✅ Índice verificado:")
        print(f"   Nombre: {SEMANTIC_LOCAL_INDEX_NAME}")
        print(f"   Documentos: {doc_count}")

        sample = es_client.search(index=SEMANTIC_LOCAL_INDEX_NAME, size=1)
        if sample['hits']['hits']:
            print(f"   Estado: Activo y funcional ✅\n")

        return True

    except Exception as e:
        print(f"❌ Error verificando índice: {e}")
        return False


def main():
    """Función principal."""
    print("\n" + "🚀"*30)
    print("  INDEXADOR SEMÁNTICO LOCAL - Sistema CFA")
    print("  LlamaIndex + Semantic Chunking (S29) + CPU")
    print("  💰 COSTO: $0 (vs $50-100 con OpenAI)")
    print("🚀"*30)

    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Libros: {BOOKS_DIR}")
    print(f"📦 Índice ES: {SEMANTIC_LOCAL_INDEX_NAME}")
    print(f"🧠 Embeddings: {LOCAL_EMBEDDING_MODEL} (LOCAL)")
    print(f"📊 Dimensiones: {LOCAL_EMBEDDING_DIMENSIONS}")
    print(f"💰 Costo: $0")
    print(f"⏱️  Tiempo estimado: 30-60 minutos\n")

    # Confirmar
    response = input("¿Deseas continuar? (s/n): ")
    if response.lower() != 's':
        print("❌ Cancelado por el usuario.")
        sys.exit(0)

    try:
        # 1. Verificar prerrequisitos
        check_prerequisites()

        # 2. Obtener cliente ES
        es_client = get_elasticsearch_client()
        if not es_client:
            print("❌ No se pudo conectar a Elasticsearch")
            sys.exit(1)

        # 3. Configurar índice
        create_or_recreate_index(es_client)

        # 4. Cargar documentos
        documents = load_documents_llamaindex()

        if not documents:
            print("❌ ERROR: No se cargaron documentos.")
            sys.exit(1)

        # 5. Fragmentación SEMÁNTICA LOCAL
        nodes = split_documents_semantic_local(documents)

        # 6. Indexar en Elasticsearch
        success = index_nodes_to_elasticsearch(nodes)

        if not success:
            print("❌ ERROR: Fallo en la indexación")
            sys.exit(1)

        # 7. Verificar
        verify_index()

        # Resumen final
        print_header("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📊 Resumen:")
        print(f"   - Documentos procesados: {len(documents)}")
        print(f"   - Nodos semánticos: {len(nodes)}")
        print(f"   - Índice Elasticsearch: {SEMANTIC_LOCAL_INDEX_NAME}")
        print(f"   - Embeddings: {LOCAL_EMBEDDING_MODEL} (LOCAL)")
        print(f"   - Dimensiones: {LOCAL_EMBEDDING_DIMENSIONS}")
        print(f"   - Costo: $0 💰")
        print(f"   - Método: Semantic Chunking (S29)")
        print(f"\n🎯 Los usuarios ya pueden consultar este material.")
        print(f"💡 Ventaja: Fórmulas financieras preservadas + Costo $0\n")

    except KeyboardInterrupt:
        print("\n\n❌ Proceso cancelado por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
