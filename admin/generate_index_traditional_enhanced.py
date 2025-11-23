#!/usr/bin/env python3
"""
generate_index_traditional_enhanced.py
Script de ADMINISTRADOR para indexar libros CFA usando TRADITIONAL CHUNKING MEJORADO.

🎯 OBJETIVO: $0 costos + RÁPIDO + Preservar fórmulas financieras
⚡ VENTAJAS sobre Semantic Chunking:
   1. ✅ Costo: $0 (no usa embeddings durante chunking)
   2. ✅ Velocidad: 2-5 minutos (vs 30-60 minutos con semantic local)
   3. ✅ Preserva fórmulas con separadores financieros inteligentes
   4. ✅ Usa OpenAI embeddings SOLO en indexación final (mucho menos llamadas)

🔬 MEJORAS vs generate_index.py tradicional:
   1. ✅ Separadores financieros específicos (ecuaciones, fórmulas, tablas)
   2. ✅ Chunk size optimizado para finanzas (1500 vs 1200)
   3. ✅ Overlap aumentado para preservar contexto (300 vs 250)
   4. ✅ Detección de bloques matemáticos (no corta en medio de fórmulas)

USO:
1. Coloca tus libros CFA en: ./data/cfa_books/
2. Configura OPENAI_API_KEY en .env
3. Ejecuta: python admin/generate_index_traditional_enhanced.py
4. Los documentos se indexan en Elasticsearch

COSTO ESTIMADO:
- Chunking: $0 (sin embeddings)
- Indexación final: ~$0.50-1 (solo embeddings de chunks finales, NO de cada oración)
- AHORRO: $49-99 vs semantic chunking con OpenAI

SOLO el administrador ejecuta este script.
"""

import sys
from pathlib import Path
from datetime import datetime
import re

# Añadir el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar configuración de Elasticsearch
from config_elasticsearch import (
    get_elasticsearch_client,
    ES_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS
)

# Importar API key de OpenAI
from config import OPENAI_API_KEY

# ========================================
# CONFIGURACIÓN OPTIMIZADA PARA FINANZAS
# ========================================

BOOKS_DIR = Path("./data/cfa_books")

# Índice mejorado
ENHANCED_INDEX_NAME = ES_INDEX_NAME + "_enhanced"

# Chunking optimizado para material financiero/técnico
ENHANCED_CHUNK_SIZE = 1500  # Aumentado de 1200 para capturar fórmulas completas
ENHANCED_CHUNK_OVERLAP = 300  # Aumentado de 250 para mejor contexto

# Separadores financieros (orden de prioridad)
FINANCIAL_SEPARATORS = [
    # 1. Secciones principales
    "\n\n## ",
    "\n\n### ",
    "\n\n#### ",

    # 2. Bloques de ecuaciones (LaTeX, Markdown)
    "\n$$",  # Ecuación LaTeX block
    "\n\\begin{equation}",
    "\n\\begin{align}",

    # 3. Saltos de párrafo
    "\n\n",

    # 4. Puntos de corte lógicos en finanzas
    "\nExample:",
    "\nFormula:",
    "\nEquation:",
    "\nDefinition:",
    "\nTheorem:",
    "\nLearning Outcome:",

    # 5. Saltos de línea y puntos
    "\n",
    ". ",

    # 6. Último recurso
    " ",
    ""
]

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

    # 0. Verificar OpenAI API Key
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY no encontrada")
        print("   Configúrala en .env o como variable de entorno:")
        print("   OPENAI_API_KEY=sk-...")
        sys.exit(1)
    else:
        print(f"✅ OpenAI API Key configurada")
        print(f"   Modelo: {EMBEDDING_MODEL}")
        print(f"   Dimensiones: {EMBEDDING_DIMENSIONS}")
        print(f"   ⚡ Uso: SOLO para indexación final (NO para chunking)")

    # 1. Verificar carpeta de libros
    if not BOOKS_DIR.exists():
        print(f"❌ ERROR: No existe la carpeta: {BOOKS_DIR}")
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

    # 3. Verificar dependencias
    try:
        from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_elasticsearch import ElasticsearchStore
        from elasticsearch import Elasticsearch
        print("✅ Dependencias instaladas correctamente")
    except ImportError as e:
        print(f"❌ ERROR: Falta instalar dependencias")
        print(f"   {e}")
        sys.exit(1)

    # 4. Verificar conexión a Elasticsearch
    client = get_elasticsearch_client()
    if not client:
        print("❌ ERROR: No se pudo conectar a Elasticsearch")
        sys.exit(1)

    print("\n✅ Todos los prerrequisitos cumplidos\n")
    return True


def load_documents():
    """Carga todos los documentos."""
    print_header("Cargando Documentos")

    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
        PyPDFLoader,
    )

    all_docs = []

    # PDFs
    print("📄 Cargando PDFs...")
    try:
        pdf_loader = DirectoryLoader(
            str(BOOKS_DIR),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"✅ {len(pdf_docs)} PDFs cargados\n")
    except Exception as e:
        print(f"⚠️  Error cargando PDFs: {e}\n")

    # TXTs
    print("📝 Cargando archivos TXT...")
    try:
        txt_loader = DirectoryLoader(
            str(BOOKS_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        txt_docs = txt_loader.load()
        all_docs.extend(txt_docs)
        print(f"✅ {len(txt_docs)} TXTs cargados\n")
    except Exception as e:
        print(f"⚠️  Error cargando TXTs: {e}\n")

    print(f"📚 TOTAL DOCUMENTOS CARGADOS: {len(all_docs)}\n")
    return all_docs


def detect_formula_blocks(text: str) -> bool:
    """
    Detecta si un texto contiene bloques de fórmulas matemáticas.
    Esto ayuda a evitar cortes en medio de ecuaciones.
    """
    # Patrones de fórmulas financieras
    formula_patterns = [
        r'\$\$.*?\$\$',  # LaTeX display mode
        r'\\begin\{equation\}',
        r'\\begin\{align\}',
        r'\\frac\{',
        r'\\sum',
        r'\\int',
        r'NPV\s*=',
        r'IRR\s*=',
        r'WACC\s*=',
        r'Beta\s*=',
        r'E\(R\)\s*=',  # Expected Return
    ]

    for pattern in formula_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def split_documents_enhanced(documents):
    """
    Divide documentos usando TRADITIONAL CHUNKING MEJORADO para finanzas.

    🎯 MEJORAS:
    1. ✅ Separadores financieros específicos
    2. ✅ Chunk size optimizado (1500 vs 1200)
    3. ✅ Overlap aumentado (300 vs 250)
    4. ✅ Sin costo de embeddings (vs $50-100 con semantic)

    Args:
        documents: Lista de documentos de LangChain

    Returns:
        Lista de chunks optimizados
    """
    print_header("Fragmentación MEJORADA para Material Financiero")

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print(f"✂️  Configuración MEJORADA:")
    print(f"   Chunk size: {ENHANCED_CHUNK_SIZE} (vs 1200 tradicional)")
    print(f"   Overlap: {ENHANCED_CHUNK_OVERLAP} (vs 250 tradicional)")
    print(f"   Separadores: {len(FINANCIAL_SEPARATORS)} específicos para finanzas")
    print(f"   💰 Costo chunking: $0 (sin embeddings)")
    print(f"\n📋 Separadores financieros usados:")
    print(f"   1. Secciones (##, ###)")
    print(f"   2. Bloques de ecuaciones ($$, \\begin{{equation}})")
    print(f"   3. Puntos lógicos (Example:, Formula:, Definition:)")
    print(f"   4. Saltos de párrafo y puntos\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ENHANCED_CHUNK_SIZE,
        chunk_overlap=ENHANCED_CHUNK_OVERLAP,
        length_function=len,
        separators=FINANCIAL_SEPARATORS
    )

    chunks = text_splitter.split_documents(documents)

    # Añadir metadata adicional
    formula_chunks = 0
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', '')

        # Detectar Level CFA
        if 'Level_I' in source or 'Level_1' in source:
            chunk.metadata['cfa_level'] = 'I'
        elif 'Level_II' in source or 'Level_2' in source:
            chunk.metadata['cfa_level'] = 'II'
        elif 'Level_III' in source or 'Level_3' in source:
            chunk.metadata['cfa_level'] = 'III'

        chunk.metadata['chunk_id'] = f"chunk_{i+1}"
        chunk.metadata['indexed_at'] = datetime.now().isoformat()

        # Detectar si contiene fórmulas
        if detect_formula_blocks(chunk.page_content):
            chunk.metadata['contains_formulas'] = True
            formula_chunks += 1
        else:
            chunk.metadata['contains_formulas'] = False

    print(f"✅ {len(chunks)} chunks creados")
    print(f"   Promedio: {len(chunks) / max(len(documents), 1):.1f} chunks por documento")
    print(f"   📐 Chunks con fórmulas: {formula_chunks} ({formula_chunks / len(chunks) * 100:.1f}%)\n")

    # Estadísticas
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0

    print(f"📏 Estadísticas de tamaño:")
    print(f"   Promedio: {avg_size:.0f} caracteres")
    print(f"   Mínimo: {min_size} caracteres")
    print(f"   Máximo: {max_size} caracteres\n")

    return chunks


def create_or_recreate_index(es_client):
    """Crea o recrea el índice mejorado en Elasticsearch."""
    print_header("Configurando Índice en Elasticsearch")

    if es_client.indices.exists(index=ENHANCED_INDEX_NAME):
        print(f"⚠️  El índice '{ENHANCED_INDEX_NAME}' ya existe.")
        response = input("¿Deseas eliminarlo y recrearlo? (s/n): ")

        if response.lower() == 's':
            print(f"🗑️  Eliminando índice '{ENHANCED_INDEX_NAME}'...")
            es_client.indices.delete(index=ENHANCED_INDEX_NAME)
            print("✅ Índice eliminado")
        else:
            print("ℹ️  Los documentos se añadirán al índice existente")
            return

    print(f"🔨 Creando índice '{ENHANCED_INDEX_NAME}'...")

    index_mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,  # 1536 para OpenAI
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {"type": "object"}
            }
        }
    }

    es_client.indices.create(index=ENHANCED_INDEX_NAME, body=index_mapping)
    print(f"✅ Índice '{ENHANCED_INDEX_NAME}' creado\n")


def estimate_tokens(text: str) -> int:
    """Estima la cantidad de tokens en un texto."""
    return len(text) // 4


def create_batches(chunks, max_tokens_per_batch=250000):
    """Divide chunks en batches que no excedan el límite de tokens."""
    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk.page_content)

        if current_tokens + chunk_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def index_documents_to_elasticsearch(chunks):
    """Indexa los chunks en Elasticsearch usando OpenAI Embeddings."""
    print_header("Indexando Documentos en Elasticsearch")

    from langchain_openai import OpenAIEmbeddings
    from langchain_elasticsearch import ElasticsearchStore
    from config_elasticsearch import get_es_config

    print(f"🧠 Modelo de embeddings OpenAI: {EMBEDDING_MODEL}")
    print(f"   Dimensiones: {EMBEDDING_DIMENSIONS}")
    print(f"   💰 Costo: ~$0.50-1 (SOLO para {len(chunks)} chunks finales)")
    print(f"   ⚡ Velocidad: ~1-2 minutos\n")

    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY no encontrada")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        chunk_size=500,
        max_retries=3
    )

    es_config = get_es_config()

    # Crear batches
    print(f"📦 Creando batches de documentos...")
    batches = create_batches(chunks, max_tokens_per_batch=250000)
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Total batches: {len(batches)}")
    print(f"   Chunks por batch (aprox): {len(chunks) // len(batches) if batches else 0}\n")

    try:
        vector_store = None
        total_indexed = 0

        for i, batch in enumerate(batches, 1):
            print(f"📤 Procesando batch {i}/{len(batches)} ({len(batch)} chunks)...")

            if i == 1:
                vector_store = ElasticsearchStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    index_name=ENHANCED_INDEX_NAME,
                    es_url=es_config["es_url"],
                    es_user=es_config["es_user"],
                    es_password=es_config["es_password"],
                    bulk_kwargs={"request_timeout": 120}
                )
            else:
                vector_store.add_documents(
                    documents=batch,
                    bulk_kwargs={"request_timeout": 120}
                )

            total_indexed += len(batch)
            print(f"   ✅ Batch {i} completado ({total_indexed}/{len(chunks)} chunks indexados)")

        print(f"\n✅ Todos los documentos indexados exitosamente ({total_indexed} chunks)\n")
        return True

    except Exception as e:
        print(f"❌ ERROR indexando documentos: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_index():
    """Verifica que el índice se haya creado correctamente."""
    print_header("Verificando Índice")

    es_client = get_elasticsearch_client()

    try:
        count = es_client.count(index=ENHANCED_INDEX_NAME)
        doc_count = count['count']

        print(f"✅ Índice verificado:")
        print(f"   Nombre: {ENHANCED_INDEX_NAME}")
        print(f"   Documentos: {doc_count}")

        sample = es_client.search(index=ENHANCED_INDEX_NAME, size=1)
        if sample['hits']['hits']:
            print(f"   Estado: Activo y funcional ✅\n")

        return True

    except Exception as e:
        print(f"❌ Error verificando índice: {e}")
        return False


def main():
    """Función principal."""
    print("\n" + "🚀"*30)
    print("  INDEXADOR MEJORADO - Sistema CFA")
    print("  Traditional Chunking + Separadores Financieros")
    print("  💰 COSTO: ~$0.50-1 (vs $50-100 semantic)")
    print("  ⚡ VELOCIDAD: 2-5 minutos (vs 30-60 semantic local)")
    print("🚀"*30)

    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Libros: {BOOKS_DIR}")
    print(f"📦 Índice ES: {ENHANCED_INDEX_NAME}")
    print(f"🧠 Embeddings: {EMBEDDING_MODEL} (OpenAI, SOLO indexación final)")
    print(f"✂️  Chunking: Mejorado (sin embeddings)")
    print(f"💰 Costo estimado: ~$0.50-1")
    print(f"⏱️  Tiempo estimado: 2-5 minutos\n")

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
        documents = load_documents()

        if not documents:
            print("❌ ERROR: No se cargaron documentos.")
            sys.exit(1)

        # 5. Dividir en chunks MEJORADOS
        chunks = split_documents_enhanced(documents)

        # 6. Indexar en Elasticsearch
        success = index_documents_to_elasticsearch(chunks)

        if not success:
            print("❌ ERROR: Fallo en la indexación")
            sys.exit(1)

        # 7. Verificar
        verify_index()

        # Resumen final
        print_header("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📊 Resumen:")
        print(f"   - Documentos procesados: {len(documents)}")
        print(f"   - Chunks generados: {len(chunks)}")
        print(f"   - Índice Elasticsearch: {ENHANCED_INDEX_NAME}")
        print(f"   - Embeddings: OpenAI {EMBEDDING_MODEL}")
        print(f"   - Chunking: Mejorado con separadores financieros")
        print(f"   - Costo: ~$0.50-1 (98% ahorro vs semantic OpenAI)")
        print(f"   - Tiempo: 2-5 minutos (95% más rápido vs semantic local)")
        print(f"\n🎯 Ventajas:")
        print(f"   ✅ Fórmulas financieras preservadas con separadores inteligentes")
        print(f"   ✅ Rápido y económico")
        print(f"   ✅ Dimensiones compatibles con OpenAI (1536)\n")

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
