# config.py
"""
Configuración consolidada del sistema.
Incluye: Anthropic, LangSmith, Elasticsearch, Paths, Logging.
Actualizado para LangChain 1.0+ con validación robusta.
"""

import os
from pathlib import Path
from typing import Optional
import streamlit as st
from langchain_anthropic import ChatAnthropic

# ========================================
# PATHS DEL PROYECTO
# ========================================

BASE_DIR = Path(__file__).resolve().parent

# Directorio compartido (persistente)
SHARED_DIR = Path("/mnt/user-data/shared")
SHARED_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectorios
DOCS_DIR = SHARED_DIR / "docs"
LOGS_DIR = SHARED_DIR / "logs"
FAISS_DIR = SHARED_DIR / "faiss_index"

for directory in [DOCS_DIR, LOGS_DIR, FAISS_DIR]:
    directory.mkdir(exist_ok=True)

# ========================================
# HELPER: CARGAR API KEYS SEGURAS
# ========================================

def load_api_key(secret_name: str, env_var_name: str, required: bool = True) -> Optional[str]:
    """
    Carga una API key desde Streamlit secrets o variables de entorno.
    
    Args:
        secret_name: Nombre en Streamlit secrets
        env_var_name: Nombre en variables de entorno
        required: Si es True, detiene la app si no encuentra la key
    
    Returns:
        API key o None si es opcional y no existe
    """
    loaded_key = None
    source = "unknown"

    # Intenta Streamlit Secrets primero
    try:
        loaded_key = st.secrets[secret_name]
        source = "Streamlit secrets"
        return loaded_key
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    
    # Intenta variables de entorno
    try:
        from dotenv import load_dotenv
        dotenv_path = BASE_DIR / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
        else:
            load_dotenv()
    except ImportError:
        pass
    
    loaded_key = os.getenv(env_var_name)
    if loaded_key:
        source = "variables de entorno"
        return loaded_key
    
    # No se encontró la key
    if required:
        error_message = (
            f"❌ {env_var_name} no encontrada.\n\n"
            f"**Opciones de configuración:**\n"
            f"1. Crear archivo `.env` con: `{env_var_name}=tu_valor`\n"
            f"2. Configurar en Streamlit Cloud secrets\n"
            f"3. Exportar variable de entorno: `export {env_var_name}=tu_valor`"
        )
        st.error(error_message)
        st.stop()
    
    return None

# ========================================
# API KEYS - ANTHROPIC & LANGSMITH
# ========================================

ANTHROPIC_API_KEY = load_api_key("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", required=True)
LANGSMITH_API_KEY = load_api_key("LANGSMITH_API_KEY", "LANGSMITH_API_KEY", required=False)

# ========================================
# LANGSMITH CONFIGURATION
# ========================================

LANGSMITH_ENABLED = LANGSMITH_API_KEY is not None

if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "financial-agent-prod")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ========================================
# ELASTICSEARCH CONFIGURATION
# ========================================

# Cargar credenciales (SIN defaults para seguridad)
ES_HOST = load_api_key("ES_HOST", "ES_HOST", required=True)
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_USERNAME = load_api_key("ES_USERNAME", "ES_USERNAME", required=True)
ES_PASSWORD = load_api_key("ES_PASSWORD", "ES_PASSWORD", required=True)
ES_SCHEME = os.getenv("ES_SCHEME", "https")

# URL completa
ES_URL = f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"

# Configuración del índice
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "cfa_documents")

# Configuración de embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Configuración de chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

# ========================================
# FUNCIÓN: CLIENTE ELASTICSEARCH
# ========================================

def get_elasticsearch_client():
    """
    Crea y retorna un cliente de Elasticsearch configurado.
    
    Returns:
        Cliente ES o None si falla la conexión
    """
    from elasticsearch import Elasticsearch
    
    try:
        # Importar certifi para SSL seguro
        try:
            import certifi
            ca_certs = certifi.where()
            verify_certs = True
        except ImportError:
            ca_certs = None
            verify_certs = False
            print("⚠️ certifi no disponible, deshabilitando verificación SSL")
        
        es_client = Elasticsearch(
            [ES_URL],
            basic_auth=(ES_USERNAME, ES_PASSWORD),
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        if es_client.ping():
            return es_client
        else:
            return None
    
    except Exception as e:
        print(f"❌ Error conectando a Elasticsearch: {e}")
        return None

# ========================================
# FUNCIÓN: CONFIG ELASTICSEARCH PARA LANGCHAIN
# ========================================

def get_es_config() -> dict:
    """
    Retorna configuración para ElasticsearchStore de LangChain.
    
    Returns:
        Diccionario con configuración ES
    """
    return {
        "es_url": ES_URL,
        "es_user": ES_USERNAME,
        "es_password": ES_PASSWORD,
        "index_name": ES_INDEX_NAME
    }

# ========================================
# LLM CONFIGURATION
# ========================================

LLM_MODEL = "claude-3-5-haiku-20241022"
LLM_TEMPERATURE = 0.1

_llm_instance = None

def get_llm():
    """
    Retorna una instancia singleton configurada del LLM.
    
    Returns:
        ChatAnthropic instance
    """
    global _llm_instance
    if _llm_instance is None:
        if not ANTHROPIC_API_KEY:
            st.error("No se pudo inicializar el LLM: API Key no disponible.")
            st.stop()
        try:
            _llm_instance = ChatAnthropic(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                api_key=ANTHROPIC_API_KEY
            )
        except Exception as e:
            st.error(f"Error al crear la instancia del LLM: {e}")
            st.stop()
    return _llm_instance

# ========================================
# CIRCUIT BREAKER CONFIGURATION
# ========================================

CIRCUIT_BREAKER_MAX_RETRIES = 2
CIRCUIT_BREAKER_COOLDOWN = 5  # segundos

# ========================================
# ADMIN CONFIGURATION
# ========================================

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

def is_admin(password: str) -> bool:
    """Verifica si el password es correcto para admin."""
    return password == ADMIN_PASSWORD

# ========================================
# HEALTH CHECK
# ========================================

def check_system_health() -> dict:
    """
    Verifica el estado de todos los componentes del sistema.
    
    Returns:
        Diccionario con estado de cada componente
    """
    health = {
        "anthropic": False,
        "langsmith": False,
        "elasticsearch": False,
        "llm": False
    }
    
    # Check Anthropic API Key
    health["anthropic"] = ANTHROPIC_API_KEY is not None
    
    # Check LangSmith
    health["langsmith"] = LANGSMITH_ENABLED
    
    # Check Elasticsearch
    try:
        es_client = get_elasticsearch_client()
        if es_client and es_client.ping():
            health["elasticsearch"] = True
    except:
        pass
    
    # Check LLM
    try:
        llm = get_llm()
        health["llm"] = llm is not None
    except:
        pass
    
    return health

# ========================================
# LOGGING HELPER
# ========================================

def log_event(event_type: str, data: dict) -> bool:
    """
    Registra eventos en el log correspondiente.
    DEPRECATED: Usar utils.logger en su lugar.
    """
    import json
    from datetime import datetime
    
    log_file = LOGS_DIR / f"{event_type}_log.json"
    
    try:
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        logs.append(event)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
    except Exception as e:
        print(f"❌ Error logging event: {e}")
        return False

# ========================================
# INICIALIZACIÓN
# ========================================

print("✅ Configuración consolidada cargada")
print(f"   LLM: {LLM_MODEL}")
print(f"   LangSmith: {'Habilitado' if LANGSMITH_ENABLED else 'Deshabilitado'}")
print(f"   Elasticsearch: {ES_URL}")
print(f"   Índice ES: {ES_INDEX_NAME}")