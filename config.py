# config.py
"""
Configuración general del sistema + LangSmith.
Actualizado para LangChain 1.0+
"""

import os
from pathlib import Path
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
DOCS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ========================================
# API KEYS
# ========================================

_ANTHROPIC_API_KEY = None
_LANGSMITH_API_KEY = None

def load_api_key(secret_name: str, env_var_name: str, required: bool = True) -> str:
    """Carga una API key desde Streamlit secrets o variables de entorno."""
    loaded_key = None
    source = "unknown"

    try:
        # Intenta Streamlit Secrets primero
        loaded_key = st.secrets[secret_name]
        source = "Streamlit secrets"
        print(f"🔑 Cargada {secret_name} desde {source}.")
        return loaded_key
    except (FileNotFoundError, KeyError, AttributeError):
        # Intenta variables de entorno
        try:
            from dotenv import load_dotenv
            dotenv_path = BASE_DIR / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path=dotenv_path)
                print("📄 Archivo .env cargado.")
            else:
                load_dotenv()
        except ImportError:
            print("⚠️ python-dotenv no instalado.")
        
        loaded_key = os.getenv(env_var_name)
        if loaded_key:
            source = "variables de entorno"
            print(f"🔑 Cargada {env_var_name} desde {source}.")
            return loaded_key
        else:
            if required:
                error_message = f"{env_var_name} no encontrada. Configúrala en secrets o .env"
                st.error(error_message)
                print(f"❌ {error_message}")
                st.stop()
            else:
                print(f"⚠️ {env_var_name} no encontrada (opcional).")
                return None
    except Exception as e:
        st.error(f"Error inesperado al cargar {secret_name}: {e}")
        print(f"❌ Error al cargar {secret_name}: {e}")
        if required:
            st.stop()
        return None

# Cargar API keys
ANTHROPIC_API_KEY = load_api_key("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", required=True)
LANGSMITH_API_KEY = load_api_key("LANGSMITH_API_KEY", "LANGSMITH_API_KEY", required=False)

# ========================================
# LANGSMITH CONFIGURATION
# ========================================

# Habilitar LangSmith si hay API key
LANGSMITH_ENABLED = LANGSMITH_API_KEY is not None

if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "financial-agent-prod")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    print("✅ LangSmith habilitado")
    print(f"   Proyecto: {os.environ['LANGCHAIN_PROJECT']}")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("⚠️ LangSmith deshabilitado (no hay API key)")

# ========================================
# LLM CONFIGURATION
# ========================================

LLM_MODEL = "claude-3-5-haiku-20241022"  # Modelo actualizado
LLM_TEMPERATURE = 0.1

_llm_instance = None

def get_llm():
    """Retorna una instancia singleton configurada del LLM."""
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
            print(f"🧠 Instancia LLM ({LLM_MODEL}, temp={LLM_TEMPERATURE}) creada.")
        except Exception as e:
            st.error(f"Error al crear la instancia del LLM: {e}")
            print(f"❌ Error al crear LLM: {e}")
            st.stop()
    return _llm_instance

# ========================================
# OTRAS CONFIGURACIONES
# ========================================

CIRCUIT_BREAKER_MAX_RETRIES = 2

# ========================================
# SISTEMA DE ROLES (OPCIONAL)
# ========================================

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # CAMBIAR EN PRODUCCIÓN

def is_admin(password: str) -> bool:
    """Verifica si el password es correcto para admin."""
    return password == ADMIN_PASSWORD

# ========================================
# LOGGING
# ========================================

def log_event(event_type: str, data: dict) -> bool:
    """Registra eventos en el log correspondiente."""
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

print("✅ Módulo config cargado (LangChain 1.0 + LangSmith).")