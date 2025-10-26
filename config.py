# config.py
import os
import streamlit as st
from langchain_anthropic import ChatAnthropic

# Variable global para almacenar la API key cargada
# para evitar cargarla múltiples veces si se importa config.py varias veces.
_ANTHROPIC_API_KEY = None

def load_api_key(secret_name: str, env_var_name: str) -> str:
    """Carga una API key desde Streamlit secrets o variables de entorno."""
    global _ANTHROPIC_API_KEY
    if _ANTHROPIC_API_KEY:
        return _ANTHROPIC_API_KEY

    loaded_key = None
    source = "unknown" # Para logging

    try:
        # Intenta leer desde Streamlit Secrets primero (más seguro en Cloud)
        loaded_key = st.secrets[secret_name]
        source = "Streamlit secrets"
        print(f"🔑 Cargada {secret_name} desde {source}.")
        _ANTHROPIC_API_KEY = loaded_key
        return loaded_key
    except (FileNotFoundError, KeyError, AttributeError): # AttributeError si st.secrets no existe (local)
        # Si falla (local o secreto no encontrado), intenta leer desde variables de entorno
        # Usar python-dotenv para cargar .env si existe
        try:
            from dotenv import load_dotenv
            # Busca el .env en el directorio actual o superior
            dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Asume .env en la raíz del proyecto
            if os.path.exists(dotenv_path):
                 load_dotenv(dotenv_path=dotenv_path) 
                 print("📄 Archivo .env cargado (si existe).")
            else:
                 # Si no está en la raíz, intentar directorio actual (menos común para apps)
                 load_dotenv() # Carga .env del directorio actual si existe
                 
        except ImportError:
            print("⚠️ python-dotenv no instalado, no se cargará .env. Leyendo variables de entorno directas.")
            
        loaded_key = os.getenv(env_var_name)
        if loaded_key:
            source = "variables de entorno"
            print(f"🔑 Cargada {env_var_name} desde {source}.")
            _ANTHROPIC_API_KEY = loaded_key
            return loaded_key
        else:
            # Si ninguna funciona, muestra error y detiene
            error_message = f"{env_var_name} no encontrada. Configúrala en Streamlit secrets (si está desplegado) o como variable de entorno/archivo .env (localmente)."
            st.error(error_message)
            print(f"❌ {error_message}")
            st.stop() # Detiene la ejecución de la app Streamlit si no hay key
    except Exception as e:
        # Captura otros posibles errores inesperados
        st.error(f"Error inesperado al cargar {secret_name}: {e}")
        print(f"❌ Error inesperado al cargar {secret_name}: {e}")
        st.stop()


# --- Configuración LLM ---
# Carga la API key al importar el módulo
ANTHROPIC_API_KEY = load_api_key("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")

LLM_MODEL = "claude-3-haiku-20240307" # Modelo económico y rápido
# Usamos la temperatura 0.1 que probaste, pero puedes ajustarla
LLM_TEMPERATURE = 0.1 

# Variable global para la instancia del LLM (Singleton)
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
                api_key=ANTHROPIC_API_KEY # Pasar la key explícitamente
            )
            print(f"🧠 Instancia LLM ({LLM_MODEL}, temp={LLM_TEMPERATURE}) creada.")
        except Exception as e:
             st.error(f"Error al crear la instancia del LLM: {e}")
             print(f"❌ Error al crear LLM: {e}")
             st.stop()
    return _llm_instance

# --- Otras Configuraciones ---
CIRCUIT_BREAKER_MAX_RETRIES = 2 # Límite de reintentos para el supervisor

print("✅ Módulo config cargado.")