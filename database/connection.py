# database/connection.py
from langchain_community.utilities import SQLDatabase
import streamlit as st # Para mostrar errores

# Importar credenciales desde config
from config import DB_URI

db_instance = None

def get_db_connection():
    """Retorna una instancia singleton de SQLDatabase, manejando errores."""
    global db_instance
    if db_instance is None:
        try:
            db_instance = SQLDatabase.from_uri(DB_URI)
            print("✅ Conexión a PostgreSQL establecida vía Langchain.") # Log
        except Exception as e:
            st.error(f"Error al conectar a PostgreSQL con Langchain: {e}")
            print(f"❌ Error al conectar a PostgreSQL: {e}") # Log
            # Podrías decidir detener la app aquí si la BD es crucial
            # st.stop()
            return None # O retornar None para manejo de error
    return db_instance

# (Opcional) Puedes añadir aquí funciones específicas para interactuar con tu BD
# def get_financial_data(query: str):
#     db = get_db_connection()
#     if db:
#         try:
#             return db.run(query)
#         except Exception as e:
#             st.error(f"Error al ejecutar consulta: {e}")
#             return None
#     return None