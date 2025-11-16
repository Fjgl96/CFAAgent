# utils/logger.py
"""
Sistema de logging centralizado.
Proporciona loggers configurados con rotación de archivos y formato estructurado.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Crear directorio de logs si no existe
LOGS_DIR = Path("/mnt/user-data/shared/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Formato de logs
LOG_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Diccionario de loggers creados (singleton pattern)
_loggers = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Obtiene o crea un logger configurado.
    
    Args:
        name: Nombre del módulo (ej: 'rag', 'agents', 'tools')
        level: Nivel de logging (default: INFO)
    
    Returns:
        Logger configurado con handlers de archivo y consola
    """
    # Si ya existe, retornarlo
    if name in _loggers:
        return _loggers[name]
    
    # Crear nuevo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar duplicación de handlers
    if logger.handlers:
        return logger
    
    # ========================================
    # HANDLER 1: ARCHIVO (con rotación)
    # ========================================
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # ========================================
    # HANDLER 2: CONSOLA
    # ========================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s | %(name)s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Guardar en caché
    _loggers[name] = logger
    
    return logger


def log_system_event(event_type: str, **kwargs):
    """
    Registra eventos del sistema en log dedicado.
    
    Args:
        event_type: Tipo de evento ('startup', 'error', 'query', etc)
        **kwargs: Datos adicionales del evento
    """
    system_logger = get_logger('system_events')
    
    event_data = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        **kwargs
    }
    
    system_logger.info(f"SYSTEM_EVENT: {event_data}")


# Logger por defecto para el módulo
logger = get_logger('main')

# Log de inicialización
logger.info("✅ Sistema de logging inicializado")
logger.debug(f"   Directorio de logs: {LOGS_DIR}")