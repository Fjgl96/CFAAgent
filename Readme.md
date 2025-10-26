# 💰 Agente Financiero Inteligente (Calculadora CFA - MVP)

Una aplicación web interactiva construida con Streamlit y LangGraph que actúa como un agente financiero inteligente. Es capaz de realizar diversos cálculos financieros estilo CFA mediante una arquitectura multi-agente supervisada.

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Langchain](https://img.shields.io/badge/Langchain-🦜🔗-green.svg)](https://python.langchain.com/)

## ✨ Características (MVP)

* **Interfaz Web Interactiva:** Creada con Streamlit para facilitar las consultas.
* **Arquitectura Multi-Agente:** Utiliza LangGraph con un agente "Supervisor" que direcciona las consultas al especialista adecuado.
* **Agentes Especialistas:**
    * Renta Fija (Valoración de Bonos)
    * Finanzas Corporativas (VAN, WACC)
    * Equity (Valoración Gordon Growth)
    * Portafolio (CAPM, Sharpe Ratio)
    * Derivados (Opciones Call Black-Scholes)
* **Modelo de Lenguaje:** Impulsado por Anthropic Claude 3 Haiku (configurable).
* **Manejo de Errores:** Incluye un "Circuit Breaker" básico para evitar bucles infinitos.
* **Seguridad:** Configuración de API Keys mediante variables de entorno y Streamlit Secrets (no hardcodeado).
* **Código Estructurado:** Organizado en módulos para mejor mantenibilidad (`config`, `tools`, `agents`, `graph`).

## 🏛️ Arquitectura

El sistema utiliza una arquitectura multi-agente supervisada implementada con LangGraph:

1.  El usuario ingresa una consulta en la interfaz de Streamlit.
2.  El agente **Supervisor** recibe la consulta y, basado en su contenido y el historial (si aplica), decide qué agente especialista debe manejarla.
3.  El **Agente Especialista** (ej. `Agente_Finanzas_Corp`) recibe la tarea, extrae los parámetros necesarios usando el LLM y sus schemas Pydantic, y ejecuta su herramienta específica (ej. `_calcular_van`).
4.  El resultado de la herramienta se devuelve al agente especialista.
5.  El agente especialista formula una respuesta final y la devuelve al Supervisor.
6.  El Supervisor recibe la respuesta. Si la tarea está completa, decide `FINISH`. Si hay pasos pendientes (en versiones futuras) o un error manejable, podría redirigir a otro agente o reintentar (limitado por el Circuit Breaker).
7.  La respuesta final se muestra al usuario en Streamlit.

## 🚀 Getting Started (Localmente)

Sigue estos pasos para ejecutar la aplicación en tu máquina local.

### Prerrequisitos

* **Python:** Versión 3.9 o superior recomendada.
* **Git:** Para clonar el repositorio.
* **Anthropic API Key:** Necesitas una clave API de Anthropic.

### Pasos de Instalación

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git) # Reemplaza con tu URL
    cd TU_REPOSITORIO 
    ```

2.  **Crear y Activar Entorno Virtual:** (Altamente recomendado)
    ```bash
    # Crear entorno
    python -m venv venv 
    
    # Activar entorno
    # Windows (CMD/PowerShell)
    .\venv\Scripts\activate 
    # Windows (Git Bash)
    # source venv/Scripts/activate
    # macOS/Linux
    # source venv/bin/activate 
    ```
    *Deberías ver `(venv)` al inicio de tu prompt.*

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar API Key (Local):**
    * Crea un archivo llamado `.env` en la raíz del proyecto (`tu_proyecto_financiero/`).
    * Añade tu API key de Anthropic dentro de este archivo:
        ```env
        # .env
        ANTHROPIC_API_KEY="sk-ant-api03-..." # Reemplaza con tu clave real
        ```
    * **IMPORTANTE:** Asegúrate de que el archivo `.env` esté listado en tu `.gitignore` para no subirlo accidentalmente a GitHub. El archivo `config.py` está configurado para leer esta variable si `python-dotenv` está instalado. Alternativamente, puedes configurar la variable de entorno `ANTHROPIC_API_KEY` directamente en tu sistema operativo.

### Ejecutar la Aplicación

1.  Asegúrate de que tu entorno virtual esté activado.
2.  Ejecuta Streamlit desde la carpeta raíz del proyecto:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Abre tu navegador y ve a la dirección que indique Streamlit (normalmente `http://localhost:8501`).

## ☁️ Despliegue en Streamlit Cloud

1.  **Sube tu Código a GitHub:** Asegúrate de que tu repositorio esté actualizado en GitHub (commit y push), **sin** incluir el archivo `.env` ni la carpeta `venv`. Puedes usar GitHub Desktop o la línea de comandos (`git add .`, `git commit -m "Listo para deploy"`, `git push origin main`).
2.  **Conecta Streamlit Cloud:**
    * Ve a [share.streamlit.io](https://share.streamlit.io/) y haz clic en "New app".
    * Selecciona tu repositorio de GitHub, la rama (`main`) y el archivo principal (`streamlit_app.py`).
3.  **Configura los Secrets:**
    * Antes de hacer clic en "Deploy!", ve a "Advanced settings..." > "Secrets".
    * Pega tu API key usando el formato TOML:
        ```toml
        ANTHROPIC_API_KEY = "sk-ant-api03-..." 
        ```
    * Guarda los secretos.
4.  **Deploy:** Haz clic en "Deploy!". Streamlit Cloud construirá la aplicación e instalará las dependencias de `requirements.txt`.

## 📁 Estructura del Proyecto