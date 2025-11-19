# 💰 Agente Financiero Inteligente (Calculadora CFA - MVP)

Una aplicación web interactiva construida con Streamlit y LangGraph que actúa como un agente financiero inteligente. Es capaz de realizar diversos cálculos financieros estilo CFA mediante una arquitectura multi-agente supervisada con sistema RAG integrado.

## 📋 Tabla de Contenidos
1.  [Características](#-características-mvp)
2.  [Arquitectura](#️-arquitectura)
3.  [Ejemplos de Uso](#-ejemplos-de-uso-guía-de-preguntas)
4.  [Instalación Local](#-getting-started-localmente)
5.  [Despliegue](#️-despliegue-en-streamlit-cloud)
6.  [Estructura del Proyecto](#-estructura-del-proyecto)
7.  [¡Contribuye!](#-contribuye)
8.  [Licencia](#-licencia)

## ✨ Características (MVP)

* **Interfaz Web Interactiva:** Creada con Streamlit para facilitar las consultas.
* **Arquitectura Multi-Agente:** Utiliza LangGraph con un agente "Supervisor" que direcciona las consultas al especialista adecuado.
* **Sistema RAG Integrado:** Búsqueda semántica en documentación CFA usando Elasticsearch + OpenAI Embeddings.
* **22 Herramientas Financieras CFA Level I:** Cálculos especializados organizados por área
* **Agentes Especialistas:**
    * **Renta Fija (6 herramientas):** Valor de bonos, Duration Macaulay/Modificada, Convexity, Current Yield, Bonos cupón cero
    * **Finanzas Corporativas (5 herramientas):** VAN, WACC, TIR (IRR), Payback Period, Profitability Index
    * **Portafolio (7 herramientas):** CAPM, Sharpe Ratio, Treynor Ratio, Jensen's Alpha, Beta/Retorno/Std Dev de Portafolio
    * **Derivados (3 herramientas):** Opciones Call/Put (Black-Scholes), Put-Call Parity
    * **Equity (1 herramienta):** Gordon Growth Model
    * **RAG:** Consultas a documentación CFA con síntesis bilingüe
    * **Ayuda:** Guía de uso con ejemplos
* **Modelo de Lenguaje:** Impulsado por Anthropic Claude 3.5 Haiku (configurable).
* **Observabilidad:** Integración opcional con LangSmith para tracing y debugging.
* **Manejo de Errores:** Incluye un "Circuit Breaker" básico para evitar bucles infinitos.
* **Seguridad:** Configuración de API Keys mediante variables de entorno y Streamlit Secrets (no hardcodeado).
* **Código Estructurado:** Organizado en módulos para mejor mantenibilidad (`config`, `tools`, `agents`, `graph`, `rag`).

## 🏛️ Arquitectura

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e3f2fd','primaryTextColor':'#1565c0','primaryBorderColor':'#1976d2','lineColor':'#424242','secondaryColor':'#fff3e0','tertiaryColor':'#f3e5f5'}}}%%

flowchart TD
    START([🚀 USUARIO<br/>Streamlit UI])
    
    START --> INPUT[📝 Input Query<br/>HumanMessage]
    
    INPUT --> SUPERVISOR{🧭 SUPERVISOR<br/>Claude 3 Haiku<br/>RouterSchema}
    
    SUPERVISOR -->|Consultas Teóricas| RAG[📚 AGENTE RAG<br/>buscar_documentacion_financiera]
    SUPERVISOR -->|Ayuda y Ejemplos| HELP[❓ AGENTE AYUDA<br/>obtener_ejemplos_de_uso]
    SUPERVISOR -->|VAN, WACC, TIR, Payback, PI| CORP[💼 AGENTE FIN. CORP<br/>5 herramientas]
    SUPERVISOR -->|Bonos, Duration, Convexity| BOND[📊 AGENTE RENTA FIJA<br/>6 herramientas]
    SUPERVISOR -->|Gordon Growth| EQUITY[📈 AGENTE EQUITY<br/>1 herramienta]
    SUPERVISOR -->|CAPM, Sharpe, Treynor, Jensen, etc.| PORT[📂 AGENTE PORTAFOLIO<br/>7 herramientas]
    SUPERVISOR -->|Call, Put, Put-Call Parity| DERIV[💹 AGENTE DERIVADOS<br/>3 herramientas]
    
    RAG --> RAGVS[(🔍 ELASTICSEARCH<br/>Vector Store<br/>Embeddings)]
    RAGVS --> RAGDOCS[📄 Docs CFA<br/>Fragmentos Relevantes]
    RAGDOCS --> RAGEND[Respuesta Contextual]
    
    HELP --> HELPEND[Guía de Preguntas]
    
    CORP --> TOOLS1[🧮 Python Tools<br/>numpy-financial]
    BOND --> TOOLS1
    EQUITY --> TOOLS1
    PORT --> TOOLS1
    DERIV --> TOOLS1
    
    TOOLS1 --> CALC[Cálculo Ejecutado<br/>JSON Result]
    
    CALC --> BACK[⬅️ AIMessage]
    BACK --> SUPERVISOR
    
    SUPERVISOR -->|Tarea Completa| FINISH{✅ FINISH?}
    
    FINISH -->|Si| END([💬 RESPUESTA FINAL<br/>Usuario ve resultado])
    FINISH -->|No Continuar| SUPERVISOR
    
    RAGEND --> END
    HELPEND --> END
    
    SUPERVISOR -.->|Error Count mayor o igual a 2| BREAK[🚨 Circuit Breaker<br/>Detener Proceso]
    BREAK --> END
    
    classDef userNode fill:#4caf50,stroke:#2e7d32,stroke-width:3px,color:#fff
    classDef supervisorNode fill:#2196f3,stroke:#1565c0,stroke-width:3px,color:#fff
    classDef agentNode fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    classDef toolNode fill:#9c27b0,stroke:#6a1b9a,stroke-width:2px,color:#fff
    classDef ragNode fill:#00bcd4,stroke:#006064,stroke-width:2px,color:#fff
    classDef endNode fill:#4caf50,stroke:#2e7d32,stroke-width:3px,color:#fff
    
    class START,END userNode
    class SUPERVISOR,FINISH supervisorNode
    class CORP,BOND,EQUITY,PORT,DERIV,HELP agentNode
    class RAG,RAGVS,RAGDOCS ragNode
    class TOOLS1,CALC toolNode
    class BREAK endNode
```

### Flujo de Ejecución:

1.  El usuario ingresa una consulta en la interfaz de Streamlit.
2.  El agente **Supervisor** recibe la consulta y, basado en su contenido y el historial, decide qué agente especialista debe manejarla.
3.  El **Agente Especialista** (ej. `Agente_Finanzas_Corp`) recibe la tarea, extrae los parámetros necesarios usando el LLM y ejecuta su herramienta específica (ej. `_calcular_van`).
4.  El resultado de la herramienta se devuelve al agente especialista.
5.  El agente especialista formula una respuesta final y la devuelve al Supervisor.
6.  El Supervisor recibe la respuesta. Si la tarea está completa, decide `FINISH`.
7.  La respuesta final se muestra al usuario en Streamlit.

### Componentes Principales:
- **Portal de Entrada:** Streamlit UI para captura de consultas
- **Supervisor:** Orquestador inteligente con Claude 3.5 Haiku
- **8 Agentes Especializados:** Renta Fija, Finanzas Corp, Equity, Portafolio, Derivados, RAG, Síntesis RAG, Ayuda
- **22 Python Tools:** Cálculos deterministas con numpy/scipy/numpy-financial
- **Sistema RAG:** Elasticsearch + OpenAI Embeddings (text-embedding-3-large) para búsqueda semántica bilingüe
- **MemorySaver:** Persistencia de contexto durante la sesión

## 🚀 Ejemplos de Uso (Guía de Preguntas)

Una vez que la aplicación esté corriendo, puedes usar estas consultas como ejemplo:

### Cálculos Financieros

* **Ayuda (Para ver la guía):**
    * `Ayuda`
    * `¿Qué puedes hacer?`

* **Finanzas Corporativas (5 herramientas):**
    * `Calcula el VAN de un proyecto. Inversión inicial 100,000. Flujos [30k, 40k, 50k] a 3 años. Tasa de descuento 10%.`
    * `Necesito calcular el WACC. Ke=12%, Kd=8%, E=60M, D=40M, y tasa impositiva 25%.`
    * `Calcula la TIR (IRR) de un proyecto con inversión inicial 50,000 y flujos [15k, 20k, 25k].`
    * `¿Cuál es el Payback Period? Inversión 80,000, flujos [20k, 30k, 35k, 25k].`
    * `Profitability Index con tasa 8%, inversión 100k, flujos [40k, 50k, 60k].`

* **Renta Fija (6 herramientas):**
    * `Precio de un bono: nominal 1,000, cupón 5% anual, 10 años, YTM 6%.`
    * `Calcula la Duration Macaulay de un bono con nominal 1000, cupón 6%, YTM 5%, 5 años, pagos semestrales.`
    * `Duration Modificada de un bono con Duration Macaulay 4.5 años, YTM 6%, pagos semestrales.`
    * `Convexity de un bono: nominal 1000, cupón 5%, YTM 6%, 10 años, semestral.`
    * `Current Yield de un bono con cupón anual $60 y precio actual $950.`
    * `Valor de un bono cupón cero: nominal 1000, YTM 5%, vencimiento 3 años.`

* **Portafolio (7 herramientas):**
    * `¿Cuál es el costo de equity (Ke) usando CAPM? La tasa libre de riesgo es 3%, el beta es 1.2 y el retorno de mercado es 10%.`
    * `Calcula el Ratio de Sharpe. Retorno 15%, tasa libre de riesgo 4%, volatilidad 20%.`
    * `Treynor Ratio con retorno portafolio 18%, tasa libre riesgo 4%, beta 1.3.`
    * `Jensen's Alpha: retorno portafolio 16%, tasa libre riesgo 3%, beta 1.1, retorno mercado 12%.`
    * `Beta de portafolio con 60% en activo (beta 1.2) y 40% en activo (beta 0.8).`
    * `Retorno esperado de portafolio: 70% activo (retorno 12%) + 30% activo (retorno 8%).`
    * `Desviación estándar de portafolio: 50% activo A (σ=20%), 50% activo B (σ=15%), correlación 0.3.`

* **Equity (1 herramienta):**
    * `Valora una acción con Gordon Growth. El dividendo esperado (D1) es $2.50, el costo de equity (Ke) es 12%, y la tasa de crecimiento (g) es 4%.`

* **Derivados (3 herramientas):**
    * `Precio de opción call: S=100, K=105, T=0.5 años, r=5%, sigma=20%.`
    * `Precio de opción put europea: S=50, K=52, T=1 año, r=4%, sigma=25%.`
    * `Verifica Put-Call Parity: call=$5, put=$3, S=$100, K=$100, T=1 año, r=5%.`

### Consultas a Documentación CFA (RAG)

* `¿Qué dice el material CFA sobre el WACC?`
* `Explica el concepto de Duration según el CFA`
* `Busca información sobre el modelo Gordon Growth en el CFA`
* `¿Qué es el Beta según la documentación CFA?`
* `¿Cómo se calcula el Treynor Ratio según el CFA?`
* `Explica la diferencia entre Duration Macaulay y Modificada`
* `¿Qué es Jensen's Alpha y cómo se interpreta?`

## ⚠️ Disclaimer Legal

**Importante: Lee antes de usar este proyecto**

Este es un **proyecto académico y experimental** desarrollado para investigación sobre IA Generativa aplicada a educación financiera.

### Avisos Importantes:

- ❌ **NO está afiliado ni respaldado por el CFA Institute®**
- ✅ Este proyecto es una **herramienta de estudio complementaria**, NO un sustituto del material oficial
- 📚 **Responsabilidad del usuario**: Si utilizas la funcionalidad RAG, debes poseer legalmente el material CFA que indexes en tu propia instancia
- 🔒 **Uso recomendado**: Configurar en modo BYOD (Bring Your Own Documents) para uso personal con material que hayas adquirido legalmente
- 🎓 **Propósito educativo**: Diseñado para ayudar a comprender conceptos, NO para reproducir contenido protegido por copyright

**Nota sobre derechos de autor:**
- El sistema está programado para explicar conceptos con sus propias palabras, no para copiar textualmente material del CFA Institute
- Si el usuario solicita copias literales de contenido protegido, el sistema rechazará la solicitud

**CFA®** y **Chartered Financial Analyst®** son marcas registradas del CFA Institute. Este proyecto no tiene ninguna afiliación con el CFA Institute.

### 🔒 Protecciones Técnicas de Copyright

El sistema implementa **múltiples capas de protección** para evitar reproducción inadvertida de contenido protegido:

**✅ Implementado:**
- **Guardrails Semánticos:** Clasificador que intercepta queries riesgosas (transcripciones, copias literales, solicitudes de capítulos completos)
- **Prompts Anti-Reproducción:** El agente RAG opera como "tutor conceptual" que genera explicaciones originales, NO como motor de recuperación documental
- **Prohibiciones explícitas:** Referencias bibliográficas específicas deshabilitadas, ejemplos numéricos generados por el sistema

**⚠️ Limitaciones arquitecturales:**
- El sistema RAG actual usa fragmentos textuales (no embeddings abstractos)
- Para máxima protección, se recomienda configuración BYOD con material adquirido legalmente

📄 **Documentación técnica completa:** Ver `/docs/COPYRIGHT_PROTECTION.md` para análisis detallado de las 4 capas de protección.

---

## 🚀 Getting Started (Localmente)

Sigue estos pasos para ejecutar la aplicación en tu máquina local.

### Prerrequisitos

* **Python:** Versión 3.9 o superior recomendada.
* **Git:** Para clonar el repositorio.
* **Anthropic API Key:** Necesitas una clave API de Anthropic.
* **(Opcional) LangSmith API Key:** Para observabilidad y debugging.

### Pasos de Instalación

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
    cd TU_REPOSITORIO
    ```

2.  **Crear y Activar Entorno Virtual:** (Altamente recomendado)
    ```bash
    # Crear entorno
    python -m venv venv
    
    # Activar entorno
    # Windows (CMD/PowerShell)
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
    Deberías ver `(venv)` al inicio de tu prompt.

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar API Keys y Elasticsearch (Local):**
    * Crea un archivo llamado `.env` en la raíz del proyecto.
    * Añade tus API keys y credenciales de Elasticsearch dentro de este archivo:
        ```ini
        # .env
        ANTHROPIC_API_KEY="sk-ant-api03-..."
        LANGSMITH_API_KEY="lsv2_pt_..."  # Opcional
        LANGCHAIN_PROJECT="financial-agent-dev"  # Opcional
        
        # Elasticsearch Configuration
        ES_HOST="tu-servidor-elasticsearch.com"
        ES_PORT="9200"
        ES_USERNAME="elastic"
        ES_PASSWORD="tu-contraseña-segura"
        ES_SCHEME="https"
        ES_INDEX_NAME="cfa_documents"
        ```
    * **IMPORTANTE:** Asegúrate de que el archivo `.env` esté listado en tu `.gitignore` para no subirlo accidentamente a GitHub.

5.  **(Opcional) Indexar Documentación CFA en Elasticsearch:**
    * Si tienes material CFA para indexar:
    ```bash
    # Crear directorio para documentos
    mkdir -p data/cfa_books
    
    # Copiar tus PDFs ahí
    # Luego ejecutar el indexador (requiere permisos de admin)
    python admin/generate_index.py
    ```
    * **Nota:** Asegúrate de tener Elasticsearch corriendo y configurado correctamente.

### Ejecutar la Aplicación

1.  Asegúrate de que tu entorno virtual esté activado.
2.  Ejecuta Streamlit desde la carpeta raíz del proyecto:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Abre tu navegador y ve a la dirección que indique Streamlit (normalmente `http://localhost:8501`).

## ☁️ Despliegue en Streamlit Cloud

1.  **Sube tu Código a GitHub:** Asegúrate de que tu repositorio esté actualizado en GitHub (`git push`), **sin** incluir el archivo `.env` ni la carpeta `venv`.
2.  **Conecta Streamlit Cloud:**
    * Ve a `share.streamlit.io` y haz clic en "New app".
    * Selecciona tu repositorio de GitHub, la rama (`main`) y el archivo principal (`streamlit_app.py`).
3.  **Configura los Secrets:**
    * Antes de hacer clic en "Deploy!", ve a "Advanced settings..." > "Secrets".
    * Pega tus API keys y configuración de Elasticsearch usando el formato TOML:
        ```toml
        ANTHROPIC_API_KEY = "sk-ant-api03-..." 
        LANGSMITH_API_KEY = "lsv2_pt_..."  # Opcional
        LANGCHAIN_PROJECT = "financial-agent-prod"  # Opcional
        
        # Elasticsearch
        ES_HOST = "tu-servidor-elasticsearch.com"
        ES_PORT = "9200"
        ES_USERNAME = "elastic"
        ES_PASSWORD = "tu-contraseña-segura"
        ES_SCHEME = "https"
        ES_INDEX_NAME = "cfa_documents"
        ```
    * Guarda los secretos.
4.  **Deploy:** Haz clic en "Deploy!".

## 📁 Estructura del Proyecto

El repositorio está organizado de forma modular para facilitar la mantenibilidad y la adición de nuevos agentes o herramientas:

```bash
tu_repositorio/
├── agents/                 # Define los agentes especialistas y el supervisor
│   ├── __init__.py
│   └── financial_agents.py
├── graph/                  # Construye y compila el StateGraph de LangGraph
│   ├── __init__.py
│   └── agent_graph.py
├── tools/                  # Define las @tools (funciones de cálculo)
│   ├── __init__.py
│   ├── financial_tools.py
│   ├── help_tools.py
│   └── schemas.py          # Pydantic schemas para las tools
├── rag/                    # Sistema RAG con Elasticsearch
│   ├── __init__.py
│   └── financial_rag_elasticsearch.py  # RAG usando Elasticsearch como vector store
├── admin/                  # Scripts de administración
│   └── generate_index.py   # Indexador de documentos CFA
├── data/                   # Datos persistentes (no en repo)
│   └── cfa_books/         # PDFs de material CFA
├── config.py              # Configuración (LLM, API keys, LangSmith)
├── config_elasticsearch.py # Configuración (legacy, no usado actualmente)
├── database/              # Conexión a BD (opcional, no usado en MVP)
│   └── connection.py
├── requirements.txt       # Dependencias del proyecto
├── streamlit_app.py       # El punto de entrada de la app web
├── .env.example          # Ejemplo de variables de entorno
├── .gitignore            # Archivos a ignorar
└── README.md             # Esta documentación
```

## 🤝 ¡Contribuye!

¡Este es un proyecto abierto y las contribuciones son bienvenidas! La arquitectura está diseñada para que añadir nuevas herramientas sea fácil.

### Cómo añadir una nueva herramienta financiera:

1.  **Fork** del repositorio y crea una rama (`git checkout -b feature/nueva-calculadora`)
2.  **Crea el Schema Pydantic** en `tools/schemas.py`:
    ```python
    class MiNuevaHerramientaInput(BaseModel):
        """Schema para mi nueva herramienta."""
        parametro1: float = Field(description="Descripción", gt=0)
    ```
3.  **Implementa la herramienta** en `tools/financial_tools.py`:
    ```python
    @tool("mi_nueva_herramienta", args_schema=MiNuevaHerramientaInput)
    def _mi_nueva_herramienta(parametro1: float) -> dict:
        """Calcula [lo que sea]."""
        # Tu lógica aquí
        return {"resultado": valor}
    ```
4.  **Añade al agente apropiado** en `agents/financial_agents.py`:
    - Importa tu herramienta
    - Agrégala a la lista de tools del agente correspondiente
    - Actualiza el PROMPT del agente
5.  **Actualiza el Supervisor** para que reconozca la nueva capacidad
6.  **(Opcional)** Añade términos bilingües en `rag/financial_rag_elasticsearch.py`
7.  **Actualiza Readme.md** con ejemplos de uso
8.  Haz **Commit** y **Push** → Abre un **Pull Request**

### Ideas para contribuir:
- ⭐ Más herramientas CFA Level II/III
- ⭐ Tests unitarios para validar cálculos
- ⭐ Interfaz mejorada en Streamlit
- ⭐ Documentación de API
- ⭐ Más ejemplos y tutoriales

## 📜 Licencia

Este proyecto está bajo la Licencia MIT.

---

**Stack Tecnológico:**
- LangChain 0.3.0+
- LangGraph 1.0.1+
- Anthropic Claude 3.5 Haiku (con fallback a GPT-4o)
- Streamlit 1.39+
- Elasticsearch 8.15+ (Vector Store)
- OpenAI Embeddings (text-embedding-3-large, 3072 dims)
- Pydantic 2.0+
- NumPy, SciPy, numpy-financial
- LangSmith (Opcional para observabilidad)

---

**Versión:** 2.0.0 (22 herramientas CFA Level I)
**Última actualización:** Noviembre 2024