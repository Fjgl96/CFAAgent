# 💰 Agente Financiero Inteligente - Versión Enterprise

Una aplicación web profesional construida con **Streamlit**, **LangGraph** y **Anthropic Claude** que actúa como un agente financiero inteligente con acceso a documentación CFA mediante RAG (Elasticsearch).

[![LangChain](https://img.shields.io/badge/LangChain-1.0+-blue)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)]()
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.15+-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.9+-purple)]()

---

## 📋 Tabla de Contenidos

1. [Características](#-características)
2. [Arquitectura](#️-arquitectura)
3. [Novedades v2.0](#-novedades-v20)
4. [Instalación](#-instalación)
5. [Configuración](#️-configuración)
6. [Uso](#-uso)
7. [Estructura del Proyecto](#-estructura-del-proyecto)
8. [Mantenimiento](#-mantenimiento)
9. [Troubleshooting](#-troubleshooting)
10. [Contribuir](#-contribuir)

---

## ✨ Características

### 🧮 **Cálculos Financieros Profesionales**
- ✅ **VAN** (Valor Actual Neto)
- ✅ **WACC** (Costo Promedio Ponderado de Capital)
- ✅ **Valoración de Bonos** (con cupones)
- ✅ **CAPM** (Costo del Equity)
- ✅ **Sharpe Ratio** (Retorno ajustado por riesgo)
- ✅ **Gordon Growth** (Valoración de acciones)
- ✅ **Black-Scholes** (Opciones Call europeas)

### 📚 **Sistema RAG con Elasticsearch**
- 🔍 Búsqueda semántica en documentación CFA
- 💾 Vector store con Elasticsearch Cloud
- 🧠 Embeddings con HuggingFace (offline capable)
- 📊 Indexación de múltiples formatos (PDF, TXT, MD)

### 🤖 **Arquitectura Multi-Agente Avanzada**
- 👔 **Supervisor Inteligente** con enrutamiento dinámico
- 🎯 **7 Agentes Especializados** (uno por dominio)
- 🔄 **Circuit Breaker** con tracking de tipos de errores
- 💬 **Memoria Conversacional** persistente por sesión

### 🛡️ **Enterprise-Grade Features**
- 📊 **Health Checks** automáticos al inicio
- 📝 **Logging Estructurado** con rotación de archivos
- 🔐 **Gestión Segura** de credenciales
- ⚡ **Retry Logic** con exponential backoff
- 🎨 **UI Mejorada** con métricas en tiempo real

---

## 🏛️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                             │
│  - Health Check Dashboard                                   │
│  - Chat Interface                                           │
│  - System Metrics                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 LANGGRAPH SUPERVISOR                        │
│  - Enrutamiento Inteligente                                 │
│  - Circuit Breaker (tipos de error)                         │
│  - Gestión de Estado                                        │
└──┬────────┬────────┬────────┬────────┬────────┬────────┬───┘
   │        │        │        │        │        │        │
   ▼        ▼        ▼        ▼        ▼        ▼        ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│Renta ││Fin.  ││Equity││Port- ││Deriva││RAG   ││Ayuda │
│Fija  ││Corp  ││      ││folio ││dos   ││      ││      │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘
   │       │       │       │       │       │       │
   ▼       ▼       ▼       ▼       ▼       ▼       │
┌──────────────────────────────────┐   ┌───────────▼────┐
│   FINANCIAL TOOLS (Python)       │   │  ELASTICSEARCH │
│  - NumPy                          │   │  Vector Store  │
│  - SciPy                          │   │  - Semantic    │
│  - numpy-financial                │   │    Search      │
└───────────────────────────────────┘   │  - CFA Docs    │
                                         └────────────────┘
```

### **Flujo de Ejecución:**

1. **Usuario** → Ingresa consulta en Streamlit
2. **Health Check** → Verifica sistemas (LLM, RAG, Tools)
3. **Supervisor** → Analiza consulta y decide agente
4. **Agente Especialista** → Ejecuta herramienta o consulta RAG
5. **Circuit Breaker** → Monitorea errores y previene bucles
6. **Respuesta** → Se muestra al usuario con contexto

---

## 🎉 Novedades v2.0

### 🔐 **Seguridad Reforzada**
- ❌ Eliminadas credenciales hardcodeadas
- ✅ Validación obligatoria de API keys
- ✅ Certificados SSL con `certifi`
- ✅ Secrets management con `.env` y Streamlit Secrets

### 📊 **Observabilidad Mejorada**
- ✅ Logging estructurado en todos los módulos
- ✅ Health checks con métricas visuales
- ✅ Sistema de eventos con timestamps
- ✅ Logs rotatorios (10MB por archivo)

### 🧠 **Circuit Breaker Inteligente**
- ✅ Tracking por tipos de error (`tool_failure`, `validation`, `capability`)
- ✅ Mensajes personalizados según tipo de fallo
- ✅ Cooldown periods configurables
- ✅ Prevención de bucles infinitos

### 🔍 **Sistema RAG Robusto**
- ✅ Retry con exponential backoff
- ✅ Fallback cuando Elasticsearch no disponible
- ✅ Cache de embeddings
- ✅ Búsqueda con filtros de metadata

### 🎨 **UI Mejorada**
- ✅ Dashboard de estado en sidebar
- ✅ Métricas en tiempo real
- ✅ Advertencias contextuales
- ✅ Mejor feedback visual

---

## 🚀 Instalación

### **Prerrequisitos**
- Python 3.9+
- Elasticsearch 8.15+ (cloud o local)
- Anthropic API Key
- Git

### **Paso 1: Clonar Repositorio**
```bash
git clone https://github.com/tu-usuario/agente-financiero.git
cd agente-financiero
```

### **Paso 2: Crear Entorno Virtual**
```bash
python -m venv venv

# Activar
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### **Paso 3: Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **Paso 4: Configurar Variables de Entorno**
```bash
# Copiar template
cp .env.example .env

# Editar .env con tus credenciales
nano .env
```

### **Paso 5: (Admin) Indexar Documentos CFA**
```bash
# Colocar PDFs en ./data/cfa_books/
mkdir -p data/cfa_books

# Ejecutar indexador
python admin/generate_index.py
```

### **Paso 6: Ejecutar Aplicación**
```bash
streamlit run streamlit_app.py
```

Abre tu navegador en `http://localhost:8501`

---

## ⚙️ Configuración

### **Variables de Entorno Requeridas**

```ini
# .env

# ===== ANTHROPIC =====
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# ===== ELASTICSEARCH =====
ES_HOST=your-cluster.es.cloud
ES_PORT=9200
ES_USERNAME=elastic
ES_PASSWORD=your-password
ES_SCHEME=https
ES_INDEX_NAME=cfa_documents

# ===== LANGSMITH (Opcional) =====
LANGSMITH_API_KEY=lsv2_pt_xxxxx
LANGCHAIN_PROJECT=financial-agent-prod

# ===== ADMIN =====
ADMIN_PASSWORD=change-in-production
```

### **Configuración Avanzada** (`config.py`)

```python
# LLM
LLM_MODEL = "claude-3-5-haiku-20241022"
LLM_TEMPERATURE = 0.1

# Circuit Breaker
CIRCUIT_BREAKER_MAX_RETRIES = 2
CIRCUIT_BREAKER_COOLDOWN = 5  # segundos

# RAG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
```

---

## 💡 Uso

### **Ejemplos de Consultas**

#### 📊 **Cálculos Numéricos**
```
Usuario: Calcula el VAN de un proyecto con inversión inicial de 
         100k, flujos anuales de [30k, 40k, 50k] y tasa de 
         descuento del 10%.

Asistente: He calculado el VAN del proyecto:
           • VAN = $14,397.18
           • Interpretación: Como VAN > 0, el proyecto es rentable 
             y crea valor.
```

#### 📚 **Consultas Conceptuales (RAG)**
```
Usuario: ¿Qué dice el material CFA sobre el WACC?

Asistente: Según la documentación CFA Level II:

           --- Fragmento 1 ---
           📚 Fuente: CFA_L2_Corporate_Finance.pdf
           📄 Contenido:
           El WACC (Weighted Average Cost of Capital) representa 
           la tasa de retorno mínima que una empresa debe obtener 
           en sus inversiones...
```

#### ❓ **Ayuda**
```
Usuario: Ayuda

Asistente: Aquí tienes ejemplos de lo que puedo hacer:

           **Cálculos Simples:**
           • WACC: "Calcula WACC con Ke=12%, Kd=8%..."
           • VAN: "Calcula VAN con inversión 100k..."
           ...
```

### **Comandos Especiales**

| Comando | Descripción |
|---------|-------------|
| `ayuda` | Muestra guía de uso completa |
| `qué puedes hacer` | Lista capacidades |
| `busca en CFA [tema]` | Consulta documentación |

---

## 📁 Estructura del Proyecto

```
agente-financiero/
│
├── streamlit_app.py          # 🎯 Punto de entrada
├── config.py                 # ⚙️ Configuración consolidada
├── requirements.txt          # 📦 Dependencias
├── .env                      # 🔐 Variables de entorno (NO commitear)
├── .env.example              # 📄 Template de .env
│
├── utils/
│   └── logger.py             # 📝 Sistema de logging
│
├── agents/
│   └── financial_agents.py   # 🤖 Agentes especialistas
│
├── graph/
│   └── agent_graph.py        # 🔄 Grafo LangGraph + Circuit Breaker
│
├── tools/
│   ├── financial_tools.py    # 🧮 Herramientas de cálculo
│   ├── help_tools.py         # ❓ Herramientas de ayuda
│   └── schemas.py            # 📋 Esquemas Pydantic
│
├── rag/
│   └── financial_rag_elasticsearch.py  # 🔍 Sistema RAG
│
├── admin/
│   └── generate_index.py     # 👨‍💼 Indexador (solo admin)
│
├── data/
│   └── cfa_books/            # 📚 PDFs CFA (no en repo)
│
└── logs/                     # 📊 Logs rotatorios (auto-generado)
```

---

## 🛠 Mantenimiento

### **Actualizar Índice de Documentación**
```bash
# Cuando agregues nuevos PDFs a data/cfa_books/
python admin/generate_index.py
```

### **Ver Logs**
```bash
# Logs en tiempo real
tail -f /mnt/user-data/shared/logs/streamlit.log

# Filtrar errores
grep "ERROR" /mnt/user-data/shared/logs/*.log
```

### **Health Check Manual**
```python
from config import check_system_health

health = check_system_health()
print(health)
```

### **Limpiar Logs Antiguos**
```bash
# Los logs rotan automáticamente (10MB)
# Para limpiar manualmente:
rm /mnt/user-data/shared/logs/*.log.1
rm /mnt/user-data/shared/logs/*.log.2
```

---

## 🐛 Troubleshooting

### **Problema: Elasticsearch no conecta**
```
❌ Error: No se pudo conectar a Elasticsearch

Solución:
1. Verifica credenciales en .env
2. Confirma que ES_HOST es accesible
3. Revisa firewall/VPN
4. Verifica que el índice existe: 
   python admin/generate_index.py
```

### **Problema: Circuit Breaker activo constantemente**
```
🚨 Sistema detenido por seguridad

Causa: Múltiples errores de validación o herramientas

Solución:
1. Revisa que tu consulta incluya todos los parámetros
2. Verifica sintaxis: "Calcula VAN: inversión 100k, flujos [30k, 40k], tasa 10%"
3. Si persiste, revisa logs: tail -f logs/graph.log
```

### **Problema: RAG siempre offline**
```
⚠️ RAG desconectado

Solución:
1. Verifica conexión a Elasticsearch
2. Confirma que el índice tiene documentos:
   curl -u elastic:password https://host:9200/cfa_documents/_count
3. Re-indexa si es necesario:
   python admin/generate_index.py
```

---

## 🤝 Contribuir

¡Contribuciones son bienvenidas!

### **Agregar Nueva Herramienta Financiera**

1. **Crear schema** en `tools/schemas.py`:
```python
class TIRInput(BaseModel):
    flujos_caja: List[float] = Field(description="...")
```

2. **Implementar tool** en `tools/financial_tools.py`:
```python
@tool("calcular_tir", args_schema=TIRInput)
def _calcular_tir(flujos_caja: List[float]) -> dict:
    logger.info("🔧 Calculando TIR...")
    # Implementación
    return {"tir": resultado}
```

3. **Agregar a lista**:
```python
financial_tool_list = [
    ...,
    _calcular_tir
]
```

4. **Actualizar agente** o crear nuevo agente en `agents/financial_agents.py`

5. **Actualizar supervisor prompt** para incluir nueva capacidad

### **Pull Request Guidelines**
- ✅ Incluir tests unitarios
- ✅ Actualizar README si aplica
- ✅ Seguir estilo de logging existente
- ✅ Documentar parámetros con docstrings

---

## 📜 Licencia

MIT License - Ver `LICENSE` para detalles

---

## 🙏 Agradecimientos

- **Anthropic** - Claude 3.5 Haiku
- **LangChain Team** - Framework LangChain/LangGraph
- **Elasticsearch** - Vector Search
- **HuggingFace** - Embeddings Models
- **Streamlit** - UI Framework

---

## 📧 Contacto

- Issues: [GitHub Issues](https://github.com/fjgl96/agente-financiero/issues)
- Documentación: [Wiki](https://github.com/fjgl96/agente-financiero/wiki)

---

**⭐ Si te gusta este proyecto, dale una estrella en GitHub!**