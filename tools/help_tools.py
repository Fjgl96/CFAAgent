# (Añade esto a tus otros imports de tools)
from langchain.tools import tool

# Define tu guía de preguntas como una constante
GUIA_DE_PREGUNTAS = """
Aquí tienes algunos ejemplos de lo que puedes pedirme:

**Cálculos Simples (1 paso):**
* **WACC:** "Necesito calcular el WACC. Ke=12%, Kd=8%, E=60M, D=40M, y tasa impositiva 25%."
* **VAN:** "Calcula el VAN de un proyecto. Inversión inicial 100,000. Flujos [30k, 40k, 50k] a 3 años. Tasa de descuento 10%."
* **Bono:** "Precio de un bono: nominal 1,000, cupón 5% anual, 10 años, YTM 6%."
* **CAPM:** "¿Cuál es el Ke usando CAPM? Tasa libre de riesgo 3%, beta 1.2, retorno de mercado 10%."
* **Sharpe:** "Calcula el Ratio de Sharpe. Retorno 15%, tasa libre de riesgo 4%, volatilidad 20%."
* **Gordon:** "Valora una acción con Gordon Growth. D1=$2.50, Ke=12%, g=4%."
* **Call:** "Precio de opción call: S=100, K=105, T=0.5 años, r=5%, sigma=0.2."

"""

@tool
def obtener_ejemplos_de_uso() -> str:
    """
    Se invoca cuando el usuario pregunta 'qué puedes hacer', 'ayuda',
    'dame ejemplos', o cómo debe preguntar algo.
    Devuelve una guía de preguntas y ejemplos de uso.
    """
    return GUIA_DE_PREGUNTAS