#!/usr/bin/env python3
"""
evaluate_rag.py
Script de evaluación automática del sistema RAG usando Ragas.

Mide la calidad de las respuestas del sistema RAG en:
- Faithfulness (Fidelidad): ¿La respuesta está basada en el contexto?
- Answer Relevancy (Relevancia): ¿La respuesta responde la pregunta?
- Context Precision (Precisión): ¿El contexto recuperado es relevante?

USO:
1. Asegúrate de tener el índice Elasticsearch generado
2. Configura OPENAI_API_KEY en .env
3. Ejecuta: python evaluate_rag.py
"""

import sys
from pathlib import Path

# Añadir el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime
from typing import List, Dict
import pandas as pd

# Importar sistema RAG
from rag.financial_rag_elasticsearch import rag_system

# Importar config
from config import get_llm, OPENAI_API_KEY

# ========================================
# CONFIGURACIÓN
# ========================================

def print_header(text: str):
    """Imprime un header bonito."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


# ========================================
# GOLDEN DATASET (5 PREGUNTAS CFA)
# ========================================

GOLDEN_DATASET = [
    {
        "question": "¿Qué es el Valor Actual Neto (VAN) y cómo se interpreta?",
        "ground_truth": (
            "El Valor Actual Neto (Net Present Value, NPV) es una métrica de "
            "valoración de proyectos que calcula el valor presente de todos los "
            "flujos de caja futuros menos la inversión inicial. Se interpreta así: "
            "si VAN > 0, el proyecto es rentable y se debe aceptar; si VAN < 0, "
            "el proyecto destruye valor y se debe rechazar; si VAN = 0, el proyecto "
            "está en equilibrio. Es una de las métricas más importantes en "
            "Corporate Finance según CFA Level I."
        ),
        "topic_hint": "Corporate Finance"
    },
    {
        "question": "Explica el concepto de Duration en bonos y su utilidad",
        "ground_truth": (
            "La Duration (Duración de Macaulay) es una medida de sensibilidad del "
            "precio de un bono ante cambios en las tasas de interés. Representa el "
            "tiempo promedio ponderado hasta recibir los flujos de caja del bono. "
            "La Duration Modificada mide el cambio porcentual aproximado en el precio "
            "del bono por cada 1% de cambio en el yield. Es fundamental en gestión "
            "de riesgo de renta fija para inmunización de portafolios y estrategias "
            "de cobertura (hedging)."
        ),
        "topic_hint": "Fixed Income"
    },
    {
        "question": "¿Qué es el CAPM y para qué se utiliza?",
        "ground_truth": (
            "El Capital Asset Pricing Model (CAPM) es un modelo de equilibrio que "
            "establece la relación entre el riesgo sistemático (beta) y el retorno "
            "esperado de un activo. La fórmula es: E(Ri) = Rf + βi * [E(Rm) - Rf], "
            "donde Rf es la tasa libre de riesgo, βi es el beta del activo, y "
            "E(Rm) es el retorno esperado del mercado. Se utiliza para calcular el "
            "costo de capital accionario (Ke) en valoración de empresas y para "
            "evaluar si un activo está sobrevalorado o infravalorado según su "
            "riesgo sistemático."
        ),
        "topic_hint": "Portfolio Management"
    },
    {
        "question": "¿Cómo funciona una opción call europea y cuándo se ejerce?",
        "ground_truth": (
            "Una opción call europea otorga al comprador el derecho (no la obligación) "
            "de comprar un activo subyacente a un precio de ejercicio (strike) "
            "predeterminado en la fecha de vencimiento. A diferencia de las opciones "
            "americanas, solo pueden ejercerse al vencimiento. Se ejerce cuando el "
            "precio spot del subyacente (S) es mayor que el strike (K), es decir, "
            "cuando está 'in the money'. El payoff al vencimiento es max(S - K, 0). "
            "Se valora usando el modelo Black-Scholes para opciones europeas."
        ),
        "topic_hint": "Derivatives"
    },
    {
        "question": "¿Qué mide el Sharpe Ratio y cómo se interpreta?",
        "ground_truth": (
            "El Sharpe Ratio mide el exceso de retorno por unidad de riesgo total. "
            "Se calcula como: (Rp - Rf) / σp, donde Rp es el retorno del portafolio, "
            "Rf es la tasa libre de riesgo, y σp es la desviación estándar del "
            "portafolio. Un Sharpe Ratio mayor indica mejor desempeño ajustado por "
            "riesgo. Valores típicos: > 1 es bueno, > 2 es muy bueno, > 3 es "
            "excelente. Se usa para comparar diferentes portafolios o estrategias "
            "de inversión en términos de eficiencia riesgo-retorno."
        ),
        "topic_hint": "Portfolio Management"
    }
]

print(f"✅ Golden Dataset cargado: {len(GOLDEN_DATASET)} preguntas CFA")


# ========================================
# FUNCIONES DE EVALUACIÓN
# ========================================

def ejecutar_consultas_rag(dataset: List[Dict]) -> pd.DataFrame:
    """
    Ejecuta las preguntas del dataset contra el sistema RAG.

    Args:
        dataset: Lista de diccionarios con preguntas y ground truth

    Returns:
        DataFrame con preguntas, respuestas, contextos y ground truth
    """
    print_header("Ejecutando Consultas contra el Sistema RAG")

    resultados = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"📝 Pregunta {i}/{len(dataset)}: {question[:50]}...")

        try:
            # Buscar contextos relevantes (simula lo que hace buscar_documentacion_financiera)
            from rag.financial_rag_elasticsearch import extraer_filtros_de_consulta, enriquecer_query_bilingue

            # Extraer filtros
            filtros = extraer_filtros_de_consulta(question)

            # Enriquecer query
            query_enriquecida = enriquecer_query_bilingue(question)

            # Buscar documentos
            docs = rag_system.search_documents(
                query_enriquecida,
                k=3,
                filter_dict=filtros if filtros else None
            )

            if not docs:
                print(f"   ⚠️ No se encontraron documentos para esta pregunta")
                contexts = ["No se encontró contexto"]
                answer = "No se pudo generar respuesta (sin contexto)"
            else:
                # Extraer contextos
                contexts = [doc.page_content for doc in docs]

                # Generar respuesta usando LLM + contextos (simula Agente_Sintesis_RAG)
                llm = get_llm()
                context_str = "\n\n".join([f"Fragmento {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

                prompt = f"""Contexto del material financiero:
{context_str}

Pregunta del usuario:
{question}

Responde en español basándote SOLO en el contexto proporcionado. Sé conciso y profesional."""

                respuesta_llm = llm.invoke(prompt)
                answer = respuesta_llm.content if hasattr(respuesta_llm, 'content') else str(respuesta_llm)

            # Guardar resultado
            resultados.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })

            print(f"   ✅ Respuesta generada ({len(answer)} caracteres)")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            resultados.append({
                "question": question,
                "answer": f"Error: {e}",
                "contexts": ["Error en búsqueda"],
                "ground_truth": ground_truth
            })

    print(f"\n✅ {len(resultados)} consultas completadas\n")

    return pd.DataFrame(resultados)


def calcular_metricas_ragas(df_resultados: pd.DataFrame) -> Dict:
    """
    Calcula métricas de Ragas sobre los resultados.

    Args:
        df_resultados: DataFrame con question, answer, contexts, ground_truth

    Returns:
        Diccionario con métricas calculadas
    """
    print_header("Calculando Métricas Ragas")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision
        )
        from datasets import Dataset

        # Convertir a formato Dataset de HuggingFace (requerido por Ragas)
        dataset_dict = {
            "question": df_resultados["question"].tolist(),
            "answer": df_resultados["answer"].tolist(),
            "contexts": df_resultados["contexts"].tolist(),
            "ground_truth": df_resultados["ground_truth"].tolist()
        }

        dataset = Dataset.from_dict(dataset_dict)

        print("📊 Métricas a calcular:")
        print("   1. Faithfulness (Fidelidad): ¿Respuesta basada en contexto?")
        print("   2. Answer Relevancy (Relevancia): ¿Respuesta responde la pregunta?")
        print("   3. Context Precision (Precisión): ¿Contexto relevante?\n")

        print("⏳ Calculando... (esto puede tomar 1-2 minutos)\n")

        # Evaluar
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision
            ]
        )

        # Convertir a diccionario
        metricas = {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"]
        }

        print("✅ Métricas calculadas exitosamente\n")

        return metricas

    except Exception as e:
        print(f"❌ Error calculando métricas Ragas: {e}")
        import traceback
        traceback.print_exc()
        return {}


def mostrar_resultados(metricas: Dict, df_resultados: pd.DataFrame):
    """Muestra los resultados de la evaluación de forma legible."""
    print_header("📊 RESULTADOS DE LA EVALUACIÓN")

    if not metricas:
        print("❌ No se pudieron calcular métricas")
        return

    print("🎯 MÉTRICAS GLOBALES (0-1, mayor es mejor):\n")

    for nombre, valor in metricas.items():
        # Interpretación
        if valor >= 0.8:
            status = "✅ Excelente"
        elif valor >= 0.6:
            status = "👍 Bueno"
        elif valor >= 0.4:
            status = "⚠️ Regular"
        else:
            status = "❌ Requiere mejora"

        print(f"  {nombre.upper()}: {valor:.3f} {status}")

    print("\n" + "=" * 60)
    print("\n📝 DETALLE POR PREGUNTA:\n")

    for i, row in df_resultados.iterrows():
        print(f"Pregunta {i+1}: {row['question'][:60]}...")
        print(f"  Respuesta generada: {row['answer'][:100]}...")
        print(f"  Contextos recuperados: {len(row['contexts'])} fragmentos")
        print()

    print("=" * 60)

    # Guardar a CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rag_evaluation_{timestamp}.csv"
    df_resultados.to_csv(output_file, index=False)
    print(f"\n💾 Resultados guardados en: {output_file}")


# ========================================
# FUNCIÓN PRINCIPAL
# ========================================

def main():
    """Función principal."""
    print("\n" + "🧪" * 30)
    print("  EVALUACIÓN AUTOMÁTICA RAG - Sistema CFA")
    print("  Framework: Ragas + OpenAI")
    print("🧪" * 30)

    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Dataset: {len(GOLDEN_DATASET)} preguntas CFA Level I")
    print(f"🧠 LLM: Claude/OpenAI (fallback)")
    print(f"🔍 RAG: Elasticsearch + OpenAI Embeddings\n")

    # Verificar API key
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY no encontrada")
        print("   Configúrala en .env o Streamlit Secrets")
        sys.exit(1)

    # Verificar conexión a RAG
    health = rag_system.get_health_status()
    if health["connection_status"] != "connected":
        print("❌ ERROR: Sistema RAG no conectado")
        print(f"   Estado: {health}")
        sys.exit(1)

    print("✅ Sistema RAG conectado y listo\n")

    try:
        # Paso 1: Ejecutar consultas
        df_resultados = ejecutar_consultas_rag(GOLDEN_DATASET)

        # Paso 2: Calcular métricas
        metricas = calcular_metricas_ragas(df_resultados)

        # Paso 3: Mostrar resultados
        mostrar_resultados(metricas, df_resultados)

        print("\n✅ EVALUACIÓN COMPLETADA EXITOSAMENTE\n")

    except KeyboardInterrupt:
        print("\n\n❌ Evaluación cancelada por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
