# tools/financial_tools.py
"""
Herramientas financieras con cálculos deterministas.
Actualizado con logging estructurado y manejo robusto de errores.
"""

import numpy as np
import numpy_financial as npf
from scipy.stats import norm
from langchain_core.tools import tool
from typing import List

# Importar schemas
from .schemas import (
    BonoInput, VANInput, OpcionCallInput, WACCInput,
    CAPMInput, SharpeRatioInput, GordonGrowthInput
)

# Importar logger
try:
    from utils.logger import get_logger
    logger = get_logger('tools')
except ImportError:
    import logging
    logger = logging.getLogger('tools')

# ========================================
# HERRAMIENTAS FINANCIERAS
# ========================================

@tool("calcular_valor_bono", args_schema=BonoInput)
def _calcular_valor_presente_bono(
    valor_nominal: float,
    tasa_cupon_anual: float,
    tasa_descuento_anual: float,
    num_anos: int,
    frecuencia_cupon: int
) -> dict:
    """Calcula el valor presente de un bono."""
    logger.info(f"🔧 Calculando valor de bono: nominal={valor_nominal}, años={num_anos}")
    
    try:
        tasa_cupon_periodo = (tasa_cupon_anual / 100) / frecuencia_cupon
        tasa_descuento_periodo = (tasa_descuento_anual / 100) / frecuencia_cupon
        num_periodos_totales = num_anos * frecuencia_cupon
        pago_cupon = valor_nominal * tasa_cupon_periodo

        # Cálculo PV cupones
        if tasa_descuento_periodo == 0:
            pv_cupones = pago_cupon * num_periodos_totales if num_periodos_totales > 0 else 0
        elif num_periodos_totales > 0:
            pv_cupones = pago_cupon * (1 - (1 + tasa_descuento_periodo)**-num_periodos_totales) / tasa_descuento_periodo
        else:
            pv_cupones = 0

        # Cálculo PV valor nominal
        pv_nominal = valor_nominal / (1 + tasa_descuento_periodo)**num_periodos_totales if num_periodos_totales > 0 else valor_nominal

        valor_bono = pv_cupones + pv_nominal
        
        logger.info(f"✅ Valor bono calculado: ${valor_bono:,.2f}")
        return {"valor_presente_bono": round(valor_bono, 2)}
        
    except OverflowError:
        logger.error("❌ Overflow en cálculo de bono")
        return {"error": "Error de cálculo: Overflow. Verifica tasas muy grandes o periodos largos."}
    except Exception as e:
        logger.error(f"❌ Error en cálculo de bono: {type(e).__name__} - {e}")
        return {"error": f"Error calculando valor del bono: {type(e).__name__}"}


@tool("calcular_van", args_schema=VANInput)
def _calcular_van(tasa_descuento: float, inversion_inicial: float, flujos_caja: List[float]) -> dict:
    """Calcula el Valor Actual Neto (VAN) de un proyecto."""
    logger.info(f"🔧 Calculando VAN: inversión={inversion_inicial}, flujos={len(flujos_caja)}")
    
    try:
        tasa = tasa_descuento / 100
        
        if not all(isinstance(fc, (int, float)) for fc in flujos_caja):
            logger.error("❌ Flujos de caja inválidos")
            return {"error": "Los flujos de caja deben ser una lista de números."}
        
        flujos_totales = [-abs(inversion_inicial)] + flujos_caja
        van = npf.npv(tasa, flujos_totales)
        
        logger.info(f"✅ VAN calculado: ${van:,.2f}")
        return {"van": round(van, 2), "interpretacion": "Si VAN > 0, el proyecto es rentable."}
        
    except Exception as e:
        logger.error(f"❌ Error en cálculo de VAN: {type(e).__name__} - {e}")
        return {"error": f"Error calculando VAN: {type(e).__name__}"}


@tool("calcular_opcion_call", args_schema=OpcionCallInput)
def _calcular_opcion_call(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Calcula el valor de una Opción Call Europea usando Black-Scholes."""
    logger.info(f"🔧 Calculando opción call: S={S}, K={K}, T={T}")
    
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            logger.error("❌ Parámetros inválidos en opción call")
            return {"error": "Tiempo (T), volatilidad (sigma), precio actual (S) y precio ejercicio (K) deben ser positivos."}
        
        r_dec = r / 100
        sigma_dec = sigma / 100
        
        if sigma_dec == 0:
            call_price = max(S - K * np.exp(-r_dec * T), 0)
            logger.info(f"✅ Opción call (σ=0): ${call_price:.4f}")
            return {"valor_opcion_call": round(call_price, 4)}
        
        denominator = sigma_dec * np.sqrt(T)
        d1 = (np.log(S / K) + (r_dec + 0.5 * sigma_dec**2) * T) / denominator
        d2 = d1 - denominator
        
        call_price = (S * norm.cdf(d1) - K * np.exp(-r_dec * T) * norm.cdf(d2))
        call_price = max(call_price, 0)
        
        logger.info(f"✅ Opción call calculada: ${call_price:.4f}")
        return {"valor_opcion_call": round(call_price, 4)}
        
    except OverflowError:
        logger.error("❌ Overflow en cálculo de opción")
        return {"error": "Error de cálculo: Overflow. Verifica inputs muy grandes/pequeños."}
    except ValueError as ve:
        logger.error(f"❌ Error matemático en opción: {ve}")
        return {"error": f"Error matemático: {ve}. Verifica los inputs (S, K > 0)."}
    except Exception as e:
        logger.error(f"❌ Error en cálculo de opción: {type(e).__name__} - {e}")
        return {"error": f"Error calculando Opción Call: {type(e).__name__}"}


@tool("calcular_wacc", args_schema=WACCInput)
def _calcular_wacc(
    tasa_impuestos: float,
    costo_deuda: float,
    costo_equity: float,
    valor_mercado_deuda: float,
    valor_mercado_equity: float
) -> dict:
    """Calcula el Costo Promedio Ponderado de Capital (WACC)."""
    logger.info(f"🔧 Calculando WACC: D={valor_mercado_deuda}, E={valor_mercado_equity}")
    
    try:
        t_c = tasa_impuestos / 100
        k_d = costo_deuda / 100
        k_e = costo_equity / 100
        D = valor_mercado_deuda
        E = valor_mercado_equity
        
        if D < 0 or E < 0:
            logger.error("❌ Valores de mercado negativos")
            return {"error": "Valores de mercado de deuda y equity no pueden ser negativos."}
        
        V = D + E
        if V <= 0:
            if D==0 and E==0:
                logger.warning("⚠️ WACC = 0 (sin capital)")
                return {"wacc_porcentaje": 0.0, "nota": "WACC es 0 ya que no hay capital."}
            logger.error("❌ Valor total de mercado inválido")
            return {"error": "El valor total de mercado (Deuda + Equity) debe ser positivo."}
        
        weight_e = E / V
        weight_d = D / V
        
        wacc = weight_e * k_e + weight_d * k_d * (1 - t_c)
        
        logger.info(f"✅ WACC calculado: {wacc*100:.4f}%")
        return {"wacc_porcentaje": round(wacc * 100, 4)}
        
    except Exception as e:
        logger.error(f"❌ Error en cálculo de WACC: {type(e).__name__} - {e}")
        return {"error": f"Error calculando WACC: {type(e).__name__}"}


@tool("calcular_capm", args_schema=CAPMInput)
def _calcular_capm(tasa_libre_riesgo: float, beta: float, retorno_mercado: float) -> dict:
    """Calcula el Costo del Equity (Ke) usando el Capital Asset Pricing Model (CAPM)."""
    logger.info(f"🔧 Calculando CAPM: rf={tasa_libre_riesgo}%, β={beta}")
    
    try:
        rf = tasa_libre_riesgo / 100
        rm = retorno_mercado / 100
        k_e = rf + beta * (rm - rf)
        
        logger.info(f"✅ Ke (CAPM) calculado: {k_e*100:.4f}%")
        return {"costo_equity_porcentaje": round(k_e * 100, 4)}
        
    except Exception as e:
        logger.error(f"❌ Error en cálculo de CAPM: {type(e).__name__} - {e}")
        return {"error": f"Error calculando CAPM: {type(e).__name__}"}


@tool("calcular_sharpe_ratio", args_schema=SharpeRatioInput)
def _calcular_sharpe_ratio(retorno_portafolio: float, tasa_libre_riesgo: float, std_dev_portafolio: float) -> dict:
    """Calcula el Ratio de Sharpe para medir el retorno ajustado al riesgo."""
    logger.info(f"🔧 Calculando Sharpe Ratio: rp={retorno_portafolio}%, σ={std_dev_portafolio}%")
    
    try:
        r_p = retorno_portafolio / 100
        r_f = tasa_libre_riesgo / 100
        std_p = std_dev_portafolio / 100
        
        if std_p <= 0:
            logger.error("❌ Desviación estándar inválida")
            return {"error": "La desviación estándar del portafolio debe ser mayor que cero."}
        
        sharpe = (r_p - r_f) / std_p
        
        logger.info(f"✅ Sharpe Ratio calculado: {sharpe:.4f}")
        return {"sharpe_ratio": round(sharpe, 4)}
        
    except Exception as e:
        logger.error(f"❌ Error en cálculo de Sharpe: {type(e).__name__} - {e}")
        return {"error": f"Error calculando Sharpe Ratio: {type(e).__name__}"}


@tool("calcular_gordon_growth", args_schema=GordonGrowthInput)
def _calcular_gordon_growth(
    dividendo_prox_periodo: float,
    tasa_descuento_equity: float,
    tasa_crecimiento_dividendos: float
) -> dict:
    """Calcula el valor de una acción usando el Modelo de Crecimiento de Gordon (DDM)."""
    logger.info(f"🔧 Calculando Gordon Growth: D1={dividendo_prox_periodo}, Ke={tasa_descuento_equity}%")
    
    try:
        D1 = dividendo_prox_periodo
        Ke = tasa_descuento_equity / 100
        g = tasa_crecimiento_dividendos / 100
        
        if D1 <= 0:
            logger.error("❌ Dividendo inválido")
            return {"error": "El dividendo del próximo periodo (D1) debe ser positivo."}
        
        if Ke <= g:
            logger.error("❌ Ke <= g (inválido para Gordon)")
            return {"error": "La tasa de descuento (Ke) debe ser estrictamente mayor que la tasa de crecimiento (g)."}
        
        denominator = Ke - g
        if denominator == 0:
            logger.error("❌ División por cero en Gordon")
            return {"error": "División por cero evitada (Ke - g es cero). Ke debe ser > g."}
        
        valor_accion = D1 / denominator
        
        if valor_accion < 0:
            logger.error("❌ Valor negativo inesperado")
            return {"error": "El cálculo resultó en un valor negativo inesperado."}
        
        logger.info(f"✅ Valor acción calculado: ${valor_accion:.2f}")
        return {"valor_intrinseco_accion": round(valor_accion, 2)}
        
    except Exception as e:
        logger.error(f"❌ Error en cálculo de Gordon: {type(e).__name__} - {e}")
        return {"error": f"Error calculando Gordon Growth: {type(e).__name__}"}


# ========================================
# LISTA EXPORTABLE
# ========================================

financial_tool_list = [
    _calcular_valor_presente_bono,
    _calcular_van,
    _calcular_opcion_call,
    _calcular_wacc,
    _calcular_capm,
    _calcular_sharpe_ratio,
    _calcular_gordon_growth,
]

logger.info(f"✅ Módulo financial_tools cargado ({len(financial_tool_list)} herramientas)")