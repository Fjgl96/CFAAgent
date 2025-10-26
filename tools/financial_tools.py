# tools/financial_tools.py
import numpy as np
import numpy_financial as npf
from scipy.stats import norm
from langchain_core.tools import tool
from typing import List # Necesario para type hinting en las funciones

# Importar los schemas definidos en el mismo directorio
from .schemas import (
    BonoInput, VANInput, OpcionCallInput, WACCInput,
    CAPMInput, SharpeRatioInput, GordonGrowthInput
)

# --- Funciones Tool ---
@tool("calcular_valor_bono", args_schema=BonoInput)
def _calcular_valor_presente_bono(valor_nominal: float, tasa_cupon_anual: float, tasa_descuento_anual: float, num_anos: int, frecuencia_cupon: int) -> dict:
    """Calcula el valor presente de un bono."""
    try:
        tasa_cupon_periodo = (tasa_cupon_anual / 100) / frecuencia_cupon
        tasa_descuento_periodo = (tasa_descuento_anual / 100) / frecuencia_cupon
        num_periodos_totales = num_anos * frecuencia_cupon
        pago_cupon = valor_nominal * tasa_cupon_periodo

        # Asegurarse que tasa_descuento_periodo no sea cero si hay periodos
        if tasa_descuento_periodo == 0:
             if num_periodos_totales == 0: pv_cupones = 0
             else: pv_cupones = pago_cupon * num_periodos_totales # Simple suma si tasa es 0
        elif num_periodos_totales > 0:
             # Cálculo PV cupones (anualidad) - Usar fórmula directa es más estable que npf.npv para anualidades
            pv_cupones = pago_cupon * (1 - (1 + tasa_descuento_periodo)**-num_periodos_totales) / tasa_descuento_periodo
        else: # Si no hay periodos, no hay cupones
             pv_cupones = 0

        # Cálculo PV valor nominal
        # Manejar caso de 0 periodos donde la potencia sería 0
        if num_periodos_totales == 0:
             pv_nominal = valor_nominal # Si vence ahora, vale su nominal
        else:
             pv_nominal = valor_nominal / (1 + tasa_descuento_periodo)**num_periodos_totales

        valor_bono = pv_cupones + pv_nominal

        return {"valor_presente_bono": round(valor_bono, 2)}
    except OverflowError:
         return {"error": "Error de cálculo: Overflow. Verifica tasas muy grandes o periodos largos."}
    except Exception as e:
        print(f"ERROR en _calcular_valor_presente_bono: {type(e).__name__} - {e}")
        return {"error": f"Error calculando valor del bono: {type(e).__name__}"}


@tool("calcular_van", args_schema=VANInput)
def _calcular_van(tasa_descuento: float, inversion_inicial: float, flujos_caja: List[float]) -> dict: # Usar List
    """Calcula el Valor Actual Neto (VAN) de un proyecto."""
    try:
        tasa = tasa_descuento / 100
        # Validar que los flujos sean números (a veces LLMs pasan None o strings si fallan)
        if not all(isinstance(fc, (int, float)) for fc in flujos_caja):
             return {"error": "Los flujos de caja deben ser una lista de números."}
        # Asegurar que la inversión inicial sea negativa
        flujos_totales = [-abs(inversion_inicial)] + flujos_caja
        van = npf.npv(tasa, flujos_totales)
        return {"van": round(van, 2), "interpretacion": "Si VAN > 0, el proyecto es rentable."}
    except Exception as e:
        print(f"ERROR en _calcular_van: {type(e).__name__} - {e}")
        return {"error": f"Error calculando VAN: {type(e).__name__}"}

@tool("calcular_opcion_call", args_schema=OpcionCallInput)
def _calcular_opcion_call(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Calcula el valor de una Opción Call Europea usando Black-Scholes."""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
             return {"error": "Tiempo (T), volatilidad (sigma), precio actual (S) y precio ejercicio (K) deben ser positivos."}
        r_dec = r / 100
        sigma_dec = sigma / 100
        
        # Evitar división por cero o logaritmo de cero
        if sigma_dec == 0: # Si no hay volatilidad
             call_price = max(S - K * np.exp(-r_dec * T), 0) # Valor intrínseco descontado
             return {"valor_opcion_call": round(call_price, 4)}
        
        # Cálculo normal de d1 y d2
        denominator = sigma_dec * np.sqrt(T)
        # Asegurar que S/K sea positivo para np.log
        if S <= 0 or K <= 0: return {"error": "S y K deben ser positivos."} 

        d1 = (np.log(S / K) + (r_dec + 0.5 * sigma_dec**2) * T) / denominator
        d2 = d1 - denominator
        
        call_price = (S * norm.cdf(d1) - K * np.exp(-r_dec * T) * norm.cdf(d2))
        call_price = max(call_price, 0) # Precio no puede ser negativo
        
        return {"valor_opcion_call": round(call_price, 4)}
    except OverflowError:
         return {"error": "Error de cálculo: Overflow. Verifica inputs muy grandes/pequeños."}
    except ValueError as ve: # Capturar errores de dominio matemático (ej. log de negativo)
         print(f"ERROR en _calcular_opcion_call: {type(ve).__name__} - {ve}")
         return {"error": f"Error matemático: {ve}. Verifica los inputs (S, K > 0)."}
    except Exception as e:
        print(f"ERROR en _calcular_opcion_call: {type(e).__name__} - {e}")
        return {"error": f"Error calculando Opción Call: {type(e).__name__}"}

@tool("calcular_wacc", args_schema=WACCInput)
def _calcular_wacc(tasa_impuestos: float, costo_deuda: float, costo_equity: float, valor_mercado_deuda: float, valor_mercado_equity: float) -> dict:
    """Calcula el Costo Promedio Ponderado de Capital (WACC)."""
    try:
        t_c = tasa_impuestos / 100
        k_d = costo_deuda / 100
        k_e = costo_equity / 100
        D = valor_mercado_deuda
        E = valor_mercado_equity
        # Validar que D y E no sean negativos (aunque schema lo hace >0)
        if D < 0 or E < 0: return {"error": "Valores de mercado de deuda y equity no pueden ser negativos."}
        V = D + E
        if V <= 0: 
             # Si ambos son 0, WACC no está definido o es 0 si no hay capital
             if D==0 and E==0: return {"wacc_porcentaje": 0.0, "nota": "WACC es 0 ya que no hay capital."}
             else: return {"error": "El valor total de mercado (Deuda + Equity) debe ser positivo."}
        
        # Calcular pesos
        weight_e = E / V
        weight_d = D / V
        
        wacc = weight_e * k_e + weight_d * k_d * (1 - t_c)
        return {"wacc_porcentaje": round(wacc * 100, 4)}
    except Exception as e:
        print(f"ERROR en _calcular_wacc: {type(e).__name__} - {e}")
        return {"error": f"Error calculando WACC: {type(e).__name__}"}

@tool("calcular_capm", args_schema=CAPMInput)
def _calcular_capm(tasa_libre_riesgo: float, beta: float, retorno_mercado: float) -> dict:
    """Calcula el Costo del Equity (Ke) usando el Capital Asset Pricing Model (CAPM)."""
    try:
        rf = tasa_libre_riesgo / 100
        rm = retorno_mercado / 100
        k_e = rf + beta * (rm - rf)
        return {"costo_equity_porcentaje": round(k_e * 100, 4)}
    except Exception as e:
        print(f"ERROR en _calcular_capm: {type(e).__name__} - {e}")
        return {"error": f"Error calculando CAPM: {type(e).__name__}"}

@tool("calcular_sharpe_ratio", args_schema=SharpeRatioInput)
def _calcular_sharpe_ratio(retorno_portafolio: float, tasa_libre_riesgo: float, std_dev_portafolio: float) -> dict:
    """Calcula el Ratio de Sharpe para medir el retorno ajustado al riesgo."""
    try:
        r_p = retorno_portafolio / 100
        r_f = tasa_libre_riesgo / 100
        std_p = std_dev_portafolio / 100
        if std_p <= 0: 
             return {"error": "La desviación estándar del portafolio debe ser mayor que cero."}
        sharpe = (r_p - r_f) / std_p
        return {"sharpe_ratio": round(sharpe, 4)}
    except Exception as e:
        print(f"ERROR en _calcular_sharpe_ratio: {type(e).__name__} - {e}")
        return {"error": f"Error calculando Sharpe Ratio: {type(e).__name__}"}

@tool("calcular_gordon_growth", args_schema=GordonGrowthInput)
def _calcular_gordon_growth(dividendo_prox_periodo: float, tasa_descuento_equity: float, tasa_crecimiento_dividendos: float) -> dict:
    """Calcula el valor de una acción usando el Modelo de Crecimiento de Gordon (DDM)."""
    try:
        D1 = dividendo_prox_periodo
        Ke = tasa_descuento_equity / 100
        g = tasa_crecimiento_dividendos / 100
        
        if D1 <= 0:
             return {"error": "El dividendo del próximo periodo (D1) debe ser positivo."}
        if Ke <= g:
            return {"error": "La tasa de descuento (Ke) debe ser estrictamente mayor que la tasa de crecimiento (g)."}
        
        denominator = Ke - g
        if denominator == 0: # Evitar división por cero explícita
             return {"error": "División por cero evitada (Ke - g es cero). Ke debe ser > g."}
             
        valor_accion = D1 / denominator
        
        if valor_accion < 0:
             # Esto no debería pasar si Ke > g y D1 > 0, pero por si acaso
             return {"error": "El cálculo resultó en un valor negativo inesperado."}
             
        return {"valor_intrinseco_accion": round(valor_accion, 2)}
    except Exception as e:
        print(f"ERROR en _calcular_gordon_growth: {type(e).__name__} - {e}")
        return {"error": f"Error calculando Gordon Growth: {type(e).__name__}"}

# Lista exportable de todas las herramientas
financial_tool_list = [
    _calcular_valor_presente_bono,
    _calcular_van,
    _calcular_opcion_call,
    _calcular_wacc,
    _calcular_capm,
    _calcular_sharpe_ratio,
    _calcular_gordon_growth,
]

print(f"✅ Módulo financial_tools cargado ({len(financial_tool_list)} herramientas).")