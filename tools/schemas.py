# tools/schemas.py
from pydantic import BaseModel, Field
from typing import List # Usar List en lugar de list para compatibilidad con Pydantic < 2.7

# --- Schemas Pydantic ---

class BonoInput(BaseModel):
    """Schema para calcular el valor presente de un bono."""
    valor_nominal: float = Field(description="Valor nominal (facial) del bono", gt=0)
    tasa_cupon_anual: float = Field(description="Tasa de interés del cupón **ANUAL** en % (ej. 6 para 6%)", ge=0, le=100)
    tasa_descuento_anual: float = Field(description="Tasa de descuento de mercado **ANUAL** (YTM) en % (ej. 5 para 5%)", ge=0, le=100)
    num_anos: int = Field(description="Número total de **AÑOS** hasta el vencimiento (ej. 5 para 5 años)", gt=0)
    frecuencia_cupon: int = Field(description="Pagos de cupón por año (ej. 1 para anual, 2 para semestral)", gt=0)

class VANInput(BaseModel):
    """Schema para calcular el Valor Actual Neto (VAN) de un proyecto."""
    tasa_descuento: float = Field(description="Tasa de descuento (WACC, TMAR) en %", ge=0, le=100)
    inversion_inicial: float = Field(description="Desembolso inicial como un número **POSITIVO** (ej. 100000)", gt=0)
    flujos_caja: List[float] = Field(description="Lista de flujos de caja futuros (ej. [25000, 30000, 35000])")

class OpcionCallInput(BaseModel):
    """Schema para calcular el valor de una Opción Call Europea usando Black-Scholes."""
    S: float = Field(description="Precio actual del activo subyacente (Stock price)", gt=0)
    K: float = Field(description="Precio de ejercicio (Strike price)", gt=0)
    T: float = Field(description="Tiempo hasta el vencimiento en años (ej. 0.5 para 6 meses)", gt=0)
    r: float = Field(description="Tasa de interés libre de riesgo anual en %", ge=0, le=100)
    sigma: float = Field(description="Volatilidad anual del activo en % (ej. 20 para 20%)", gt=0, le=200)

class WACCInput(BaseModel):
    """Schema para calcular el Costo Promedio Ponderado de Capital (WACC)."""
    tasa_impuestos: float = Field(description="Tasa de impuestos corporativos en %", ge=0, le=100)
    costo_deuda: float = Field(description="Costo de la deuda (tasa de interés) en %", ge=0, le=100)
    costo_equity: float = Field(description="Costo del equity (capital propio) en %", ge=0, le=100)
    valor_mercado_deuda: float = Field(description="Valor de mercado total de la deuda en dólares", gt=0)
    valor_mercado_equity: float = Field(description="Valor de mercado total del equity (capital) en dólares", gt=0)

class CAPMInput(BaseModel):
    """Schema para calcular el Costo del Equity usando CAPM."""
    tasa_libre_riesgo: float = Field(description="Tasa libre de riesgo (ej. bonos del tesoro) en %", ge=0, le=100)
    beta: float = Field(description="Beta del activo (medida de volatilidad)", gt=0)
    retorno_mercado: float = Field(description="Retorno esperado del mercado (ej. S&P 500) en %", ge=0, le=100)

class SharpeRatioInput(BaseModel):
    """Schema para calcular el Ratio de Sharpe de un portafolio."""
    retorno_portafolio: float = Field(description="Retorno esperado del portafolio en %", ge=0, le=100)
    tasa_libre_riesgo: float = Field(description="Tasa libre de riesgo en %", ge=0, le=100)
    std_dev_portafolio: float = Field(description="Desviación estándar (volatilidad) del portafolio en %", gt=0, le=200)

class GordonGrowthInput(BaseModel):
    """Schema para calcular el valor de una acción usando el Modelo de Crecimiento de Gordon (DDM)."""
    dividendo_prox_periodo: float = Field(description="Dividendo esperado en el próximo periodo (D1) en dólares", gt=0)
    tasa_descuento_equity: float = Field(description="Tasa de descuento o costo del equity (Ke) en %", gt=0, le=100)
    tasa_crecimiento_dividendos: float = Field(description="Tasa de crecimiento constante de los dividendos (g) en %", ge=0, le=100)

print("✅ Módulo schemas cargado.")