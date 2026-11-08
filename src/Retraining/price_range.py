"""Modul 5 - Penentuan Rentang Harga (Murah / Wajar / Mahal).

`halfwidth` berasal dari bundle model (hasil VehiclePricePredictor.pipeline.calibrate_sd):
  - relative_residual : SD residual relatif -> batas = prediksi * (1 +/- halfwidth)
  - residual_std / global_std : SD absolut dalam Rp -> batas = prediksi +/- halfwidth
Kategori hanya bisa dihitung untuk HARGA PEMBANDING (asking price), bukan untuk prediksi.
"""
from __future__ import annotations
import math


def compute_range(prediction: float, halfwidth: float,
                  strategy: str = "relative_residual") -> dict:
    if strategy not in {'relative_residual', 'residual_std', 'global_std'}:
        raise ValueError('Strategi rentang tidak dikenal')
    if not all(math.isfinite(v) and v >= 0 for v in [prediction, halfwidth]):
        raise ValueError('Prediksi dan halfwidth harus nonnegatif dan finite')
    delta = prediction * halfwidth if strategy == "relative_residual" else halfwidth
    return {
        "batas_bawah": round(max(prediction - delta, 0.0)),
        "batas_atas": round(prediction + delta),
        "strategi": strategy,
    }


def categorize(price: float, low: float, high: float) -> str:
    if price < low:
        return "Murah"
    if price > high:
        return "Mahal"
    return "Wajar"
