"""
CATS - Marketplace Price Statistics (Experimental)

Menghitung:
- Jumlah iklan relevan
- Mean
- Median
- SD sampel
- Rentang +/-1, +/-2, +/-3 SD
- Posisi prediksi AI terhadap pasar

Read-only.
Tidak mengubah model produksi maupun evidence.py.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.Retraining.evidence import get_index
from src.Training.pipeline import normalize_features


def market_statistics(vehicle, payload, predicted_price=None):
    index = get_index()
    data = index.frames[vehicle].copy()

    q = normalize_features(
        pd.DataFrame([payload])
    ).iloc[0]

    # Filter fitur kategorikal yang harus sama.
    required = [
        "Merk",
        "Model",
        "JenisMesin",
        "JenisTransmisi",
    ]

    for col in required:
        data = data.loc[data[col].eq(q[col])]

    # Varian harus sama jika tersedia.
    if q["Varian"] not in {"UNKNOWN", "LAINNYA"}:
        data = data.loc[data["Varian"].eq(q["Varian"])]

    # Tahun +/- 1.
    if pd.notna(q["TahunUnit"]):
        data = data.loc[
            data["TahunUnit"]
            .sub(q["TahunUnit"])
            .abs()
            .le(1)
        ]

    # Kilometer +/- 20.000.
    if pd.notna(q["Kilometer"]):
        data = data.loc[
            data["Kilometer"]
            .sub(q["Kilometer"])
            .abs()
            .le(20000)
        ]

    # EV tidak memiliki kapasitas mesin.
    if q["JenisMesin"] != "ELECTRIC":
        if pd.notna(q["KapasitasMesin"]):
            data = data.loc[
                data["KapasitasMesin"]
                .sub(q["KapasitasMesin"])
                .abs()
                .le(50)
            ]

    data = data.loc[
        data["url_kind"].eq("individual_listing")
    ]

    prices = pd.to_numeric(
        data["harga"], errors="coerce"
    )

    prices = prices[
        prices.notna() & prices.gt(0)
    ]

    result = {
        "status": "insufficient_data",
        "jumlah_iklan": int(len(prices)),
        "harga_minimum": None,
        "harga_maksimum": None,
        "mean": None,
        "median": None,
        "sd": None,
        "rentang_sd": None,
        "posisi_prediksi_sd": None,
    }

    # Batas eksploratif, bukan aturan final.
    if len(prices) < 10:
        return result

    mean = float(prices.mean())
    median = float(prices.median())
    sd = float(prices.std(ddof=1))

    bands = {}

    for multiplier in [1, 2, 3]:
        bands[str(multiplier)] = {
            "batas_bawah": round(
                max(0, mean - multiplier * sd)
            ),
            "batas_atas": round(
                mean + multiplier * sd
            ),
        }

    position = None

    if predicted_price is not None and sd > 0:
        position = float(
            (predicted_price - mean) / sd
        )

    result.update({
        "status": "available",
        "jumlah_iklan": int(len(prices)),
        "harga_minimum": round(float(prices.min())),
        "harga_maksimum": round(float(prices.max())),
        "mean": round(mean),
        "median": round(median),
        "sd": round(sd),
        "rentang_sd": bands,
        "posisi_prediksi_sd": (
            round(position, 3)
            if position is not None
            else None
        ),
    })

    return result


