"""
CATS - EV Inference Experiment

Menggunakan model EV khusus yang sudah disimpan.
Tidak mengubah model aktif, service.py, atau API produksi.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

from src.Training.pipeline import normalize_features, predict_branch

MODEL_PATH = (
    BASE
    / "ev_specialist_experiment"
    / "ev_specialist_candidate.joblib"
)


def predict_ev(payload):
    bundle = joblib.load(MODEL_PATH)

    if bundle.get("experiment_only") is not True:
        raise ValueError("Bundle bukan model EV eksperimen")

    if bundle.get("vehicle") != "mobil_ev":
        raise ValueError("Jenis model tidak sesuai")

    fuel = str(payload.get("JenisMesin", "")).strip().upper()

    if fuel not in {"ELECTRIC", "LISTRIK"}:
        raise ValueError(
            "Adaptor ini hanya menerima kendaraan listrik murni"
        )

    required = [
        "Merk",
        "Model",
        "TahunUnit",
        "Kilometer",
        "JenisTransmisi",
    ]

    for field in required:
        if payload.get(field) is None:
            raise ValueError(f"{field} wajib diisi untuk EV")

    if payload.get("KapasitasMesin") not in (None, 0):
        raise ValueError(
            "KapasitasMesin EV harus kosong atau 0"
        )

    row = {
        **payload,
        "JenisMesin": "ELECTRIC",
        "KapasitasMesin": None,
    }

    frame = normalize_features(
        pd.DataFrame([row])
    )

    predicted = predict_branch(
        bundle["branch"],
        frame,
    )

    price = float(np.asarray(predicted)[0])

    if not np.isfinite(price) or price <= 0:
        raise ValueError("Prediksi EV tidak valid")

    sd = float(bundle["band_halfwidth_sd"])

    return {
        "prediksi_harga": round(price),
        "rentang": {
            "batas_bawah": round(max(0, price * (1 - sd))),
            "batas_atas": round(price * (1 + sd)),
            "strategi": "ev_relative_residual",
        },
        "model_digunakan": {
            "rute": "mobil-ev-specialist-experiment",
            "algoritma": bundle["algorithm"],
        },
        "experiment_only": True,
        "catatan": (
            "Prediksi EV eksperimental. "
            "Belum digunakan pada API produksi CATS."
        ),
    }


