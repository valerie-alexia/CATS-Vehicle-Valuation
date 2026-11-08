"""
CATS Demo API

- Endpoint produksi yang sudah tersedia
- Prediksi + analisis pasar eksperimen
- Prediksi EV eksperimen

Tidak menulis database atau mengganti active.json.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from fastapi import FastAPI
from fastapi.routing import APIRoute

from src.Retraining.api import app as original_app
from cats_market_api_experiment import app as experiment_app

app = FastAPI(
    title="CATS Vehicle Valuation Demo",
    version="0.1.0",
    description=(
        "Demo prediksi harga mobil dan motor, "
        "analisis marketplace, serta EV eksperimen. "
        "Tidak melakukan penulisan database CATS."
    ),
)

# Salin route dari dua aplikasi ke satu entrypoint.
# Pertahankan implementasi endpoint yang sudah diuji.
registered = set()

for source in (original_app, experiment_app):
    for route in source.routes:
        if not isinstance(route, APIRoute):
            continue

        for method in route.methods:
            key = (method, route.path)

            if key in registered:
                raise RuntimeError(
                    f"Duplikasi endpoint: {key}"
                )

            registered.add(key)

        app.router.routes.append(route)


if __name__ == "__main__":
    print("=== CATS DEMO API ===")

    for method, path in sorted(registered):
        print(method, path)

    print("\nTotal pasangan method/path:", len(registered))
    print("Tidak ada database atau model aktif diubah.")