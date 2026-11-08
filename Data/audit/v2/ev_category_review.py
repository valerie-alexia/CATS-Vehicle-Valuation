from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE
    / "ev_specialist_experiment"
    / "ev_specialist_candidate.joblib"
)

COLUMNS = [
    "Merk",
    "Model",
    "Varian",
    "JenisMesin",
    "JenisTransmisi",
]


@lru_cache(maxsize=1)
def get_known_categories():
    bundle = joblib.load(MODEL_PATH)

    encoder = bundle["branch"]["preprocessor"] \
        .named_transformers_["category"]

    return {
        column: {
            str(value).strip().upper()
            for value in values
            if not (
                value is None
                or (
                    isinstance(value, (float, np.floating))
                    and np.isnan(value)
                )
            )
        }
        for column, values in zip(
            COLUMNS,
            encoder.categories_
        )
    }


def review_ev_categories(payload):
    categories = get_known_categories()

    unknown = []

    for column in COLUMNS:
        value = payload.get(column)

        if value is None:
            continue

        normalized = str(value).strip().upper()

        if normalized not in categories[column]:
            unknown.append(column)

    return {
        "required": bool(unknown),
        "unknown_columns": unknown,
        "reason": (
            "Terdapat kategori input yang tidak dikenal "
            "oleh encoder model EV."
            if unknown
            else None
        ),
        "note": (
            "Pemeriksaan ini hanya memvalidasi kategori "
            "secara individual, bukan kombinasi "
            "merek-model-varian atau distribusi numerik."
        ),
    }