from pathlib import Path
from functools import lru_cache
import json

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "ev_specialist_experiment"

COMBINATIONS_PATH = OUTPUT / "ev_training_combinations.json"
REPORT_PATH = OUTPUT / "ev_experiment_report.json"


def normalize(value):
    return str(value or "").strip().upper()


@lru_cache(maxsize=1)
def load_training_combinations():
    coverage = json.loads(
        COMBINATIONS_PATH.read_text(encoding="utf-8")
    )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    counts = report["data_counts"]

    expected = (
        counts["train"] + counts["validation"]
    )

    actual = coverage["training_rows"]

    if expected != actual:
        raise ValueError(
            f"Jumlah training EV tidak cocok: "
            f"report={expected}, reconstruction={actual}"
        )

    if coverage["seed"] != report["seed"]:
        raise ValueError("Seed eksperimen EV tidak cocok")

    combinations = {
        (
            normalize(row["Merk"]),
            normalize(row["Model"]),
            normalize(row["Varian"]),
        )
        for row in coverage["combinations"]
    }

    return {
        "training_rows": actual,
        "combination_count": len(combinations),
        "combinations": combinations,
    }


def review_ev_combination(payload):
    reference = load_training_combinations()

    key = (
        normalize(payload.get("Merk")),
        normalize(payload.get("Model")),
        normalize(payload.get("Varian")),
    )

    # Jika varian tidak tersedia, evaluasi merek-model.
    if not key[2]:
        known = any(
            row[:2] == key[:2]
            for row in reference["combinations"]
        )
        level = "merk_model"
    else:
        known = key in reference["combinations"]
        level = "merk_model_varian"

    return {
        "required": not known,
        "known_combination": known,
        "match_level": level,
        "reason": (
            None if known else
            "Kombinasi kendaraan tidak ditemukan "
            "dalam data final training EV yang direkonstruksi."
        ),
        "reference_training_rows": reference["training_rows"],
    }