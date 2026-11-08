"""
Eksperimen integrasi API CATS + statistik marketplace.

Tidak mengubah:
- src/Retraining/api.py
- Models/*/active.json
- Database CATS
- Dataset CLEAN/RAW
"""

from pathlib import Path
import logging
import sys
from typing import Annotated

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))

from fastapi import FastAPI, HTTPException
from pydantic import Field

from src.Retraining.api import (
    CatsVehicleIn,
    _predict_cats_one,
)

from market_price_statistics import market_statistics
# from test_market_note_missing_data import market_category
from market_price_category import market_category

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CATS Market Analysis Experiment",
    version="0.1",
)


@app.post("/cats/predict-market")
def predict_market(item: CatsVehicleIn):
    try:
        # Jalur prediksi CATS yang sudah ada.
        result = _predict_cats_one(
            item,
            with_shap=item.with_shap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (FileNotFoundError, OSError):
        raise HTTPException(
            status_code=503,
            detail="Model atau data evidence tidak tersedia",
        ) from None

    normalized = result["input_normalization"]["input_model"]
    vehicle = normalized["JenisAset"]

    try:
        # Statistik berdasarkan seluruh iklan relevan,
        # bukan hanya lima pembanding yang ditampilkan.
        stats = market_statistics(
            vehicle,
            normalized,
            predicted_price=result["prediksi_harga"],
        )

        category = market_category(stats)
        market_required = stats["status"] != "available"
        market_count = stats["jumlah_iklan"]
    except Exception as exc:
        # Do not log exception messages, tracebacks, payloads or local paths.
        logger.warning("Marketplace statistics unavailable (%s)", type(exc).__name__)
        stats = {
            "status": "unavailable",
            "jumlah_iklan": None,
            "harga_minimum": None,
            "harga_maksimum": None,
            "mean": None,
            "median": None,
            "sd": None,
            "rentang_sd": None,
            "posisi_prediksi_sd": None,
        }
        category = None
        market_required = True
        market_count = None

    result["market_analysis"] = {
        **stats,
        "kategori_prediksi_terhadap_pasar": category,
        "catatan": (
            "Kategori adalah posisi estimasi AI "
            "terhadap harga penawaran marketplace, "
            "bukan bukti harga transaksi."
        ),
    }
    result["market_review"] = {
        "required": market_required,
        "reason": (
            "Statistik pasar tidak tersedia atau jumlah iklan relevan tidak mencukupi."
            if market_required else None
        ),
        "jumlah_iklan_relevan": market_count,
    }
    result["requires_manual_review"] = (
        result["requires_manual_review"] or market_required
    )
    return result
        
@app.post('/cats/predict-market/batch')
def predict_market_batch(items: Annotated[list[CatsVehicleIn], Field(min_length=1, max_length=10)]) -> dict:
    """Sequential single-endpoint parity; isolate expected HTTP 422/503 per row.

    Pydantic schema errors reject the entire request before execution. Unexpected
    internal errors propagate as HTTP 500, rather than becoming marketplace or
    validation failures. Each successful item preserves the complete single result.
    """
    results = []
    for index, item in enumerate(items):
        try:
            results.append({'index': index, 'hasil': predict_market(item)})
        except HTTPException as exc:
            if exc.status_code not in {422, 503}:
                raise
            results.append({'index': index, 'AsetBergerakID': item.AsetBergerakID,
                            'status_code': exc.status_code, 'error': exc.detail})
    return {'jumlah': len(items), 'berhasil': sum('hasil' in row for row in results), 'items': results}


# ============================================================
# EV EXPERIMENTAL ENDPOINT
# Tidak terhubung ke API produksi CATS
# ============================================================

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

from ev_inference_experiment import predict_ev
from ev_category_review import review_ev_categories
from ev_combination_review import review_ev_combination

class EVVehicleIn(BaseModel):
    AsetBergerakID: Optional[str] = None

    JenisAsetID: int = Field(default=1, ge=0)

    Merk: str = Field(min_length=1)
    Model: str = Field(min_length=1)

    Varian: Optional[str] = None

    TahunUnit: int = Field(ge=2000, le=2027)

    Kilometer: float = Field(
        ge=0,
        le=2_000_000,
        allow_inf_nan=False
    )

    JenisMesin: Literal["ELECTRIC", "LISTRIK"]

    JenisTransmisi: str = Field(min_length=1)

    KapasitasMesin: Optional[float] = None

    @model_validator(mode="after")
    def validate_ev_fields(self):
        for field in ["Merk", "Model", "JenisTransmisi"]:
            value = getattr(self, field)

            if not value.strip():
                raise ValueError(
                    f"{field} tidak boleh kosong atau hanya spasi"
                )

        if self.JenisAsetID == 5:
            raise ValueError(
                "Endpoint EV ini hanya mendukung mobil"
            )

        if self.KapasitasMesin not in (None, 0):
            raise ValueError(
                "KapasitasMesin kendaraan listrik harus kosong atau 0"
            )

        return self


@app.post("/cats/predict-market-ev")
def predict_market_ev(item: EVVehicleIn):
    if item.JenisAsetID == 5:
        raise HTTPException(
            status_code=422,
            detail="Endpoint EV eksperimen hanya mendukung mobil."
        )

    if item.KapasitasMesin not in (None, 0):
        raise HTTPException(
            status_code=422,
            detail="KapasitasMesin EV harus kosong atau 0."
        )

    payload = item.model_dump()

    result = predict_ev(payload)

    stats = market_statistics(
        "mobil",
        {
            **payload,
            "JenisMesin": "ELECTRIC",
            "KapasitasMesin": None,
        },
        predicted_price=result["prediksi_harga"],
    )

    category = market_category(stats)

    result["AsetBergerakID"] = item.AsetBergerakID

    result["market_analysis"] = {
        **stats,
        "kategori_prediksi_terhadap_pasar": category,
    }

    result["market_review"] = {
        "required": stats["status"] != "available",
        "reason": (
            "Iklan relevan tidak mencukupi."
            if stats["status"] != "available"
            else None
        ),
        "jumlah_iklan_relevan": stats["jumlah_iklan"],
    }

    # Periksa kategori input terhadap encoder training EV.
    # Pemeriksaan kategori individual terhadap encoder EV.
    category_review = review_ev_categories({
        **payload,
        "JenisMesin": "ELECTRIC",
    })

    # Pemeriksaan kombinasi merek-model-varian
    # terhadap final training EV yang direkonstruksi.
    combination_review = review_ev_combination(payload)

    result["category_review"] = category_review
    result["combination_review"] = combination_review

    # Review wajib jika salah satu pemeriksaan bermasalah.
    result["requires_manual_review"] = any([
        result["market_review"]["required"],
        category_review["required"],
        combination_review["required"],
    ])

    # Evidence ditampilkan terpisah dari statistik pasar.
    # Peringkat evidence tidak memengaruhi prediksi.
    from src.Retraining.evidence import get_index

    result.update(
        get_index().comparables("mobil", {
            **payload,
            "JenisMesin": "ELECTRIC",
            "KapasitasMesin": None,
        })
    )

    result["shap"] = None
    result["shap_status"] = "not_implemented_for_ev_experiment"

    return result
