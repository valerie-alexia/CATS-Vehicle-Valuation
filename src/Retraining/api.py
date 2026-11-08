"""Modul 4 - API Prediksi (FastAPI).

Jalankan dari root repo:
    uvicorn src.Retraining.api:app --reload
atau dari folder src/Retraining:
    uvicorn api:app --reload
Dokumentasi interaktif: http://127.0.0.1:8000/docs

CATS cukup mengirim POST /predict; penulisan hasil ke NilaiAppraisalAI,
AppraisalAINote, dan AppraisalAIDate dilakukan sisi CATS (di luar lingkup proyek).
"""
from __future__ import annotations

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))
# install fastapi, pydantic, uvicorn, joblib, numpy, pandas
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402

import inference  # noqa: E402
from src.Retraining.evidence import get_index  # noqa: E402
from src.Retraining.cats_input import CatsVehicleIn, adapt_cats  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CATS Vehicle Valuation API",
    version="2.1",
    description="Estimasi nilai pasar kendaraan (mobil/motor) berbasis XGBoost/LightGBM.",
)

from src.Retraining.ev_api import predict_ev_experimental
app.add_api_route('/predict/ev-experimental', predict_ev_experimental,
                  methods=['POST'], tags=['EV experimental'])


class VehicleIn(BaseModel):
    JenisAset: Literal["mobil", "motor"]
    Merk: str
    Model: str
    Varian: Optional[str] = None
    TahunUnit: Union[Annotated[int, Field(ge=1990, le=datetime.now().year + 1)], Literal['Lainnya']]
    JenisMesin: str
    JenisTransmisi: str
    KapasitasMesin: Union[Annotated[float, Field(gt=0, allow_inf_nan=False)], Literal['Lainnya']]
    Kilometer: Union[Annotated[float, Field(ge=0, allow_inf_nan=False)], Literal['Lainnya']]
    KapasitasPenumpang: Optional[float] = None
    TipeBodi: Optional[str] = None
    Warna: Optional[str] = None
    # opsional: harga iklan yang ingin dinilai Murah/Wajar/Mahal
    harga_pembanding: Optional[float] = Field(default=None, gt=0)
    with_shap: bool = False


def _dump(item: BaseModel, exclude: set) -> dict:
    fn = getattr(item, "model_dump", None) or item.dict  # pydantic v2 / v1
    return fn(exclude=exclude)


def _predict_one(item: VehicleIn, with_shap: bool, bundle=None) -> dict:
    payload = _dump(item, {"harga_pembanding", "with_shap"})
    t0 = time.perf_counter()
    result = inference.predict_price(
        payload, harga_pembanding=item.harga_pembanding, with_shap=with_shap,
        **({'bundle': bundle} if bundle is not None else {}))
    try:
        result.update(get_index().comparables(item.JenisAset, payload))
    except Exception as exc:
        # No payload, exception message, filesystem path or dataset contents in logs/API.
        logger.warning('Comparable lookup failed (%s)', type(exc).__name__)
        result.update(comparables=[], evidence_status='unavailable', evidence_metadata=None)
    result["waktu_proses_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def _prepare_cats(item: CatsVehicleIn):
    vehicle = 'motor' if item.JenisAsetID == 5 else 'mobil'
    bundle = inference.load_active(vehicle)
    return adapt_cats(item, bundle['catalog']), bundle


@app.post('/cats/normalize')
def normalize_cats(item: CatsVehicleIn) -> dict:
    """Preview deterministic mappings without computing a price or writing to CATS."""
    try:
        normalized, _ = _prepare_cats(item)
        return {'AsetBergerakID': item.AsetBergerakID, **normalized}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=503, detail='Model/katalog aktif belum tersedia') from None


def _numeric_review_required(payload: dict, bundle: dict) -> bool:
    """Check supplied numbers against fitted bounds, independently of warning text.

    Unknown numeric sentinels are reviewed by the existing normalization/fallback
    rules. Bounds are inclusive and come from the same bundle used for inference.
    """
    return any(
        isinstance(payload.get(field), (int, float))
        and not bounds['min'] <= payload[field] <= bounds['max']
        for field, bounds in bundle.get('numeric_bounds', {}).items()
    )


def _predict_cats_one(item: CatsVehicleIn, with_shap: bool) -> dict:
    normalized, bundle = _prepare_cats(item)
    result = _predict_one(VehicleIn(**normalized['input_model']), with_shap, bundle=bundle)
    result['AsetBergerakID'] = item.AsetBergerakID
    result['input_normalization'] = normalized
    result['requires_manual_review'] = (
        normalized['status'] == 'needs_review'
        or result['fallback_global']
        or _numeric_review_required(normalized['input_model'], bundle)
        or bool(result.get('enrichment_details', {}).get('missing_features'))
    )
    return result


@app.post('/cats/predict')
def predict_cats(item: CatsVehicleIn) -> dict:
    """Accept a CATS database row; ignore name, plate and historical financial columns."""
    try:
        return _predict_cats_one(item, item.with_shap)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=503, detail='Model/katalog aktif belum tersedia') from None


@app.post('/cats/predict/batch')
def predict_cats_batch(items: Annotated[List[CatsVehicleIn], Field(min_length=1, max_length=500)]) -> dict:
    """Per-row normalization/inference errors are isolated; SHAP disabled for batches."""
    results = []
    for i, item in enumerate(items):
        try:
            results.append({'index': i, 'hasil': _predict_cats_one(item, False)})
        except ValueError as exc:
            results.append({'index': i, 'AsetBergerakID': item.AsetBergerakID, 'error': str(exc)})
        except (FileNotFoundError, OSError):
            results.append({'index': i, 'AsetBergerakID': item.AsetBergerakID,
                            'error': 'Model/katalog aktif belum tersedia'})
    return {'jumlah': len(items), 'berhasil': sum('hasil' in row for row in results), 'items': results}


class VehicleSearch(BaseModel):
    vehicle_type: Literal['mobil', 'motor']
    merk: Optional[str] = None
    model: Optional[str] = None
    varian: Optional[str] = None
    tipe_bodi: Optional[str] = None
    jenis_transmisi: Optional[str] = None
    tahun_min: Optional[int] = Field(default=None, ge=1900, le=datetime.now().year + 1)
    tahun_max: Optional[int] = Field(default=None, ge=1900, le=datetime.now().year + 1)
    harga_min: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    harga_max: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    kilometer_max: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    kapasitas_mesin_min: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    kapasitas_mesin_max: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def ordered_ranges(self):
        for name in ['tahun', 'harga', 'kapasitas_mesin']:
            lower, upper = getattr(self, name + '_min'), getattr(self, name + '_max')
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(name + '_min harus <= ' + name + '_max')
        if self.vehicle_type == 'motor' and self.tipe_bodi is not None:
            raise ValueError('tipe_bodi hanya tersedia untuk mobil')
        return self


@app.get('/vehicles/search')
def search_vehicles(params: Annotated[VehicleSearch, Query()]) -> dict:
    try:
        return get_index().search(params.vehicle_type, **params.model_dump(exclude={'vehicle_type'}))
    except Exception as exc:
        logger.warning('Vehicle search unavailable (%s)', type(exc).__name__)
        raise HTTPException(status_code=503, detail='Data pencarian sementara tidak tersedia') from None


@app.get("/health")
def health() -> dict:
    states = {}
    for vehicle in ['mobil', 'motor']:
        try:
            bundle = inference.load_active(vehicle)
            states[vehicle] = {'ready': True, 'version': bundle['version'], 'batch': bundle['batch']}
        except (FileNotFoundError, ValueError, OSError) as exc:
            states[vehicle] = {'ready': False, 'reason': 'Model sementara tidak tersedia'}
    return {'status': 'ready' if all(s['ready'] for s in states.values()) else 'partial',
            'models': states}


@app.get('/catalog/{vehicle}')
def catalog(vehicle: Literal['mobil', 'motor'], Merk: Optional[str] = None,
            Model: Optional[str] = None, Varian: Optional[str] = None,
            TahunUnit: Optional[str] = None, JenisMesin: Optional[str] = None,
            JenisTransmisi: Optional[str] = None, KapasitasMesin: Optional[str] = None,
            Kilometer: Optional[str] = None) -> dict:
    """Opsi dropdown dari katalog model aktif; query sebelumnya menyaring pilihan berikutnya."""
    import numpy as np
    import pandas as pd
    from src.Retraining.catalog import INPUT_ORDER, options_for
    from src.Retraining.input_values import is_other
    from src.Training.pipeline import NUMERIC, normalize_features
    filters = {key: value for key, value in locals().copy().items()
               if key in INPUT_ORDER and value is not None}
    try:
        for key in NUMERIC:
            if key in filters and not is_other(filters[key]):
                filters[key] = float(filters[key])
                if not np.isfinite(filters[key]):
                    raise ValueError(f'{key} harus berupa angka finite')
        normalized = normalize_features(pd.DataFrame([filters])).iloc[0]
        filters = {key: 'Lainnya' if is_other(value) else normalized[key] for key, value in filters.items()}
        bundle = inference.load_active(vehicle)
        return {'jenis_aset': vehicle, 'batch': bundle['batch'], 'versi_model': bundle['version'],
                'opsi': {field: options_for(bundle['catalog'],
                         {k: v for k, v in filters.items() if k != field}, field)
                         for field in INPUT_ORDER}}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail='Model/tabel referensi belum tersedia')


@app.post("/predict")
def predict(item: VehicleIn) -> dict:
    try:
        return _predict_one(item, item.with_shap)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail='Model/tabel referensi belum tersedia')


@app.post("/predict/batch")
def predict_batch(items: List[VehicleIn]) -> dict:
    """Untuk mengisi banyak aset sekaligus. SHAP dimatikan agar cepat."""
    t0 = time.perf_counter()
    hasil = []
    for i, item in enumerate(items):
        try:
            hasil.append({"index": i, "hasil": _predict_one(item, with_shap=False)})
        except FileNotFoundError:
            hasil.append({"index": i, "error": 'Model/tabel referensi belum tersedia'})
        except ValueError as exc:
            hasil.append({"index": i, "error": str(exc)})
    return {
        "jumlah": len(items),
        "berhasil": sum("hasil" in h for h in hasil),
        "waktu_total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "items": hasil,
    }
