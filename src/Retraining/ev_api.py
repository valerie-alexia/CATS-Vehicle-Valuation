"""Explicit experimental EV route, isolated from active vehicle model routing."""
from datetime import datetime
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.Training.pipeline import FEATURES, normalize_features, predict_branch
from src.Retraining.explanations import explain_branch

MODEL_PATH = Path(__file__).resolve().parents[2] / 'Data/audit/v2/ev_specialist_experiment/ev_specialist_candidate.joblib'
logger = logging.getLogger(__name__)


class EVInput(BaseModel):
    JenisAset: str = 'mobil'
    Merk: str
    Model: str
    Varian: str | None = None
    TahunUnit: int = Field(ge=2000, le=datetime.now().year + 1, strict=True)
    JenisMesin: str
    JenisTransmisi: str
    KapasitasMesin: float | None = Field(default=None, allow_inf_nan=False)
    Kilometer: float = Field(ge=0, le=2_000_000, allow_inf_nan=False)
    with_shap: bool = False

    @field_validator('Merk', 'Model', 'JenisTransmisi', 'JenisMesin', 'JenisAset')
    @classmethod
    def text(cls, value):
        value = value.strip().upper()
        if value in {'', 'UNKNOWN', 'NULL', 'NONE', 'N/A', 'LAINNYA', '-'}:
            raise ValueError('Atribut EV wajib diketahui')
        return value

    @model_validator(mode='after')
    def electric_only(self):
        if self.JenisAset != 'MOBIL' or self.JenisMesin not in {'ELECTRIC', 'LISTRIK'}:
            raise ValueError('Endpoint eksperimen ini hanya menerima mobil listrik murni')
        if self.KapasitasMesin not in (None, 0):
            raise ValueError('KapasitasMesin EV harus null atau 0; tidak digunakan sebagai cc')
        return self


@lru_cache(maxsize=1)
def load_ev():
    bundle = joblib.load(MODEL_PATH)
    if (bundle.get('experiment_only') is not True or bundle.get('vehicle') != 'mobil_ev'
            or bundle.get('algorithm') != 'xgboost' or bundle.get('features') != FEATURES):
        raise ValueError('Identitas/fitur bundle EV tidak kompatibel')
    branch = bundle['branch']
    if list(branch['preprocessor'].feature_names_in_) != FEATURES:
        raise ValueError('Urutan fitur preprocessor EV tidak kompatibel')
    if branch['model'].n_features_in_ != len(branch['preprocessor'].get_feature_names_out()):
        raise ValueError('Dimensi model/preprocessor EV tidak kompatibel')
    sd = float(bundle['band_halfwidth_sd'])
    if not np.isfinite(sd) or sd < 0:
        raise ValueError('Kalibrasi EV tidak valid')
    return bundle, hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


def explain_ev(bundle, frame):
    branch = bundle['branch']
    prep = branch['preprocessor']
    if list(prep.feature_names_in_) != FEATURES or branch['algorithm'] != 'xgboost':
        raise ValueError('SHAP EV tidak kompatibel dengan urutan fitur/model')
    explanation = explain_branch(branch, frame)
    items = explanation['kontribusi_fitur']
    if ({item['fitur'] for item in items} != set(FEATURES) or len(items) != len(FEATURES)
            or not np.isfinite([explanation['baseline_log_harga'], explanation['prediksi_log_harga'],
                                *[item['kontribusi_log_harga'] for item in items]]).all()):
        raise ValueError('SHAP EV tidak lengkap atau tidak finite')
    for item in items:
        if item['fitur'] == 'KapasitasMesin':
            item['label'] = 'KapasitasMesin (missing, bukan cc aktual)'
    explanation['catatan'] += (
        ' KapasitasMesin EV selalu missing: nilai numerik dan missing indicator ditangani '
        'oleh preprocessor tersimpan; kontribusinya bukan pengaruh kapasitas mesin fisik. '
        'Dummy one-hot dan missing indicator dijumlahkan ke fitur asal.')
    return explanation


def predict_ev_experimental(item: EVInput):
    started = time.perf_counter()
    try:
        bundle, sha = load_ev()
        payload = item.model_dump()
        # Identical to the existing experimental adapter: cc is missing, never invented.
        payload.update(JenisMesin='ELECTRIC', KapasitasMesin=None)
        frame = normalize_features(pd.DataFrame([payload]))
        price = float(predict_branch(bundle['branch'], frame)[0])
        if not np.isfinite(price) or price <= 0:
            raise ValueError('Prediksi EV tidak valid')
    except Exception as exc:
        logger.warning('Experimental EV unavailable (%s)', type(exc).__name__)
        raise HTTPException(503, 'Model EV eksperimental tidak tersedia atau tidak kompatibel; tidak ada fallback ke model non-EV') from None
    sd = float(bundle['band_halfwidth_sd'])
    result = dict(prediksi_harga=round(price),
                rentang=dict(batas_bawah=round(max(0, price*(1-sd))), batas_atas=round(price*(1+sd)),
                             strategi='ev_relative_residual'),
                model_digunakan=dict(rute='mobil-ev-specialist-experiment', algoritma=bundle['algorithm']),
                experiment_only=True, versi_model='ev-experiment-sha256:' + sha,
                metode_rentang='1 x SD residual relatif calibration EV',
                requires_manual_review=True, shap=None, shap_status='not_requested',
                warnings=['Model EV eksperimental; hasil bukan jaminan harga transaksi atau akurasi setiap kendaraan.',
                          'Delapan fitur baseline; tidak memakai kesehatan/kapasitas baterai atau jarak tempuh baterai.',
                          'KapasitasMesin diproses sebagai missing menggunakan preprocessor EV tersimpan; bukan cc kendaraan.',
                          'Kategori atau kombinasi baru dapat berada di luar cakupan eksperimen; appraisal manual diperlukan.'],
                waktu_proses_ms=round((time.perf_counter()-started)*1000, 1))
    if item.with_shap:
        try:
            result['shap'] = explain_ev(bundle, frame)
            result['shap_status'] = 'available'
        except Exception as exc:
            logger.warning('Experimental EV SHAP unavailable (%s)', type(exc).__name__)
            result['shap_status'] = 'unavailable'
            result['warnings'].append('Penjelasan SHAP EV tidak tersedia; prediksi dan rentang tetap dapat digunakan.')
    result['waktu_proses_ms'] = round((time.perf_counter()-started)*1000, 1)
    return result
