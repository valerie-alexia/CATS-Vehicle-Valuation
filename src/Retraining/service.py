"""Versioned model loader and prediction service shared by UI and API."""
from __future__ import annotations
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.Training.pipeline import CATEGORICAL, NUMERIC, normalize_features, predict_bundle
from src.Retraining.price_range import categorize, compute_range
from src.Retraining.input_values import is_other

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRS = {'mobil': ROOT / 'Models/Mobil123', 'motor': ROOT / 'Models/Momotor'}


def active_path(vehicle='mobil') -> Path:
    if vehicle not in MODEL_DIRS:
        raise ValueError('JenisAset harus mobil atau motor')
    if vehicle == 'mobil':
        mode = os.environ.get('CATS_MOBIL_MODEL', 'active8')
        if mode not in {'active8', 'granular26'}:
            raise ValueError('CATS_MOBIL_MODEL harus active8 atau granular26')
        if mode == 'granular26':
            location = os.environ.get('CATS_MOBIL_26_BUNDLE')
            if not location:
                raise ValueError('CATS_MOBIL_26_BUNDLE wajib untuk mode granular26')
            path = (ROOT / location).resolve()
            if not path.is_relative_to((ROOT / 'Experiments/feature_comparison').resolve()):
                raise ValueError('Bundle 26 fitur harus berada di Experiments/feature_comparison')
            return path
    directory = MODEL_DIRS[vehicle]
    pointer = directory / 'active.json'
    if not pointer.exists():
        raise FileNotFoundError(f'Model {vehicle} belum tersedia. Jalankan python src/Training/final_train.py {vehicle}')
    manifest = json.loads(pointer.read_text(encoding='utf-8'))
    path = (directory / manifest['bundle']).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError('Lokasi artefak di luar direktori model')
    return path


@lru_cache(maxsize=4)
def _load_version(path: str) -> dict:
    bundle = joblib.load(path)
    if bundle.get('bundle_format') == 'cats-granular-experiment':
        from src.Retraining.granular_runtime import adapt_bundle
        return adapt_bundle(bundle, path)
    if bundle.get('schema_version') != 2:
        raise ValueError('Bundle lama tidak didukung; training ulang dengan final_train.py')
    return bundle


def load_active(vehicle='mobil') -> dict:
    bundle = _load_version(str(active_path(vehicle)))
    if vehicle == 'mobil' and os.environ.get('CATS_MOBIL_MODEL', 'active8') == 'granular26':
        if 'experimental_bundle' not in bundle:
            raise ValueError('Konfigurasi granular26 harus menunjuk bundle eksperimen C')
    return bundle


def validate_payload(payload: dict) -> tuple[str, pd.DataFrame]:
    vehicle = str(payload.get('JenisAset', 'mobil')).strip().lower()
    if vehicle not in MODEL_DIRS:
        raise ValueError('JenisAset harus mobil atau motor')
    for field in ['Merk', 'Model', 'JenisMesin', 'JenisTransmisi'] + NUMERIC:
        value = payload.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f'{field} wajib diisi')
    for field in ['Merk', 'Model', 'JenisMesin', 'JenisTransmisi']:
        if not isinstance(payload[field], str):
            raise ValueError(f'{field} harus berupa teks')
    if str(payload['JenisMesin']).strip().upper() in {'ELECTRIC', 'LISTRIK'}:
        raise ValueError('Kendaraan listrik murni belum didukung model demo ini')
    for field in NUMERIC:
        if is_other(payload[field]):
            continue
        try:
            number = float(payload[field])
        except (ValueError, TypeError):
            raise ValueError(f'{field} harus berupa angka') from None
        if not np.isfinite(number):
            raise ValueError(f'{field} harus berupa angka finite')
    year = None if is_other(payload['TahunUnit']) else float(payload['TahunUnit'])
    minimum = 2000 if vehicle == 'mobil' else 2010
    if year is not None and (not year.is_integer() or not minimum <= year <= datetime.now().year + 1):
        raise ValueError(f'TahunUnit harus bilangan bulat {minimum} sampai {datetime.now().year + 1}')
    cc = None if is_other(payload['KapasitasMesin']) else float(payload['KapasitasMesin'])
    lower, upper = (500, 8000) if vehicle == 'mobil' else (50, 2000)
    if cc is not None and not lower <= cc <= upper:
        raise ValueError(f'KapasitasMesin di luar cakupan {lower}–{upper} cc')
    if not is_other(payload['Kilometer']) and not 0 <= float(payload['Kilometer']) <= 2_000_000:
        raise ValueError('Kilometer harus antara 0 dan 2.000.000')
    return vehicle, normalize_features(pd.DataFrame([payload]))


def predict_price(payload: dict, harga_pembanding=None, with_shap=False, bundle=None) -> dict:
    vehicle, row = validate_payload(payload)
    if harga_pembanding is not None:
        try:
            harga_pembanding = float(harga_pembanding)
        except (ValueError, TypeError):
            raise ValueError('Harga pembanding harus berupa angka positif') from None
        if not np.isfinite(harga_pembanding) or harga_pembanding <= 0:
            raise ValueError('Harga pembanding harus positif dan finite')
    bundle = bundle if bundle is not None else load_active(vehicle)
    if bundle['vehicle'] != vehicle:
        raise ValueError('Jenis aset tidak cocok dengan model')
    granular = 'experimental_bundle' in bundle
    enrichment_details = None
    if granular:
        from src.Retraining.granular_runtime import prepare_row
        row, enrichment_details = prepare_row(payload, bundle)
    catalog = bundle['catalog']
    fallback_reasons = [f'{field}=Lainnya' for field in CATEGORICAL + NUMERIC if is_other(payload.get(field))]
    for field in CATEGORICAL:
        if row[field].iloc[0] not in set(catalog[field]):
            fallback_reasons.append(f'{field} belum dikenal model')
    pair = catalog.Merk.eq(row.Merk.iloc[0]) & catalog.Model.eq(row.Model.iloc[0])
    if not pair.any():
        fallback_reasons.append('Kombinasi merek/model belum dikenal model')
    elif not (pair & catalog.Varian.eq(row.Varian.iloc[0])).any():
        fallback_reasons.append('Kombinasi model/varian belum dikenal model')
    force_global = bool(fallback_reasons)
    if granular:
        from src.Training.experiment_export import predict_features
        predictions = predict_features(bundle['experimental_bundle'], row)
        routes = ['global']
    else:
        predictions, routes = predict_bundle(bundle, row, force_global=force_global)
    prediction = float(predictions[0])
    if not np.isfinite(prediction):
        raise ValueError('Model menghasilkan prediksi tidak valid')
    band = compute_range(prediction, bundle['band_halfwidth'])
    route = routes[0]
    branch = bundle['global'] if route == 'global' else bundle['brands'][route[5:]]
    warnings = []
    if granular:
        missing = enrichment_details['missing_features']
        if missing:
            warnings.append('Referensi April tidak lengkap: ' + ', '.join(missing) +
                            '. Numerik memakai imputasi training tersimpan; kategori kosong memakai encoder tersimpan.')
        for field in ['TipeBodi', 'Warna', 'KapasitasPenumpang']:
            if pd.isna(payload.get(field)):
                warnings.append(f'{field} tidak tersedia; memakai penanganan missing preprocessor training.')
        warnings.append('Evaluasi historis eksperimen C bukan jaminan akurasi setiap kendaraan. '
                        'Urutan temporal April belum terbukti karena timestamp scraping tidak tersedia.')
    if force_global:
        warnings.append('Model global digunakan karena input Lainnya atau kategori belum dikenal.')
        if bundle['scheme'] == 'hybrid':
            warnings.append('Coverage test yang tercatat memakai routing evaluasi standar; '
                            'coverage khusus input Lainnya belum diverifikasi.')
    for field in NUMERIC:
        if is_other(payload.get(field)):
            warnings.append(f'{field}=Lainnya diisi dengan median training model global, bukan nilai kendaraan sebenarnya.')
    for field in CATEGORICAL:
        if row[field].iloc[0] not in set(catalog[field]):
            warnings.append(f'{field} belum terlihat pada data training: {row[field].iloc[0]}')
    if not pair.any():
        warnings.append('Kombinasi merek/model belum terlihat pada training; lakukan appraisal manual.')
    elif not (pair & catalog.Varian.eq(row.Varian.iloc[0])).any():
        warnings.append('Kombinasi model/varian belum terlihat pada training; periksa spesifikasi kendaraan.')
    for field, bounds in bundle['numeric_bounds'].items():
        if not pd.isna(row[field].iloc[0]) and not bounds['min'] <= row[field].iloc[0] <= bounds['max']:
            warnings.append(f'{field} di luar rentang data training.')
    result = {'prediksi_harga': round(prediction), 'rentang': band,
              'model_digunakan': {'rute': f'{vehicle}-{route}', 'algoritma': branch['algorithm']},
              'versi_model': bundle['version'], 'batch': bundle['batch'],
              'metode_rentang': bundle.get('band_method', 'quantile kalibrasi (bundle lama)'),
              'target_coverage': bundle.get('coverage_target'),
              'coverage_test': bundle['report']['final_test']['coverage'],
              'enrichment': 'fitur_payload_langsung',
              'fallback_global': force_global, 'alasan_fallback_global': fallback_reasons,
              'catatan': 'Estimasi harga penawaran marketplace, bukan harga transaksi atau keputusan appraisal final.',
              'peringatan': warnings}
    if granular:
        result['enrichment'] = 'referensi_april_15_fitur_per_atribut'
        result['enrichment_details'] = enrichment_details
        result['model_digunakan']['jumlah_fitur'] = 26
    if harga_pembanding is not None:
        result['kategori'] = categorize(harga_pembanding, band['batas_bawah'], band['batas_atas'])
    if with_shap:
        try:
            from src.Retraining.explanations import explain_branch
            result['shap'] = explain_branch(branch, row)
            result['shap_top'] = result['shap']['kontribusi_fitur']
        except Exception as exc:
            result['peringatan'].append(f'Penjelasan SHAP tidak tersedia ({type(exc).__name__}); prediksi tetap valid.')
    return result
