"""Runtime-only definitions retained from the legacy module; training/export CLI omitted."""

from __future__ import annotations

import re

from urllib.parse import urlsplit, urlunsplit

import numpy as np

import pandas as pd

CATEGORICAL = ['Merk', 'Model', 'Varian', 'JenisMesin', 'JenisTransmisi']

NUMERIC = ['TahunUnit', 'KapasitasMesin', 'Kilometer']

FEATURES = CATEGORICAL + NUMERIC

def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exactly the same normalization for training, API and demo input."""
    out = frame.copy()
    if 'Kilometer' not in out and 'Kilometer_Num' in out:
        out['Kilometer'] = out['Kilometer_Num']
    for col in CATEGORICAL:
        if col not in out:
            out[col] = 'UNKNOWN'
        out[col] = (out[col].fillna('UNKNOWN').astype(str).str.strip()
                    .str.replace(r'\s+', ' ', regex=True).str.upper().replace('', 'UNKNOWN'))
    out['Merk'] = out['Merk'].replace({'MERCEDES': 'MERCEDES-BENZ', 'LAND': 'LAND ROVER'})
    out['JenisMesin'] = out['JenisMesin'].replace({'PETROL': 'BENSIN', 'GASOLINE': 'BENSIN'})
    out['JenisTransmisi'] = out['JenisTransmisi'].replace({
        'AUTOMATIC': 'OTOMATIS', 'AUTO': 'OTOMATIS', 'MATIC': 'OTOMATIS',
        'AT': 'OTOMATIS', 'MT': 'MANUAL'})
    for col in NUMERIC:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return out[FEATURES]

def listing_identity(value) -> str:
    if pd.isna(value) or str(value).strip().lower() in {'', 'unknown', 'nan'}:
        return ''
    parsed = urlsplit(str(value).strip())
    path = parsed.path.rstrip('/')
    match = re.search(r'/([0-9]+)$', path)
    if 'mobil123.com' in parsed.netloc.lower() and match:
        return 'mobil123:' + match.group(1)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, '', ''))

def predict_branch(branch: dict, data: pd.DataFrame) -> np.ndarray:
    X = branch['preprocessor'].transform(normalize_features(data))
    return np.maximum(np.expm1(branch['model'].predict(X)), 1.0)

def predict_bundle(bundle: dict, frame: pd.DataFrame, force_global=False) -> tuple[np.ndarray, list[str]]:
    data = normalize_features(frame)
    predictions = predict_branch(bundle['global'], data)
    routes = np.full(len(data), 'global', dtype=object)
    if bundle['scheme'] == 'hybrid' and not force_global:
        for brand, branch in bundle['brands'].items():
            mask = data['Merk'].eq(brand).to_numpy()
            if mask.any():
                predictions[mask] = predict_branch(branch, data.loc[mask])
                routes[mask] = 'merk:' + brand
    return predictions, routes.tolist()
