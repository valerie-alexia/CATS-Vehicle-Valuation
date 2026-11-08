"""Verified portable EV fitting categories; no training CSV or reconstruction at runtime."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pandas as pd
from src.Training.pipeline import normalize_features

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'Data/audit/v2'
FIELDS = ['Merk', 'Model', 'Varian', 'JenisTransmisi']
SOURCE_SHA = 'ccac710d2b4f83e2699d4a8f9b7453251143e45d0d3ee3d99b3323e72792b405'
BUNDLE_SHA = 'ae423f7cb08077a49193128cbf08cf2f5cd4937d43a18bec761536dcad4d4e23'
CATALOG_PATH = BASE / 'ev_specialist_experiment/ev_fitting_catalog.json'
CATALOG_SHA = 'e64b626d4b556ffb527418928cbe609a28b2333932afcbf65afeadb9e3302c01'


@lru_cache(maxsize=1)
def _verified_rows():
    try:
        content = CATALOG_PATH.read_bytes()
    except OSError:
        raise ValueError('Katalog fitting EV JSON tidak tersedia atau tidak dapat dibaca') from None
    if hashlib.sha256(content).hexdigest() != CATALOG_SHA:
        raise ValueError('Integritas katalog fitting EV JSON tidak cocok; perlu diverifikasi ulang')
    try:
        catalog = json.loads(content)
    except (ValueError, UnicodeError):
        raise ValueError('Katalog fitting EV JSON tidak valid') from None
    keys = {'schema_version', 'bundle_sha256', 'source_sha256', 'fields',
            'combination_count', 'combinations'}
    if (not isinstance(catalog, dict) or set(catalog) != keys
            or type(catalog['schema_version']) is not int or catalog['schema_version'] != 1
            or catalog['fields'] != FIELDS
            or catalog['bundle_sha256'] != BUNDLE_SHA or catalog['source_sha256'] != SOURCE_SHA
            or type(catalog['combination_count']) is not int
            or not isinstance(catalog['combinations'], list) or not catalog['combinations']
            or catalog['combination_count'] != len(catalog['combinations'])):
        raise ValueError('Schema atau provenance katalog fitting EV JSON tidak cocok')
    rows = []
    for row in catalog['combinations']:
        if (not isinstance(row, dict) or set(row) != set(FIELDS)
                or any(not isinstance(row[field], str) or not row[field].strip() for field in FIELDS)):
            raise ValueError('Kategori katalog fitting EV JSON tidak valid')
        rows.append(tuple(row[field] for field in FIELDS))
    if rows != sorted(set(rows)):
        raise ValueError('Urutan atau keunikan katalog fitting EV JSON tidak valid')
    model = BASE / 'ev_specialist_experiment/ev_specialist_candidate.joblib'
    if hashlib.sha256(model.read_bytes()).hexdigest() != BUNDLE_SHA:
        raise ValueError('Bundle EV berubah; katalog perlu diverifikasi ulang')
    return tuple(rows)


def fitting_catalog():
    # Return owned data: widgets/callers cannot mutate the cached tuples.
    return pd.DataFrame(_verified_rows(), columns=FIELDS)


def options(catalog, selected, field):
    narrowed = catalog
    normalized = normalize_features(pd.DataFrame([selected])).iloc[0]
    for parent in FIELDS[:FIELDS.index(field)]:
        value = selected.get(parent)
        # Empty variant is allowed; transmission then uses the verified brand-model scope.
        if value:
            narrowed = narrowed.loc[narrowed[parent].eq(normalized[parent])]
    return sorted(narrowed[field].unique().tolist())
