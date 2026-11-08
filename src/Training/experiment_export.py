"""Runtime-only definitions retained from the legacy module; training/export CLI omitted."""

from __future__ import annotations

import numpy as np

import pandas as pd

from src.Training import experiment_granular as exp

from src.Training.pipeline import normalize_features

BUNDLE_FORMAT = 'cats-granular-experiment'

BUNDLE_VERSION = 1

def granular_features(raw, reference, *, enriched=None):
    """Reproduce C from raw canonical inputs; enrich BEFORE baseline key aliases.

    Only the baseline eight columns pass through normalize_features. Direct CATS
    columns always come from raw, and supplied granular fields/targets are ignored.
    """
    raw = raw.reset_index(drop=True)
    required = exp.SCENARIOS['B']
    if set(required) - set(raw.columns):
        raise ValueError(f'Missing direct CATS inputs: {sorted(set(required)-set(raw.columns))}')
    # Serving may supply the exact enrichment computed with immutable static
    # aggregates. The default experimental/export path is unchanged.
    if enriched is None:
        granular, _, _ = exp.enrich(raw, reference)
    else:
        granular = enriched
        if len(granular) != len(raw) or list(granular.columns) != exp.SPECS:
            raise ValueError('Enrichment rows/feature order do not match raw input')
    result = normalize_features(raw)
    for c in ['TipeBodi', 'Warna']:
        result[c] = exp.clean_category(raw[c]).fillna('UNKNOWN').astype(str)
    seats = pd.to_numeric(raw.KapasitasPenumpang, errors='coerce')
    result['KapasitasPenumpang'] = seats.where(seats > 0)
    for c in exp.EXTRA_NUM:
        result[c] = pd.to_numeric(granular[c], errors='coerce')
    for c in exp.EXTRA_CAT:
        result[c] = granular[c].where(granular[c].notna(), np.nan).astype(object)
    for c in exp.NUMERIC + ['KapasitasPenumpang'] + exp.EXTRA_NUM:
        result[c] = result[c].replace([np.inf, -np.inf], np.nan)
    return result[exp.SCENARIOS['C']]

def predict_features(bundle, features):
    if bundle.get('bundle_format') != BUNDLE_FORMAT or bundle.get('bundle_format_version') != BUNDLE_VERSION:
        raise ValueError('Unsupported experimental bundle format')
    X = bundle['preprocessor'].transform(features[bundle['feature_order']])
    return np.maximum(np.expm1(bundle['model'].predict(X)), 1.0)

def predict_raw(bundle, raw):
    return predict_features(bundle, granular_features(raw, bundle['enrichment']['reference']))
