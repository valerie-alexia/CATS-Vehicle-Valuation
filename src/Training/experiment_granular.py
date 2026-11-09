"""Runtime-only definitions retained from the legacy module; training/export CLI omitted."""

from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from src.Training.pipeline import NUMERIC, FEATURES

SOURCE = ROOT / 'Data/clean/CLEAN_MOBIL123_2026-10.csv'
SOURCE_PROVENANCE = 'Data/clean/CLEAN_MOBIL123_Oktober.csv'  # Frozen artifact key; never rewritten.

EXTRA_NUM = ['Pintu', 'Tenaga Puncak (hp)', 'Tenaga Putaran Puncak (Nm)',
             'Panjang (mm)', 'Lebar (mm)', 'Tinggi (mm)', 'Jarak Roda (mm)',
             'Velg Depan (inci)', 'Veld Belakang (inci)']

EXTRA_CAT = ['Tipe Gir', 'Aspirasi', 'Rem Depan', 'Rem Belakang',
             'Suspensi Depan', 'Suspensii Belakang']

SPECS = EXTRA_NUM + EXTRA_CAT

SCENARIOS = {'A': FEATURES, 'B': FEATURES + ['TipeBodi', 'KapasitasPenumpang', 'Warna'],
             'C': FEATURES + ['TipeBodi', 'KapasitasPenumpang', 'Warna'] + SPECS}

LEVELS = [['Merk', 'Model', 'Varian', 'TahunUnit'],
          ['Merk', 'Model', 'TahunUnit'], ['Merk', 'Model']]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clean_category(series):
    values = series.astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.upper()
    return values.mask(values.isin(['', 'UNKNOWN', 'NAN', 'NONE', 'NULL', '<NA>']))

def normalize_keys(frame):
    out = frame.copy()
    for c in ['Merk', 'Model', 'Varian']:
        out[c] = clean_category(out[c])
    out['Varian'] = out['Varian'].fillna('UNKNOWN')
    out['TahunUnit'] = pd.to_numeric(out['TahunUnit'], errors='coerce')
    # Match the reference builder on BOTH sides; do not apply baseline brand aliases here.
    return out

def enrich(target, reference):
    """Per-attribute fallback; only explicit keys and specification columns are read."""
    keys = LEVELS[0]
    result = normalize_keys(target[keys]).reset_index(drop=True)
    ref = normalize_keys(reference[keys + SPECS])
    for c in EXTRA_NUM:
        ref[c] = pd.to_numeric(ref[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        ref[c] = ref[c].where(ref[c] > 0)
    for c in EXTRA_CAT:
        ref[c] = clean_category(ref[c])
    values = pd.DataFrame({c: pd.Series(np.nan, index=result.index, dtype=object) for c in SPECS})
    levels = pd.DataFrame(0, index=result.index, columns=SPECS)
    for number, level_keys in enumerate(LEVELS, 1):
        lookup = ref.groupby(level_keys, dropna=True).agg({
            **{c: 'median' for c in EXTRA_NUM},
            **{c: lambda x: x.mode().iloc[0] if x.notna().any() else np.nan for c in EXTRA_CAT}})
        matched = result.merge(lookup.reset_index(), on=level_keys, how='left', validate='many_to_one')
        for c in SPECS:
            fill = values[c].isna() & matched[c].notna()
            values.loc[fill, c] = matched.loc[fill, c]
            levels.loc[fill, c] = number
    coverage = []
    for c in SPECS:
        coverage.append({'feature': c, 'rows': len(result), 'missing': int(values[c].isna().sum()),
                         'coverage': float(values[c].notna().mean()), **{
                             f'level_{i}_rows': int(levels[c].eq(i).sum()) for i in range(1, 4)}, **{
                             f'level_{i}_proportion': float(levels[c].eq(i).mean()) for i in range(1, 4)}})
    audit = {'rows_without_any_reference': int(values.isna().all(axis=1).sum()),
             'rows_with_all_specs': int(values.notna().all(axis=1).sum()),
             'rows_receiving_any_attribute_by_level_nonexclusive': {
                 str(i): int(levels.eq(i).any(axis=1).sum()) for i in range(1, 4)}}
    return values, pd.DataFrame(coverage), audit
