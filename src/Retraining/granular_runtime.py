"""Read-only adapter for the experimental C bundle; no serving schema-v2 relabeling."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.Training import experiment_granular as exp
from src.Training.experiment_export import BUNDLE_FORMAT, BUNDLE_VERSION, granular_features
from src.Training.pipeline import normalize_features, NUMERIC
from src.Retraining.granular_lookup import prepare_reference, enrich_prepared


def adapt_bundle(bundle, path):
    path = Path(path)
    metadata = json.loads((path.parent / 'export_metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('status') != 'completed' or metadata.get('bundle_sha256') != exp.sha256(path)
            or bundle.get('bundle_format') != BUNDLE_FORMAT
            or bundle.get('bundle_format_version') != BUNDLE_VERSION
            or bundle.get('metadata', {}).get('status') != 'completed'
            or bundle.get('feature_order') != exp.SCENARIOS['C']
            or bundle.get('architecture') != 'global-only'):
        raise ValueError('Bundle eksperimen C tidak valid atau ekspor belum selesai')
    ref = bundle['enrichment']
    if hashlib.sha256(ref['reference_csv_bytes']).hexdigest() != ref['reference_sha256']:
        raise ValueError('Hash snapshot referensi April tidak cocok')
    snapshot = pd.read_csv(io.BytesIO(ref['reference_csv_bytes']))
    try:
        pd.testing.assert_frame_equal(snapshot[exp.LEVELS[0]+exp.SPECS], ref['reference'], check_dtype=False)
    except AssertionError as exc:
        raise ValueError('Snapshot referensi April tidak konsisten') from exc
    if ref['levels'] != exp.LEVELS or ref['granular_inputs'] != exp.SPECS:
        raise ValueError('Aturan enrichment bundle tidak didukung')
    prep = bundle['preprocessor']
    if list(prep.feature_names_in_) != bundle['feature_order']:
        raise ValueError('Urutan fitted preprocessor tidak cocok')
    if list(prep.get_feature_names_out()) != bundle['transformed_feature_order']:
        raise ValueError('Urutan fitur transformasi tidak cocok')
    config = bundle['metadata']['frozen_config']
    # The experiment did not save a catalog. Reconstruct fitting identities from
    # its frozen source_row manifest; no fitting, deduplication, or split creation.
    inputs = {k.replace('\\', '/'): v for k, v in config['inputs'].items()}
    if exp.sha256(exp.SOURCE) != inputs[exp.SOURCE.relative_to(exp.ROOT).as_posix()]:
        raise ValueError('Dataset katalog berbeda dari sumber eksperimen')
    manifest = bundle['split_manifest']
    rows = manifest.loc[manifest.split.isin(['train', 'validation']), 'source_row'].to_numpy(dtype=int)
    raw = pd.read_csv(exp.SOURCE, low_memory=False).iloc[rows].reset_index(drop=True)
    catalog = normalize_features(raw)
    for c in ['TipeBodi', 'Warna']:
        catalog[c] = exp.clean_category(raw[c]).fillna('UNKNOWN').astype(str)
    catalog['KapasitasPenumpang'] = pd.to_numeric(raw.KapasitasPenumpang, errors='coerce')
    catalog = catalog.drop_duplicates().reset_index(drop=True)
    test = next(r for r in config['results']['test'] if r['scenario'] == 'C')
    calibration = bundle['calibration']
    if not np.isclose(calibration['halfwidth'], test['band_halfwidth'], rtol=1e-12):
        raise ValueError('Kalibrasi eksperimen tidak cocok')
    branch = {'preprocessor': prep, 'model': bundle['model'], 'algorithm': 'xgboost',
              'features': bundle['feature_order'], 'categorical': bundle['categorical_features']}
    prepared_reference = prepare_reference(ref['reference'],
        (metadata['bundle_sha256'], ref['reference_sha256'], bundle['bundle_format_version']))
    return {'vehicle': 'mobil', 'scheme': 'global', 'global': branch, 'brands': {},
            'version': path.parent.name, 'batch': 'Oktober', 'catalog': catalog,
            'numeric_bounds': {c: {'min': float(catalog[c].min()), 'max': float(catalog[c].max())}
                               for c in NUMERIC},
            'band_halfwidth': calibration['halfwidth'], 'coverage_target': None,
            'band_method': '1 x SD residual relatif (calibration eksperimen C)',
            'report': {'audit': config['audit']['preparation'], 'final_test': test},
            'experimental_bundle': bundle, 'prepared_reference': prepared_reference}


def prepare_row(payload, bundle):
    # Preserve raw keys until the shared training enrichment helper handles them.
    raw = pd.DataFrame([{**payload, **{c: payload.get(c) for c in
                           ['Varian', 'TipeBodi', 'KapasitasPenumpang', 'Warna']}}])
    reference = bundle['experimental_bundle']['enrichment']['reference']
    values, coverage, audit = enrich_prepared(raw, bundle['prepared_reference'])
    row = granular_features(raw, reference, enriched=values)
    missing = [c for c in exp.SPECS if pd.isna(row[c].iloc[0])]
    levels = {r['feature']: next((i for i in range(1, 4) if r[f'level_{i}_rows']), None)
              for r in coverage.to_dict('records')}
    return row, {'reference_sha256': bundle['experimental_bundle']['enrichment']['reference_sha256'],
                 'source': 'embedded_april_reference', 'missing_features': missing,
                 'matched_level_per_feature': levels, **audit}
