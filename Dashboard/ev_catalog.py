"""Read-only reconstruction of the saved EV fitting catalog, never train/refit."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.Training.pipeline import normalize_features

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'Data/audit/v2'
FIELDS = ['Merk', 'Model', 'Varian', 'JenisTransmisi']
SOURCE_SHA = 'ccac710d2b4f83e2699d4a8f9b7453251143e45d0d3ee3d99b3323e72792b405'
BUNDLE_SHA = 'ae423f7cb08077a49193128cbf08cf2f5cd4937d43a18bec761536dcad4d4e23'


@lru_cache(maxsize=1)
def _verified_rows():
    source = ROOT / 'Data/clean/CLEAN_MOBIL123_Oktober.csv'
    artifact = BASE / 'ev_specialist_experiment'
    model = artifact / 'ev_specialist_candidate.joblib'
    for path, expected in [(source, SOURCE_SHA), (model, BUNDLE_SHA)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Snapshot/bundle EV berubah; katalog perlu diverifikasi ulang')
    report = json.loads((artifact / 'ev_experiment_report.json').read_text(encoding='utf-8'))
    coverage = json.loads((artifact / 'ev_training_combinations.json').read_text(encoding='utf-8'))
    # This module contains definitions only, unlike the training/reconstruction scripts.
    spec = importlib.util.spec_from_file_location('ev_catalog_pipeline', BASE / 'pipeline_ev_experiment.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    data, _ = pipeline.prepare_data(pd.read_csv(source, low_memory=False), 'mobil')
    parts = pipeline.split_data(data, seed=report['seed'])
    for split in ['train', 'validation', 'calibration', 'test']:
        parts[split] = parts[split].loc[parts[split].JenisMesin.eq('ELECTRIC')]
        if len(parts[split]) != report['data_counts'][split]:
            raise ValueError('Split EV tidak cocok dengan report')
    fitting = pd.concat([parts['train'], parts['validation']], ignore_index=True)
    if len(fitting) != coverage['training_rows'] or len(fitting) != report['data_counts']['refit']:
        raise ValueError('Jumlah fitting EV tidak cocok')
    combos = set(fitting[['Merk', 'Model', 'Varian']].itertuples(index=False, name=None))
    if combos != {(r['Merk'], r['Model'], r['Varian']) for r in coverage['combinations']}:
        raise ValueError('Kombinasi fitting EV tidak cocok')
    bundle = joblib.load(model)
    prep = bundle['branch']['preprocessor']
    encoder = prep.named_transformers_['category']
    for feature, categories in zip(pipeline.CATEGORICAL, encoder.categories_):
        if set(fitting[feature]) != set(categories):
            raise ValueError('Kategori tidak cocok dengan fitted encoder EV')
    if not normalize_features(fitting).equals(pipeline.normalize_features(fitting)):
        raise ValueError('Normalisasi training dan inference EV berbeda')
    medians = fitting[pipeline.NUMERIC].median().fillna(0).to_numpy()
    imputer = prep.named_transformers_['numeric'].named_steps['impute']
    if not np.array_equal(medians, imputer.statistics_):
        raise ValueError('Statistik fitting tidak cocok dengan preprocessor EV')
    return tuple(sorted(set(fitting[FIELDS].itertuples(index=False, name=None))))


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
