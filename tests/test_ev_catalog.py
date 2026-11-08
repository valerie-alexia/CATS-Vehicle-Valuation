"""Portable EV catalog integrity and runtime regression; never fits a model."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from Dashboard import ev_catalog


@pytest.fixture(autouse=True)
def clear_catalog_cache():
    ev_catalog._verified_rows.cache_clear()
    yield
    ev_catalog._verified_rows.cache_clear()


def test_catalog_is_portable_and_returns_owned_data():
    with patch('pandas.read_csv', side_effect=AssertionError('CSV read forbidden')), patch(
        'joblib.load', side_effect=AssertionError('Model deserialization not needed for dropdowns')
    ):
        catalog = ev_catalog.fitting_catalog()
        assert catalog.columns.tolist() == ev_catalog.FIELDS
        assert len(catalog) == 116
        assert catalog.nunique().to_dict() == {
            'Merk': 31, 'Model': 63, 'Varian': 93, 'JenisTransmisi': 1,
        }
        catalog.iloc[0, 0] = 'CALLER MUTATION'
        assert ev_catalog.fitting_catalog().iloc[0, 0] != 'CALLER MUTATION'


def test_hyundai_models_are_distinct_and_empty_variant_keeps_scope():
    catalog = ev_catalog.fitting_catalog()
    models = ev_catalog.options(catalog, {'Merk': 'HYUNDAI'}, 'Model')
    assert {'IONIQ', 'IONIQ 5', 'IONIQ 6'} <= set(models)
    for model in ['IONIQ', 'IONIQ 5', 'IONIQ 6']:
        selected = {'Merk': 'HYUNDAI', 'Model': model}
        expected = sorted(catalog.loc[
            catalog.Merk.eq('HYUNDAI') & catalog.Model.eq(model), 'Varian'
        ].unique().tolist())
        assert ev_catalog.options(catalog, selected, 'Varian') == expected
        assert ev_catalog.options(catalog, {**selected, 'Varian': ''}, 'JenisTransmisi') == ['OTOMATIS']


def test_relationships_match_existing_training_combinations():
    catalog = ev_catalog.fitting_catalog()
    coverage = json.loads((ev_catalog.CATALOG_PATH.parent / 'ev_training_combinations.json').read_text())
    assert set(catalog[['Merk', 'Model', 'Varian']].itertuples(index=False, name=None)) == {
        (r['Merk'], r['Model'], r['Varian']) for r in coverage['combinations']
    }


def test_missing_catalog_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ev_catalog, 'CATALOG_PATH', tmp_path / 'missing.json')
    with pytest.raises(ValueError, match='JSON tidak tersedia'):
        ev_catalog.fitting_catalog()


def test_tampered_catalog_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'catalog.json'
    path.write_bytes(ev_catalog.CATALOG_PATH.read_bytes() + b' ')
    monkeypatch.setattr(ev_catalog, 'CATALOG_PATH', path)
    with pytest.raises(ValueError, match='Integritas'):
        ev_catalog.fitting_catalog()


@pytest.mark.parametrize('change', [
    'schema', 'boolean_schema', 'source', 'bundle', 'count', 'boolean_count',
    'fields', 'duplicate', 'ordering', 'missing_field', 'extra_field', 'nonstring', 'empty',
])
def test_invalid_schema_is_rejected_even_with_matching_file_hash(tmp_path, monkeypatch, change):
    data = json.loads(ev_catalog.CATALOG_PATH.read_bytes())
    if change == 'schema': data['schema_version'] = 2
    elif change == 'boolean_schema': data['schema_version'] = True
    elif change == 'source': data['source_sha256'] = '0' * 64
    elif change == 'bundle': data['bundle_sha256'] = '0' * 64
    elif change == 'count': data['combination_count'] += 1
    elif change == 'boolean_count': data['combination_count'] = True
    elif change == 'fields': data['fields'] = list(reversed(data['fields']))
    elif change == 'duplicate': data['combinations'][1] = copy.deepcopy(data['combinations'][0])
    elif change == 'ordering': data['combinations'].reverse()
    elif change == 'missing_field': del data['combinations'][0]['Varian']
    elif change == 'extra_field': data['combinations'][0]['unexpected'] = 'REJECT'
    elif change == 'nonstring': data['combinations'][0]['Varian'] = None
    elif change == 'empty': data['combinations'][0]['Varian'] = ''
    content = json.dumps(data).encode()
    path = tmp_path / 'catalog.json'
    path.write_bytes(content)
    monkeypatch.setattr(ev_catalog, 'CATALOG_PATH', path)
    # Exercise schema independently of the pinned catalog digest.
    monkeypatch.setattr(ev_catalog, 'CATALOG_SHA', hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match='katalog fitting EV JSON'):
        ev_catalog.fitting_catalog()


def test_wrong_model_hash_is_rejected(monkeypatch):
    original = Path.read_bytes

    def read(path):
        if path.name == 'ev_specialist_candidate.joblib': return b'not the verified bundle'
        return original(path)

    with patch.object(Path, 'read_bytes', read), pytest.raises(ValueError, match='Bundle EV berubah'):
        ev_catalog.fitting_catalog()


@pytest.mark.parametrize('invalid', [False, True])
def test_dashboard_warns_and_keeps_manual_input_when_catalog_unavailable(tmp_path, monkeypatch, invalid):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / 'unavailable.json'
    if invalid:
        path.write_bytes(b'{}')
    monkeypatch.setattr(ev_catalog, 'CATALOG_PATH', path)
    app_path = Path(__file__).resolve().parents[1] / 'Dashboard/app_demo.py'
    with patch('pandas.read_csv', side_effect=AssertionError('CSV fallback forbidden')):
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        app.selectbox(key='demo_mode').select('EV (eksperimen)').run()
        assert not app.exception
        assert any('Hanya input manual tersedia' in item.value for item in app.warning)
        app.selectbox(key='demo_ev_Merk').select('Lainnya / input manual').run()
        assert not app.exception
        assert app.text_input(key='demo_ev_Merk_manual') is not None
