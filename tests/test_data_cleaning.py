"""Cleaning configuration, entrypoint safety, and existing safe-helper regression."""
import ast
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.Preprocessing import DataCleaning as cleaning

ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name='cleaning_test_module'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Compile in memory so the read-only legacy repository never receives .pyc files.
    exec(compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec'), module.__dict__)
    return module


def test_import_has_no_data_io():
    with patch('pandas.read_csv', side_effect=AssertionError('Unexpected input read')), patch(
        'pandas.DataFrame.to_csv', side_effect=AssertionError('Unexpected output write')
    ), patch('os.makedirs', side_effect=AssertionError('Unexpected directory creation')):
        load_module(ROOT / 'src/Preprocessing/DataCleaning.py')


def test_default_paths(monkeypatch):
    for name in ['CATS_BATCH', 'CATS_RAW_DIR', 'CATS_CLEAN_DIR']:
        monkeypatch.delenv(name, raising=False)
    module = load_module(ROOT / 'src/Preprocessing/DataCleaning.py')
    assert module.BATCH is None
    assert module.CSV_MOBIL == ROOT / 'Data/raw/MOBIL123_raw_dataset_None.csv'
    assert module.CSV_MOTOR == ROOT / 'Data/raw/MOMOTOR_raw_dataset_None.csv'
    assert module.DIR_CLEAN == ROOT / 'Data/clean'


def test_external_paths_and_batch(monkeypatch, tmp_path):
    monkeypatch.setenv('CATS_BATCH', '2026-11')
    monkeypatch.setenv('CATS_RAW_DIR', str(tmp_path / 'input'))
    monkeypatch.setenv('CATS_CLEAN_DIR', str(tmp_path / 'output'))
    module = load_module(ROOT / 'src/Preprocessing/DataCleaning.py')
    assert module.CSV_MOBIL == tmp_path / 'input/MOBIL123_raw_dataset_2026-11.csv'
    assert module.CSV_MOTOR == tmp_path / 'input/MOMOTOR_raw_dataset_2026-11.csv'
    for name in ['OUT_MOBIL_BERSIH', 'OUT_MOBIL_ANTIK', 'OUT_MOBIL_ANOMALI',
                 'OUT_MOTOR_BERSIH', 'OUT_MOTOR_ANTIK', 'OUT_MOTOR_ANOMALI']:
        assert getattr(module, name).parent == tmp_path / 'output'
        assert getattr(module, name).name.endswith('_2026-11.csv')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('vehicle', ['mobil', 'motor', 'all'])
def test_missing_input_fails_before_any_cleaning(monkeypatch, tmp_path, capsys, vehicle):
    monkeypatch.setattr(cleaning, 'CSV_MOBIL', tmp_path / 'missing_car.csv')
    monkeypatch.setattr(cleaning, 'CSV_MOTOR', tmp_path / 'missing_motor.csv')
    monkeypatch.setattr(cleaning, 'BATCH', 'missing')
    monkeypatch.setattr(cleaning, 'DIR_RAW', tmp_path)
    car, motor = Mock(), Mock()
    monkeypatch.setattr(cleaning, 'clean_mobil', car)
    monkeypatch.setattr(cleaning, 'clean_motor', motor)
    with pytest.raises(SystemExit) as error:
        cleaning.main(['--vehicle', vehicle])
    assert error.value.code == 2
    car.assert_not_called()
    motor.assert_not_called()
    captured = capsys.readouterr()
    assert 'File RAW tidak ditemukan' in captured.err
    assert 'SELURUH PIPELINE SELESAI' not in captured.out


def test_all_preflight_prevents_partial_output(monkeypatch, tmp_path):
    present = tmp_path / 'present.csv'
    present.touch()  # Path-only dispatch fixture; never parsed as RAW data.
    monkeypatch.setattr(cleaning, 'BATCH', 'test')
    monkeypatch.setattr(cleaning, 'DIR_RAW', tmp_path)
    (tmp_path / 'MOBIL123_raw_dataset_test.csv').touch()
    monkeypatch.setattr(cleaning, 'CSV_MOBIL', present)
    monkeypatch.setattr(cleaning, 'CSV_MOTOR', tmp_path / 'missing.csv')
    with patch.object(cleaning, 'clean_mobil') as car, patch.object(cleaning, 'clean_motor') as motor:
        with pytest.raises(SystemExit): cleaning.main([])
        car.assert_not_called()
        motor.assert_not_called()


@pytest.mark.parametrize('vehicle,expected', [('mobil', (1, 0)), ('motor', (0, 1)), ('all', (1, 1))])
def test_entrypoint_dispatch_only(monkeypatch, tmp_path, vehicle, expected):
    placeholder = tmp_path / 'present.csv'
    placeholder.touch()  # No dataset is generated or processed in this test.
    monkeypatch.setattr(cleaning, 'BATCH', 'test')
    monkeypatch.setattr(cleaning, 'DIR_RAW', tmp_path)
    for prefix in ['MOBIL123', 'MOMOTOR']:
        pd.DataFrame({name: ['OFFLINE_DISPATCH_FIXTURE'] for name in ['link', 'nama_mobil', 'nama_motor', 'merk', 'harga', 'Kilometer', 'Transmisi', 'Tipe Bahan Bakar', 'Bahan bakar']}).to_csv(tmp_path / f'{prefix}_raw_dataset_test.csv', index=False)
    monkeypatch.setattr(cleaning, 'CSV_MOBIL', placeholder)
    monkeypatch.setattr(cleaning, 'CSV_MOTOR', placeholder)
    with patch.object(cleaning, 'clean_mobil') as car, patch.object(cleaning, 'clean_motor') as motor:
        cleaning.main(['--vehicle', vehicle])
        assert (car.call_count, motor.call_count) == expected


@pytest.mark.parametrize('name', ['clean_mobil', 'clean_motor'])
def test_direct_function_does_not_silently_skip_missing_raw(monkeypatch, tmp_path, name):
    monkeypatch.setattr(cleaning, 'CSV_MOBIL', tmp_path / 'missing_car.csv')
    monkeypatch.setattr(cleaning, 'CSV_MOTOR', tmp_path / 'missing_motor.csv')
    with pytest.raises(FileNotFoundError, match='File RAW'):
        getattr(cleaning, name)()


def test_existing_safe_validity_fixture_and_temporary_output(tmp_path, monkeypatch):
    # Exact fixture from legacy tests/test_pipeline.py:
    # test_cleaning_retains_valid_extreme_prices_without_fitting.
    frame = pd.DataFrame({'harga': [10_000_000, 15_000_000_000, -1, np.inf],
                          'Kilometer': [50000, 1000, 100, 200]})
    kept, invalid = cleaning.pisahkan_outlier(frame, 'Kilometer')
    assert kept.harga.tolist() == [10_000_000, 15_000_000_000]
    assert len(invalid) == 2
    monkeypatch.setattr(cleaning, 'DIR_CLEAN', tmp_path / 'target')
    outputs = [cleaning.DIR_CLEAN / name for name in ['clean.csv', 'antik.csv', 'anomali.csv']]
    cleaning.simpan_hasil(kept, frame.iloc[0:0], invalid, *outputs)
    assert all(path.is_file() for path in outputs)
    pd.testing.assert_frame_equal(pd.read_csv(outputs[0]), kept.reset_index(drop=True), check_dtype=False)


def test_legacy_logic_and_safe_fixture_parity(tmp_path, monkeypatch):
    legacy_path = ROOT.parent / 'Tugas Akhir c14230205/src/Preprocessing/DataCleaning.py'
    if not legacy_path.is_file():
        pytest.skip('Read-only legacy source not available for migration parity check')
    original = load_module(legacy_path, 'legacy_cleaning_readonly')
    before = ast.parse(legacy_path.read_text(encoding='utf-8-sig'))
    after = ast.parse((ROOT / 'src/Preprocessing/DataCleaning.py').read_text(encoding='utf-8'))
    definitions = {node.name: node for node in before.body if isinstance(node, ast.FunctionDef)}
    for node in after.body:
        if not isinstance(node, ast.FunctionDef) or node.name in {'main', 'simpan_hasil', 'batch_provenance'}: continue
        old = definitions[node.name]
        if node.name in {'clean_mobil', 'clean_motor'}:
            # Only the first read_csv FileNotFoundError handler is intentionally changed.
            old.body[1].handlers = node.body[1].handlers
            # Batch provenance is preserved when a historical CLEAN was renamed.
            old.body[2] = node.body[2]
        assert ast.dump(old, include_attributes=False) == ast.dump(node, include_attributes=False), node.name
    frame = pd.DataFrame({'harga': [10_000_000, 15_000_000_000, -1, np.inf],
                          'Kilometer': [50000, 1000, 100, 200]})
    pairs = [original.pisahkan_outlier(frame.copy(), 'Kilometer'),
             cleaning.pisahkan_outlier(frame.copy(), 'Kilometer')]
    for old, new in zip(*pairs): pd.testing.assert_frame_equal(old, new, check_exact=True)
    written = []
    for label, module, (kept, invalid) in zip(['original', 'target'], [original, cleaning], pairs):
        monkeypatch.setattr(module, 'DIR_CLEAN', tmp_path / label)
        paths = [module.DIR_CLEAN / name for name in ['clean.csv', 'antik.csv', 'anomali.csv']]
        module.simpan_hasil(kept, frame.iloc[0:0], invalid, *paths)
        written.append([path.read_bytes() for path in paths])
    assert written[0] == written[1]
