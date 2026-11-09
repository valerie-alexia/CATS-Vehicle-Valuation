"""Offline monthly batch, resume and failure regression."""
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src import data_batch
from src.Scraping import scrape_mobil123, scrape_momotor


@pytest.mark.parametrize('module', [scrape_mobil123, scrape_momotor])
def test_month_rollover_and_batch_frozen_at_start(monkeypatch, tmp_path, module):
    monkeypatch.delenv('CATS_BATCH', raising=False)
    clock = Mock()
    clock.now.return_value = datetime(2026, 11, 30, 23, 59)
    monkeypatch.setattr(data_batch, 'datetime', clock)
    batches = []
    def run(batch, directory, stage):
        clock.now.return_value = datetime(2026, 12, 1)
        batches.append(batch)  # The running invocation retains November.
    monkeypatch.setattr(module, 'run_scraper', run)
    module.main(['--output-dir', str(tmp_path)])
    module.main(['--output-dir', str(tmp_path)])
    module.main(['--output-dir', str(tmp_path)])
    module.main(['--batch', '2026-04', '--output-dir', str(tmp_path)])
    assert batches == ['2026-11', '2026-12', '2026-12', '2026-04']


def test_cleaner_selects_raw_not_calendar_and_rejects_mixed_batches(tmp_path):
    for prefix in ['MOBIL123', 'MOMOTOR']:
        for batch in ['2026-04', '2026-10']:
            (tmp_path / f'{prefix}_raw_dataset_{batch}.csv').write_text('link\nfixture\n')
    assert data_batch.latest_raw_batch(tmp_path, ['MOBIL123', 'MOMOTOR']) == '2026-10'
    (tmp_path / 'MOBIL123_raw_dataset_2026-11.csv').write_text('link\nfixture\n')
    with pytest.raises(ValueError, match='berbeda'):
        data_batch.latest_raw_batch(tmp_path, ['MOBIL123', 'MOMOTOR'])
    assert data_batch.latest_raw_batch(tmp_path, ['MOBIL123']) == '2026-11'


def test_legacy_year_is_not_inferred(tmp_path):
    (tmp_path / 'MOBIL123_raw_dataset_April.csv').write_text('link\nfixture\n')
    with pytest.raises(ValueError, match='CATS_BATCH'):
        data_batch.latest_raw_batch(tmp_path, ['MOBIL123'])


def test_atomic_resume_dedup_and_distinct_months(tmp_path):
    old = tmp_path / 'raw_2026-10.csv'
    new = tmp_path / 'raw_2026-11.csv'
    data_batch.save_scrape(pd.DataFrame({'link': ['a', 'b'], 'harga': [1, 2]}), old)
    original = old.read_bytes()
    data_batch.save_scrape(pd.DataFrame({'link': ['a', 'a']}), new, append=True)
    data_batch.save_scrape(pd.DataFrame({'link': ['a', 'c']}), new, append=True)
    assert pd.read_csv(new).link.tolist() == ['a', 'c']
    assert old.read_bytes() == original


@pytest.mark.parametrize('failure', ['serialize', 'validate', 'replace'])
def test_failed_regeneration_preserves_existing_clean(monkeypatch, tmp_path, failure):
    paths = [tmp_path / f'clean_{n}.csv' for n in range(3)]
    for path in paths:
        path.write_bytes(b'link\noriginal\n')
    if failure == 'serialize':
        monkeypatch.setattr(pd.DataFrame, 'to_csv', Mock(side_effect=OSError('disk full')))
    elif failure == 'validate':
        monkeypatch.setattr(pd, 'read_csv', Mock(return_value=pd.DataFrame({'wrong': []})))
    else:
        replace = data_batch.os.replace
        calls = 0
        def interrupted(source, dest):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('interrupted publication')
            return replace(source, dest)
        monkeypatch.setattr(data_batch.os, 'replace', interrupted)
    with pytest.raises((OSError, ValueError)):
        data_batch.atomic_csvs([(p, pd.DataFrame({'link': ['new']})) for p in paths])
    assert all(p.read_bytes() == b'link\noriginal\n' for p in paths)
    assert not list(tmp_path.glob('*.tmp'))


def test_scrape_write_failure_preserves_prior_rows(monkeypatch, tmp_path):
    path = tmp_path / 'raw.csv'
    path.write_bytes(b'link\na\n')
    monkeypatch.setattr(data_batch.os, 'replace', Mock(side_effect=OSError('write failed')))
    with pytest.raises(OSError):
        data_batch.save_scrape(pd.DataFrame({'link': ['b']}), path, append=True)
    assert path.read_bytes() == b'link\na\n'


@pytest.mark.parametrize('site', ['mobil123', 'momotor'])
def test_detail_restart_after_failure_and_duplicate_queue(monkeypatch, tmp_path, site):
    # Reuse the pre-existing safe offline fixtures, never real marketplace rows.
    import importlib
    fixtures = importlib.import_module('test_scrape_' + site)
    module = fixtures.scraper
    prefix = site.upper()
    links = [fixtures.LINK + 'a', fixtures.LINK + 'b']
    pd.DataFrame({'link': [links[0], links[0], links[1]], 'terjual': 0}).to_csv(
        tmp_path / f'{prefix}_daftar_link_2026-11.csv', index=False)
    def page(url):
        if url == links[1]:
            raise module.AccessControlError('Offline interruption fixture')
        return fixtures.detail_page()
    driver = fixtures.Driver(page)
    install = fixtures.install_browser_mocks if site == 'mobil123' else fixtures.install_mocks
    install(monkeypatch, driver)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    with pytest.raises(module.AccessControlError):
        module.run_scraper('2026-11', tmp_path, 'detail')
    output = tmp_path / f'{prefix}_raw_dataset_2026-11.csv'
    assert pd.read_csv(output).link.tolist() == [links[0]]
    assert driver.closed
    driver = fixtures.Driver(lambda url: fixtures.detail_page())
    install(monkeypatch, driver)
    module.run_scraper('2026-11', tmp_path, 'detail')
    assert driver.visits == [links[1]]
    assert driver.closed
    assert pd.read_csv(output).link.tolist() == links


def test_historical_provenance_retained_but_new_month_is_canonical(tmp_path, monkeypatch):
    from src.Preprocessing import DataCleaning as cleaning
    path = tmp_path / 'CLEAN_MOBIL123_2026-04.csv'
    monkeypatch.setattr(cleaning, 'BATCH', '2026-04')
    pd.DataFrame({'batch_id': ['April']}).to_csv(path, index=False)
    assert cleaning.batch_provenance(path) == 'April'
    monkeypatch.setattr(cleaning, 'BATCH', '2026-11')
    assert cleaning.batch_provenance(path) == '2026-11'


@pytest.mark.parametrize('content', ['link\n', 'wrong\nvalue\n'])
def test_unready_raw_does_not_start_cleaning(tmp_path, monkeypatch, content):
    from src.Preprocessing import DataCleaning as cleaning
    (tmp_path / 'MOBIL123_raw_dataset_2026-11.csv').write_text(content)
    monkeypatch.setattr(cleaning, 'BATCH', None)
    monkeypatch.setattr(cleaning, 'DIR_RAW', tmp_path)
    clean = Mock()
    monkeypatch.setattr(cleaning, 'clean_mobil', clean)
    with pytest.raises(SystemExit) as exc:
        cleaning.main(['--vehicle', 'mobil'])
    assert exc.value.code == 2
    clean.assert_not_called()
