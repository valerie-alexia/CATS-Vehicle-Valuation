"""Offline-only Mobil123 CLI tests; browser and transport are always mocked."""
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.Scraping import scrape_mobil123 as scraper

SCRIPT = Path(scraper.__file__).resolve()
LINK = 'https://www.mobil123.com/dijual/offline-fixture/1'


def node(text):
    return SimpleNamespace(get_text=lambda *a, **k: text)


def detail_page():
    page = Mock()
    page.get_text.return_value = 'Offline synthetic Toyota Avanza G listing'
    key_items = []
    for label, value in [('Tahun Kendaraan', '2022'), ('Kilometer', '30 - 35K km'),
                         ('Cakupan mesin', '1500 cc'), ('Transmisi', 'Automatic'),
                         ('Tipe Bahan Bakar', 'Bensin')]:
        item = Mock()
        item.find.side_effect = lambda *a, label=label, value=value, **kw: node(
            label if kw.get('class_') == 'u-text-7' else value)
        key_items.append(item)
    specs = Mock()
    spec_row = Mock()
    spec_row.find_all.return_value = [node('Pintu'), node('5')]
    specs.find_all.return_value = [spec_row]
    equipment = Mock()
    equipment_row = Mock()
    equipment_row.find_all.return_value = [node('ABS')]
    equipment.find_all.return_value = [equipment_row]

    def find(name=None, **kw):
        if name == 'h1': return node('2022 Toyota Avanza G')
        if name == 'div' and kw.get('id') == 'tab-specifications': return specs
        if name == 'div' and kw.get('id') is not None: return equipment
        if name is None and 'class_' in kw: return node('Rp 191.000.000')
        return None

    page.find.side_effect = find
    page.find_all.side_effect = lambda name=None, **kw: key_items if name == 'div' else []
    return page


class Driver:
    def __init__(self, page_for):
        self.page_for = page_for
        self.page_source = None
        self.visits = []
        self.closed = False

    def get(self, url):
        self.visits.append(url)
        self.page_source = self.page_for(url)

    def quit(self):
        self.closed = True


def install_browser_mocks(monkeypatch, driver):
    class Wait:
        def __init__(self, *a): pass
        def until(self, *a): return object()

    deps = (pd, lambda content, parser: content,
            SimpleNamespace(ID='id', CLASS_NAME='class'), Wait,
            SimpleNamespace(presence_of_element_located=lambda value: value), object())
    monkeypatch.setattr(scraper, '_load_dependencies', lambda: deps)
    monkeypatch.setattr(scraper, '_create_driver', lambda uc: driver)
    monkeypatch.setattr(scraper.time, 'sleep', lambda seconds: None)


def test_import_does_not_load_browser_dependencies_or_write(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *a, **kw):
        assert name.split('.')[0] not in {'selenium', 'undetected_chromedriver', 'bs4', 'pandas'}
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(Path, 'mkdir', Mock(side_effect=AssertionError('Unexpected write')))
    spec = importlib.util.spec_from_file_location('offline_import_check', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    exec(compile(SCRIPT.read_text(encoding='utf-8'), str(SCRIPT), 'exec'), module.__dict__)


def test_cli_help_runs_without_browser_packages():
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert all(option in result.stdout for option in ['--batch', '--stage', '--output-dir'])


@pytest.mark.parametrize('override', [False, True])
def test_batch_and_output_configuration(monkeypatch, tmp_path, override):
    monkeypatch.setenv('CATS_BATCH', '2026-11')
    monkeypatch.setenv('CATS_RAW_DIR', str(tmp_path / 'environment'))
    run = Mock()
    monkeypatch.setattr(scraper, 'run_scraper', run)
    argv = ['--stage', 'links']
    if override: argv += ['--batch', 'April', '--output-dir', str(tmp_path / 'cli')]
    scraper.main(argv)
    run.assert_called_once_with('April' if override else '2026-11',
                               tmp_path / ('cli' if override else 'environment'), 'links')


def test_default_configuration_matches_cleaner(monkeypatch):
    for key in ['CATS_BATCH', 'CATS_RAW_DIR']: monkeypatch.delenv(key, raising=False)
    run = Mock()
    monkeypatch.setattr(scraper, 'run_scraper', run)
    scraper.main([])
    run.assert_called_once_with(__import__('datetime').datetime.now().strftime('%Y-%m'), scraper.ROOT / 'Data/raw', 'all')


@pytest.mark.parametrize('batch', ['../escape', '', 'a/b', 'a\\b'])
def test_invalid_batch_rejected_before_any_execution(monkeypatch, batch):
    run = Mock()
    monkeypatch.setattr(scraper, 'run_scraper', run)
    with pytest.raises(SystemExit) as exc: scraper.main(['--batch', batch])
    assert exc.value.code == 2
    run.assert_not_called()


def test_missing_links_does_not_load_browser_or_create_output(monkeypatch, tmp_path):
    deps = Mock(side_effect=AssertionError('Unexpected dependencies/browser'))
    monkeypatch.setattr(scraper, '_load_dependencies', deps)
    out = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError): scraper.run_scraper('Oktober', out, 'detail')
    deps.assert_not_called()
    assert not out.exists()


def test_missing_dependency_exits_without_creating_output_or_browser(monkeypatch, tmp_path, capsys):
    real_import = builtins.__import__

    def missing_bs4(name, *args, **kwargs):
        if name == 'bs4': raise ImportError('offline dependency check')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing_bs4)
    browser = Mock(side_effect=AssertionError('Unexpected browser startup'))
    monkeypatch.setattr(scraper, '_create_driver', browser)
    output = tmp_path / 'not-created'
    with pytest.raises(SystemExit) as exc:
        scraper.main(['--stage', 'links', '--output-dir', str(output)])
    assert exc.value.code == 1
    assert 'Dependency scraper belum tersedia' in capsys.readouterr().err
    browser.assert_not_called()
    assert not output.exists()


def test_detail_schema_resume_and_autosave(monkeypatch, tmp_path):
    links = [f'https://www.mobil123.com/dijual/offline-fixture/{i}' for i in range(25)]
    pd.DataFrame({'link': links, 'terjual': 0}).to_csv(tmp_path / 'MOBIL123_daftar_link_Oktober.csv', index=False)
    output = tmp_path / 'MOBIL123_raw_dataset_Oktober.csv'
    pd.DataFrame({'link': links[:5], 'nama_mobil': '2022 Toyota Avanza G',
                  'harga': 'Rp 191.000.000'}).to_csv(output, index=False)
    driver = Driver(lambda url: detail_page())
    install_browser_mocks(monkeypatch, driver)
    writes = []
    original = pd.DataFrame.to_csv

    def record_write(frame, path, *a, **kw):
        writes.append(Path(path))
        return original(frame, path, *a, **kw)

    monkeypatch.setattr(pd.DataFrame, 'to_csv', record_write)
    scraper.run_scraper('Oktober', tmp_path, 'detail')
    raw = pd.read_csv(output)
    assert driver.visits == links[5:]
    assert driver.closed
    assert len(raw) == 25 and raw.link.is_unique
    assert len(writes) == 1  # Twenty records are autosaved via a staged sibling file.
    assert writes[0].parent == output.parent and writes[0].name.startswith('.' + output.name + '.')
    assert {'link', 'tanggal_scrape', 'nama_mobil', 'harga', 'Tahun Kendaraan', 'Kilometer',
            'Cakupan mesin', 'Transmisi', 'Tipe Bahan Bakar', 'Pintu', 'Fitur_ABS',
            'tanggal_update'} == set(raw.columns)
    saved = raw.iloc[-1]
    assert saved['harga'] == 'Rp 191.000.000'
    assert saved['Kilometer'] == '30 - 35K km'
    assert saved['Cakupan mesin'] == '1500 cc'
    assert saved['Fitur_ABS'] == 1
    assert isinstance(saved['tanggal_scrape'], str)


def test_links_stage_deduplicates_and_does_not_fetch_details(monkeypatch, tmp_path):
    def article():
        a = {'href': LINK}
        heading = Mock()
        heading.find.return_value = a
        art = Mock()
        art.find.side_effect = lambda name=None, **kw: heading if name == 'h2' else 'Rp'
        return art

    def page_for(url):
        page = Mock()
        page.get_text.return_value = 'Offline fixture search page'
        section = Mock()
        section.find_all.return_value = [] if 'page_number=2' in url else [article(), article()]
        page.find.side_effect = lambda name=None, **kw: section if name == 'section' else None
        page.find_all.return_value = ['2022 (2)']
        return page

    driver = Driver(page_for)
    install_browser_mocks(monkeypatch, driver)
    scraper.run_scraper('Oktober', tmp_path, 'links')
    links = pd.read_csv(tmp_path / 'MOBIL123_daftar_link_Oktober.csv')
    assert links.link.tolist() == [LINK]
    assert not (tmp_path / 'MOBIL123_raw_dataset_Oktober.csv').exists()
    assert LINK not in driver.visits
    assert driver.closed


@pytest.mark.parametrize('stage', ['links', 'detail', 'all'])
def test_access_control_stops_without_retry(monkeypatch, tmp_path, stage):
    pd.DataFrame({'link': [LINK], 'terjual': [0]}).to_csv(
        tmp_path / 'MOBIL123_daftar_link_Oktober.csv', index=False)
    page = Mock()
    page.get_text.return_value = 'Access denied. CAPTCHA: verify you are human'
    driver = Driver(lambda url: page)
    install_browser_mocks(monkeypatch, driver)
    with pytest.raises(scraper.AccessControlError): scraper.run_scraper('Oktober', tmp_path, stage)
    assert len(driver.visits) == 1
    assert driver.closed
    assert not (tmp_path / 'MOBIL123_raw_dataset_Oktober.csv').exists()


def test_real_html_parser_with_local_synthetic_fixture():
    bs4 = pytest.importorskip('bs4', reason='beautifulsoup4 is not installed; dependency unchanged by task')
    html = '<html><h1>2022 Toyota Avanza G</h1><span class="price">Rp 191.000.000</span></html>'
    page = scraper._page_soup(SimpleNamespace(page_source=html), bs4.BeautifulSoup)
    assert page.find('h1').get_text(strip=True) == '2022 Toyota Avanza G'
    assert page.find(class_='price').get_text(strip=True) == 'Rp 191.000.000'
