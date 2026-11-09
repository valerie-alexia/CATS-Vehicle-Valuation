"""Momotor offline CLI tests: no real Selenium, browser, or network."""
import builtins
import datetime as real_datetime
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.Scraping import scrape_momotor as scraper

SCRIPT = Path(scraper.__file__).resolve()
LINK = 'https://momotor.id/motor-bekas/offline-fixture/1'


def node(text):
    return SimpleNamespace(get_text=lambda *a, **k: text, text=text)


class FrozenDateTime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None): return cls(2026, 11, 9, 12, 0, 0, tzinfo=tz)


@pytest.fixture(autouse=True)
def no_live_io(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', Mock(side_effect=AssertionError('Network forbidden')))
    monkeypatch.setattr(scraper, '_create_driver', Mock(side_effect=AssertionError('Real browser forbidden')))
    monkeypatch.setattr(scraper.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(scraper, 'datetime', SimpleNamespace(datetime=FrozenDateTime,
                                                           timedelta=real_datetime.timedelta))


class Driver:
    def __init__(self, page_for):
        self.page_for = page_for
        self.page_source = None
        self.visits = []
        self.closed = False
        self.clicks = 0

    def get(self, url):
        self.visits.append(url)
        self.page_source = self.page_for(url)

    def find_element(self, *args): return node('2022')

    def execute_script(self, script, *args):
        if 'arguments[0].click()' in script:
            self.clicks += 1
            self.page_source = self.page_for(self.visits[-1])

    def quit(self): self.closed = True


def install_mocks(monkeypatch, driver):
    class Wait:
        def __init__(self, *args): pass
        def until(self, locator):
            if locator[0] == 'id': return node('2022')
            if locator[0] == 'xpath':
                return SimpleNamespace(get_attribute=lambda name: None if name == 'disabled' else '')
            return object()

    deps = (pd, lambda content, parser: content,
            SimpleNamespace(ID='id', TAG_NAME='tag', XPATH='xpath'), Wait,
            SimpleNamespace(presence_of_element_located=lambda value: value), object())
    monkeypatch.setattr(scraper, '_load_dependencies', lambda: deps)
    monkeypatch.setattr(scraper, '_create_driver', lambda uc: driver)


def detail_page(empty=False, mobile_price=False):
    page = Mock()
    page.get_text.return_value = '' if empty else 'Offline synthetic motor listing'
    info = Mock()
    items = []
    for label, value in [('Bahan bakar', 'Bensin'), ('Kilometer', '30.000 - 35.000 KM'),
                         ('Transmisi', 'Manual'), ('Kapasitas mesin', '110 cc'),
                         ('Penjual', 'FIXTURE LOKAL')]:
        item = Mock()
        item.find.side_effect = lambda name=None, label=label, value=value, **kw: node(
            label if kw.get('class_') == 'caption' else value)
        items.append(item)

    def info_find(name, **kw):
        if name == 'h1': return node('Honda Revo FIT (2020)')
        if name == 'p': return node('HONDA')
        if name == 'div': return node('LOKASI FIKTIF')
        return None

    info.find.side_effect = info_find
    info.find_all.return_value = items

    def find(name=None, **kw):
        if empty: return None
        if name == 'div': return info
        if name == 'h2': return None if mobile_price else node('Rp 10.000.000')
        if name == 'p': return node('Rp 10.000.000')
        return None

    page.find.side_effect = find
    return page


def test_import_has_no_browser_dependency_or_file_write(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kw):
        assert name.split('.')[0] not in {'selenium', 'undetected_chromedriver', 'bs4', 'pandas'}
        return original(name, *args, **kw)

    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(Path, 'mkdir', Mock(side_effect=AssertionError('Unexpected directory')))
    spec = importlib.util.spec_from_file_location('momotor_import_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    exec(compile(SCRIPT.read_text(encoding='utf-8'), str(SCRIPT), 'exec'), module.__dict__)


def test_cli_help_without_scraper_packages():
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert all(s in result.stdout for s in ['--batch', '--stage', '--output-dir'])


@pytest.mark.parametrize('override', [False, True])
def test_configuration(monkeypatch, tmp_path, override):
    monkeypatch.setenv('CATS_BATCH', '2026-11')
    monkeypatch.setenv('CATS_RAW_DIR', str(tmp_path / 'env'))
    run = Mock()
    monkeypatch.setattr(scraper, 'run_scraper', run)
    args = ['--stage', 'detail']
    if override: args += ['--batch', 'April', '--output-dir', str(tmp_path / 'cli')]
    scraper.main(args)
    run.assert_called_once_with('April' if override else '2026-11',
                               tmp_path / ('cli' if override else 'env'), 'detail')


def test_defaults_match_cleaner_and_mobil123(monkeypatch):
    for name in ['CATS_BATCH', 'CATS_RAW_DIR']: monkeypatch.delenv(name, raising=False)
    run = Mock()
    monkeypatch.setattr(scraper, 'run_scraper', run)
    scraper.main([])
    run.assert_called_once_with(__import__('datetime').datetime.now().strftime('%Y-%m'), scraper.ROOT / 'Data/raw', 'all')


@pytest.mark.parametrize('batch', ['', '../escape', 'a/b', 'a\\b'])
def test_bad_batch_rejected_before_execution(monkeypatch, batch):
    run = Mock()
    monkeypatch.setattr(scraper, 'run_scraper', run)
    with pytest.raises(SystemExit) as exc: scraper.main(['--batch', batch])
    assert exc.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize('text,expected', [(None, 'Unknown'), ('Unknown', 'Unknown'),
    ('hari ini', '2026-11-09'), ('3 hari yang lalu', '2026-11-06'),
    ('25 okt', '2026-10-25'), ('31 feb', '31 feb'), ('TIDAK DIKETAHUI', 'tidak diketahui')])
def test_posting_date_parser(text, expected):
    assert scraper.konversi_tanggal_momotor(text) == expected


def test_missing_links_stops_before_dependencies(monkeypatch, tmp_path):
    dependencies = Mock(side_effect=AssertionError('Unexpected dependency load'))
    monkeypatch.setattr(scraper, '_load_dependencies', dependencies)
    output = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError): scraper.run_scraper('Oktober', output, 'detail')
    dependencies.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize('mobile_price', [False, True])
def test_raw_schema_resume_autosave_and_final_flush(monkeypatch, tmp_path, mobile_price):
    links = [f'https://momotor.id/motor-bekas/offline-fixture/{i}' for i in range(26)]
    pd.DataFrame({'link': links, 'tanggal_posting': '2026-11-06', 'terjual': 0}).to_csv(
        tmp_path / 'MOMOTOR_daftar_link_Oktober.csv', index=False)
    output = tmp_path / 'MOMOTOR_raw_dataset_Oktober.csv'
    pd.DataFrame({'link': links[:5], 'nama_motor': 'Honda Revo FIT (2020)',
                  'harga': 'Rp 10.000.000'}).to_csv(output, index=False)
    driver = Driver(lambda url: detail_page(mobile_price=mobile_price))
    install_mocks(monkeypatch, driver)
    original = pd.DataFrame.to_csv
    writes = []

    def capture(frame, path, *args, **kwargs):
        writes.append((Path(path), len(frame)))
        return original(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'to_csv', capture)
    scraper.run_scraper('Oktober', tmp_path, 'detail')
    raw = pd.read_csv(output)
    assert driver.visits == links[5:] and driver.closed
    assert [n for _, n in writes] == [25, 26]
    assert all(p.parent == output.parent and p.name.startswith('.' + output.name + '.') for p, _ in writes)
    assert len(raw) == 26 and raw.link.is_unique
    assert set(raw.columns) == {'link', 'tanggal_posting', 'waktu_scrape', 'nama_motor',
        'merk', 'lokasi', 'Bahan bakar', 'Kilometer', 'Transmisi', 'Kapasitas mesin', 'Penjual', 'harga'}
    row = raw.iloc[-1]
    assert row['waktu_scrape'] == '2026-11-09' and row['tanggal_posting'] == '2026-11-06'
    assert row['harga'] == 'Rp 10.000.000' and row['Kapasitas mesin'] == '110 cc'
    assert row['Kilometer'] == '30.000 - 35.000 KM'


def test_load_more_deduplication_relative_urls_and_existing_links(monkeypatch, tmp_path):
    pd.DataFrame({'link': [LINK], 'tanggal_posting': ['2026-11-01'], 'terjual': [0]}).to_csv(
        tmp_path / 'MOMOTOR_daftar_link_Oktober.csv', index=False)

    class Anchor(dict):
        def find(self, *args, **kwargs): return node('3 hari yang lalu')

    driver = None

    def page_for(url):
        page = Mock()
        page.get_text.return_value = 'Offline search fixture'
        page.find.return_value = node('Motor bekas (2 pencarian)')
        anchors = [Anchor(href=LINK)]
        if driver.clicks:
            anchors += [Anchor(href=LINK), Anchor(href='/motor-bekas/offline-fixture/2')]
        page.find_all.return_value = anchors
        return page

    driver = Driver(page_for)
    install_mocks(monkeypatch, driver)
    scraper.run_scraper('Oktober', tmp_path, 'links')
    data = pd.read_csv(tmp_path / 'MOMOTOR_daftar_link_Oktober.csv')
    assert data.link.tolist() == [LINK, 'https://momotor.id/motor-bekas/offline-fixture/2']
    assert data.tanggal_posting.tolist() == ['2026-11-01', '2026-11-06']
    assert driver.clicks == 1 and driver.closed
    assert not (tmp_path / 'MOMOTOR_raw_dataset_Oktober.csv').exists()


def test_empty_or_changed_html_retains_legacy_unknown_fallback(monkeypatch, tmp_path):
    pd.DataFrame({'link': [LINK], 'tanggal_posting': ['Unknown'], 'terjual': [0]}).to_csv(
        tmp_path / 'MOMOTOR_daftar_link_Oktober.csv', index=False)
    driver = Driver(lambda url: detail_page(empty=True))
    install_mocks(monkeypatch, driver)
    scraper.run_scraper('Oktober', tmp_path, 'detail')
    raw = pd.read_csv(tmp_path / 'MOMOTOR_raw_dataset_Oktober.csv')
    assert raw.harga.tolist() == ['Unknown']
    assert 'nama_motor' not in raw.columns and 'merk' not in raw.columns
    assert driver.closed  # Inherited fallback is NOT a guarantee of cleaner-ready RAW.


def test_navigation_failure_does_not_create_raw_and_closes_driver(monkeypatch, tmp_path):
    pd.DataFrame({'link': [LINK], 'tanggal_posting': ['Unknown'], 'terjual': [0]}).to_csv(
        tmp_path / 'MOMOTOR_daftar_link_Oktober.csv', index=False)

    def fail(url): raise OSError('Offline simulated navigation failure')

    driver = Driver(fail)
    install_mocks(monkeypatch, driver)
    scraper.run_scraper('Oktober', tmp_path, 'detail')
    assert driver.closed
    assert not (tmp_path / 'MOMOTOR_raw_dataset_Oktober.csv').exists()


@pytest.mark.parametrize('stage', ['links', 'detail', 'all'])
def test_access_restriction_stops_without_retry(monkeypatch, tmp_path, stage):
    pd.DataFrame({'link': [LINK], 'tanggal_posting': ['Unknown'], 'terjual': [0]}).to_csv(
        tmp_path / 'MOMOTOR_daftar_link_Oktober.csv', index=False)
    page = Mock()
    page.get_text.return_value = 'CAPTCHA Access denied: verify you are human'
    driver = Driver(lambda url: page)
    install_mocks(monkeypatch, driver)
    with pytest.raises(scraper.AccessControlError): scraper.run_scraper('Oktober', tmp_path, stage)
    assert len(driver.visits) == 1 and driver.closed


def test_invalid_links_schema_is_reported(monkeypatch, tmp_path):
    pd.DataFrame({'wrong': [LINK]}).to_csv(tmp_path / 'MOMOTOR_daftar_link_Oktober.csv', index=False)
    driver = Driver(lambda url: detail_page())
    install_mocks(monkeypatch, driver)
    with pytest.raises(ValueError, match='kolom link'): scraper.run_scraper('Oktober', tmp_path, 'detail')
    assert driver.closed and not driver.visits


def test_missing_dependency_has_clear_cli_error(monkeypatch, tmp_path, capsys):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'bs4': raise ImportError('Offline dependency check')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing)
    output = tmp_path / 'absent'
    with pytest.raises(SystemExit) as exc:
        scraper.main(['--stage', 'links', '--output-dir', str(output)])
    assert exc.value.code == 1 and 'Dependency scraper belum tersedia' in capsys.readouterr().err
    assert not output.exists()


def test_optional_real_html_parser_uses_local_fixture_only():
    bs4 = pytest.importorskip('bs4', reason='beautifulsoup4 unavailable; no dependency installation authorized')
    page = scraper._page_soup(SimpleNamespace(page_source='<html><h1>Honda Revo FIT (2020)</h1></html>'),
                              bs4.BeautifulSoup)
    assert page.find('h1').get_text(strip=True) == 'Honda Revo FIT (2020)'
