"""Presentation-only capture dates; API records remain unchanged."""
import copy

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from Dashboard.market import listing_capture_date


@pytest.mark.parametrize('batch,expected', [
    ('2026-04', 'April 2026'), ('2026-10', 'Oktober 2026'),
    ('April', 'April 2026'), ('Oktober', 'Oktober 2026'),
    ('2027-01', 'Januari 2027'), ('2027-02', 'Februari 2027'),
    ('2027-03', 'Maret 2027'), ('2027-05', 'Mei 2027'),
    ('2027-06', 'Juni 2027'), ('2027-07', 'Juli 2027'),
    ('2027-08', 'Agustus 2027'), ('2027-09', 'September 2027'),
    ('2027-11', 'November 2027'), ('2027-12', 'Desember 2027'),
])
def test_period_is_not_a_fabricated_capture_date(batch, expected):
    assert listing_capture_date(None, batch) == expected + ' (tanggal tidak tercatat)'


@pytest.mark.parametrize('batch', [None, '', '2026-00', '2026-13', 'unknown'])
def test_unknown_date(batch):
    assert listing_capture_date(None, batch) == 'Tanggal pengambilan tidak diketahui'


@pytest.mark.parametrize('missing', [None, '', ' ', float('nan'), pd.NA, pd.NaT])
def test_missing_date_uses_batch(missing):
    assert listing_capture_date(missing, '2026-04') == 'April 2026 (tanggal tidak tercatat)'


def test_actual_timestamp_is_preserved():
    actual = '2026-10-09 21:06:07'
    assert listing_capture_date(actual, '2026-04') == actual


def test_listing_table_renders_fallback_without_mutating_api_records():
    items = [dict(harga_iklan=100, url_kind='individual_listing', scraped_at=date, batch=batch)
             for date, batch in [('2026-10-09 21:06:07', '2026-04'),
                                 (None, '2026-04'), (None, '2026-10'),
                                 (None, 'April'), (None, 'Oktober'), (None, None)]]
    before = copy.deepcopy(items)
    def render(records):
        from Dashboard.market import show_listings
        show_listings(records)

    app = AppTest.from_function(render, args=(items,)).run()
    assert not app.exception
    assert app.dataframe[0].value['Tanggal scraping'].tolist() == [
        '2026-10-09 21:06:07', 'April 2026 (tanggal tidak tercatat)',
        'Oktober 2026 (tanggal tidak tercatat)', 'April 2026 (tanggal tidak tercatat)',
        'Oktober 2026 (tanggal tidak tercatat)', 'Tanggal pengambilan tidak diketahui',
    ]
    assert items == before
