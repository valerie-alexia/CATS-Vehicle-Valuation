"""Streamlit views over the shared, read-only marketplace index."""
import logging
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.Retraining.evidence import get_index
from Dashboard.display_catalog import HIDDEN_VARIANTS

logger = logging.getLogger(__name__)


def listing_capture_date(scraped_at, batch=None):
    """Format capture provenance for display without inventing a per-listing date."""
    if pd.notna(scraped_at) and str(scraped_at).strip():
        return scraped_at
    months = ('Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
              'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember')
    if isinstance(batch, str):
        batch = batch.strip()
        # These legacy snapshots were confirmed as 2026 during the naming migration.
        batch = {'April': '2026-04', 'Oktober': '2026-10'}.get(batch, batch)
        match = re.fullmatch(r'(\d{4})-(0[1-9]|1[0-2])', batch)
        if match:
            year, month = match.groups()
            return f'{months[int(month) - 1]} {year} (tanggal tidak tercatat)'
    return 'Tanggal pengambilan tidak diketahui'


def show_snapshot(metadata):
    if not metadata:
        return
    start = metadata.get('snapshot_start') or 'tidak tersedia'
    end = metadata.get('snapshot_end') or 'tidak tersedia'
    st.caption(f'Sumber: {metadata["source"]} | Snapshot bertanggal: {start} — {end}. '
               'Waktu mengikuti catatan sumber. Jika tanggal iklan tidak tercatat, '
               'periode batch ditampilkan bila tersedia.')
    st.caption('Harga adalah penawaran iklan, bukan harga transaksi. Tautan belum diperiksa aktif; '
               'tautan kategori/unknown bukan halaman iklan individual.')


def show_listings(items):
    data = pd.DataFrame(items)
    data['scraped_at'] = [listing_capture_date(item.get('scraped_at'), item.get('batch'))
                          for item in items]
    columns = {'merk': 'Merek', 'model': 'Model', 'varian': 'Varian', 'tahun': 'Tahun',
               'kilometer': 'Kilometer', 'jenis_transmisi': 'Transmisi',
               'kapasitas_mesin': 'Mesin (cc)', 'tipe_bodi': 'Tipe bodi',
               'harga_iklan': 'Harga iklan', 'source': 'Sumber', 'scraped_at': 'Tanggal scraping',
               'url_kind': 'Jenis tautan', 'url': 'Tautan sumber',
               'similarity_score': 'Skor kecocokan', 'match_level': 'Tingkat kecocokan', 'reason': 'Alasan'}
    data = data[[col for col in columns if col in data]].rename(columns=columns)
    data['Harga iklan'] = data['Harga iklan'].map(lambda value: 'Rp ' + f'{value:,.0f}'.replace(',', '.'))
    data['Jenis tautan'] = data['Jenis tautan'].replace({'individual_listing': 'Pola iklan individual',
                                                       'category_or_unknown': 'Kategori / belum diketahui'})
    if 'Tingkat kecocokan' in data:
        data['Tingkat kecocokan'] = data['Tingkat kecocokan'].replace({
            'same_brand_model_variant': 'Merek, model, varian sama',
            'same_brand_model': 'Merek dan model sama',
            'same_brand_alternative': 'Alternatif merek sama',
            'cross_brand_alternative': 'Alternatif merek berbeda'})
    st.dataframe(data, hide_index=True, column_config={
        'Tautan sumber': st.column_config.LinkColumn(display_text='Buka sumber'),
        'Tahun': st.column_config.NumberColumn(format='%d'),
        'Skor kecocokan': st.column_config.NumberColumn(format='%.3f'),
    })


def find_evidence(vehicle, payload):
    try:
        return get_index().comparables(vehicle, payload)
    except Exception as exc:
        logger.warning('Dashboard evidence unavailable (%s)', type(exc).__name__)
        return {'comparables': [], 'evidence_status': 'unavailable', 'evidence_metadata': None}


def show_evidence(result):
    with st.container(border=True):
        st.subheader('Iklan pembanding dari dataset')
        st.caption('Pembanding memberi konteks pasar. Kontribusi fitur model dijelaskan oleh SHAP; '
                   'iklan ini bukan penyebab matematis angka prediksi. Satu iklan ekstrem tidak membuktikan harga wajar.')
        if result.get('evidence_status', 'unavailable') == 'unavailable':
            st.warning('Data pembanding sementara tidak tersedia. Hasil prediksi tetap dapat digunakan.')
        elif not result.get('comparables'):
            st.info('Belum ditemukan iklan pembanding yang cukup relevan dalam snapshot data.')
        else:
            show_listings(result['comparables'])
            st.caption('Skor 0–1 mengukur kecocokan atribut, bukan probabilitas atau akurasi prediksi. '
                       'Merek/model/varian sama didahulukan; alternatif diberi tingkat kecocokan lebih rendah.')
        show_snapshot(result.get('evidence_metadata'))


def reset_search(vehicle, parent=None):
    """Callbacks run before widgets render; clear submitted query and pagination."""
    st.session_state.pop('market_search', None)
    st.session_state.pop('market_total', None)
    fields = ('model', 'varian') if parent == 'merk' else ('varian',) if parent == 'model' else ()
    for key in list(st.session_state):
        if (parent is None and key.startswith(f'search_{vehicle}_')) or key in {
                f'search_{vehicle}_{field}' for field in fields}:
            st.session_state.pop(key, None)


def search_category(index, vehicle, field, merk=None, model=None):
    options = list(index.category_options(vehicle, field, merk, model))
    if field == 'Varian':
        hidden = {tuple(v.casefold() for v in row) for row in HIDDEN_VARIANTS}
        options = [v for v in options if (str(merk).casefold(), str(model).casefold(), v.casefold()) not in hidden]
    key = f'search_{vehicle}_{field.lower()}'
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    labels = {'Merk': 'Semua merek', 'Model': 'Semua model', 'Varian': 'Semua varian'}
    return st.selectbox({'Merk': 'Merek (opsional)', 'Model': 'Model (opsional)', 'Varian': 'Varian (opsional)'}[field],
                        [None] + options, key=key, format_func=lambda v: labels[field] if v is None else v,
                        on_change=reset_search, args=(vehicle, field.lower()))


def search_view(vehicle):
    st.subheader('Cari iklan berdasarkan budget dan spesifikasi')
    st.caption('Pencarian memakai harga asli iklan dari dataset scraping. Kosongkan filter yang tidak diperlukan. '
               'Filter harga dan tahun tidak diperluas secara otomatis.')
    st.button('Hapus filter', key=f'clear_search_{vehicle}', on_click=reset_search, args=(vehicle,))
    try:
        index = get_index()
    except Exception as exc:
        logger.warning('Dashboard search index unavailable (%s)', type(exc).__name__)
        st.error('Indeks marketplace tidak tersedia. Silakan coba lagi.')
        return
    # Cascades must rerun outside st.form; no dataset reread or AI prediction.
    merk = search_category(index, vehicle, 'Merk')
    model = search_category(index, vehicle, 'Model', merk) if merk else None
    varian = search_category(index, vehicle, 'Varian', merk, model) if merk and model else None
    if not merk:
        st.caption('Pilih merek untuk mempersempit pilihan model dan varian.')
    with st.form(f'market_filters_{vehicle}'):
        left, right = st.columns(2)
        with left:
            body = st.text_input('Tipe bodi (opsional)', placeholder='Contoh: SUV atau MPV',
                                 key='search_mobil_body') if vehicle == 'mobil' else None
            transmission = st.selectbox('Transmisi', ['Semua', 'Otomatis', 'Manual'],
                                        key=f'search_{vehicle}_transmission')
            year_min = st.number_input('Tahun minimum', min_value=1900, max_value=datetime.now().year + 1,
                                       value=None, step=1, key=f'search_{vehicle}_year_min')
            year_max = st.number_input('Tahun maksimum', min_value=1900, max_value=datetime.now().year + 1,
                                       value=None, step=1, key=f'search_{vehicle}_year_max')
        with right:
            price_min = st.number_input('Harga minimum (Rp)', min_value=0, value=None, step=1000000,
                                        key=f'search_{vehicle}_price_min')
            price_max = st.number_input('Harga maksimum (Rp)', min_value=0, value=None, step=1000000,
                                        key=f'search_{vehicle}_price_max')
            km_max = st.number_input('Kilometer maksimum', min_value=0, value=None, step=1000,
                                     key=f'search_{vehicle}_km_max')
            cc_min = st.number_input('Kapasitas mesin minimum (cc)', min_value=1, value=None, step=50,
                                     key=f'search_{vehicle}_cc_min')
            cc_max = st.number_input('Kapasitas mesin maksimum (cc)', min_value=1, value=None, step=50,
                                     key=f'search_{vehicle}_cc_max')
            limit = st.selectbox('Hasil per halaman', [5, 10, 20, 50], index=1, key=f'search_{vehicle}_limit')
        submitted = st.form_submit_button('Cari iklan', type='primary')

    if submitted:
        from src.Retraining.api import VehicleSearch
        try:
            params = VehicleSearch(vehicle_type=vehicle, merk=merk,
                model=model, varian=varian,
                tipe_bodi=body.strip() or None if body is not None else None,
                jenis_transmisi=None if transmission == 'Semua' else transmission,
                tahun_min=year_min, tahun_max=year_max, harga_min=price_min, harga_max=price_max,
                kilometer_max=km_max, kapasitas_mesin_min=cc_min, kapasitas_mesin_max=cc_max, limit=limit)
            st.session_state['market_search'] = params.model_dump()
            st.session_state.pop('market_total', None)
        except ValidationError:
            st.session_state.pop('market_search', None)
            st.error('Periksa rentang filter: nilai minimum harus lebih kecil atau sama dengan maksimum.')

    saved = st.session_state.get('market_search')
    if not saved or saved['vehicle_type'] != vehicle:
        return
    filters = {key: value for key, value in saved.items() if key != 'vehicle_type'}
    try:
        with st.spinner('Mencari iklan dalam snapshot data...'):
            result = index.search(vehicle, **filters)
        st.session_state['market_total'] = result['total']
        prev, following = st.columns(2)
        if prev.button('Sebelumnya', key='search_previous', disabled=saved['offset'] == 0):
            saved['offset'] = max(0, saved['offset'] - saved['limit'])
            st.rerun()
        if following.button('Berikutnya', key='search_next',
                            disabled=saved['offset'] + saved['limit'] >= result['total']):
            saved['offset'] += saved['limit']
            st.rerun()
        st.subheader('Filter kategori perlu diperiksa' if result.get('search_status') in {
            'ambiguous_alias', 'unrecognized_category'} else f'{result["total"]:,} iklan ditemukan')
        with st.expander('Filter hasil yang sedang ditampilkan'):
            st.json({key: value for key, value in saved.items() if value is not None})
            if result.get('search_normalization', {}).get('changes'):
                st.json(result['search_normalization'])
        if result['items']:
            st.caption(f'Menampilkan {saved["offset"] + 1}–{saved["offset"] + len(result["items"])} '
                       f'dari {result["total"]:,} iklan, diurutkan harga termurah.')
            show_listings(result['items'])
        else:
            status = result.get('search_status')
            if status == 'ambiguous_alias':
                st.warning('Alias kategori ambigu; tidak dipilih otomatis. Pilih kategori dari dropdown.')
            elif status == 'unrecognized_category':
                st.warning('Merek/model/varian tidak dikenali dalam indeks. Periksa filter aktif atau pilih dari dropdown.')
            else:
                st.info('Tidak ada iklan yang memenuhi seluruh filter. Kategori dikenali; ubah filter lalu tekan Cari iklan.')
        show_snapshot(result['metadata'])
    except Exception as exc:
        logger.warning('Dashboard search unavailable (%s)', type(exc).__name__)
        st.error('Data pencarian sementara tidak tersedia. Silakan coba lagi.')
