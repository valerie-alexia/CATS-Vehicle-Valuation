"""Office demo: HTTP prediction, local catalog options and local listing search."""
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from Dashboard import demo_client
from Dashboard import ev_catalog
from Dashboard.ev_market import render_market as render_ev_market
from Dashboard.display_catalog import options_for_display
from Dashboard.market import search_view, show_listings, show_snapshot
from src.Retraining.cats_input import CatsVehicleIn
from src.Retraining.evidence import url_kind
from src.Retraining.service import active_path, _load_version, validate_payload

st.set_page_config(page_title='CATS — Demo API', page_icon=':material/directions_car:', layout='wide')
st.title('CATS · Estimasi harga kendaraan')
st.caption('Prediksi melalui HTTP API CATS · Harga penawaran marketplace, bukan harga transaksi')

CATEGORIES = ['Merk', 'Model', 'Varian', 'JenisMesin', 'JenisTransmisi']
LABELS = {'Merk': 'Merek', 'Model': 'Model', 'Varian': 'Varian',
          'JenisMesin': 'Bahan bakar', 'JenisTransmisi': 'Transmisi',
          'TahunUnit': 'Tahun kendaraan', 'Kilometer': 'Kilometer (km)', 'KapasitasMesin': 'Kapasitas mesin (cc)'}


@st.cache_data(max_entries=4)
def catalog_for(version_path):
    # The local bundle supplies dropdowns only. Prices always come from HTTP.
    return _load_version(version_path)['catalog'].copy()


def reset_after(field, vehicle):
    for following in CATEGORIES[CATEGORIES.index(field) + 1:]:
        st.session_state.pop(f'demo_{vehicle}_{following}', None)
    st.session_state.pop('demo_prediction', None)


def reset_vehicle():
    for key in list(st.session_state):
        if key.startswith(('demo_mobil_', 'demo_motor_')):
            st.session_state.pop(key, None)
    st.session_state.pop('demo_prediction', None)
    st.session_state.pop('demo_related_form', None)
    st.session_state.pop('demo_ev_prediction', None)


def reset_ev_after(field):
    for child in ev_catalog.FIELDS[ev_catalog.FIELDS.index(field)+1:]:
        st.session_state.pop(f'demo_ev_{child}', None)
        st.session_state.pop(f'demo_ev_{child}_manual', None)
    st.session_state.pop('demo_ev_prediction', None)


def open_related_search(result, vehicle):
    """Navigate with explicit filters; never request or recalculate a prediction."""
    from src.Retraining.api import VehicleSearch
    saved = st.session_state.get('demo_prediction') or {}
    query = (result.get('input_normalization') or {}).get('input_model') or saved.get('payload') or {}

    def known(field):
        value = query.get(field)
        return value if isinstance(value, str) and value.strip().casefold() not in {'', 'lainnya', 'unknown'} else None

    year = query.get('TahunUnit')
    year = int(year) if isinstance(year, (int, float)) and math.isfinite(year) and float(year).is_integer() else None
    try:
        params = VehicleSearch(vehicle_type=vehicle, merk=known('Merk'), model=known('Model'),
                               varian=known('Varian'), tahun_min=year, tahun_max=year)
    except ValueError:
        st.error('Filter kendaraan belum valid untuk pencarian terkait. Gunakan menu Cari iklan secara manual.')
        return
    # The callback runs before the next script rerun, before the search widgets
    # exist. Initialize both the form and submitted query to exactly these filters.
    st.session_state['demo_related_form'] = {
        'vehicle': vehicle,
        'values': {key: st.session_state[key] for key in st.session_state
                   if key.startswith(f'demo_{vehicle}_') or key == 'demo_with_shap'},
    }
    for suffix, value in {
        'merk': params.merk or '', 'model': params.model or '', 'varian': params.varian or '',
        'year_min': year, 'year_max': year,
        'price_min': None, 'price_max': None, 'km_max': None, 'cc_min': None, 'cc_max': None,
    }.items():
        st.session_state[f'search_{vehicle}_{suffix}'] = value
    # Reuse search_view defaults (Semua / 10), without conflicting widget defaults.
    for suffix in ('transmission', 'limit'):
        st.session_state.pop(f'search_{vehicle}_{suffix}', None)
    if vehicle == 'mobil':
        st.session_state['search_mobil_body'] = ''
    st.session_state['market_search'] = params.model_dump()
    st.session_state.pop('market_total', None)
    st.session_state['demo_mode'] = 'Cari iklan'


def rupiah(value):
    if value is None:
        return 'Tidak diketahui'
    return 'Rp ' + f'{value:,.0f}'.replace(',', '.')


def render_result(result, vehicle):
    st.subheader('Hasil estimasi dari API')
    price, low, high = st.columns(3)
    price.metric('Estimasi harga penawaran', rupiah(result['prediksi_harga']))
    low.metric('Batas bawah AI', rupiah(result['rentang']['batas_bawah']))
    high.metric('Batas atas AI', rupiah(result['rentang']['batas_atas']))
    st.caption('Rentang AI berasal dari kalibrasi residual model; berbeda dari rentang statistik iklan di bawah.')
    model = result.get('model_digunakan') or {}
    st.caption(f'Algoritma: {model.get("algoritma", "Tidak tersedia")} · Rute: {model.get("rute", "Tidak tersedia")}')
    st.caption(f'Versi model API: {result.get("versi_model", "Tidak tersedia")}')
    normal = result.get('input_normalization') or {}
    st.caption(f'Status normalisasi: {normal.get("status", "Tidak tersedia")}')
    if result.get('requires_manual_review'):
        st.warning('PERLU PEMERIKSAAN MANUAL — tinjau alasan di bawah sebelum memakai hasil appraisal.')
    else:
        st.info('Tidak ada flag review otomatis. Persetujuan appraisal tetap mengikuti kebijakan kantor.')
    reasons = list(normal.get('warnings') or []) + list(result.get('peringatan') or [])
    reasons += list(result.get('alasan_fallback_global') or [])
    review = result.get('market_review') or {}
    if review.get('reason'):
        reasons.append(review['reason'])
    for reason in dict.fromkeys(str(r) for r in reasons):
        st.warning(reason)
    with st.expander('Normalisasi dan detail model'):
        st.json({'normalisasi': normal, 'model': model, 'metode_rentang': result.get('metode_rentang'),
                 'coverage_test': result.get('coverage_test'), 'fallback_global': result.get('fallback_global')})

    with st.container(border=True):
        st.subheader('Statistik harga penawaran marketplace')
        market = result.get('market_analysis') or {}
        status = market.get('status', 'unavailable')
        count = market.get('jumlah_iklan')
        st.caption(f'Status pasar: {status} · Jumlah iklan relevan: {count if count is not None else "Tidak diketahui"}')
        if status == 'available':
            mean, median, sd = st.columns(3)
            mean.metric('Rata-rata harga iklan', rupiah(market.get('mean')))
            median.metric('Median harga iklan', rupiah(market.get('median')))
            sd.metric('Standar deviasi harga iklan', rupiah(market.get('sd')))
            st.text(f'Kategori prediksi terhadap pasar: {market.get("kategori_prediksi_terhadap_pasar") or "Belum dapat ditentukan"}')
            with st.expander('Rentang statistik iklan: mean ± SD'):
                for multiplier in ['1', '2', '3']:
                    band = (market.get('rentang_sd') or {}).get(multiplier)
                    if isinstance(band, dict):
                        st.text(f'{multiplier} SD: {rupiah(band.get("batas_bawah"))} – {rupiah(band.get("batas_atas"))}')
        elif status == 'insufficient_data':
            st.info('Sampel iklan relevan tidak cukup untuk statistik pasar. Kategori harga tidak ditentukan.')
        else:
            st.warning('Statistik marketplace tidak tersedia. Harga AI tetap ditampilkan; kategori harga tidak ditentukan.')
        st.caption('Mean ± SD adalah sebaran harga penawaran iklan, bukan rentang ketidakpastian AI atau harga transaksi.')

    with st.expander('Penjelasan fitur model · SHAP'):
        explanation = result.get('shap') or {}
        items = explanation.get('kontribusi_fitur') or result.get('shap_top') or []
        if items:
            st.caption('Kontribusi pada skala log1p(harga), bukan penyesuaian rupiah atau hubungan sebab-akibat.')
            st.dataframe(pd.DataFrame(items).astype({'nilai_input': str}) if 'nilai_input' in items[0] else pd.DataFrame(items), hide_index=True)
        else:
            st.info('SHAP tidak diminta atau belum tersedia pada respons API.')

    with st.container(border=True):
        st.subheader('Iklan pembanding · maksimal 5')
        safe = []
        for item in result.get('comparables') or []:
            if not isinstance(item, dict):
                continue
            kind = url_kind(item.get('url'), vehicle)
            amount = item.get('harga_iklan')
            if kind and isinstance(amount, (int, float)) and math.isfinite(amount) and amount > 0:
                safe.append({**item, 'url_kind': kind})
        if safe:
            show_listings(safe[:5])
        else:
            st.info('Tidak ada referensi dengan URL sumber dan harga valid yang dapat ditampilkan.')
        metadata = result.get('evidence_metadata')
        if isinstance(metadata, dict) and metadata.get('source'):
            show_snapshot(metadata)
        st.caption('Pembanding merupakan konteks pasar, bukan penyebab matematis prediksi. '
                   'Tautan adalah snapshot scraping; status aktif atau terjual belum diverifikasi. '
                   'HTTP 200 bukan bukti iklan masih tersedia.')
        st.button('Cari lebih banyak iklan', key='demo_more_listings',
                  on_click=open_related_search, args=(result, vehicle))
        st.caption('Membuka indeks lokal dengan merek, model, varian dan tahun yang diketahui. '
                   'Filter harga, kilometer, cc dan filter pencarian lama direset; harga AI tidak dihitung ulang.')


vehicle = st.selectbox('Jenis kendaraan', ['mobil', 'motor'], format_func=str.title,
                       key='demo_vehicle', on_change=reset_vehicle)
mode = st.selectbox('Fitur', ['Prediksi API', 'Cari iklan', 'EV (eksperimen)'], key='demo_mode')
if mode == 'Cari iklan':
    st.info('Cari Iklan membaca indeks CSV lokal. Mode prediksi menggunakan HTTP API CATS. Pencarian tidak menjalankan AI.')
    st.caption('Tautan adalah snapshot scraping; status aktif/terjual belum diverifikasi. HTTP 200 bukan bukti ketersediaan iklan.')
    search_view(vehicle)
    st.stop()
if mode == 'EV (eksperimen)':
    st.session_state.pop('demo_prediction', None)
    st.warning('Eksperimental — memerlukan pemeriksaan manual')
    if vehicle != 'mobil':
        st.info('EV eksperimental hanya mendukung mobil. Pilih jenis kendaraan mobil untuk melanjutkan.')
        st.stop()
    st.caption('Prediksi melalui endpoint EV khusus. Tidak memakai model mobil non-EV; '
               'fitur kesehatan baterai dan range belum tersedia pada model.')
    try:
        ev_reference = ev_catalog.fitting_catalog()
        st.caption('Katalog EV: 617 baris fitting train + validation; kombinasi diverifikasi terhadap '
                   'artefak eksperimen dan fitted encoder. Bukan seluruh CLEAN Oktober.')
    except Exception:
        ev_reference = pd.DataFrame(columns=ev_catalog.FIELDS)
        st.warning('Katalog fitting EV tidak dapat diverifikasi. Hanya input manual tersedia; '
                   'kategori mungkin tidak dikenal model. Tidak memakai katalog non-EV.')
    ev_values = {}
    for field in ev_catalog.FIELDS:
        choices = ev_catalog.options(ev_reference, ev_values, field)
        if field == 'Varian':
            choices = [value for value in choices if value != 'UNKNOWN']
            choices = ['(Kosong)', *choices]
        choices.append('Lainnya / input manual')
        key = f'demo_ev_{field}'
        if st.session_state.get(key) is not None and st.session_state[key] not in choices:
            st.session_state.pop(key, None)
        selection = st.selectbox(LABELS[field], choices, index=None, key=key,
            on_change=reset_ev_after, args=(field,),
            disabled=any(ev_values.get(parent) is None for parent in ev_catalog.FIELDS[:ev_catalog.FIELDS.index(field)]
                         if parent != 'Varian'))
        if selection == 'Lainnya / input manual':
            ev_values[field] = st.text_input(f'{LABELS[field]} EV manual', key=key+'_manual',
                                            on_change=reset_ev_after, args=(field,)).strip()
            st.warning(f'{LABELS[field]} diisi manual; kategori/kombinasi mungkin tidak dikenal model EV.')
        else:
            ev_values[field] = '' if selection == '(Kosong)' else selection
    ev_values['TahunUnit'] = st.number_input('Tahun kendaraan EV', min_value=2000,
        max_value=datetime.now().year + 1, value=None, step=1, key='demo_ev_TahunUnit')
    ev_values['Kilometer'] = st.number_input('Kilometer EV (km)', min_value=0.0,
        max_value=2000000.0, value=None, step=1.0, key='demo_ev_Kilometer')
    st.caption('Bahan bakar: ELECTRIC. KapasitasMesin dikirim null; varian boleh kosong.')
    ev_with_shap = st.checkbox('Sertakan penjelasan SHAP EV', key='demo_ev_with_shap')
    ev_payload = {'JenisAset': 'mobil', 'JenisMesin': 'ELECTRIC', 'KapasitasMesin': None,
                  **ev_values, 'Varian': ev_values['Varian'] or None, 'with_shap': ev_with_shap}
    ev_signature = json.dumps([ev_payload, demo_client.api_base_url()], sort_keys=True)
    ev_ready = all(ev_values[f] for f in ['Merk', 'Model', 'JenisTransmisi']) and all(
        ev_values[f] is not None for f in ['TahunUnit', 'Kilometer'])
    if st.button('Prediksi EV melalui API', type='primary', disabled=not ev_ready, key='demo_ev_submit'):
        st.session_state.pop('demo_ev_prediction', None)
        try:
            with st.spinner('Meminta prediksi EV eksperimental...'):
                ev_result = demo_client.predict_ev(ev_payload)
            st.session_state['demo_ev_prediction'] = {'signature': ev_signature, 'result': ev_result}
            try:
                with st.spinner('Meminta informasi pasar dan kandidat iklan EV...'):
                    market_result = demo_client.predict_ev_market(ev_payload)
                st.session_state['demo_ev_prediction']['market'] = market_result
            except demo_client.DemoAPIError as exc:
                st.session_state['demo_ev_prediction']['market_error'] = str(exc)
        except demo_client.DemoAPIError as exc:
            st.error(str(exc))
            st.caption('Pastikan API yang sudah memuat endpoint /predict/ev-experimental berjalan.')
            st.code(demo_client.START_API, language='powershell')
    ev_saved = st.session_state.get('demo_ev_prediction')
    if ev_saved and ev_saved['signature'] == ev_signature:
        result = ev_saved['result']
        price, low, high = st.columns(3)
        price.metric('Estimasi harga penawaran EV', rupiah(result['prediksi_harga']))
        low.metric('Batas bawah AI', rupiah(result['rentang']['batas_bawah']))
        high.metric('Batas atas AI', rupiah(result['rentang']['batas_atas']))
        for warning in result.get('warnings', []):
            st.warning(warning)
        st.caption(f'Model EV: {result["model_digunakan"].get("algoritma")} · '
                   f'{result.get("versi_model", "Eksperimen")}')
        # Market context is separate; never render non-EV SHAP or replace the primary price.
        if ev_saved.get('market') is not None:
            render_ev_market(ev_saved['market'], result, ev_payload, rupiah)
        elif ev_saved.get('market_error'):
            st.warning('Informasi pasar EV tidak tersedia; harga dan rentang utama tetap ditampilkan. '
                       + ev_saved['market_error'])
        if ev_with_shap:
            with st.expander('Penjelasan prediksi EV — SHAP'):
                explanation = result.get('shap') or {}
                items = explanation.get('kontribusi_fitur') or []
                if result.get('shap_status') == 'available' and items:
                    st.caption('Kontribusi pada skala log1p(harga), bukan Rupiah. Positif menaikkan dan '
                               'negatif menurunkan output relatif terhadap base value; bukan hubungan sebab-akibat.')
                    st.dataframe(pd.DataFrame(items).astype({'nilai_input': str}), hide_index=True)
                    st.caption(f'Base value (log): {explanation["baseline_log_harga"]:.6f} · '
                               f'Output model (log): {explanation["prediksi_log_harga"]:.6f} · '
                               f'Galat rekonstruksi (log): {explanation["galat_rekonstruksi_log"]:.8f}')
                    st.caption(explanation.get('catatan', ''))
                else:
                    st.info('Penjelasan SHAP EV tidak tersedia. Harga, rentang AI, dan konteks pasar tetap dapat digunakan.')
    elif ev_saved:
        st.info('Input EV berubah. Tekan Prediksi EV melalui API untuk hasil yang sesuai.')
    st.stop()

related_form = st.session_state.pop('demo_related_form', None)
if related_form and related_form['vehicle'] == vehicle:
    # Streamlit cleans up widgets hidden on another view. Restore the captured
    # inputs before rendering them, so returning can show the saved result safely.
    st.session_state.update(related_form['values'])

try:
    version_path = str(active_path(vehicle))
    catalog = catalog_for(version_path)
except Exception:
    st.error('Katalog model aktif lokal belum dapat dibaca. Periksa artefak model tanpa menjalankan training ulang.')
    st.stop()

st.subheader('Identitas kendaraan')
st.caption('Katalog lokal hanya untuk pilihan. Angka prediksi dan normalisasi dikembalikan oleh API.')
selected = {}
for field in CATEGORIES:
    options = options_for_display(catalog, selected, field)
    key = f'demo_{vehicle}_{field}'
    if st.session_state.get(key) is not None and st.session_state[key] not in options:
        st.session_state.pop(key, None)
    selected[field] = st.selectbox(LABELS[field], options, index=None,
        placeholder='Pilih atau ketik untuk mencari', accept_new_options=False,
        disabled=any(selected.get(parent) is None for parent in CATEGORIES[:CATEGORIES.index(field)]),
        key=key, on_change=reset_after, args=(field, vehicle))
if any(value == 'Lainnya' for value in selected.values()):
    st.info('Lainnya berarti atribut tidak diketahui. API dapat memakai fallback global dan meminta review; varian tidak ditebak.')

st.subheader('Pemakaian dan spesifikasi')
numeric = {}
for field, lower, upper, integer in [
    ('TahunUnit', 2000 if vehicle == 'mobil' else 2010, datetime.now().year + 1, True),
    ('Kilometer', 0.0, 2000000.0, False),
    ('KapasitasMesin', 500.0 if vehicle == 'mobil' else 50.0, 8000.0 if vehicle == 'mobil' else 2000.0, False),
]:
    unknown = st.checkbox(f'{LABELS[field]} tidak diketahui (Lainnya)', key=f'demo_{vehicle}_{field}_unknown')
    value = st.number_input(LABELS[field], min_value=lower, max_value=upper, value=None,
                            step=1 if integer else 1.0, disabled=unknown, key=f'demo_{vehicle}_{field}')
    numeric[field] = 'Lainnya' if unknown else value
if any(value == 'Lainnya' for value in numeric.values()):
    st.caption('Numerik Lainnya dikirim sebagai sentinel; median training mungkin dipakai oleh model, bukan dianggap data aktual kendaraan.')
with_shap = st.checkbox('Sertakan penjelasan SHAP', key='demo_with_shap')
payload = {'AsetBergerakID': f'DEMO-{vehicle.upper()}', 'JenisAsetID': 1 if vehicle == 'mobil' else 5,
           **selected, **numeric, 'with_shap': with_shap}
signature = json.dumps([payload, version_path, demo_client.api_base_url()], sort_keys=True)
ready = all(value is not None for value in selected.values()) and all(value is not None for value in numeric.values())
if not ready:
    st.info('Lengkapi pilihan dan angka yang kosong, atau pilih Lainnya secara eksplisit, sebelum meminta prediksi.')
submit = st.button('Prediksi melalui API', type='primary', disabled=not ready, key='demo_submit')
if submit:
    st.session_state.pop('demo_prediction', None)
    try:
        # Local schema/range validation only; this never calculates a price.
        CatsVehicleIn(**payload)
        validate_payload({'JenisAset': vehicle, **selected, **numeric})
    except ValueError:
        st.error('Input belum valid. Periksa tahun, kilometer, kapasitas mesin dan pilihan kategori.')
    else:
        try:
            with st.spinner('Meminta prediksi, normalisasi dan analisis pasar dari API...'):
                result = demo_client.predict_market(payload)
            st.session_state['demo_prediction'] = {'signature': signature, 'result': result, 'payload': dict(payload)}
        except demo_client.DemoAPIError as exc:
            st.error(str(exc))
            st.caption('Jalankan API demo dari root proyek. Untuk port lain, sesuaikan CATS_DEMO_API_URL.')
            st.code(demo_client.START_API, language='powershell')
saved = st.session_state.get('demo_prediction')
if saved and saved['signature'] == signature:
    render_result(saved['result'], vehicle)
elif saved:
    st.info('Input berubah. Tekan Prediksi melalui API untuk memperoleh hasil yang sesuai.')
