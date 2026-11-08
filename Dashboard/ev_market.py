"""Display API EV market context without recalculating statistics or price."""
import math
import pandas as pd
import streamlit as st
from Dashboard.market import show_listings, show_snapshot
from src.Retraining.evidence import url_kind
from src.Training.pipeline import normalize_features


def candidate_rows(response, payload):
    """Preserve API order/scores; label identity differences, never fuzzy merge."""
    query = normalize_features(pd.DataFrame([payload])).iloc[0]
    safe = []
    for item in response.get('comparables', []):
        amount = item.get('harga_iklan')
        kind = url_kind(item.get('url'), 'mobil')
        if (not kind or not isinstance(amount, (int, float)) or isinstance(amount, bool)
                or not math.isfinite(amount) or amount <= 0):
            continue
        row = normalize_features(pd.DataFrame([dict(Merk=item.get('merk'), Model=item.get('model'),
                    Varian=item.get('varian'), JenisMesin=item.get('jenis_mesin'),
                    JenisTransmisi=item.get('jenis_transmisi'))])).iloc[0]
        differences = [field for field in ['Merk', 'Model', 'JenisMesin', 'JenisTransmisi']
                       if query[field] != row[field]]
        if query.Varian not in {'UNKNOWN', 'LAINNYA'} and query.Varian != row.Varian:
            differences.append('Varian')
        identity = ('Berbeda pada ' + ', '.join(differences) + '; bukan kendaraan identik'
                    if differences else 'Identitas kategorikal cocok; relevansi tahun/km mengikuti detail API')
        if response['market_analysis']['status'] == 'insufficient_data':
            identity = 'KANDIDAT KONTEKS — belum cukup bukti relevansi untuk statistik pasar. ' + identity
        safe.append({**item, 'url_kind': kind, 'reason': identity + '. ' + str(item.get('reason') or '')})
        if len(safe) == 5:
            break
    return safe


def render_market(response, primary, payload, rupiah):
    st.subheader('Informasi pasar EV dari API')
    mismatch = response['prediksi_harga'] != primary['prediksi_harga'] or any(
        response['rentang'][key] != primary['rentang'][key] for key in ['batas_bawah', 'batas_atas'])
    if mismatch:
        st.warning('Hasil kedua endpoint EV berbeda. Harga dan rentang utama tetap dari /predict/ev-experimental. '
                   f'Endpoint pasar mengembalikan {rupiah(response["prediksi_harga"])} '
                   f'({rupiah(response["rentang"]["batas_bawah"])}–{rupiah(response["rentang"]["batas_atas"])}). '
                   'Kategori pasar di bawah merujuk prediksi endpoint pasar, bukan angka utama.')
    market = response['market_analysis']
    st.caption(f'Status pasar: {market["status"]} · Jumlah iklan relevan: {market["jumlah_iklan"]}')
    if market['status'] == 'available':
        mean, median, sd = st.columns(3)
        mean.metric('Rata-rata harga iklan EV', rupiah(market['mean']))
        median.metric('Median harga iklan EV', rupiah(market['median']))
        sd.metric('SD harga iklan EV', rupiah(market['sd']))
        st.text('Kategori prediksi endpoint pasar: ' + str(market.get('kategori_prediksi_terhadap_pasar') or 'Belum dapat ditentukan'))
    else:
        st.warning('Iklan relevan belum mencukupi (ambang endpoint: 10). Mean, median, SD, dan kategori pasar '
                   'tidak ditampilkan; kandidat di bawah bukan bukti harga pasar kendaraan yang sama.')
    for name in ['market_review', 'category_review', 'combination_review']:
        review = response.get(name) or {}
        if review.get('required') and review.get('reason'):
            st.warning(str(review['reason']))
    st.caption('Statistik dihitung API dari iklan relevan, bukan dari lima kandidat berikut. '
               'Harga penawaran bukan harga transaksi; tidak menunjukkan sebab matematis prediksi.')
    st.subheader('Kandidat iklan EV · maksimal 5')
    rows = candidate_rows(response, payload)
    if rows:
        show_listings(rows)
    else:
        st.info('Tidak ada kandidat dengan harga dan tautan sumber valid yang tersedia.')
    metadata = response.get('evidence_metadata')
    if isinstance(metadata, dict) and metadata.get('source'):
        show_snapshot(metadata)
