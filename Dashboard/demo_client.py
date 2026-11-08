"""Small HTTP-only client for the demo API; never runs local inference."""
import json
import math
import os
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_API_URL = 'http://127.0.0.1:8000'
START_API = ('python -m uvicorn cats_demo_app:app --app-dir Data/audit/v2 '
             '--host 127.0.0.1 --port 8000')


class DemoAPIError(Exception):
    """A sanitized, user-facing failure without raw server response contents."""


def api_base_url():
    return os.environ.get('CATS_DEMO_API_URL', DEFAULT_API_URL).strip().rstrip('/')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def predict_market(payload, *, base_url=None, timeout=30.0):
    return _predict(payload, base_url=base_url, timeout=timeout)


def predict_ev(payload, *, base_url=None, timeout=30.0):
    # Never use the non-EV endpoint or invent engine capacity.
    return _predict({**payload, 'KapasitasMesin': None}, base_url=base_url, timeout=timeout, ev=True)


def predict_ev_market(payload, *, base_url=None, timeout=30.0):
    # No fabricated CATS asset ID. JenisAsetID is omitted: endpoint defaults to car.
    fields = ['Merk', 'Model', 'Varian', 'TahunUnit', 'JenisTransmisi', 'Kilometer']
    body = {key: payload.get(key) for key in fields}
    body.update(JenisMesin='ELECTRIC', KapasitasMesin=None)
    return _predict(body, base_url=base_url, timeout=timeout, ev=True, ev_market=True)


def _predict(payload, *, base_url=None, timeout=30.0, ev=False, ev_market=False):
    base = api_base_url() if base_url is None else base_url.strip().rstrip('/')
    try:
        parsed = urlsplit(base)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or any(c.isspace() for c in base)):
            raise ValueError('URL')
        parsed.port  # Validate invalid port syntax before performing I/O.
        if not _number(timeout) or timeout <= 0:
            raise ValueError('timeout')
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError):
        raise DemoAPIError('Konfigurasi API atau payload tidak valid; periksa URL, timeout dan angka input.') from None
    endpoint = '/cats/predict-market-ev' if ev_market else ('/predict/ev-experimental' if ev else '/cats/predict-market')
    request = Request(base + endpoint, data=body,
                      headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        messages = {
            422: 'Input ditolak API (HTTP 422). Periksa atribut dan rentang angka kendaraan.',
            404: 'Endpoint demo tidak ditemukan (HTTP 404). Jalankan cats_demo_app, bukan hanya API resmi.',
            503: 'Model atau layanan API belum tersedia (HTTP 503). Prediksi belum berhasil.',
        }
        raise DemoAPIError(messages.get(exc.code, f'API gagal memproses permintaan (HTTP {exc.code}).')) from None
    except (URLError, TimeoutError, OSError, HTTPException):
        raise DemoAPIError('Tidak dapat menghubungi API atau waktu tunggu habis. Pastikan server demo berjalan.') from None
    except (ValueError, UnicodeError):
        raise DemoAPIError('Respons API tidak berisi JSON yang valid.') from None
    try:
        if (not isinstance(result, dict) or not _number(result['prediksi_harga'])
                or result['prediksi_harga'] <= 0):
            raise ValueError('price')
        band = result['rentang']
        if (not _number(band['batas_bawah']) or not _number(band['batas_atas'])
                or not 0 <= band['batas_bawah'] <= band['batas_atas']):
            raise ValueError('range')
        if ev:
            if (result.get('experiment_only') is not True
                    or not isinstance(result.get('requires_manual_review'), bool)
                    or not isinstance(result.get('model_digunakan'), dict)
                    or result.get('model_digunakan', {}).get('rute') != 'mobil-ev-specialist-experiment'):
                raise ValueError('EV contract')
            if ev_market:
                market = result['market_analysis']
                if (not isinstance(market, dict) or market.get('status') not in {'available', 'insufficient_data'}
                        or not isinstance(market.get('jumlah_iklan'), int) or isinstance(market['jumlah_iklan'], bool)
                        or market['jumlah_iklan'] < 0 or not isinstance(result.get('comparables'), list)
                        or not all(isinstance(item, dict) for item in result['comparables'])):
                    raise ValueError('EV market contract')
                if market['status'] == 'available' and (market['jumlah_iklan'] == 0
                        or any(not _number(market.get(k)) or market[k] < 0 for k in ['mean', 'median', 'sd'])):
                    raise ValueError('EV market statistics')
            elif (result.get('requires_manual_review') is not True
                    or not isinstance(result.get('warnings'), list)
                    or not all(isinstance(w, str) for w in result['warnings'])):
                raise ValueError('EV warnings')
        elif (not isinstance(result['requires_manual_review'], bool)
                or not isinstance(result['model_digunakan'], dict)
                or not isinstance(result['input_normalization'], dict)
                or not isinstance(result['market_analysis'], dict)):
            raise ValueError('contract')
    except (KeyError, TypeError, ValueError):
        raise DemoAPIError('Struktur respons API tidak sesuai kontrak demo; hasil tidak ditampilkan.') from None
    return result
