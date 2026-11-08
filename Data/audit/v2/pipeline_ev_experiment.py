"""Read-only EV catalog reconstruction helpers; fitting and prediction helpers omitted."""

from __future__ import annotations

import hashlib

import re

from datetime import datetime

from urllib.parse import urlsplit, urlunsplit

import numpy as np

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

CATEGORICAL = ['Merk', 'Model', 'Varian', 'JenisMesin', 'JenisTransmisi']

NUMERIC = ['TahunUnit', 'KapasitasMesin', 'Kilometer']

FEATURES = CATEGORICAL + NUMERIC

SEED = 42

def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exactly the same normalization for training, API and demo input."""
    out = frame.copy()
    if 'Kilometer' not in out and 'Kilometer_Num' in out:
        out['Kilometer'] = out['Kilometer_Num']
    for col in CATEGORICAL:
        if col not in out:
            out[col] = 'UNKNOWN'
        out[col] = (out[col].fillna('UNKNOWN').astype(str).str.strip()
                    .str.replace(r'\s+', ' ', regex=True).str.upper().replace('', 'UNKNOWN'))
    out['Merk'] = out['Merk'].replace({'MERCEDES': 'MERCEDES-BENZ', 'LAND': 'LAND ROVER'})
    out['JenisMesin'] = out['JenisMesin'].replace({'PETROL': 'BENSIN', 'GASOLINE': 'BENSIN'})
    out['JenisTransmisi'] = out['JenisTransmisi'].replace({
        'AUTOMATIC': 'OTOMATIS', 'AUTO': 'OTOMATIS', 'MATIC': 'OTOMATIS',
        'AT': 'OTOMATIS', 'MT': 'MANUAL'})
    for col in NUMERIC:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return out[FEATURES]

def listing_identity(value) -> str:
    if pd.isna(value) or str(value).strip().lower() in {'', 'unknown', 'nan'}:
        return ''
    parsed = urlsplit(str(value).strip())
    path = parsed.path.rstrip('/')
    match = re.search(r'/([0-9]+)$', path)
    if 'mobil123.com' in parsed.netloc.lower() and match:
        return 'mobil123:' + match.group(1)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, '', ''))

def prepare_data(raw: pd.DataFrame, vehicle='mobil') -> tuple[pd.DataFrame, dict]:
    if 'link' not in raw or not ({'tanggal_scrape', 'waktu_scrape'} & set(raw)):
        raise ValueError('Metadata link/waktu scraping hilang. Jalankan ulang DataCleaning.py yang baru.')
    data = normalize_features(raw)
    data['harga'] = pd.to_numeric(raw['harga'], errors='coerce')
    valid = (np.isfinite(data['harga']) & data['harga'].gt(0)
             & data['TahunUnit'].notna() & data['Merk'].ne('UNKNOWN') & data['Model'].ne('UNKNOWN'))
    minimum, cc_min, cc_max = (2000, 500, 8000) if vehicle == 'mobil' else (2010, 50, 2000)
    valid &= data.TahunUnit.between(minimum, datetime.now().year + 1)
    valid &= data.TahunUnit.mod(1).eq(0)
    valid &= data.Kilometer.isna() | data.Kilometer.between(0, 2_000_000)
    valid &= data.KapasitasMesin.isna() | data.KapasitasMesin.between(cc_min, cc_max)
    # EXPERIMENT ONLY: EV diizinkan masuk training
    data['source_row'] = np.arange(len(raw))
    data['link'] = raw['link'].map(listing_identity)
    col = 'tanggal_scrape' if 'tanggal_scrape' in raw else 'waktu_scrape'
    data['scraped_at'] = pd.to_datetime(raw[col], errors='coerce')
    data['batch_id'] = raw.get('batch_id', pd.Series('unknown', index=raw.index))
    data = data.loc[valid].copy()
    data['feature_hash'] = [hashlib.sha256(row.to_json().encode()).hexdigest()
                            for _, row in data[FEATURES].iterrows()]
    before = len(data)
    data = data.sort_values('scraped_at', na_position='first', kind='stable')
    data = data.reset_index(drop=True)
    # Connected components: same listing OR identical payload features stay together.
    parents = list(range(len(data)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for column in ['link', 'feature_hash']:
        seen = {}
        for i, key in enumerate(data[column]):
            if not key:
                continue
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    data['group_id'] = [f'g{find(i)}' for i in range(len(data))]
    # Deduplicate AFTER unions, so removed listing IDs cannot break connections.
    data = data.drop_duplicates(['feature_hash', 'harga'], keep='last').reset_index(drop=True)
    audit = {'input_rows': len(raw), 'invalid_rows': int((~valid).sum()),
             'duplicate_payload_and_price_removed': before - len(data),
             'retained_rows': len(data), 'groups': data['group_id'].nunique(),
             'missing_listing_id': int(data['link'].eq('').sum()),
             'invalid_scrape_time': int(data['scraped_at'].isna().sum()),
             'scrape_start': str(data['scraped_at'].min()), 'scrape_end': str(data['scraped_at'].max())}
    return data, audit

def split_data(data: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    if data['group_id'].nunique() < 20:
        raise ValueError('Minimal 20 kelompok independen diperlukan untuk empat bagian evaluasi.')
    remaining, splits = data, {}
    for offset, (name, fraction) in enumerate([('test', .15), ('calibration', .10 / .85),
                                               ('validation', .15 / .75)]):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + offset)
        keep, take = next(splitter.split(remaining, groups=remaining['group_id']))
        splits[name] = remaining.iloc[take].copy()
        remaining = remaining.iloc[keep].copy()
    splits['train'] = remaining.copy()
    for i, (name, part) in enumerate(splits.items()):
        if len(part) < 10:
            raise ValueError(f'Bagian {name} terlalu kecil ({len(part)} baris).')
        for other in list(splits.values())[i + 1:]:
            for col in ['group_id', 'feature_hash']:
                if set(part[col]) & set(other[col]):
                    raise AssertionError(f'Overlap {col}')
            if (set(part['link']) - {''}) & (set(other['link']) - {''}):
                raise AssertionError('Overlap identitas iklan')
    return splits
