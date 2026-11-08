"""Read-only marketplace evidence. Ranking never affects model inference."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src.Training.pipeline import FEATURES, listing_identity, normalize_features

ROOT = Path(__file__).resolve().parents[2]
MISSING = {'', 'UNKNOWN', 'NAN', 'NONE', 'LAINNYA', 'N/A'}


def text(value) -> str:
    if value is None or pd.isna(value):
        return ''
    value = re.sub(r'\s+', ' ', str(value).strip()).upper()
    return '' if value in MISSING else value


def url_kind(value, vehicle):
    """Validate source hosts; classify only recognized individual-page patterns."""
    if not isinstance(value, str):
        return None
    try:
        p = urlsplit(value.strip())
        hosts = {'mobil': {'mobil123.com', 'www.mobil123.com'},
                 'motor': {'momotor.id', 'www.momotor.id'}}
        if (p.scheme not in {'http', 'https'} or p.hostname not in hosts[vehicle]
                or p.username or p.password or p.port not in {None, 80, 443}
                or any(c.isspace() or ord(c) < 32 for c in value)):
            return None
        pattern = r'/dijual/[^/]+/\d+/?' if vehicle == 'mobil' else r'/motor-bekas/[^/]+/[^/]+/?'
        return 'individual_listing' if re.fullmatch(pattern, p.path) else 'category_or_unknown'
    except ValueError:
        return None


class EvidenceIndex:
    """Immutable after construction; synthetic frames can be injected in tests."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = {}
        for vehicle, raw in frames.items():
            if vehicle not in {'mobil', 'motor'}:
                raise ValueError('vehicle_type harus mobil atau motor')
            raw = raw.reset_index(drop=True)
            data = normalize_features(raw)
            for col in ['Merk', 'Model', 'Varian', 'JenisMesin', 'JenisTransmisi']:
                data[col] = data[col].map(text)
            data['TipeBodi'] = raw.get('TipeBodi', pd.Series('', index=raw.index)).map(text)
            data['harga'] = pd.to_numeric(raw.get('harga', pd.Series(index=raw.index, dtype=float)), errors='coerce')
            data['url'] = raw.get('link', pd.Series('', index=raw.index))
            data['url_kind'] = data.url.map(lambda v: url_kind(v, vehicle))
            dates = raw.get('tanggal_scrape', pd.Series(None, index=raw.index, dtype=object))
            dates = dates.where(dates.notna(), raw.get('waktu_scrape', pd.Series(None, index=raw.index, dtype=object)))
            data['scraped_at'] = pd.to_datetime(dates, errors='coerce', utc=True, format='mixed')
            # Preserve the source's timestamp text; naive source times have no known timezone.
            data['scraped_label'] = dates.where(data.scraped_at.notna(), None)
            data['batch'] = raw.get('batch_id', pd.Series(None, index=raw.index, dtype=object))
            for col in ['Kilometer', 'KapasitasMesin']:
                data.loc[data[col].lt(0) | ~np.isfinite(data[col]), col] = np.nan
            valid = (data.url_kind.notna() & np.isfinite(data.harga) & data.harga.gt(0)
                     & data.Merk.ne('') & data.Model.ne(''))
            data = data.loc[valid].copy()
            data['fingerprint'] = (data[FEATURES + ['TipeBodi', 'harga']].astype(str).agg('|'.join, axis=1)
                                   if len(data) else pd.Series(index=data.index, dtype=str))
            data['identity'] = [listing_identity(u) if k == 'individual_listing' else 'row:' + f
                                for u, k, f in zip(data.url, data.url_kind, data.fingerprint)]
            # Newest observation wins; lexical ties make order independent of input order.
            data = data.sort_values(['scraped_at', 'url', 'fingerprint'], na_position='first', kind='stable')
            data = data.drop_duplicates('identity', keep='last').drop_duplicates('fingerprint', keep='last')
            data['id'] = data.identity.map(lambda s: hashlib.sha256((vehicle + ':' + s).encode()).hexdigest())
            self.frames[vehicle] = data.sort_values('id', kind='stable').reset_index(drop=True)

    def metadata(self, vehicle):
        data = self.frames[vehicle]
        dates = data.scraped_at.dropna()
        return {'source': 'Mobil123' if vehicle == 'mobil' else 'Momotor',
                'snapshot_start': str(data.loc[dates.idxmin(), 'scraped_label']) if len(dates) else None,
                'snapshot_end': str(data.loc[dates.idxmax(), 'scraped_label']) if len(dates) else None,
                'timestamp_timezone': 'as_recorded_in_source; unspecified_if_no_offset',
                'records': len(data), 'missing_scrape_dates': int(data.scraped_at.isna().sum()),
                'category_or_unknown_urls': int(data.url_kind.eq('category_or_unknown').sum()),
                'price_type': 'advertised_asking_price', 'url_live_status': 'not_verified',
                'note': 'Referensi pasar, bukan penyebab prediksi atau bukti harga transaksi. '
                        'Satu iklan, termasuk harga ekstrem, tidak membuktikan kewajaran harga.'}

    @staticmethod
    def records(data, vehicle):
        result = []
        mapping = {'Merk': 'merk', 'Model': 'model', 'Varian': 'varian', 'TahunUnit': 'tahun',
                   'Kilometer': 'kilometer', 'JenisTransmisi': 'jenis_transmisi',
                   'JenisMesin': 'jenis_mesin', 'KapasitasMesin': 'kapasitas_mesin',
                   'TipeBodi': 'tipe_bodi', 'harga': 'harga_iklan', 'url': 'url',
                   'url_kind': 'url_kind', 'batch': 'batch', 'id': 'id'}
        for _, row in data.iterrows():
            item = {target: None if pd.isna(row[col]) or row[col] == '' else row[col]
                    for col, target in mapping.items()}
            item.update(vehicle_type=vehicle, source='Mobil123' if vehicle == 'mobil' else 'Momotor',
                        scraped_at=str(row.scraped_label) if pd.notna(row.scraped_at) else None)
            result.append(item)
        return result

    def comparables(self, vehicle, payload, limit=5):
        if not 1 <= limit <= 10:
            raise ValueError('limit harus 1 sampai 10')
        q = normalize_features(pd.DataFrame([payload])).iloc[0].to_dict()
        for field in ['Merk', 'Model', 'Varian', 'JenisMesin', 'JenisTransmisi']:
            q[field] = text(q[field])
        body = text(payload.get('TipeBodi'))
        data = self.frames[vehicle].copy()
        brand = data.Merk.eq(q['Merk']) & bool(q['Merk'])
        model = brand & data.Model.eq(q['Model']) & bool(q['Model'])
        variant = model & data.Varian.eq(q['Varian']) & bool(q['Varian'])
        # Cross-model alternatives require known body, engine, fuel, transmission and year.
        # Motor data has no reliable category, so cross-model motorcycle fallback is disabled.
        alternative = pd.Series(False, index=data.index)
        if vehicle == 'mobil' and body and q['JenisMesin'] and q['JenisTransmisi']:
            cc, year = q['KapasitasMesin'], q['TahunUnit']
            if pd.notna(cc) and cc > 0 and pd.notna(year):
                alternative = (data.TipeBodi.eq(body) & data.JenisMesin.eq(q['JenisMesin'])
                               & data.JenisTransmisi.eq(q['JenisTransmisi'])
                               & data.KapasitasMesin.sub(cc).abs().le(cc * .2)
                               & data.TahunUnit.sub(year).abs().le(3))
        data['tier'] = np.select([variant, model, brand & alternative, alternative], [1, 2, 3, 4], default=5)
        data = data.loc[data.tier.lt(5)].copy()
        # Fixed denominator: missing values contribute zero, never an artificial perfect match.
        score = (data.Merk.eq(q['Merk']) * .20 + data.Model.eq(q['Model']) * bool(q['Model']) * .25
                 + data.Varian.eq(q['Varian']) * bool(q['Varian']) * .10
                 + data.JenisTransmisi.eq(q['JenisTransmisi']) * bool(q['JenisTransmisi']) * .10
                 + data.TipeBodi.eq(body) * bool(body) * .05)
        for field, weight, scale in [('TahunUnit', .15, 10), ('Kilometer', .10, 100000),
                                     ('KapasitasMesin', .05, max(q['KapasitasMesin'], 1) if pd.notna(q['KapasitasMesin']) else 1)]:
            if pd.notna(q[field]):
                score += weight * (1 - data[field].sub(q[field]).abs() / scale).clip(0, 1).fillna(0)
        data['score'] = score
        data = data.sort_values(['tier', 'score', 'id'], ascending=[True, False, True], kind='stable').head(limit)
        records = self.records(data, vehicle)
        levels = {1: 'same_brand_model_variant', 2: 'same_brand_model',
                  3: 'same_brand_alternative', 4: 'cross_brand_alternative'}
        for item, (_, row) in zip(records, data.iterrows()):
            reason = ('Merek, model, dan varian sama' if row.tier == 1 else 'Merek dan model sama') if row.tier <= 2 else (
                'Alternatif kecocokan lebih rendah: tipe bodi, bahan bakar dan transmisi sama; kapasitas mesin berbeda maksimal 20%')
            for field, label, unit in [('TahunUnit', 'tahun', 'tahun'), ('Kilometer', 'kilometer', 'km')]:
                if pd.notna(q[field]) and pd.notna(row[field]):
                    reason += f'; selisih {label} {abs(row[field] - q[field]):g} {unit}'
            item.update(similarity_score=round(float(row.score), 6), match_level=levels[row.tier], reason=reason)
        return {'comparables': records, 'evidence_status': 'available' if records else 'insufficient_comparables',
                'evidence_metadata': self.metadata(vehicle)}

    def search(self, vehicle, *, limit=10, offset=0, **filters):
        data = self.frames[vehicle]
        filters = dict(filters)
        changes, issues = [], []
        for name, col in [('merk', 'Merk'), ('model', 'Model'), ('varian', 'Varian')]:
            value = filters.get(name)
            if value is None:
                continue
            values = sorted(set(data[col]) - {''})
            target = text(value)
            matches = [v for v in values if v == target]
            # Do not erase meaningful punctuation/numbers in variants or numeric
            # model names. Letter-only models allow spaces/hyphens, within brand.
            if not matches and name != 'varian' and (name == 'merk' or filters.get('merk')):
                def key(v):
                    return re.sub(r'[\s-]', '', v) if re.fullmatch(r'[A-Z\s-]+', v) else v
                matches = [v for v in values if key(v) == key(target)]
            if len(matches) == 1:
                canonical = matches[0]
                if canonical != value:
                    changes.append({'field': name, 'from': value, 'to': canonical})
                filters[name] = canonical
                data = data.loc[data[col].eq(canonical)]
            else:
                issues.append({'field': name, 'status': 'ambiguous_alias' if len(matches) > 1 else 'unrecognized_category'})
                data = data.iloc[:0]
                break
        for name, col in [('merk', 'Merk'), ('model', 'Model'), ('varian', 'Varian'),
                          ('tipe_bodi', 'TipeBodi'), ('jenis_transmisi', 'JenisTransmisi')]:
            value = filters.get(name)
            if value is not None:
                normalized = normalize_features(pd.DataFrame([{col: value}])).iloc[0][col] if col in FEATURES else value
                data = data.loc[data[col].eq(text(normalized))]
        for name, col, op in [('tahun_min', 'TahunUnit', 'ge'), ('tahun_max', 'TahunUnit', 'le'),
                              ('harga_min', 'harga', 'ge'), ('harga_max', 'harga', 'le'),
                              ('kilometer_max', 'Kilometer', 'le'),
                              ('kapasitas_mesin_min', 'KapasitasMesin', 'ge'),
                              ('kapasitas_mesin_max', 'KapasitasMesin', 'le')]:
            if filters.get(name) is not None:
                data = data.loc[getattr(data[col], op)(filters[name])]
        data = data.sort_values(['harga', 'id'], kind='stable')
        return {'items': self.records(data.iloc[offset:offset + limit], vehicle), 'total': len(data),
                'limit': limit, 'offset': offset, 'metadata': self.metadata(vehicle),
                'search_status': issues[0]['status'] if issues else ('available' if len(data) else 'no_matching_listings'),
                'search_normalization': {'changes': changes, 'issues': issues},
                'applied_filters': filters}

    @lru_cache(maxsize=512)
    def category_options(self, vehicle, field, merk=None, model=None):
        """Cached index categories, independent of numeric filters or inference."""
        if field not in {'Merk', 'Model', 'Varian'}:
            raise ValueError('Unsupported search category')
        data = self.frames[vehicle]
        if merk is not None:
            data = data.loc[data.Merk.eq(merk)]
        if model is not None:
            data = data.loc[data.Model.eq(model)]
        return tuple(sorted(set(data[field]) - {''}))


@lru_cache(maxsize=1)
def get_index():
    """Load once per worker. Restart or explicitly clear cache to refresh snapshots."""
    directory = Path(os.environ.get('VEHICLE_EVIDENCE_DIR', str(ROOT / 'Data/clean')))
    frames = {}
    for vehicle, prefix in [('mobil', 'MOBIL123'), ('motor', 'MOMOTOR')]:
        parts = []
        for path in sorted(directory.glob(f'CLEAN_{prefix}_*.csv')):
            raw = pd.read_csv(path, low_memory=False)
            if 'link' in raw:
                parts.append(raw)
        frames[vehicle] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return EvidenceIndex(frames)
