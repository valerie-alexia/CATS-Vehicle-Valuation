"""Deterministic CATS boundary adapter; never modifies training normalization."""
from __future__ import annotations

import math
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.Training.pipeline import FEATURES, normalize_features
from src.Retraining.service import validate_payload

EMPTY = {'', 'NULL', 'NONE', 'N/A', 'UNKNOWN', '-', 'LAINNYA'}


def clean_text(value):
    """Only explicitly marked license annotations are removed. Never delete a bare B."""
    value = re.sub(r'\s+', ' ', str(value).strip()).upper()
    value = re.sub(r'\s*\(\s*(?:NOPOL|NO\.?\s*POL(?:ISI)?|PLAT(?: NOMOR)?)\b[^)]*\)', '', value)
    value = re.sub(r'\s+\b(?:NOPOL|NO\.?\s*POL(?:ISI)?|PLAT(?: NOMOR)?)\b.*$', '', value)
    return value.strip()


def parse_number(value, field):
    if value is None or isinstance(value, str) and value.strip().upper() in EMPTY:
        return 'Lainnya'
    if isinstance(value, bool):
        raise ValueError(f'{field} harus angka, bukan boolean')
    if isinstance(value, str):
        value = value.strip().lower()
        # Units are field-specific; cc is never inferred from an unlabelled litre value.
        factor = 1
        if field == 'KapasitasMesin' and re.search(r'\s*(?:l|liter)$', value):
            factor = 1000
            value = re.sub(r'\s*(?:l|liter)$', '', value)
        else:
            unit = {'KapasitasMesin': r'\s*(?:cc|cm3|cm³)$', 'Kilometer': r'\s*km$'}.get(field)
            if unit:
                value = re.sub(unit, '', value)
        # Explicit Indonesian formatting. Comma + exactly 3 decimals is ambiguous: reject.
        if re.fullmatch(r'\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?', value):
            value = value.replace('.', '').replace(',', '.')
        elif re.fullmatch(r'\d+(?:[.,]\d{1,2})?', value):
            value = value.replace(',', '.')
        else:
            raise ValueError(f'{field}: gunakan angka polos atau format Indonesia yang tidak ambigu')
        number = float(value) * factor
    else:
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError(f'{field} harus angka') from None
    if not math.isfinite(number):
        raise ValueError(f'{field} harus angka finite')
    return number


class CatsVehicleIn(BaseModel):
    # Only these fields can ever reach the predictor. Other database columns are ignored.
    AsetBergerakID: int | str | None = None
    JenisAsetID: int = Field(ge=0)
    JenisAset: Literal['mobil', 'motor'] | None = None
    Merk: str
    Model: str
    Varian: str | None = None
    TahunUnit: int | Literal['Lainnya']
    JenisMesin: str
    JenisTransmisi: str
    KapasitasMesin: float | Literal['Lainnya']
    Kilometer: float | Literal['Lainnya']
    KapasitasPenumpang: float | None = None
    TipeBodi: str | None = None
    Warna: str | None = None
    with_shap: bool = False

    @field_validator('JenisAsetID', mode='before')
    @classmethod
    def asset_code(cls, value):
        if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value).strip()):
            raise ValueError('JenisAsetID harus kode integer nonnegatif')
        return int(str(value).strip())

    @field_validator('TahunUnit', 'KapasitasMesin', 'Kilometer', mode='before')
    @classmethod
    def numbers(cls, value, info):
        return parse_number(value, info.field_name)

    @field_validator('KapasitasPenumpang', mode='before')
    @classmethod
    def seats(cls, value):
        result = parse_number(value, 'KapasitasPenumpang')
        if result == 'Lainnya':
            return None
        if result <= 0 or not result.is_integer():
            raise ValueError('KapasitasPenumpang harus integer positif')
        return result

    @field_validator('Varian', 'TipeBodi', 'Warna', mode='before')
    @classmethod
    def optional_text(cls, value):
        if value is None or isinstance(value, str) and value.strip().upper() in EMPTY:
            return None
        return value


def compact(value):
    return re.sub(r'[^A-Z0-9]', '', value)


def engine_capacity_support(payload, catalog):
    """Observational support, never a manufacturer specification or accuracy claim.

    Exact equality permits only floating representation error (1e-6 cc). No
    physical tolerance or rounding is accepted as support. All unobserved values
    require verification, including possible rounded inputs. Catalog rows are
    deduplicated fitting feature combinations, NOT sample counts; no frequency
    threshold can safely be inferred. Year/km are deliberately not dependencies.
    Fuel/transmission restrict the segment when those categories are known.
    """
    result = {'status': 'not_assessed', 'requires_review': True,
              'observed_cc': [], 'catalog_feature_rows': 0,
              'sample_count': None, 'source': 'active_bundle_catalog',
              'note': 'Nilai teramati pada katalog fitting, bukan spesifikasi resmi pabrikan atau bukti akurasi.'}
    cc = payload.get('KapasitasMesin')
    if not isinstance(cc, (int, float)) or not math.isfinite(cc):
        return result, 'Dukungan kapasitas mesin tidak dapat diperiksa karena cc tidak diketahui.'
    segment = catalog
    for field in ['Merk', 'Model', 'Varian']:
        value = payload.get(field)
        if value is None or str(value).upper() in EMPTY:
            return result, 'Dukungan kapasitas mesin tidak dapat disimpulkan: merek/model/varian belum dikenal.'
        segment = segment.loc[segment[field].eq(value)]
    if segment.empty:
        return result, 'Dukungan kapasitas mesin tidak dapat disimpulkan: segmen merek/model/varian belum dikenal.'
    for field in ['JenisMesin', 'JenisTransmisi']:
        value = payload.get(field)
        if value in set(catalog[field]):
            segment = segment.loc[segment[field].eq(value)]
    observed = pd.to_numeric(segment.KapasitasMesin, errors='coerce')
    values = sorted({float(value) for value in observed if pd.notna(value) and math.isfinite(value) and value > 0})
    result.update(observed_cc=values, catalog_feature_rows=len(segment))
    if not values:
        result['status'] = 'insufficient_support'
        return result, 'Dukungan kapasitas mesin segmen tidak memadai dalam katalog aktif; verifikasi spesifikasi.'
    if any(math.isclose(cc, value, rel_tol=0, abs_tol=1e-6) for value in values):
        result.update(status='observed', requires_review=False)
        return result, None
    nearest = min(values, key=lambda value: abs(value - cc))
    result.update(status='unobserved', nearest_observed_cc=nearest, difference_cc=abs(nearest - cc))
    return result, (f'KapasitasMesin {cc:g} cc belum teramati pada segmen katalog aktif; '
                    f'nilai teramati terdekat {nearest:g} cc. Selisih, termasuk kemungkinan pembulatan, '
                    'memerlukan verifikasi; bukan bukti spesifikasi salah. Input dan harga tidak dikoreksi.')


def adapt_cats(item: CatsVehicleIn, catalog: pd.DataFrame):
    vehicle = 'motor' if item.JenisAsetID == 5 else 'mobil'
    if item.JenisAset is not None and item.JenisAset != vehicle:
        raise ValueError('JenisAset bertentangan dengan routing JenisAsetID (5=motor; lainnya=mobil)')
    changes, warnings = [], []
    payload = {'JenisAset': vehicle}
    for field in FEATURES + ['KapasitasPenumpang', 'TipeBodi', 'Warna']:
        value = getattr(item, field)
        if isinstance(value, str):
            cleaned = clean_text(value)
            if cleaned != value:
                changes.append({'field': field, 'rule': 'trim_case_and_marked_license_annotation'})
            value = cleaned or None
        payload[field] = value
    for field in ['Merk', 'Model', 'JenisMesin', 'JenisTransmisi']:
        if not payload[field]:
            raise ValueError(f'{field} wajib diisi pada kolom terstruktur; Nama tidak digunakan untuk menebak')
    # Reuse baseline aliases first, then narrowly defined additional CATS aliases.
    row = normalize_features(pd.DataFrame([payload])).iloc[0]
    for field in ['Merk', 'Model', 'Varian', 'JenisMesin', 'JenisTransmisi']:
        if payload[field] is not None and row[field] != payload[field]:
            changes.append({'field': field, 'rule': 'baseline_alias'})
            payload[field] = row[field]
    aliases = {'JenisTransmisi': {'A/T': 'OTOMATIS', 'CVT': 'OTOMATIS', 'M/CVT': 'OTOMATIS',
                                 'M/T': 'MANUAL'},
               'JenisMesin': {'SOLAR': 'DIESEL'}}
    for field, mapping in aliases.items():
        if payload[field] in mapping:
            payload[field] = mapping[payload[field]]
            changes.append({'field': field, 'rule': 'explicit_cats_alias'})
    
    # Match punctuation variants only when the active catalog contains
    # exactly one matching canonical category.
    for field in ['Merk', 'Model']:
        candidates = (
            catalog
            if field == 'Merk'
            else catalog.loc[catalog.Merk.eq(payload['Merk'])]
        )

        values = sorted(
            set(candidates[field].dropna()) - {'UNKNOWN', 'LAINNYA'}
        )

        if payload[field] in values:
            continue

        # Default: compare punctuation-insensitive names.
        target = compact(payload[field])

        matches = [
            value for value in values
            if compact(value) == target
        ]

        rule = 'unique_catalog_punctuation_match'

        # A model field may repeat its matching brand prefix.
        #
        # Only strip the matching brand prefix when the complete input
        # has no unique catalog match.
        if field == 'Model' and len(matches) != 1:
            brand = str(payload['Merk']).strip()
            model_text = str(payload['Model']).strip()

            brand_prefix = brand + ' '

            if model_text.upper().startswith(brand_prefix.upper()):
                model_without_brand = model_text[
                    len(brand_prefix):
                ].strip()

                stripped_matches = [
                    value for value in values
                    if compact(value) == compact(model_without_brand)
                ]

                if len(stripped_matches) == 1:
                    matches = stripped_matches
                    rule = 'unique_catalog_model_without_brand_prefix'

        if len(matches) == 1:
            payload[field] = matches[0]
            changes.append({
                'field': field,
                'rule': rule,
            })
        else:
            warnings.append(
                f'{field} tidak dapat dipetakan secara unik '
                'ke katalog aktif; tidak dilakukan fuzzy guessing.'
            )       
                
    pair = catalog.loc[catalog.Merk.eq(payload['Merk']) & catalog.Model.eq(payload['Model'])]
    known_variants = set(pair.Varian.dropna()) - {'UNKNOWN', 'LAINNYA', ''}
    variant = payload['Varian']
    if variant and variant not in known_variants:
        tokens = variant.split()
        remaining, stripped, conflict = [], False, False
        transmission_tokens = {'CVT': 'OTOMATIS', 'AT': 'OTOMATIS', 'A/T': 'OTOMATIS',
                               'AUTOMATIC': 'OTOMATIS', 'MT': 'MANUAL', 'M/T': 'MANUAL'}
        for token in tokens:
            if token in {'CKD', 'CBU'}:
                stripped = True
            elif token in transmission_tokens:
                if payload['JenisTransmisi'] != transmission_tokens[token]:
                    conflict = True
                stripped = True
            elif re.fullmatch(r'\d[.,]\d', token):
                cc = payload['KapasitasMesin']
                if not isinstance(cc, (int, float)) or abs(float(token.replace(',', '.')) * 1000 - cc) > 50:
                    conflict = True
                stripped = True
            else:
                remaining.append(token)
        reduced = ' '.join(remaining)
        if stripped and not conflict and reduced in known_variants:
            payload['Varian'] = reduced
            changes.append({'field': 'Varian', 'rule': 'catalog_variant_without_consistent_engine_transmission_assembly_tokens'})
        else:
            warnings.append('Varian tidak dipetakan: tidak unik/tidak dikenal atau bertentangan dengan mesin/transmisi terstruktur.')
    elif not variant:
        warnings.append('Varian kosong; model dapat menggunakan fallback global.')
    if payload['TipeBodi'] in {'MICRO/MINIBUS', 'MICRO / MINIBUS', 'MINIBUS', 'MICROBUS'}:
        payload['TipeBodi'] = None
        warnings.append('Kategori bodi administratif MICRO/MINIBUS tidak menentukan SUV/MPV; tidak dipakai untuk fallback pembanding lintas model.')
        changes.append({'field': 'TipeBodi', 'rule': 'ambiguous_administrative_body_excluded'})
    for field in ['TahunUnit', 'KapasitasMesin', 'Kilometer']:
        if str(payload[field]).upper() == 'LAINNYA':
            payload[field] = 'Lainnya'
            warnings.append(f'{field} tidak diketahui; baseline mengimputasi median training model global.')
    # Apply exactly the baseline range/EV checks; boundary cleaning never bypasses validation.
    validate_payload(payload)
    normalized = normalize_features(pd.DataFrame([payload])).iloc[0]
    for field in ['JenisMesin', 'JenisTransmisi']:
        if normalized[field] not in set(catalog[field]):
            warnings.append(f'{field} belum dikenal model; prediksi memerlukan pemeriksaan manual.')
    if not pair.empty and payload['Varian'] in known_variants:
        compatible = pair.loc[pair.Varian.eq(payload['Varian']) & pair.JenisTransmisi.eq(payload['JenisTransmisi'])]
        ###
        for field in ['TahunUnit', 'KapasitasMesin']:
            value = payload[field]

            if not isinstance(value, (int, float)):
                continue

            if field == 'TahunUnit':
                compatible = compatible.loc[
                    compatible[field].eq(value)
                ]

            elif field == 'KapasitasMesin':
                # Toleransi pembulatan kapasitas mesin.
                # Hanya untuk validasi katalog,
                # tidak mengubah fitur input model.
                compatible = compatible.loc[
                    compatible[field].sub(value).abs().le(50)
                ]
        ###
        if compatible.empty:
            warnings.append('Kombinasi model/varian/transmisi/tahun/kapasitas mesin belum ditemukan dalam katalog; periksa input.')
    support, support_warning = engine_capacity_support(payload, catalog)
    if support_warning:
        warnings.append(support_warning)
    return {'input_model': payload, 'status': 'needs_review' if warnings else 'normalized',
            'engine_capacity_support': support,
            'changes': changes, 'warnings': warnings,
            'routing_rule': 'JenisAsetID=5 adalah motor; kode nonnegatif lain adalah mobil.',
            'ignored_for_prediction': ['Nama', 'HargaPerolehan', 'NilaiAppraisalAI', 'AppraisalAINote', 'AppraisalAIDate'],
            'note': 'Kolom terstruktur menjadi sumber input. Nama/nopol tidak dipakai untuk menebak merek, model, varian atau harga.'}
