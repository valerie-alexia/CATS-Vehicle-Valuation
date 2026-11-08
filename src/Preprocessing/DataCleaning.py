"""
CATS - UNIFIED DATA CLEANING PIPELINE
========================================================================
Menggabungkan pembersihan data Mobil dan Motor dalam satu kali run:
  - standarisasi teks (Merk, Model, Varian) dan casing per merk
  - pembersihan angka murni (cc, hp, velg) dan parsing kilometer
  - deduplikasi berbasis fingerprint
  - pemisahan kendaraan niaga (mobil) dan kendaraan antik
  - validasi dasar tanpa fitting distribusi harga sebelum split
  - ringkasan sanity check di akhir eksekusi

Catatan keputusan (untuk bab metodologi):
  - lama_belum_laku_hari TIDAK dipakai (tanggal_posting motor tidak valid,
    dan fitur ini tidak tersedia saat memprediksi harga iklan baru).
  - Kilometer adalah titik tengah rentang kategori, bukan angka odometer.
  - Nilai numerik kosong dibiarkan NaN (XGBoost/LightGBM menanganinya).
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ======================================================
# KONFIGURASI
# ======================================================
ROOT = Path(__file__).resolve().parents[2]
BATCH = os.environ.get('CATS_BATCH', 'Oktober')
DIR_RAW = Path(os.environ.get('CATS_RAW_DIR', str(ROOT / 'Data/raw')))
CSV_MOBIL = DIR_RAW / f'MOBIL123_raw_dataset_{BATCH}.csv'
CSV_MOTOR = DIR_RAW / f'MOMOTOR_raw_dataset_{BATCH}.csv'

DIR_CLEAN = Path(os.environ.get('CATS_CLEAN_DIR', str(ROOT / 'Data/clean')))
OUT_MOBIL_BERSIH = DIR_CLEAN / f'CLEAN_MOBIL123_{BATCH}.csv'
OUT_MOBIL_ANTIK = DIR_CLEAN / f'CLEAN_Antik_MOBIL123_{BATCH}.csv'
OUT_MOBIL_ANOMALI = DIR_CLEAN / f'CLEAN_Anomali_MOBIL123_{BATCH}.csv'
OUT_MOTOR_BERSIH = DIR_CLEAN / f'CLEAN_MOMOTOR_{BATCH}.csv'
OUT_MOTOR_ANTIK = DIR_CLEAN / f'CLEAN_Antik_MOMOTOR_{BATCH}.csv'
OUT_MOTOR_ANOMALI = DIR_CLEAN / f'CLEAN_Anomali_MOMOTOR_{BATCH}.csv'

TAHUN_BATAS_MOBIL = 2000       # di bawah ini dianggap antik/klasik
TAHUN_BATAS_MOTOR = 2010
TAHUN_MAKSIMAL = datetime.now().year + 1
RENTANG_CC_MOBIL = (500, 8000)
RENTANG_CC_MOTOR = (50, 2000)

# CONTAMINATION = 0.05           # proporsi data yang dibuang Isolation Forest
# RANDOM_STATE = 42

# True = cetak seluruh daftar unik Model/Varian di ringkasan (panjang sekali)
CETAK_DAFTAR_LENGKAP = False

# ======================================================
# DATA REFERENSI: MOBIL
# ======================================================
MERK_VALID = [
    'ABARTH', 'AUDI', 'BAIC', 'BENTLEY', 'BMW', 'BYD', 'CADILLAC', 'CHANGAN', 'CHERY', 'CHEVROLET',
    'CHRYSLER', 'CITROEN', 'DAIHATSU', 'DATSUN', 'DENZA', 'DFSK', 'DODGE', 'FIAT', 'FORD', 'GAC',
    'GEELY', 'GENESIS', 'GWM', 'HINO', 'HONDA', 'HUMMER', 'HYUNDAI', 'INFINITI', 'ISUZU', 'JAECOO',
    'JAGUAR', 'JEEP', 'JETOUR', 'KIA', 'LAND ROVER', 'LEXUS', 'MASERATI', 'MAXUS', 'MAZDA',
    'MERCEDES-BENZ', 'MG', 'MINI', 'MITSUBISHI', 'NETA', 'NISSAN', 'PEUGEOT', 'PORSCHE', 'PROTON',
    'RENAULT', 'SMART', 'SUBARU', 'SUZUKI', 'TESLA', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO', 'WULING',
    'XPENG', 'ZEEKR',
]

# Fallback kalau penjual hanya menulis nama model
MODEL_KE_MERK = {
    'AVANZA': 'TOYOTA', 'INNOVA': 'TOYOTA', 'KIJANG': 'TOYOTA', 'VELOZ': 'TOYOTA', 'RUSH': 'TOYOTA',
    'AGYA': 'TOYOTA', 'CALYA': 'TOYOTA', 'FORTUNER': 'TOYOTA', 'ALPHARD': 'TOYOTA', 'VELLFIRE': 'TOYOTA',
    'RAIZE': 'TOYOTA', 'HILUX': 'TOYOTA',
    'BRIO': 'HONDA', 'MOBILIO': 'HONDA', 'JAZZ': 'HONDA', 'CIVIC': 'HONDA',
    'HR-V': 'HONDA', 'CR-V': 'HONDA', 'BR-V': 'HONDA',
    'XENIA': 'DAIHATSU', 'TERIOS': 'DAIHATSU', 'SIGRA': 'DAIHATSU', 'AYLA': 'DAIHATSU',
    'ERTIGA': 'SUZUKI', 'XL7': 'SUZUKI', 'APV': 'SUZUKI', 'JIMNY': 'SUZUKI', 'KARIMUN': 'SUZUKI',
    'XPANDER': 'MITSUBISHI', 'PAJERO': 'MITSUBISHI', 'RANGE ROVER': 'LAND ROVER',
}

# Kata promosi / copywriting penjual yang dibuang dari judul
KATA_PROMOSI = [
    'DIJUAL', 'ISTIMEWA', 'LIKE', 'NEW', 'LIKENEW', 'LOW', 'KM', 'RARE',
    'TERMURAH', 'PAKAIAN', 'PERAWAT', 'TDP', 'NETT', 'NIK', 'ODO',
    'ORISINAL', 'ORISINIL', 'ALL', 'CUKUP', 'CBU', 'FACELIFT', 'ALL NEW',
    'KREDIT', 'CASH', 'MURAH', 'DP', 'TT', 'PERFECT', 'MULUS', 'PAJAK',
    'PANJANG', 'TERAWAT', 'SIAP PAKAI', 'NEGO', 'GARANSI', 'RECORD',
    'TANGAN PERTAMA', 'KONDISI', 'AUTO APPROVE', 'JUAL CEPAT', 'MINT',
    'FULL', 'BEBAS BANJIR', 'TABRAK', 'PROMO', 'SERVICE', 'GOOD', 'CONDITION',
    'ANTIK', 'MATIC', 'SOLAR', 'MANUAL', 'HARGA', 'JUAL', 'MOBIL', 'OVER', 'TUKAR',
    'ISTIMEWAH', 'ISTIMWEA',
    'PEMAKAIAN', 'BONUS', 'SANGAT', 'SIAP', 'LANGSUNG', 'PAKAI', 'CREDIT', 'TERJUAL',
    'KENDARAAN', 'CAKEP', 'KEREN', 'NEGOIN', 'DEAL', 'SAMPAI', 'ASLI', 'TAHUN', 'TGN',
    'PRIBADI', 'RAWATAN', 'RENDAH', 'HITAM', 'PUTIH', 'ABU', 'MERAH', 'BIRU', 'COKLAT',
    'KUNING', 'HIJAU', 'ORANYE',
]

DAFTAR_MODEL_MOBIL = [
    'Kijang Innova Zenix', 'Kijang Innova', 'Land Cruiser Prado', 'Land Cruiser', 'Alphard', 'Vellfire',
    'Veloz', 'Agya', 'Calya', 'Rush', 'Fortuner', 'Voxy', 'Raize', 'Yaris Cross', 'Corolla Cross',
    'Corolla Altis', 'Camry', 'Hilux', 'Sienta', 'Avanza', 'Yaris', 'HR-V', 'BR-V', 'WR-V',
    'CR-V', 'Brio', 'Jazz', 'Mobilio', 'Civic Type R', 'Civic', 'Accord', 'City Hatchback',
    'City', 'Odyssey', 'Xpander Cross', 'Xpander', 'XFORCE', 'Pajero Sport', 'Outlander Sport',
    'Triton', 'Grand Vitara', 'Ertiga', 'XL7', 'Jimny', 'Fronx', 'Ignis', 'Baleno', 'APV',
    'Creta', 'Palisade', 'Santa Fe', 'Stargazer X', 'Stargazer', 'Ioniq 5', 'Ioniq 6',
    'Grand Sedona', 'Sonet', 'Seltos', 'Carnival', 'Carens', 'Sealion 7', 'Cloud EV', 'Binguo EV',
    'Air EV', 'Almaz RS', 'Almaz', 'Cortez', 'Confero', 'Omoda E5', 'Omoda 5', 'Omoda J6',
    'Tiggo 8 Pro', 'Tiggo 8', 'Tiggo 7 Pro', 'Tiggo Cross', 'Haval H6', 'Tank 500', 'Tank 300',
    'Denza D9', 'Hyptec HT', 'Range Rover Evoque', 'Range Rover Sport', 'Range Rover Velar',
    'Range Rover', 'Land Rover Defender', 'CX-90', 'CX-80', 'CX-60', 'CX-50', 'CX-30', 'CX-5', 'CX-3',
    'CX-8', 'CX-7', 'CX-9', 'Mazda 2', 'Mazda 3', 'Mazda 6', 'MINI Countryman', 'MINI Clubman',
    'MINI Cabrio', 'MINI Cooper', 'ID. Buzz', 'Tiguan Allspace', 'Tiguan', 'Golf GTI', 'Scirocco',
    'T Cross', 'Grand Cherokee', 'Gran Max', 'Granmax',
    'Macan', '911', 'Luxio', 'Ayla', 'Copen', 'Wrangler', 'J8', 'LX 700h', 'RX 500H', 'RX 270',
    'LM 350h', 'M6',
    'Innova Zenix', 'Innova', 'C-HR', 'X-Trail', 'Prado', 'Defender', 'Discovery', 'Terios', 'Xenia',
    'Sigra', 'Karimun', 'Serena', 'Livina', 'Mustang', 'Rubicon',
]
DAFTAR_MODEL_MOBIL.sort(key=len, reverse=True)

# Mapping penyatuan Model (kunci dicocokkan tanpa peduli huruf besar/kecil)
MODEL_MAPPING_MOBIL = {
    'BRV': 'BR-V', 'CRV': 'CR-V', 'HRV': 'HR-V',
    'Innovva': 'Innova', 'Reborn': 'Innova',
    'Kijang Innova': 'Innova', 'Kijang Innova Zenix': 'Innova Zenix',
    'Nbeetle': 'Beetle', 'Tbeetle': 'Beetle', 'Beatle': 'Beetle',
    'Prado': 'Land Cruiser Prado',
}
_MODEL_MAPPING_MOBIL_LOWER = {k.lower(): v for k, v in MODEL_MAPPING_MOBIL.items()}

# Casing akhir untuk kode model yang title-case-nya salah
CASING_MODEL_MOBIL = {
    'RX8': 'RX-8', 'Ev6': 'EV6', 'Ev9': 'EV9', 'Xc40': 'XC40', 'Xc60': 'XC60', 'Xc90': 'XC90',
    'Rs': 'RS', 'Rs5': 'RS5', 'Hs': 'HS', 'Zs': 'ZS', 'Sx4': 'SX4', 'Xe': 'XE', 'Xf': 'XF',
    'Xj': 'XJ', 'Xkr': 'XKR', 'Fj': 'FJ', 'Bj40': 'BJ40', 'Bz4X': 'bZ4X', 'Ix': 'iX', 'Ix1': 'iX1',
}

MERK_KODE_HURUF_BESAR = ['MERCEDES-BENZ', 'LEXUS', 'MAZDA']   # Model ditulis huruf besar semua
KECUALI_HURUF_BESAR = {'VITO', 'VIANO', 'VANEO', 'BIANTE'}      # ...kecuali nama model ini
MERK_MODEL_SATU_HURUF = ['NETA', 'ZEEKR']                       # Neta V, Zeekr X

MODEL_SAMPAH_MOBIL = {'abu2', 'model', 'super', 'gran', 'grand', 'amg', 'tfsi', 'gr'}

# Kendaraan niaga (di luar lingkup penelitian)
MODEL_NIAGA = {'dutro', 'dyna', 'elf', 'giga', 'fuso', 'traga', 'sprinter',
               'gran max', 'granmax', 'carry', 'colt', 'rangga'}
VARIAN_NIAGA_REGEX = (r'(?i)\b(?:truck|microbus|mini bus|blind van|cab chassis|flat deck|wide deck|'
                      r'pick up (?:flat|wide|standard)|PU|carry|l300)\b')

# ======================================================
# DATA REFERENSI: MOTOR
# ======================================================
DAFTAR_MODEL_MOTOR = [
    'Beat Street', 'Vario', 'PCX', 'Scoopy', 'Beat', 'Stylo', 'Genio', 'ADV', 'CRF', 'CBR', 'Nmax',
    'Aerox', 'Fazzio', 'Lx', 'Callisto', 'King', 'Supra', 'Revo', 'Blade', 'Sonic', 'Verza', 'Tiger',
    'Ninja', 'Vixion', 'R15', 'R25', 'MT-15', 'Xmax', 'Lexi', 'Mio', 'Jupiter', 'Satria', 'GSX', 'KLX',
    'CB150R', 'Vespa', 'Burgman', 'Freego', 'Gear', 'Fino', 'Soul', 'Filano', 'Grand Filano',
    'D-Tracker', 'X-Ride',
]
DAFTAR_MODEL_MOTOR.sort(key=len, reverse=True)

MODEL_MAPPING_MOTOR = {
    'Nbeat': 'Beat', 'Newbeat': 'Beat', 'Beatfisprtcbsissplus': 'Beat',
    'Newbeatpespcbsisspix': 'Beat', 'Beatpespcbsisspix': 'Beat',
    'N-max': 'Nmax', 'Nscoopy': 'Scoopy', 'Nvario': 'Vario', 'Vario110': 'Vario',
    'Pcx125': 'PCX', 'Pcx150': 'PCX', 'Cb150r': 'CB150R', 'Cb1000r': 'CB1000R',
    'Newsupra': 'Supra', 'X-ride': 'X-Ride',
}
CASING_MODEL_MOTOR = {
    'Pcx': 'PCX', 'Adv': 'ADV', 'Crf': 'CRF', 'Cbr': 'CBR', 'Gsx': 'GSX', 'Klx': 'KLX',
    'Xsr': 'XSR', 'Mt': 'MT', 'Cb': 'CB', 'Wr': 'WR',
}
# Motor besar: cc di atas 1000 itu valid, jangan dibagi 10
MODEL_MOGE = {'NINJA', 'Z', 'Z125', 'XSR', 'MT', 'GSX', 'CAFERACER', 'LEONCINO',
              'SCORPIO', 'PATAGONIAN', '502', 'TNT', 'MOTOBI'}

# ======================================================
# REGEX YANG DI-COMPILE SEKALI
# ======================================================
PROMO_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(KATA_PROMOSI, key=len, reverse=True)) + r')\b',
    re.IGNORECASE)

MODEL_PATTERNS_MOBIL = [
    (m, re.compile(rf'(?<![\w-]){re.escape(m)}(?![\w-])', re.IGNORECASE))
    for m in DAFTAR_MODEL_MOBIL
]
MODEL_PATTERNS_MOTOR = [
    (m, re.compile(rf'\b{re.escape(m)}\b', re.IGNORECASE))
    for m in DAFTAR_MODEL_MOTOR
]

POLA_ANGKA_HARDCORE = r'(?i)(\d+[,.]?\d*\s*(juta|jt|rb|ribu|perak)|tdp\s*\d+|km\s*\d+|\d+\s*kilometer)'
POLA_ANGKA_PROMOSI = r'(?i)\b(\d+[,.]?\d*\s*(juta|jt|rb|ribu)|tdp\s*\d+|km\s*\d+)\b'
POLA_SIMBOL_JUDUL = r'[!*\[\]#|()]|\s-\s|^-+|-+$'
POLA_TAHUN = r'\b(19\d{2}|20\d{2})\b'
TIPE_BODI_LIST = 'SUV|MPV|HATCHBACK|SEDAN|COUPE|CONVERTIBLE|CABRIOLET|VAN|PICKUP|WAGON'

# ======================================================
# FUNGSI BANTU (SHARED UTILITIES)
# ======================================================
def cari_merk(nama):
    """Cari merk valid di mana pun dalam judul; yang muncul paling awal menang."""
    t = str(nama).upper()
    best = None
    for m in MERK_VALID:
        mt = re.search(rf'\b{re.escape(m)}\b', t)
        if mt and (best is None or mt.start() < best[0] or
                   (mt.start() == best[0] and len(m) > len(best[1]))):
            best = (mt.start(), m)
    if best:
        return best[1]
    for model, merk in MODEL_KE_MERK.items():
        if re.search(rf'\b{re.escape(model)}\b', t):
            return merk
    return np.nan


def parse_kilometer_unified(val):
    """'5 - 10K km' / '1.000 - 5.000 KM' / '> 200K km' -> angka (titik tengah rentang)."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace('km', '').strip()
    is_thousand = 'k' in s
    s_clean = s.replace('k', '')
    s_clean = s_clean.replace(',', '.') if is_thousand else s_clean.replace('.', '').replace(',', '')
    nums = re.findall(r'\d+(?:\.\d+)?', s_clean)
    if not nums:
        return np.nan
    faktor = 1000 if is_thousand else 1
    if 'lebih dari' in s_clean or '>' in s_clean:
        return float(nums[0]) * faktor
    if '-' in s_clean and len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2 * faktor
    return float(nums[0]) * faktor


def _kolom_teks(seri):
    return pd.api.types.is_object_dtype(seri) or pd.api.types.is_string_dtype(seri)


def normalisasi_teks_kategori(df, kolom, mode='title'):
    """Menghilangkan spasi berlebih dan menyeragamkan huruf."""
    if kolom in df.columns and _kolom_teks(df[kolom]):
        s = df[kolom].astype(str).str.strip()
        if mode == 'upper':
            s = s.str.upper()
        elif mode == 'title':
            s = s.str.title()
        df[kolom] = s.replace(r'(?i)^nan$|^unknown$', np.nan, regex=True)
    return df


def ekstrak_angka_murni(series):
    """'1.498 cc'->1498 | '124,8 cc'->124.8 | '1500 cc'->1500 | '92.4 hp'->92.4"""
    token = series.astype(str).str.extract(r'(\d[\d.,]*)', expand=False)

    def _ke_float(x):
        if pd.isna(x):
            return np.nan
        x = x.rstrip('.,')
        try:
            if re.fullmatch(r'\d{1,3}([.,]\d{3})+', x):   # pemisah ribuan
                return float(re.sub(r'[.,]', '', x))
            return float(x.replace(',', '.'))
        except ValueError:
            return np.nan

    return token.map(_ke_float)


def parse_velg(series):
    """Ukuran velg (inci): terima '17', '17.0', '17 inci', '235/60 R17', 'R17', 'ring 17'."""
    s = series.astype(str)
    dari_r = s.str.extract(r'(?i)(?:r|ring)\s*(\d{2})', expand=False)
    angka = s.str.extract(r'^\s*(\d{2}(?:\.\d+)?)\s*(?:inci|inch|")?\s*$', expand=False)
    out = pd.to_numeric(dari_r.fillna(angka), errors='coerce')
    return out.where(out.between(10, 30))


def bersihkan_harga(df):
    """'Rp150.000.000' -> 150000000.0; baris tanpa harga dibuang."""
    if _kolom_teks(df['harga']):
        df['harga'] = (df['harga'].astype(str)
                       .str.replace('Rp', '', regex=False)
                       .str.replace('.', '', regex=False))
    df['harga'] = pd.to_numeric(df['harga'], errors='coerce')
    return df.dropna(subset=['harga'])


def deduplikasi(df, kolom_abaikan):
    """Buang baris yang identik di semua kolom kecuali kolom_abaikan."""
    kolom_fp = [c for c in df.columns if c not in kolom_abaikan]
    n_awal = len(df)
    df = df.loc[~df.duplicated(subset=kolom_fp, keep='first')].copy()
    print(f"  Dedup: {n_awal - len(df)} baris duplikat dibuang dari {n_awal}")
    return df


def pisahkan_outlier(df_modern, kolom_km):
    """Hanya aturan validitas; tidak belajar distribusi harga sebelum split.

    Harga ekstrem yang valid tetap disertakan dalam evaluasi.
    Nama fungsi dipertahankan untuk kompatibilitas notebook lama.
    """
    valid = np.isfinite(df_modern['harga']) & df_modern['harga'].gt(0)
    return df_modern[valid].copy(), df_modern[~valid].copy()


def finalisasi_bersih(df_bersih):
    """Isi teks kosong dengan 'Unknown' (numerik dibiarkan NaN), lalu buang Merk/Model Unknown."""
    kol_teks = df_bersih.select_dtypes(include=['object', 'string']).columns
    df_bersih[kol_teks] = df_bersih[kol_teks].fillna('Unknown')
    n_sebelum = len(df_bersih)
    df_bersih = df_bersih[(df_bersih['Merk'] != 'Unknown') & (df_bersih['Model'] != 'Unknown')].copy()
    print(f"  Dibuang karena Merk/Model Unknown: {n_sebelum - len(df_bersih)} baris")
    return df_bersih


def simpan_hasil(df_bersih, df_antik, df_anomali, path_bersih, path_antik, path_anomali):
    os.makedirs(DIR_CLEAN, exist_ok=True)
    df_bersih.to_csv(path_bersih, index=False)
    df_antik.to_csv(path_antik, index=False)
    df_anomali.to_csv(path_anomali, index=False)


# ======================================================
# RINGKASAN HASIL
# ======================================================
def _print_kategori(df, col, penuh=True, top=15):
    if col not in df.columns:
        return
    if penuh:
        unik = sorted(str(x) for x in df[col].unique())
        print(f"- {col} ({len(unik)}): {', '.join(unik)}")
    else:
        vc = df[col].value_counts()
        teratas = ', '.join(f"{k} ({v})" for k, v in vc.head(top).items())
        print(f"- {col} ({len(vc)} unik) | {top} teratas: {teratas}")


def _print_numerik(df, col):
    if col not in df.columns:
        return
    s = df[col]
    print(f"- {col}: {s.nunique()} unik | min={s.min():,.1f} | max={s.max():,.1f} "
          f"| kosong={s.isna().mean():.1%}")


def print_cleaning_summary(tipe_kendaraan, df_bersih, df_antik, df_anomali):
    """Mencetak ringkasan hasil cleaning ke terminal."""
    print("\n" + "=" * 60)
    print(f"RINGKASAN DATA {tipe_kendaraan.upper()}")
    print("=" * 60)
    print(f"Total Bersih (Siap ML) : {len(df_bersih):,} baris")
    print(f"Total Antik/Klasik     : {len(df_antik):,} baris")
    print(f"Total Anomali/Outlier  : {len(df_anomali):,} baris")

    if len(df_bersih) > 0:
        kolom_tampil = [c for c in ['Merk', 'Model', 'Varian', 'TahunUnit', 'harga'] if c in df_bersih.columns]
        print("\nINFO HARGA TERTINGGI & TERENDAH (Data Bersih):")
        print("-> HARGA TERTINGGI:")
        print(df_bersih.loc[[df_bersih['harga'].idxmax()], kolom_tampil].to_string(index=False))
        print("\n-> HARGA TERENDAH:")
        print(df_bersih.loc[[df_bersih['harga'].idxmin()], kolom_tampil].to_string(index=False))

        print("\nDAFTAR KATEGORI (Sanity Check):")
        for col in ['Merk', 'JenisTransmisi', 'JenisMesin', 'TipeBodi']:
            _print_kategori(df_bersih, col, penuh=True)
        for col in ['Model', 'Varian']:
            _print_kategori(df_bersih, col, penuh=CETAK_DAFTAR_LENGKAP)
        if 'TipeBodi' in df_bersih.columns:
            share = (df_bersih['TipeBodi'] == 'Unknown').mean()
            print(f"  (TipeBodi 'Unknown': {share:.1%} dari data bersih)")

        print("\nFITUR NUMERIK:")
        for col in ['TahunUnit', 'KapasitasMesin', 'KapasitasPenumpang', 'Pintu', 'Tenaga Puncak (hp)',
                    'Velg Depan (inci)', 'Veld Belakang (inci)', 'Kilometer', 'Kilometer_Num']:
            _print_numerik(df_bersih, col)
        for col in ['Kilometer', 'Kilometer_Num', 'KapasitasMesin']:
            if col in df_bersih.columns:
                print(f"- {col:<15}: Rata-rata = {df_bersih[col].mean():,.0f} "
                      f"| Max = {df_bersih[col].max():,.0f}")
    print("=" * 60 + "\n")


# ======================================================
# PIPELINE 1: MOBIL
# ======================================================
def normalisasi_judul_mobil(df):
    """Seragamkan penulisan merk/model populer di judul mentah."""
    df['nama_mobil'] = (df['nama_mobil'].astype(str)
        .str.replace('_', ' ', regex=False)
        .str.replace(r'(?i)\bmercedes[\s-]*benz\b', 'Mercedes-Benz', regex=True)
        .str.replace(r'(?i)\bland[\s-]*rover\b', 'Land Rover', regex=True)
        .str.replace(r'(?i)\b(hr|cr|br|wr)[\s-]*v\b', r'\1-V', regex=True)
        .str.replace(r'(?i)\bx[\s-]*trail\b', 'X-Trail', regex=True)
        .str.replace(r'(?i)\bc[\s-]*hr\b', 'C-HR', regex=True)
        .str.replace(r'(?i)garansi\s*7g\+?\s*selama\s*\d+\s*tahun', ' ', regex=True)
        .str.replace(r'(?i)\bjt\b[\sA-Za-z/]*?\btermurah\b', ' ', regex=True)
        .str.replace(r'(?i)\bcx[\s-]?(\d+)\b', r'CX-\1', regex=True)
        .str.replace(r'(?i)\b(SUV|MPV|SEDAN|HATCHBACK)(?=[A-Z])', r'\1 ', regex=True))
    return df


def buat_judul_bersih(df):
    """Buat kolom nama_mobil_bersih: judul tanpa kata promosi, harga, dan simbol iklan."""
    s = (df['nama_mobil'].str.replace(r'\(.*?\)', ' ', regex=True)
                         .str.replace(POLA_SIMBOL_JUDUL, ' ', regex=True))
    s = s.str.replace(PROMO_PATTERN, '', regex=True)
    s = s.str.replace(PROMO_PATTERN, '', regex=True)               # 2x: kata promosi bersebelahan
    s = s.str.replace(POLA_ANGKA_HARDCORE, '', regex=True)
    s = s.str.replace(POLA_SIMBOL_JUDUL, ' ', regex=True)
    s = s.str.replace(r'^[^a-zA-Z0-9]+', '', regex=True)           # simbol nyasar di paling depan
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    s = s.str.replace(POLA_ANGKA_PROMOSI, '', regex=True)
    df['nama_mobil_bersih'] = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return df


def ekstrak_tahun_mobil(df):
    """Tahun utama dari kolom 'Tahun Kendaraan'; fallback dari judul."""
    tahun_judul = df['nama_mobil_bersih'].str.extract(POLA_TAHUN, expand=False).astype(float)
    if 'Tahun Kendaraan' in df.columns:
        tahun_kolom = df['Tahun Kendaraan'].astype(str).str.extract(POLA_TAHUN, expand=False).astype(float)
    else:
        tahun_kolom = pd.Series(np.nan, index=df.index)
    df['TahunUnit_Ekstrak'] = tahun_kolom.fillna(tahun_judul)
    df.loc[df['TahunUnit_Ekstrak'] > TAHUN_MAKSIMAL, 'TahunUnit_Ekstrak'] = np.nan
    return df


def ekstrak_model_varian(row):
    """Pisahkan judul bersih menjadi (Model, Varian)."""
    teks = str(row['nama_mobil_bersih'])
    teks = re.sub(r'^(19|20)\d{2}\s+', '', teks)
    if row['Merk'] != 'PEUGEOT':                          # Peugeot 2008 itu model, bukan tahun
        teks = re.sub(r'\b(19|20)\d{2}\b', ' ', teks)
    if pd.notna(row['TipeBodi']):
        teks = re.split(rf"(?i)\b{re.escape(str(row['TipeBodi']))}\b", teks)[0]
    if pd.notna(row['Merk']):                              # buang merk di posisi mana pun
        teks = re.sub(rf"(?i)\b{re.escape(str(row['Merk']))}\b", ' ', teks)
    teks = re.sub(r'\b\d+[\.,]\d+\b', ' ', teks)           # buang "1.5" / "2,0"
    teks = re.sub(r'\s+', ' ', teks).strip()

    model_found, varian_found = 'Unknown', teks
    for model, pat in MODEL_PATTERNS_MOBIL:
        if pat.search(teks):
            model_found = model
            varian_found = pat.sub(' ', teks)
            break

    if model_found == 'Unknown':                           # fallback: kata pertama = model
        kata = teks.split()
        ada_amg = False
        while len(kata) > 1 and kata[0].upper() in {'AMG', 'BENZ'}:
            ada_amg = ada_amg or kata[0].upper() == 'AMG'
            kata = kata[1:]
        if kata:
            model_found = kata[0]
            varian_found = ('AMG ' if ada_amg else '') + ' '.join(kata[1:])

    varian_found = re.sub(r'\s+', ' ', varian_found).strip()
    return pd.Series([model_found, varian_found])


def postproses_model_mobil(df):
    """Perbaikan Model/Varian setelah ekstraksi (urutan langkah penting)."""
    # 1. Model satu huruf: biasanya trim (G, V, X) -> pindah ke Varian.
    #    Pengecualian Neta V / Zeekr X: huruf itu memang nama model.
    satu_huruf = df['Model'].str.fullmatch(r'[A-Za-z]')
    merk_huruf = df['Merk'].isin(MERK_MODEL_SATU_HURUF)
    m_trim = satu_huruf & ~merk_huruf
    df.loc[m_trim, 'Varian'] = df.loc[m_trim, 'Model'] + ' ' + df.loc[m_trim, 'Varian']
    df.loc[m_trim, 'Model'] = 'Unknown'
    m_nama = satu_huruf & merk_huruf
    df.loc[m_nama, 'Model'] = df.loc[m_nama, 'Merk'].str.title() + ' ' + df.loc[m_nama, 'Model']

    # 2. Rapikan Varian: simbol nyasar, maksimal 4 kata, buang kata ganda
    df['Varian'] = (df['Varian'].str.replace(r'^[^a-zA-Z0-9]+', '', regex=True).str.strip()
                    .str.replace(r'[,•?;]+', ' ', regex=True)
                    .str.split().str[:4].str.join(' '))
    df['Varian'] = df['Varian'].str.replace(r'(?i)\b(\w+)(?:\s+\1\b)+', r'\1', regex=True)

    # 3. Tesla: "Model 3 Standard Range" -> Model='Model 3', Varian='Standard Range'
    m_tesla = (df['Merk'] == 'TESLA') & (df['Model'] == 'Model')
    df.loc[m_tesla, 'Model'] = 'Model ' + df.loc[m_tesla, 'Varian'].str.split().str[0]
    df.loc[m_tesla, 'Varian'] = df.loc[m_tesla, 'Varian'].str.split(n=1).str[1].fillna('')

    # 4. Penyatuan nama model (tidak peduli huruf besar/kecil)
    df['Model'] = df['Model'].map(lambda x: _MODEL_MAPPING_MOBIL_LOWER.get(str(x).lower(), x))

    # 5. Casing: title-case, kecuali model kanonik; kode merk tertentu dibuat huruf besar
    kanonik = df['Model'].isin(DAFTAR_MODEL_MOBIL) | (df['Model'] == 'Unknown')
    df['Model'] = df['Model'].where(kanonik, df['Model'].str.title())

    kode_besar = (df['Merk'].isin(MERK_KODE_HURUF_BESAR) & ~kanonik
                  & ~df['Model'].str.upper().isin(KECUALI_HURUF_BESAR))
    df.loc[kode_besar, 'Model'] = df.loc[kode_besar, 'Model'].str.upper()

    m_bmw = (df['Merk'] == 'BMW') & ~kanonik                # 320I -> 320i, M135I -> M135i
    df.loc[m_bmw, 'Model'] = df.loc[m_bmw, 'Model'].str.replace(
        r'(?<=\d)([ID])\b', lambda mt: mt.group(1).lower(), regex=True)
    df['Model'] = df['Model'].replace(CASING_MODEL_MOBIL)

    # 6. Model angka tunggal: Mazda 3, BMW 3 Series; selain itu tidak bermakna
    angka = df['Model'].str.fullmatch(r'\d')
    df.loc[angka & (df['Merk'] == 'MAZDA'), 'Model'] = 'Mazda ' + df['Model']
    df.loc[angka & (df['Merk'] == 'BMW'), 'Model'] = df['Model'] + ' Series'
    df.loc[angka & ~df['Merk'].isin(['MAZDA', 'BMW']), 'Model'] = 'Unknown'

    # 7. Model yang hanya kata awalan / sampah
    lower = df['Model'].str.lower()
    sampah = lower.isin(MODEL_SAMPAH_MOBIL) | ((lower == 'cross') & (df['Merk'] != 'DATSUN'))
    df.loc[sampah, 'Model'] = 'Unknown'
    return df


def filter_niaga(df):
    """Buang kendaraan niaga (truk, van niaga, pickup niaga)."""
    mask = ((df['Merk'] == 'HINO')
            | df['Model'].str.lower().isin(MODEL_NIAGA)
            | df['Varian'].str.contains(VARIAN_NIAGA_REGEX, na=False))
    print(f"  Niaga dibuang: {mask.sum()} baris")
    return df[~mask].copy()


def clean_mobil():
    print(">>> MEMULAI PEMBERSIHAN DATA MOBIL...")
    try:
        df = pd.read_csv(CSV_MOBIL)
    except FileNotFoundError:
        raise FileNotFoundError(f"File RAW mobil tidak ditemukan: {CSV_MOBIL}") from None

    df['batch_id'] = BATCH

    # --- Judul, tahun, merk, tipe bodi ---
    df = normalisasi_judul_mobil(df)
    df = buat_judul_bersih(df)
    df = ekstrak_tahun_mobil(df)
    df['TipeBodi'] = df['nama_mobil'].str.extract(f'(?i)\\b({TIPE_BODI_LIST})\\b', expand=False)
    df['Merk'] = df['nama_mobil'].apply(cari_merk)

    df = normalisasi_teks_kategori(df, 'Merk', mode='upper')
    df = normalisasi_teks_kategori(df, 'TipeBodi', mode='upper')
    df = normalisasi_teks_kategori(df, 'Transmisi', mode='title')
    df = normalisasi_teks_kategori(df, 'Tipe Bahan Bakar', mode='title')

    # --- Model & Varian ---
    df[['Model', 'Varian']] = df.apply(ekstrak_model_varian, axis=1)
    df = postproses_model_mobil(df)
    df = df.drop(columns=['nama_mobil_bersih'])

    # --- Angka murni ---
    kolom_numerik_mobil = [
        'Cakupan mesin', 'Kapasitas Kursi', 'Pintu', 'Tenaga Puncak (hp)',
        'Panjang (mm)', 'Lebar (mm)', 'Tinggi (mm)', 'Jarak Roda (mm)',
        'Berat Kosong (kg)', 'Ruang Boot (Liter)',
    ]
    for col in kolom_numerik_mobil:
        if col in df.columns:
            df[col] = ekstrak_angka_murni(df[col])
    for col in ['Velg Depan (inci)', 'Veld Belakang (inci)']:
        if col in df.columns:
            df[col] = parse_velg(df[col])

    df = bersihkan_harga(df)

    df['Kilometer'] = df['Kilometer'].apply(parse_kilometer_unified)
    df['km_missing'] = df['Kilometer'].isna().astype(int)

    for col in [c for c in df.columns if c.startswith('Fitur_')]:
        df[col] = df[col].notna().astype(int)

    df = df.rename(columns={
        'TahunUnit_Ekstrak': 'TahunUnit', 'Tipe Bahan Bakar': 'JenisMesin',
        'Transmisi': 'JenisTransmisi', 'Cakupan mesin': 'KapasitasMesin',
        'Kapasitas Kursi': 'KapasitasPenumpang',
    })

    # --- Dedup (sebelum Penjual/lokasi dibuang), lalu buang kolom tak terpakai ---
    df = deduplikasi(df, ['waktu_scrape', 'tanggal_scrape', 'tanggal_update', 'tanggal_posting'])

    drop_cols = ['nama_mobil', 'Penjual', 'lokasi', 'Kondisi',
                 'Tahun Kendaraan', 'Engine CC', 'Penumpang']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = filter_niaga(df)

    # --- Antik vs modern, filter cc, outlier ---
    n_tahun_nan = df['TahunUnit'].isna().sum()
    if n_tahun_nan:
        print(f"  [!] {n_tahun_nan} baris tanpa TahunUnit dibuang (tidak masuk antik/modern)")

    df_antik = df[df['TahunUnit'] < TAHUN_BATAS_MOBIL]
    df_modern = df[df['TahunUnit'] >= TAHUN_BATAS_MOBIL]

    if 'KapasitasMesin' in df_modern.columns:
        lo, hi = RENTANG_CC_MOBIL
        valid_cc = df_modern['KapasitasMesin'].isna() | df_modern['KapasitasMesin'].between(lo, hi)
        df_modern = df_modern[valid_cc].copy()

    df_bersih, df_anomali = pisahkan_outlier(df_modern, 'Kilometer')
    df_bersih = finalisasi_bersih(df_bersih)

    simpan_hasil(df_bersih, df_antik, df_anomali, OUT_MOBIL_BERSIH, OUT_MOBIL_ANTIK, OUT_MOBIL_ANOMALI)
    print_cleaning_summary('Mobil', df_bersih, df_antik, df_anomali)


# ======================================================
# PIPELINE 2: MOTOR
# ======================================================
def ekstrak_model_varian_motor(row):
    """Pisahkan nama motor menjadi (Model, Varian)."""
    teks = re.sub(r'\(\d{4}\)', '', str(row['nama_motor']))
    if pd.notna(row['merk']):
        teks = re.sub(rf"^{re.escape(str(row['merk']))}\s+", '', teks, flags=re.IGNORECASE)
    teks = teks.strip()

    model_found, varian_found = 'Unknown', teks
    for model, pat in MODEL_PATTERNS_MOTOR:
        if pat.search(teks):
            model_found = model
            varian_found = pat.sub('', teks).strip()
            break

    if model_found == 'Unknown' and teks.split():
        kata = teks.split()
        model_found, varian_found = kata[0], ' '.join(kata[1:])
    return pd.Series([model_found, varian_found])


def postproses_model_motor(df):
    """Penyatuan nama model motor, taksonomi Vespa, dan CB150R/CB150X."""
    df['Model'] = df['Model'].str.title()
    df['Model'] = df['Model'].str.replace(r'(?i)^(n|new)?beat\w+$', 'Beat', regex=True)
    df['Model'] = df['Model'].replace({k.title(): v for k, v in MODEL_MAPPING_MOTOR.items()})
    df['Model'] = df['Model'].str.replace(r'(?i)^grandfilano\w*$', 'Grand Filano', regex=True)
    df['Model'] = df['Model'].replace(CASING_MODEL_MOTOR)

    # Vespa: satu taksonomi -> Model 'Vespa LX', 'Vespa GTS', 'Vespa Primavera', dst.
    piaggio = df['merk'].eq('PIAGGIO')
    teks_v = (df['Model'] + ' ' + df['Varian']).where(piaggio)
    seri = teks_v.str.extract(r'(?i)(lxv|lx|gts|primavera|sprint|liberty|medley)\b\s*(.*)$')
    ada = seri[0].notna()
    nama_seri = seri.loc[ada, 0].str.upper().replace(
        {'PRIMAVERA': 'Primavera', 'SPRINT': 'Sprint', 'LIBERTY': 'Liberty', 'MEDLEY': 'Medley'})
    df.loc[ada, 'Model'] = 'Vespa ' + nama_seri
    df.loc[ada, 'Varian'] = seri.loc[ada, 1].str.strip()

    # Honda CB: "CB" + Varian "150 R ..." -> Model CB150R, CB150X
    cb = df['Model'] == 'CB'
    ext = df.loc[cb, 'Varian'].str.extract(r'^(\d{3})\s*([RrXx])\b\s*(.*)$')
    idx = ext.index[ext[0].notna()]
    df.loc[idx, 'Model'] = 'CB' + ext.loc[idx, 0] + ext.loc[idx, 1].str.upper()
    df.loc[idx, 'Varian'] = ext.loc[idx, 2]

    df['Varian'] = (df['Varian'].str.replace(r'\s+', ' ', regex=True)
                    .str.replace(r'^[^A-Za-z0-9]+', '', regex=True).str.strip())
    return df


def perbaiki_cc_motor(df):
    """cc motor: tahun nyasar -> NaN; titik desimal hilang (1125 -> 112.5), kecuali moge."""
    if 'Kapasitas mesin' not in df.columns:
        return df
    cc = ekstrak_angka_murni(df['Kapasitas mesin'])
    cc = cc.mask(cc.between(1990, 2030))
    moge = df['Model'].str.upper().isin(MODEL_MOGE)
    cc = cc.mask((cc > 1000) & (cc <= 2000) & ~moge, cc / 10)
    df['Kapasitas mesin'] = cc
    return df


def clean_motor():
    print(">>> MEMULAI PEMBERSIHAN DATA MOTOR...")
    try:
        df = pd.read_csv(CSV_MOTOR)
    except FileNotFoundError:
        raise FileNotFoundError(f"File RAW motor tidak ditemukan: {CSV_MOTOR}") from None

    df['batch_id'] = BATCH

    df['TahunUnit'] = df['nama_motor'].str.extract(POLA_TAHUN, expand=False).astype(float)
    df = normalisasi_teks_kategori(df, 'merk', mode='upper')
    df = normalisasi_teks_kategori(df, 'Transmisi', mode='title')
    df = normalisasi_teks_kategori(df, 'Bahan bakar', mode='title')

    # Seragamkan penulisan di judul
    df['nama_motor'] = (df['nama_motor']
        .str.replace(r'(?i)\b(all\s*new|new|facelift|aln)\b', '', regex=True)
        .str.replace(r'(?i)^(allnew|new|aln)(?=[a-z])', '', regex=True)
        .str.replace(r'(?i)\bd[\s-]*tracker\b', 'D-Tracker', regex=True)
        .str.replace(r'(?i)\bx\s*-\s*ride\b', 'X-Ride', regex=True))

    df[['Model', 'Varian']] = df.apply(ekstrak_model_varian_motor, axis=1)
    df = postproses_model_motor(df)
    df = perbaiki_cc_motor(df)
    df = bersihkan_harga(df)

    df['Kilometer_Num'] = df['Kilometer'].apply(parse_kilometer_unified)
    df['km_missing'] = df['Kilometer_Num'].isna().astype(int)

    df = df.rename(columns={
        'merk': 'Merk', 'Bahan bakar': 'JenisMesin',
        'Transmisi': 'JenisTransmisi', 'Kapasitas mesin': 'KapasitasMesin',
    })

    df = deduplikasi(df, ['waktu_scrape', 'tanggal_posting', 'batch_id'])

    drop_cols = ['nama_motor', 'Kilometer', 'Penjual', 'lokasi']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    n_tahun_nan = df['TahunUnit'].isna().sum()
    if n_tahun_nan:
        print(f"  [!] {n_tahun_nan} baris tanpa TahunUnit dibuang (tidak masuk antik/modern)")

    df_antik = df[df['TahunUnit'] < TAHUN_BATAS_MOTOR]
    df_modern = df[df['TahunUnit'] >= TAHUN_BATAS_MOTOR]

    if 'KapasitasMesin' in df_modern.columns:
        lo, hi = RENTANG_CC_MOTOR
        valid_cc = df_modern['KapasitasMesin'].isna() | df_modern['KapasitasMesin'].between(lo, hi)
        df_modern = df_modern[valid_cc].copy()

    df_bersih, df_anomali = pisahkan_outlier(df_modern, 'Kilometer_Num')
    df_bersih = finalisasi_bersih(df_bersih)

    simpan_hasil(df_bersih, df_antik, df_anomali, OUT_MOTOR_BERSIH, OUT_MOTOR_ANTIK, OUT_MOTOR_ANOMALI)
    print_cleaning_summary('Motor', df_bersih, df_antik, df_anomali)


# ======================================================
# MAIN EXECUTION
# ======================================================
def main(argv=None):
    parser = argparse.ArgumentParser(description='Cleaning RAW Mobil123/Momotor tanpa training.')
    parser.add_argument('--vehicle', choices=['mobil', 'motor', 'all'], default='all')
    args = parser.parse_args(argv)
    selected = [('mobil', CSV_MOBIL, clean_mobil), ('motor', CSV_MOTOR, clean_motor)]
    if args.vehicle != 'all':
        selected = [item for item in selected if item[0] == args.vehicle]
    missing = [str(path) for _, path, _ in selected if not path.is_file()]
    if missing:
        parser.error('File RAW tidak ditemukan: ' + ', '.join(missing))

    print("=" * 60)
    print("MEMULAI PIPELINE PEMBERSIHAN DATA CATS")
    print("=" * 60)

    for _, _, clean in selected:
        clean()

    print("SELURUH PIPELINE SELESAI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
