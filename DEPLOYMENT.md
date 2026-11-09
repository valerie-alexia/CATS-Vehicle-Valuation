# CATS Vehicle Valuation — Deployment Guide

## 1. Gambaran Sistem

CATS Vehicle Valuation adalah layanan standalone untuk
mengestimasi harga penawaran pasar kendaraan melalui FastAPI.

Aplikasi CATS dapat memanggil API menggunakan HTTP JSON.
Tidak diperlukan koneksi langsung ke database CATS.

Komponen:
- FastAPI: layanan prediksi utama.
- Streamlit: dashboard demo dan pengujian.
- XGBoost/LightGBM: model prediksi terlatih.
- Data CLEAN: pencarian iklan dan statistik pasar.
- Scraper dan DataCleaning: proses CLI terpisah.

Model aktif tidak melakukan retraining otomatis.

Harga hasil prediksi adalah estimasi harga penawaran pasar,
bukan harga transaksi atau appraisal final.

## 2. Persyaratan

- Python 3.11
- uv package manager
- Git
- Model bundle aktif yang sesuai repository
- Dataset CLEAN jika membutuhkan fitur marketplace
- Docker jika akan menggunakan container
- Chrome dan ChromeDriver kompatibel jika menjalankan scraper

Repository bersifat private. Jangan mengunggah data internal,
credential, atau dataset tanpa izin distribusi.

## 3. Struktur Penting

```text
CATS-Vehicle-Valuation/
├── Dashboard/
├── Data/
│   ├── audit/v2/
│   ├── raw/
│   └── clean/
├── Models/
│   ├── Mobil123/
│   └── Momotor/
├── src/
│   ├── Retraining/
│   ├── Scraping/
│   ├── Preprocessing/
│   └── Training/
├── Dockerfile
├── .dockerignore
├── .env.example
├── pyproject.toml
└── uv.lock
```

Folder RAW dan CLEAN disimpan lokal dan diabaikan Git.
Salin data melalui saluran privat yang diizinkan.

## 4. Instalasi Lokal

Clone repository private melalui akses yang diberikan
perusahaan, kemudian buka terminal di root repository.

Instal dependencies sesuai lockfile:

```powershell
uv sync --locked
```

Untuk menjalankan scraper:

```powershell
uv sync --locked --group scraping
```

Pastikan model aktif dan dataset yang dibutuhkan tersedia.

## 5. Menjalankan FastAPI

Windows PowerShell:

```powershell
$env:VEHICLE_EVIDENCE_DIR = (Resolve-Path ".\Data\clean").Path

uv run python -B -m uvicorn cats_demo_app:app --app-dir Data/audit/v2 --host 127.0.0.1 --port 8000
```

Endpoint pemeriksaan:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/openapi.json

Port 8000 hanya terbuka di komputer lokal pada contoh ini.

## 6. Menjalankan Streamlit

Buka terminal kedua pada root repository:

```powershell
$env:VEHICLE_EVIDENCE_DIR = (Resolve-Path ".\Data\clean").Path
$env:CATS_DEMO_API_URL = "http://127.0.0.1:8000"

uv run python -B -m streamlit run Dashboard/app_demo.py
```

Buka URL lokal yang ditampilkan Streamlit,
biasanya http://localhost:8501.

FastAPI dan Streamlit harus menggunakan snapshot CLEAN
yang sama untuk pengujian marketplace.

## 7. Endpoint API Utama

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | /health | Status model |
| GET | /catalog/{vehicle} | Katalog fitting mobil/motor |
| POST | /predict | Prediksi dasar |
| POST | /predict/batch | Batch prediksi dasar |
| POST | /cats/normalize | Normalisasi input CATS |
| POST | /cats/predict | Prediksi CATS |
| POST | /cats/predict/batch | Batch prediksi CATS |
| POST | /cats/predict-market | Prediksi dan statistik pasar |
| POST | /cats/predict-market/batch | Batch prediksi dan statistik pasar |
| POST | /cats/predict-market-ev | Eksperimen prediksi EV |
| GET | /vehicles/search | Pencarian iklan |

Endpoint CATS basic batch mendukung sampai 500 item
menurut kontrak API yang diuji sebelumnya.
Market batch dibatasi 10 item per request.

Gunakan /openapi.json sebagai sumber schema aktual.

## 8. Contoh Request Prediksi

Windows PowerShell:

```powershell
$payload = @{
    JenisAsetID = 1
    Merk = "HONDA"
    Model = "HRV"
    Varian = "1.5 S CVT CKD"
    TahunUnit = 2015
    Kilometer = 98000
    KapasitasMesin = 1497
    JenisMesin = "Bensin"
    JenisTransmisi = "AUTOMATIC"
    with_shap = $false
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/cats/predict-market" `
    -Method POST `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 30
```

Periksa prediksi_harga, rentang, peringatan,
requires_manual_review, comparables,
market_analysis dan market_review.

HTTP 200 bukan persetujuan appraisal.

## 9. Data RAW dan CLEAN

Konfigurasi lingkungan:

- CATS_BATCH: opsional; kosong untuk batch otomatis. Scraper memakai bulan lokal `YYYY-MM` saat mulai. Cleaner memilih batch RAW bulanan terbaru untuk kendaraan terpilih; bila batch terbaru mobil/motor berbeda, gunakan `--vehicle` atau override eksplisit.
- CATS_RAW_DIR: default Data/raw
- CATS_CLEAN_DIR: default Data/clean
- VEHICLE_EVIDENCE_DIR: lokasi snapshot CLEAN

Snapshot CLEAN lokal yang telah diuji:
- CLEAN_MOBIL123_2026-04.csv
- CLEAN_MOBIL123_2026-10.csv
- CLEAN_MOMOTOR_2026-04.csv
- CLEAN_MOMOTOR_2026-10.csv

Jangan mengubah dataset CLEAN yang digunakan sistem
tanpa validasi kualitas dan backup.

### Data Cleaning

Contoh pengujian menggunakan folder output terpisah:

```powershell
Remove-Item Env:CATS_BATCH -ErrorAction SilentlyContinue # pilih RAW bulanan terbaru
$env:CATS_RAW_DIR = (Resolve-Path ".\Data\raw").Path
$env:CATS_CLEAN_DIR = (Join-Path (Get-Location) "Data\clean_test")

uv run python src/Preprocessing/DataCleaning.py --vehicle all
```

DataCleaning akan membuat output sesuai konfigurasi
direktori. Jangan arahkan output pengujian ke Data/clean
jika tidak ingin menimpa snapshot yang dipakai API.

### Scraping

CLI tersedia:

```powershell
uv run python src/Scraping/scrape_mobil123.py --help
uv run python src/Scraping/scrape_momotor.py --help
```

Eksekusi normal (hanya setelah izin scraping dan kesiapan browser diverifikasi):

```powershell
uv run --group scraping python src/Scraping/scrape_mobil123.py --stage all
uv run --group scraping python src/Scraping/scrape_momotor.py --stage all
```

`--stage links` dan `--stage detail` memakai batch yang sama dalam satu bulan.
Untuk melanjutkan batch lama setelah bulan berganti, berikan `--batch 2026-10`.
Batch dipatok saat proses mulai; pergantian bulan di tengah proses tidak mengganti output.
Link dan RAW disimpan melalui file sementara, divalidasi, lalu diganti secara atomik;
URL identik tidak ditambahkan lagi. Listing yang sudah tersimpan dipertahankan saat error.
Satu penulis saja per batch; jangan menjalankan dua scraper situs yang sama bersamaan.
Penghentian paksa dapat kehilangan hasil sejak autosave terakhir, tetapi tidak menimpa
CSV sebelumnya dengan file setengah tertulis. Resume akan mengunjungi ulang hasil yang belum disimpan.

Cleaner meregenerasi CLEAN, antik, dan anomali dari RAW; tidak melakukan append CLEAN.
Ketiganya disiapkan sebelum publikasi; kegagalan Python saat publikasi melakukan rollback.
Atomisitas berlaku per file, bukan transaksi tahan power-loss untuk tiga file sekaligus.
Restart API/Streamlit setelah mengganti snapshot karena indeks evidence di-cache per worker.
Nama dataset historis sudah memakai `2026-04`/`2026-10`; isi dan metadata batch historis
CSV dipertahankan saat rename. Regenerasi mempertahankan label provenance April/Oktober
bila CLEAN historis yang bersangkutan sudah tersedia; output baru tanpa baseline memakai YYYY-MM. Gunakan `CATS_BATCH=2026-04` untuk cleaning historis secara eksplisit.
Tidak ada APScheduler atau refresh terjadwal pada tahap ini.

Scraping langsung membutuhkan Chrome/ChromeDriver yang
kompatibel dan izin penggunaan sumber data.

ChromeDriver lokal yang cocok sudah tersedia di luar repository, tetapi konfigurasi
scraper masih memilih driver otomatis melalui undetected-chromedriver. Session browser
dengan konfigurasi existing belum terverifikasi. Jangan menyatakan live scraping sudah PASS.

Jangan melewati CAPTCHA atau pembatasan akses situs.

## 10. Docker — FastAPI Standalone

Dockerfile telah dibuat tetapi build dan runtime Linux
belum diverifikasi karena Docker tidak tersedia pada
lingkungan pengujian sebelumnya.

Contoh build:

```powershell
docker build -t cats-vehicle-valuation-api .
```

Contoh run pada Windows PowerShell dengan Docker Desktop:

```powershell
$evidence = (Resolve-Path ".\Data\clean").Path

docker run --rm `
    -p 127.0.0.1:8000:8000 `
    --mount "type=bind,source=$evidence,target=/data/evidence,readonly" `
    -e VEHICLE_EVIDENCE_DIR=/data/evidence `
    cats-vehicle-valuation-api
```

Perintah di atas harus diverifikasi pada Docker yang
benar-benar tersedia.

Pastikan direktori evidence dapat dibaca oleh proses
non-root dalam container.

Dataset marketplace tidak dimasukkan ke Docker image.
Scraping dan cleaning dijalankan terpisah dari API.

API belum memiliki autentikasi internal. Jangan
mengekspos API langsung ke internet. Integrasi jarak
jauh membutuhkan pengamanan jaringan, TLS, atau proxy
sesuai kebijakan perusahaan.

## 11. Kesiapan Evidence dan Cache

GET /health memeriksa kesiapan model, bukan
kelengkapan snapshot marketplace.

Jika evidence kosong atau hilang, prediksi inti
dapat tetap berjalan, sedangkan fitur pencarian
dan statistik pasar dapat menjadi tidak tersedia
atau menghasilkan data yang tidak mencukupi.

Saat snapshot berubah, restart proses API dan
Streamlit untuk memastikan cache dimuat ulang
sesuai kondisi data terbaru.

## 12. Hasil Pengujian Lokal

Pengujian HTTP localhost yang telah dilakukan:

- Seluruh 12 endpoint yang diuji merespons HTTP 200.
- Masing-masing endpoint diuji 3 kali.
- Tidak terjadi timeout pada 36 request tersebut.
- Timeout klien pengujian: 30 detik.

Pengujian batch 40 kendaraan perusahaan:
- Total: 40 kendaraan.
- Berhasil diproses: 33.
- Ditolak validasi: 7.
- Total durasi HTTP empat request: 5,671 detik.
- Timeout: 0.
- Semua 33 prediksi ditandai membutuhkan review manual.
- Statistik pasar tersedia untuk 3 dari 33 prediksi.

Penolakan validasi meliputi:
- 3 kendaraan dengan kapasitas mesin di luar rentang
  model mobil karena routing kategori aset.
- 2 motor dengan tahun sebelum 2010.
- 2 kendaraan listrik pada endpoint model utama.

Ini adalah pengujian fungsional dan durasi HTTP lokal,
bukan akurasi model atau durasi kerja manual RM2.

Pengujian cold start, konkurensi, jaringan kantor,
Docker Linux, dan scraper live masih belum dilakukan.

## 13. Batasan

- Model utama mendukung mobil dan motor sesuai
  rentang fitur yang divalidasi.
- Model mobil EV terpisah masih eksperimental.
- Motor listrik belum didukung model utama.
- Sebagian input dapat memerlukan review manual.
- Iklan pembanding berasal dari snapshot marketplace.
- Status aktif URL iklan belum diverifikasi.
- Prediksi harga tidak otomatis ditulis ke database CATS.
- Tidak ada approval appraisal otomatis.

## 14. Troubleshooting

### API tidak dapat dibuka
Periksa terminal Uvicorn, port 8000, dan /health.

### Model tidak ditemukan
Periksa active.json dan bundle aktif pada Models.
Jangan mengubah pointer model tanpa pengujian regresi.

### Marketplace tidak menghasilkan iklan
Periksa VEHICLE_EVIDENCE_DIR, nama file CLEAN,
hak akses, dan restart service.

### HTTP 422
Periksa schema request, jenis kendaraan,
kapasitas mesin, tahun, dan bahan bakar.

### Request timeout
Periksa durasi HTTP aktual, ukuran batch,
penggunaan CPU/RAM, SHAP, dan timeout klien/proxy.

### Browser scraper gagal
Periksa kesesuaian versi Chrome/ChromeDriver,
dependency, izin akses situs, serta pesan error.

## 15. Status Serah-Terima

Sudah diuji:
- FastAPI melalui HTTP localhost.
- Streamlit headless.
- Prediksi model mobil, motor dan EV pada fixture.
- Pencarian serta statistik marketplace.
- Data Cleaning RAW ke CLEAN April/Oktober.
- Pengujian offline kedua scraper.
- Batch API menggunakan 40 input perusahaan.

Belum diverifikasi:
- Docker build dan runtime Linux.
- Scraping live menggunakan browser.
- Integrasi langsung ke database produksi CATS.
- Keamanan akses dari jaringan perusahaan.
- Pengujian concurrent load dan SLA produksi.
