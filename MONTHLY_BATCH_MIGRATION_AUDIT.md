# Audit migrasi batch bulanan RAW/CLEAN

Tanggal: 9 November 2026. Python 3.11.18. Tidak ada APScheduler, live scraping, training, commit, atau push.

## Source dan konfigurasi

- `src/data_batch.py` (baru): bulan lokal saat mulai, pemilihan RAW bulanan aktual, CSV staging/read-back/replace, rollback exception, deduplikasi URL.
- `src/Scraping/scrape_mobil123.py`, `src/Scraping/scrape_momotor.py`: batch otomatis, persistensi atomik, antrean detail unik, flush hasil parsial saat exception dengan browser tetap ditutup. Parser/browser tidak diubah.
- `src/Preprocessing/DataCleaning.py`: batch RAW terbaru untuk kendaraan terpilih, preflight file/schema, validasi batch, output atomik per kendaraan, provenance historis dipertahankan. Aturan transformasi tidak diubah.
- `src/Training/experiment_granular.py`, `src/Retraining/granular_runtime.py`: lokasi fisik Oktober menjadi nama bulanan; key provenance frozen config tetap nama lama dan hash tetap diverifikasi. Tidak mengubah bundle/frozen metadata.
- `.env.example`, `DEPLOYMENT.md`: default otomatis dan prosedur resume/regenerasi.
- `tests/test_scrape_mobil123.py`, `tests/test_scrape_momotor.py`, `tests/test_data_cleaning.py`: menyesuaikan kontrak batch/persistensi. `tests/test_monthly_batch.py` baru.
- Evidence loader tidak diubah: wildcard `CLEAN_MOBIL123_*.csv` / `CLEAN_MOMOTOR_*.csv` sudah mencakup suffix baru. Dockerfile, dependency lock, dashboard, bobot model, serta .gitignore tidak diubah.

## Tahun dan integritas historis

Timestamp scraping membuktikan Momotor April 2026 dan kedua Oktober 2026. Mobil123 April tidak memiliki timestamp scraping; tanggal_update tidak dianggap tanggal pengambilan. Pengguna mengonfirmasi April 2026 secara eksplisit sebelum rename.

Semua nama lama di tabel berada di folder yang sama dengan nama baru. Rename Windows dilakukan tanpa overwrite, dengan preflight delapan sumber/destinasi/hash dan rollback terbalik bila ada exception. Tidak ada file CSV yang dihapus secara permanen atau diubah isinya. Manifest pemulihan/hash disimpan hanya di temporary directory lokal.

| Sebelum | Sesudah | Baris | Kolom | SHA-256 sebelum = sesudah |
|---|---|---:|---:|---|
| `MOBIL123_raw_dataset_April.csv` | `Data\raw\MOBIL123_raw_dataset_2026-04.csv` | 42828 | 72 | `46edd4d91d3c7ae84d9e81ecdc21f832e7531fbe828252effbc07af88c8f93e3` |
| `MOBIL123_raw_dataset_Oktober.csv` | `Data\raw\MOBIL123_raw_dataset_2026-10.csv` | 43547 | 59 | `f27d4e420c9f47856710935c1183dfc77faf969f822ebe85ccda62701a2af308` |
| `MOMOTOR_raw_dataset_April.csv` | `Data\raw\MOMOTOR_raw_dataset_2026-04.csv` | 16291 | 12 | `79018a173e342109d7d3708bd3bf79fbed797f82a65b5480c640bc44baa39214` |
| `MOMOTOR_raw_dataset_Oktober.csv` | `Data\raw\MOMOTOR_raw_dataset_2026-10.csv` | 34934 | 12 | `05dd8b1c654cf39582c0c5fdcfe2ecd97a113d744797c10b7ef604511a401ad2` |
| `CLEAN_MOBIL123_April.csv` | `Data\clean\CLEAN_MOBIL123_2026-04.csv` | 41674 | 74 | `be0365ee11e5cb47f0eb9e01989afbb5314e8aba2b7d63eefeb2002372640792` |
| `CLEAN_MOBIL123_Oktober.csv` | `Data\clean\CLEAN_MOBIL123_2026-10.csv` | 41970 | 61 | `ccac710d2b4f83e2699d4a8f9b7453251143e45d0d3ee3d99b3323e72792b405` |
| `CLEAN_MOMOTOR_April.csv` | `Data\clean\CLEAN_MOMOTOR_2026-04.csv` | 16283 | 14 | `abe80d524f3471b4c769caf629db0e9a3c39b66d6292fa1080dde9f3e73bdf38` |
| `CLEAN_MOMOTOR_Oktober.csv` | `Data\clean\CLEAN_MOMOTOR_2026-10.csv` | 34898 | 14 | `d977f2f58fe529d1a2bae670d33cb50865de8c7c20e9f8caa4ef503b1fb6745e` |

Delapan hash, jumlah baris, nama/urutan kolom: PASS. Seluruh CSV tetap ignored dan tidak tracked; staged CSV=0. Isi `batch_id` April/Oktober dipertahankan saat rename. Tidak menyalin atau memodifikasi repository TA lama.

## Cleaning aktual

Seluruh run memakai RAW lokal, output temporary directory, dan subprocess CLI Python 3.11.18. CLEAN baseline di temporary directory disediakan terlebih dahulu untuk menguji regenerasi dan pelestarian provenance historis. `batch_provenance` mempertahankan label April/Oktober hanya pada dua batch historis terkonfirmasi ketika CLEAN sebelumnya masih menyimpan label tersebut; batch baru tanpa baseline menggunakan YYYY-MM. Tanpa baseline lama, hasil pertama batch historis memakai label YYYY-MM, sehingga hash berbeda hanya karena metadata batch_id; bukan perubahan transformasi.

| Batch | Kendaraan | Baris CLEAN | Kolom | Antik | Anomali | SHA vs baseline |
|---|---|---:|---:|---:|---:|---|
| 2026-04 | mobil | 41674 | 74 | 214 | 0 | Identik, PASS |
| 2026-04 | motor | 16283 | 14 | 0 | 0 | Identik, PASS |
| 2026-10 | mobil | 41970 | 61 | 153 | 0 | Identik, PASS |
| 2026-10 | motor | 34898 | 14 | 0 | 0 | Identik, PASS |

Run otomatis tanpa CATS_BATCH, --vehicle all, juga memilih 2026-10 dari RAW yang tersedia, mempertahankan hash kedua CLEAN Oktober dan tidak menulis ulang April: PASS. RAW/CLEAN asli tidak dipakai sebagai output pengujian.

## Marketplace dan prediksi

Indeks penuh sebelum/sesudah identik (DataFrame exact dibandingkan terhadap snapshot bernama lama dari sumber read-only): mobil 48.958, motor 26.975; EV di mobil 1.267. Identitas, deduplikasi, semua nilai/urutan indeks identik.

| Fixture lama | Harga AI (Rp) | Rentang (Rp) | Iklan relevan | Mean / median / SD (Rp) | Kategori |
|---|---:|---|---:|---|---|
| Toyota Avanza G | 191948320 | 160582967?223313673 | 93 | 199332258 / 198000000 / 13781787 | Wajar |
| Honda Revo FIT | 10384952 | 9179986?11589918 | 8 | null / null / null; insufficient_data | null |
| Hyundai Ioniq 5 Signature Long Range | 418624032 | 321686326?515561738 | 129 | 435625581 / 435000000 / 37303494 | Wajar |

Fixture diekstrak dari definisi safe test lama melalui AST, tanpa menjalankan scraper atau memilih baris aset perusahaan. /predict mobil/motor dan /predict/ev-experimental diuji melalui FastAPI TestClient. Seluruh dictionary harga/rentang/routing/SHAP/warning, hasil search, pembanding (termasuk URL dan urutan), kategori, dan statistik dibandingkan dengan baseline sebelum rename: identik. Hanya waktu_proses_ms dikecualikan karena merupakan durasi request. Perbandingan awal yang menyertakan durasi gagal; setelah batas perbandingan diperbaiki, field relevan PASS.

Endpoint `/cats/predict-market` (mobil/motor) dan `/cats/predict-market-ev` juga dibandingkan melalui TestClient dengan payload test lama dan snapshot lama vs nama bulanan: **3/3 PASS**, seluruh field relevan identik.

## Pengujian dan batas operasional

`python -B -m pytest tests -q --tb=short -p no:cacheprovider`: **94 passed**, FAIL=0, SKIP=0. Mencakup offline scraper, schema/parser, resume/deduplikasi, pergantian bulan, batch dipatok saat mulai, kegagalan tulis/validasi/replace dengan rollback, RAW belum siap, pelestarian batch historis, dan katalog/UI EV.

Integrasi tambahan: cleaning aktual 4/4 PASS, pemilihan otomatis all PASS, perbandingan indeks penuh 2/2 PASS, prediksi/SHAP/pasar tiga fixture 3/3 PASS. Model/manifest hash tetap identik. git diff --check PASS; Git memberi peringatan konversi LF/CRLF biasa, bukan kegagalan whitespace.

NOT RUN: live scraping, browser nyata, HTTP TCP eksternal, mode granular26 end-to-end (bundle eksperimen tidak disertakan). Endpoint diuji in-process, bukan browser/server eksternal. Warning deprecation Starlette/httpx tidak diatasi dengan upgrade dependency.

Resume existing sebelumnya membaca daftar URL dan melewati URL yang sudah ada di RAW; link discovery sudah memakai set. Sekarang antrean detail dibuat unik, hasil baru ditambahkan ke daftar selesai dalam sesi, penulisan link/RAW atomik dan URL duplikat tidak ditambahkan lagi. Normalisasi identitas baru lintas URL/query tidak diterapkan; kontraknya URL persis yang memang tersedia pada scraper.

Satu writer per situs/batch saja. Atomic replace melindungi tiap CSV, rollback multi-file menangani exception Python; bukan transaksi tahan power-loss atau lock lintas proses. Penghentian paksa dapat kehilangan hasil sejak checkpoint terakhir, lalu resume akan mengambil ulang. Error situs/parser tidak membuktikan listing terhapus; listing tersimpan tidak dihapus saat discovery/detail gagal. Tidak ada append CLEAN.

API/Streamlit harus direstart atau cache evidence dibersihkan setelah snapshot berubah. Keterbatasan statistik motor pada fixture (8 iklan) sudah ada sebelum migrasi; tidak dibuat hasil statistik pengganti. Izin scraping/redistribusi tetap perlu diverifikasi.

## Kesiapan tahap berikutnya

Migrasi penamaan dan regresi yang dijalankan PASS. Secara teknis dapat dilanjutkan ke perancangan APScheduler dengan single-writer/job overlap protection, recovery, dan refresh cache sebagai persyaratan tahap berikutnya. Tidak ada scheduler yang diimplementasikan atau live scraping yang diklaim berhasil.
