"""Mobil123 notebook scraper exposed as an import-safe CLI; no training or cleaning."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
import datetime
import random

ROOT = Path(__file__).resolve().parents[2]


class AccessControlError(RuntimeError):
    """Stop instead of retrying a visible challenge or access restriction."""


def _load_dependencies():
    try:
        import pandas as pd
        from bs4 import BeautifulSoup
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import undetected_chromedriver as uc
    except ImportError:
        raise RuntimeError(
            'Dependency scraper belum tersedia: pandas, beautifulsoup4, selenium, '
            'undetected-chromedriver. Tidak ada browser yang dijalankan.'
        ) from None
    return pd, BeautifulSoup, By, WebDriverWait, EC, uc


def _page_soup(driver, BeautifulSoup):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    visible = soup.get_text(' ', strip=True).casefold()
    markers = ('captcha', 'verify you are human', 'access denied', 'too many requests',
               'akses ditolak', 'verifikasi bahwa anda manusia', 'rate limit exceeded',
               '403 forbidden', '429 too many requests')
    if any(marker in visible for marker in markers):
        raise AccessControlError('Kontrol akses/CAPTCHA/pembatasan terdeteksi; scraping dihentikan.')
    return soup


def get_chrome_version():
    """Mendeteksi versi Google Chrome yang terinstall di Windows."""
    
    kemungkinan_path = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
    ]

    for chrome_path in kemungkinan_path:
        if os.path.exists(chrome_path):
            try:
                output = subprocess.check_output(
                    [
                        "powershell",
                        "-Command",
                        f"(Get-Item '{chrome_path}').VersionInfo.ProductVersion"
                    ],
                    stderr=subprocess.DEVNULL
                ).decode().strip()

                match = re.search(r"(\d+)", output)

                if match:
                    return int(match.group(1))

            except Exception:
                pass

    return None


def _create_driver(uc):
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheet": 2
    }
    options.add_experimental_option("prefs", prefs)
    options.page_load_strategy = 'eager' 

    CHROME_VERSION = get_chrome_version()

    if CHROME_VERSION:
        print(f"Chrome terdeteksi: versi {CHROME_VERSION}")
    
        driver = uc.Chrome(
            options=options,
            version_main=CHROME_VERSION
        )
    else:
        print("⚠️ Versi Chrome tidak berhasil dideteksi.")
        print("   Mencoba menjalankan undetected_chromedriver secara otomatis...")
    
        driver = uc.Chrome(options=options)
    return driver


def run_scraper(batch, output_dir, stage='all'):
    output_dir = Path(output_dir)
    FILE_LINKS = output_dir / f'MOBIL123_daftar_link_{batch}.csv'
    FILE_HASIL = output_dir / f'MOBIL123_raw_dataset_{batch}.csv'
    if stage == 'detail' and not FILE_LINKS.is_file():
        raise FileNotFoundError('Daftar link Mobil123 tidak ditemukan; jalankan stage links terlebih dahulu.')
    pd, BeautifulSoup, By, WebDriverWait, EC, uc = _load_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    PROVINSI = 'indonesia'
    JALANKAN_TAHAP_1 = stage in {'links', 'all'}
    print(f'File links : {FILE_LINKS}')
    print(f'File hasil : {FILE_HASIL}')
    driver = _create_driver(uc)
    try:
        wait = WebDriverWait(driver, 10)
        if JALANKAN_TAHAP_1:
            print(f"\n Filter tahun di {PROVINSI}...")
            try:
                driver.get(f"https://www.mobil123.com/mobil-bekas-dijual/{PROVINSI}")
                _page_soup(driver, BeautifulSoup)
                time.sleep(5) 
        
                soup_awal = _page_soup(driver, BeautifulSoup)
        
                # ========================================================
                # [MODIFIKASI 2]: Print H1 Headline Total Mobil di Indonesia
                # ========================================================
                headline_elem = soup_awal.find("h1", class_=re.compile("headline", re.IGNORECASE))
                if headline_elem:
                    print(f"\n{'='*50}")
                    print(f"INFO SCRAPING: {headline_elem.get_text(strip=True)}")
                    print(f"{'='*50}\n")
                # ========================================================
        
                list_tahun = []
        
                # Regex mencari pola: "2025 (1,234)" atau "1998 (5)"
                pola_filter = re.compile(r"^(19[4-9]\d|20[0-3]\d)\s*\([\d\.,]+\)$")
                semua_label_filter = soup_awal.find_all(string=pola_filter)
        
                for label in semua_label_filter:
                    tahun = int(label.strip().split()[0]) 
                    list_tahun.append(tahun)
            
                if list_tahun:
                    list_tahun = sorted(list(set(list_tahun)), reverse=True)
                    print(f"Sukses! Ditemukan {len(list_tahun)} tahun aktif di filter.")
                    print(f"   ➜ Data: {list_tahun[0]} sampai {list_tahun[-1]}")
                else:
                    print("⚠️ Gagal menemukan filter. Menggunakan fallback (50 tahun ke belakang).")
                    tahun_sekarang = datetime.datetime.now().year
                    list_tahun = list(range(tahun_sekarang, tahun_sekarang - 50, -1))
            
            except AccessControlError:
                raise
            except Exception as e:
                print(f"⚠️ Error saat ngintip filter: {e}. Menggunakan fallback.")
                tahun_sekarang = datetime.datetime.now().year
                list_tahun = list(range(tahun_sekarang, tahun_sekarang - 50, -1))

        # ==========================================
        # TAHAP 1: KUMPULKAN LINK DARI TIAP TAHUN 
        # ==========================================
        if JALANKAN_TAHAP_1:
            print("Membangun/Melanjutkan Database Link...")

    
            all_links = set()

            if os.path.exists(FILE_LINKS):
                df_existing = pd.read_csv(FILE_LINKS, low_memory=False)

                if "terjual" not in df_existing.columns:
                    df_existing["terjual"] = 0
                    df_existing.to_csv(FILE_LINKS, index=False)

                if "link" in df_existing.columns:
                    all_links = set(df_existing["link"].dropna().tolist())

                print(f"Ditemukan {len(all_links)} link dari sesi sebelumnya.")

            for tahun in list_tahun:
                print(f"\nSedang mencari mobil tahun {tahun}...")
                link_baru_tahun_ini = 0
                base_url_tahun = f"https://www.mobil123.com/mobil-bekas-dijual/{PROVINSI}?type=used&min_year={tahun}&max_year={tahun}&page_size=25&page_number={{}}"
        
                page = 1 
                while True:
                    sukses_muat_halaman = False
                    halaman_habis = False 
                    max_retries = 3 
            
                    for attempt in range(max_retries):
                        try:
                            driver.get(base_url_tahun.format(page))
                            _page_soup(driver, BeautifulSoup)
                            wait.until(EC.presence_of_element_located((By.ID, "classified-listings-result")))
                            soup = _page_soup(driver, BeautifulSoup)
                    
                            section = soup.find("section", id="classified-listings-result")
                            if not section: 
                                halaman_habis = True 
                                break 
                    
                            articles = section.find_all("article")
                    
                            valid_articles = []
                            for art in articles:
                                if art.find(string=re.compile("Rp")):
                                    valid_articles.append(art)
                            
                            if len(valid_articles) == 0: 
                                print(f"   ➜ Mentok di halaman {page} (Kosong/Habis). Pindah tahun.")
                                sukses_muat_halaman = True 
                                halaman_habis = True 
                                break 
                    
                            new_links_in_page = []
                            for art in valid_articles:
                                a_tag = art.find("h2").find("a") if art.find("h2") else None
                                if a_tag:
                                    href = a_tag["href"]
                                    if href not in all_links:
                                        all_links.add(href)
                                        new_links_in_page.append(href)
                                        link_baru_tahun_ini += 1
                    
                            if new_links_in_page:
                                df_new_links = pd.DataFrame({
                                    "link": new_links_in_page,
                                    "terjual": 0
                                })
                                write_header = not os.path.exists(FILE_LINKS)
                                df_new_links.to_csv(FILE_LINKS, mode='a', index=False, header=write_header)
                            
                            time.sleep(random.uniform(1.0, 2.5))
                            sukses_muat_halaman = True
                            break
                    
                        except AccessControlError:
                            raise
                        except Exception as e:
                            print(f"   ⚠️ Halaman {page} nyangkut/504 (Percobaan {attempt + 1}/{max_retries}). Istirahat 10 detik...")
                            time.sleep(10) 
            
                    if halaman_habis:
                        break
            
                    # Jika sudah dicoba 3 kali dan masih gagal karena server lemot
                    if not sukses_muat_halaman:
                        print(f"   ❌ Server nyerah di halaman {page}. Pindah ke tahun sebelumnya.")
                        break 

                    page += 1

                print(f"   ➜ Total dapat {link_baru_tahun_ini} link baru untuk tahun {tahun}.")

            print("\nTAHAP 1 SELESAI. Database Link Aman.")
        else:
            print("Tahap 1 di-skip. Lanjut ke Tahap 2...")
        if stage == 'links':
            return
        print("\nMemulai Ekstrak Detail...")

        if os.path.exists(FILE_LINKS):
            df_semua_link = pd.read_csv(FILE_LINKS)
    
            # ========================================================
            # [MODIFIKASI 3]: Cek kolom 'terjual'
            # ========================================================
            if 'terjual' not in df_semua_link.columns:
                df_semua_link['terjual'] = 0
    
            daftar_link = df_semua_link[df_semua_link['terjual'] != 1]['link'].tolist()
        else:
            daftar_link = []
            print("⚠️ File Link tidak ditemukan. Pastikan Tahap 1 berjalan dengan benar.")

        link_sudah_selesai = []
        if os.path.exists(FILE_HASIL):
            df_selesai = pd.read_csv(FILE_HASIL)
            if 'link' in df_selesai.columns:
                link_sudah_selesai = df_selesai['link'].tolist()

        target_sisa = len(daftar_link) - len(link_sudah_selesai)

        data_baru = [] 
        count_diproses = 0  
        count_disimpan = 0  

        for link in daftar_link:
            if link in link_sudah_selesai:
                continue 
        
            try:
                count_diproses += 1
        
                # ========================================================
                # [MODIFIKASI 4]: Print 1 Baris Saja dengan \r
                # ========================================================
                print(f"\r🔄 Mengekstrak {count_diproses}/{target_sisa} link berjalan...", end="", flush=True)

                driver.get(link)
                _page_soup(driver, BeautifulSoup)
                time.sleep(1) 

                soup = _page_soup(driver, BeautifulSoup)
        
                # ========================================================
                # [MODIFIKASI 5]: Deteksi "Mobil ini sudah terjual"
                # ========================================================
                terjual_elem = soup.find(string=re.compile("Mobil ini sudah terjual", re.IGNORECASE))
                if terjual_elem:
                    df_semua_link.loc[df_semua_link['link'] == link, 'terjual'] = 1
                    df_semua_link.to_csv(FILE_LINKS, index=False)
                    continue 
                # ========================================================
        
                try:
                    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "c-key-details__item")))
                    soup = _page_soup(driver, BeautifulSoup) 
                except AccessControlError:
                    raise
                except:
                    pass 

                details = {"link": link}
        
                # ========================================================
                # [MODIFIKASI 6]: Tambahkan Kolom Tanggal Scrape
                # ========================================================
                details["tanggal_scrape"] = datetime.datetime.now().strftime("%Y-%m-%d")
                # ========================================================
        
                # Nama Mobil
                header = soup.find("h1")
                details["nama_mobil"] = header.get_text(strip=True) if header else "Unknown"
        
                # Harga
                harga_final = "Unknown"

                price_elem = soup.find(class_=re.compile("price", re.IGNORECASE))
                if price_elem and "Rp" in price_elem.get_text():
                    harga_final = price_elem.get_text(strip=True)
                else:
                    teks_rp = soup.find(string=re.compile(r"Rp\s*[\d\.,]+"))
                    if teks_rp:
                        harga_final = teks_rp.strip()

                details["harga"] = harga_final
        
                # Tanggal Update (di website)
                date_tag = soup.find("span", string=re.compile("Diperbarui pada:"))
                details["tanggal_update"] = date_tag.get_text(strip=True).replace("Diperbarui pada:", "").strip() if date_tag else None
        
                # Detail Lainnya (CC, Tahun, Kilometer, dll)
                key_sections = soup.find_all("div", class_="c-key-details__item")
                for sec in key_sections:
                    label = sec.find("span", class_="u-text-7")
                    value = sec.find("span", class_="u-text-bold")
                    if label and value:
                        details[label.get_text(strip=True)] = value.get_text(strip=True)

                # Tab Spesifikasi 
                spec_tab = soup.find("div", id="tab-specifications")
                if spec_tab:
                    rows = spec_tab.find_all("div", class_="u-border-bottom")
                    for row in rows:
                        spans = row.find_all("span")
                        if len(spans) >= 2:
                            details[spans[0].get_text(strip=True)] = spans[1].get_text(strip=True)

                # ========================================================
                # [MODIFIKASI 1]: Tab Kelengkapan diubah menjadi Flag 1
                # ========================================================
                equipment_tab = soup.find("div", id=re.compile("tab-equipment|equipment", re.IGNORECASE))
                if equipment_tab:
                    rows_eq = equipment_tab.find_all("div", class_=re.compile("u-border-bottom", re.IGNORECASE))
                    for r in rows_eq:
                        spans = r.find_all("span")
                        if len(spans) >= 1:
                            label_fitur = spans[0].get_text(strip=True)
                            if label_fitur: 
                                details[f"Fitur_{label_fitur}"] = "1"
                # ========================================================

                data_baru.append(details)
                count_disimpan += 1
        
                if count_disimpan % 20 == 0:
                    print(f"\r   ✅ [Auto-Save] {count_disimpan} mobil berhasil diekstrak dan disimpan...")
                    df_temp_baru = pd.DataFrame(data_baru)
            
                    if os.path.exists(FILE_HASIL):
                        df_lama = pd.read_csv(FILE_HASIL)
                        df_gabungan = pd.concat([df_lama, df_temp_baru], ignore_index=True)
                        df_gabungan.to_csv(FILE_HASIL, index=False)
                    else:
                        df_temp_baru.to_csv(FILE_HASIL, index=False)
            
                    data_baru = []
            
                if count_disimpan % 100 == 0:
                    print("\n   ⏳ 15 sec break ...")
                    time.sleep(15) 

                time.sleep(random.uniform(1.0, 2.5)) 
        
            except AccessControlError:
                raise
            except Exception as e:
                pass

        if len(data_baru) > 0:
            df_temp_baru = pd.DataFrame(data_baru)
            if os.path.exists(FILE_HASIL):
                df_lama = pd.read_csv(FILE_HASIL)
                df_gabungan = pd.concat([df_lama, df_temp_baru], ignore_index=True)
                df_gabungan.to_csv(FILE_HASIL, index=False)
            else:
                df_temp_baru.to_csv(FILE_HASIL, index=False)

        print("\n\nDone! Scraping Selesai.")
    finally:
        driver.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scraper Mobil123; hanya jalankan dengan izin penggunaan situs.')
    parser.add_argument('--batch', default=os.environ.get('CATS_BATCH', 'Oktober'))
    parser.add_argument('--stage', choices=['links', 'detail', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(os.environ.get('CATS_RAW_DIR', str(ROOT / 'Data/raw'))))
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.batch):
        parser.error('--batch harus berupa suffix nama file tanpa separator direktori')
    try:
        run_scraper(args.batch, args.output_dir, args.stage)
    except (AccessControlError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
