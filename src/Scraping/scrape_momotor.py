"""Momotor notebook scraper exposed as an import-safe CLI; no training or cleaning."""
import argparse
import sys
import os
from pathlib import Path
import re
import subprocess
import time
import datetime
import random

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))
from src.data_batch import default_batch, save_scrape



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


def konversi_tanggal_momotor(teks_tanggal):
    if not teks_tanggal or teks_tanggal == "Unknown":
        return "Unknown"
        
    teks_tanggal = str(teks_tanggal).lower().strip()
    hari_ini = datetime.datetime.now() 
    
    if "hari ini" in teks_tanggal:
        return hari_ini.strftime("%Y-%m-%d")
        
    elif "hari yang lalu" in teks_tanggal:
        try:
            angka_hari = int(re.search(r'\d+', teks_tanggal).group())
            tanggal_target = hari_ini - datetime.timedelta(days=angka_hari)
            return tanggal_target.strftime("%Y-%m-%d")
        except:
            return teks_tanggal
            
    else:
        try:
            teks_dengan_tahun = f"{teks_tanggal} {hari_ini.year}"
            bulan_indo_ke_eng = {
                'jan': 'jan', 'feb': 'feb', 'mar': 'mar', 'apr': 'apr',
                'mei': 'may', 'jun': 'jun', 'jul': 'jul', 'agu': 'aug',
                'sep': 'sep', 'okt': 'oct', 'nov': 'nov', 'des': 'dec'
            }
            for id_bulan, en_bulan in bulan_indo_ke_eng.items():
                if id_bulan in teks_dengan_tahun:
                    teks_dengan_tahun = teks_dengan_tahun.replace(id_bulan, en_bulan)
                    break
                    
            tanggal_target = datetime.datetime.strptime(teks_dengan_tahun, "%d %b %Y")
            return tanggal_target.strftime("%Y-%m-%d")
        except:
            return teks_tanggal


def run_scraper(batch, output_dir, stage='all'):
    output_dir = Path(output_dir)
    FILE_LINKS = output_dir / f'MOMOTOR_daftar_link_{batch}.csv'
    FILE_HASIL = output_dir / f'MOMOTOR_raw_dataset_{batch}.csv'
    if stage == 'detail' and not FILE_LINKS.is_file():
        raise FileNotFoundError('Daftar link Momotor tidak ditemukan; jalankan stage links terlebih dahulu.')
    pd, BeautifulSoup, By, WebDriverWait, EC, uc = _load_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    KATEGORI = 'motor'
    JALANKAN_TAHAP_1 = stage in {'links', 'all'}
    print(f'File links : {FILE_LINKS}')
    print(f'File hasil : {FILE_HASIL}')
    data_baru = []
    driver = _create_driver(uc)
    try:
        wait = WebDriverWait(driver, 10)
        all_links = set()

        if os.path.exists(FILE_LINKS):
            df_existing = pd.read_csv(FILE_LINKS, low_memory=False)

            if "link" not in df_existing.columns:
                raise ValueError("File daftar link tidak memiliki kolom link.")

            expected_cols = ["link", "tanggal_posting", "terjual"]
            extra_cols = set(df_existing.columns) - set(expected_cols)

            if extra_cols:
                raise ValueError(
                    f"Kolom tidak dikenal pada file daftar link: {extra_cols}"
                )

            if "tanggal_posting" not in df_existing.columns:
                df_existing["tanggal_posting"] = "Unknown"

            if "terjual" not in df_existing.columns:
                df_existing["terjual"] = 0

            if df_existing.columns.tolist() != expected_cols:
                df_existing = df_existing[expected_cols]
                save_scrape(df_existing, FILE_LINKS)

            all_links = set(df_existing["link"].dropna().tolist())

            print(f"Ditemukan {len(all_links)} link dari sesi sebelumnya.")


        # ==========================================
        # TAHAP 0: AUTO-DETECT TAHUN & INFO GLOBAL
        # ==========================================
        list_tahun = []
        if JALANKAN_TAHAP_1:
            print("\n[TAHAP 0] Membaca rentang tahun dari website...")
            try:
                driver.get("https://momotor.id/motor-bekas")
                _page_soup(driver, BeautifulSoup)
        
                # ========================================================
                # [MODIFIKASI 1]: Print H2 Global (Semua Unit Motor Bekas)
                # ========================================================
                for _ in range(15): # Tunggu maksimal 15 detik agar server selesai render angka
                    soup_awal = _page_soup(driver, BeautifulSoup)
                    h2_global = soup_awal.find("h2", class_=re.compile("titleSearch", re.IGNORECASE))
                    if h2_global:
                        teks_global = h2_global.get_text(separator=" ", strip=True)
                        match_global = re.search(r'\(\s*([\d\.]+)\s*pencarian', teks_global, re.IGNORECASE)
                        if match_global:
                            val_global = int(match_global.group(1).replace('.', ''))
                            if val_global > 0: # Pastikan angkanya bukan 0 bawaan loading
                                print(f"\n{'='*50}")
                                print(f"INFO SCRAPING: {teks_global}")
                                print(f"{'='*50}\n")
                                break
                    time.sleep(1)
                # ========================================================

                min_year_elem = wait.until(EC.presence_of_element_located((By.ID, "min-year-select")))
                max_year_elem = driver.find_element(By.ID, "max-year-select")
        
                min_year = int(min_year_elem.text.strip())
                max_year = int(max_year_elem.text.strip())
        
                list_tahun = list(range(max_year, min_year - 1, -1))
        
                print(f"Ditemukan rentang tahun: {list_tahun[0]} sampai {list_tahun[-1]}")
            
            except AccessControlError:
                raise
            except Exception as e:
                print(f"⚠️ Gagal membaca filter tahun: {e}. Menggunakan fallback manual (2026-2000).")
                list_tahun = list(range(datetime.datetime.now().year, 1989, -1))


        # ==========================================
        # TAHAP 1: EKSTRAKSI LINK MOMOTOR 
        # ==========================================
        if JALANKAN_TAHAP_1 and len(list_tahun) > 0:
            print(f"\n[TAHAP 1] Membangun Database Link Momotor...")

            for tahun in list_tahun:
                print(f"\n=======================================================")
                print(f"🏍️ MENCARI MOTOR TAHUN: {tahun}")
                print(f"=======================================================")
        
                url_slicing = f"https://momotor.id/motor-bekas?year={tahun}%2C{tahun}"
                driver.get(url_slicing)
                _page_soup(driver, BeautifulSoup)
        
                # ========================================================
                # [MODIFIKASI 2]: Dynamic Polling untuk Target Unit (Mencegah "0 Unit")
                # ========================================================
                jumlah_unit_target = 0 
                for _ in range(15): # Cek berulang selama maksimal 15 detik
                    soup_cek = _page_soup(driver, BeautifulSoup)
                    h2_title = soup_cek.find("h2", class_=re.compile("titleSearch"))
            
                    if h2_title:
                        teks_h2 = h2_title.get_text(separator=" ", strip=True)
                        match_pencarian = re.search(r'\(\s*([\d\.]+)\s*pencarian', teks_h2, re.IGNORECASE)
                
                        if match_pencarian:
                            val_target = int(match_pencarian.group(1).replace('.', ''))
                            if val_target > 0: # Cek apakah server Momotor sudah memberi angka asli
                                jumlah_unit_target = val_target
                                print(f"    Target: {jumlah_unit_target} unit")
                                break
                    time.sleep(1)
            
                if jumlah_unit_target == 0:
                    print(f"    Target: 0 unit (Tidak ada motor atau server lambat memuat)")
                # ========================================================
        
                last_link_count = 0
                retry_stuck = 0
                max_retries_stuck = 5 
        
                for i in range(400): 
                    current_soup = _page_soup(driver, BeautifulSoup)
                    current_links = current_soup.find_all("a", href=re.compile(r"/motor-bekas/.+/.+"))
                    current_count = len(current_links)
            
                    if jumlah_unit_target > 0 and current_count >= jumlah_unit_target:
                        print(f"\n      Target {jumlah_unit_target} unit sudah tercapai di layar. Stop scroll.")
                        break
            
                    if current_count > last_link_count:
                        last_link_count = current_count
                        retry_stuck = 0
            
                    try:
                        xpath_teks = "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'muat') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'lebih banyak')]"
                
                        wait_lokal = WebDriverWait(driver, 5)
                        btn_load_more = wait_lokal.until(EC.presence_of_element_located((By.XPATH, xpath_teks)))
                
                        is_disabled = btn_load_more.get_attribute("disabled") is not None or "disabled" in btn_load_more.get_attribute("class").lower()
                
                        if is_disabled:
                            raise Exception("Tombol disabled sementara")
                
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn_load_more)
                        time.sleep(1.5)
                        driver.execute_script("arguments[0].click();", btn_load_more)
                
                        print(f"\r      Menggali: {current_count} dari {jumlah_unit_target} item terdeteksi... (Klik ke-{i+1})", end="", flush=True)
                
                        time.sleep(random.uniform(3.5, 5.0)) 
                
                    except AccessControlError:
                        raise
                    except Exception as e:
                        if retry_stuck < max_retries_stuck:
                            print(f"\r      Server lambat! Memancing loading data... (Coba ulang {retry_stuck+1}/{max_retries_stuck})", end="", flush=True)
                    
                            driver.execute_script("window.scrollBy(0, 500);")
                            time.sleep(1)
                            driver.execute_script("window.scrollBy(0, -200);")
                            time.sleep(4) 
                    
                            retry_stuck += 1
                        else:
                            print(f"\n      Ekspedisi selesai/Mentok di {current_count} item.")
                            break
                
                print()
        
                soup_list = _page_soup(driver, BeautifulSoup)
                semua_link_iklan = soup_list.find_all("a", href=re.compile(r"/motor-bekas/.+/.+"))
        
                link_baru_tahun_ini = 0
                new_data_in_page = []
        
                for a_tag in semua_link_iklan:
                    href_raw = a_tag["href"]
                    href_full = href_raw if href_raw.startswith("http") else "https://momotor.id" + href_raw
            
                    if href_full not in all_links:
                        all_links.add(href_full)
                
                        date_elem = a_tag.find("p", class_=re.compile(r"date", re.IGNORECASE))
                        posting_date_raw = date_elem.get_text(strip=True) if date_elem else "Unknown"
                
                        # ========================================================
                        # [MODIFIKASI 3]: Memanggil fungsi pembersih tanggal
                        # ========================================================
                        posting_date_bersih = konversi_tanggal_momotor(posting_date_raw)
                
                        new_data_in_page.append({
                            "link": href_full,
                            "tanggal_posting": posting_date_bersih,
                            "terjual": 0
                        })
                        link_baru_tahun_ini += 1

                if new_data_in_page:
                    df_new_links = pd.DataFrame(new_data_in_page)
                    write_header = not os.path.exists(FILE_LINKS)
                    save_scrape(df_new_links, FILE_LINKS, append=True)
            
                print(f"    Tersimpan {link_baru_tahun_ini} link baru untuk tahun {tahun}.")
                time.sleep(2)

            print(f"\nTAHAP 1 SELESAI.")
        else:
            print("\nTahap 1 di-skip. Lanjut ke Tahap 2...")
        if stage == 'links':
            return
        print(f"\n[TAHAP 2] Memulai Ekstrak Detail {KATEGORI.upper()} di Momotor...")

        df_links = pd.read_csv(FILE_LINKS, low_memory=False)

        if 'terjual' not in df_links.columns:
            df_links['terjual'] = 0

        daftar_link = df_links[df_links['terjual'] != 1]['link'].dropna().drop_duplicates().tolist()

        if 'tanggal_posting' in df_links.columns:
            dict_tgl_posting = dict(zip(df_links['link'], df_links['tanggal_posting']))
        else:
            dict_tgl_posting = {}

        link_sudah_selesai = []
        if os.path.exists(FILE_HASIL):
            df_selesai = pd.read_csv(FILE_HASIL, low_memory=False)
            if 'link' in df_selesai.columns:
                link_sudah_selesai = df_selesai['link'].dropna().drop_duplicates().tolist()

        target_sisa = len(daftar_link) - len(link_sudah_selesai)
        data_baru = [] 
        count_diproses = 0
        count_disimpan = 0

        print(f"➜ Ada {len(daftar_link)} total link aktif. {len(link_sudah_selesai)} sudah selesai. Sisa {target_sisa} link.")

        for link in daftar_link:
            if link in link_sudah_selesai:
                continue 
        
            try:
                count_diproses += 1
                print(f"\r🔄 Mengekstrak {count_diproses}/{target_sisa} link berjalan...", end="", flush=True)

                driver.get(link)
                _page_soup(driver, BeautifulSoup)
                time.sleep(1) 

                soup = _page_soup(driver, BeautifulSoup)
        
                terjual_elem = soup.find(string=re.compile("Mobil ini sudah terjual|Motor ini sudah terjual", re.IGNORECASE))
                if terjual_elem:
                    df_links.loc[df_links['link'] == link, 'terjual'] = 1
                    save_scrape(df_links, FILE_LINKS)
                    continue 
        
                try:
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
                    time.sleep(random.uniform(0.5, 1.5))
                    soup = _page_soup(driver, BeautifulSoup) 
                except AccessControlError:
                    raise
                except:
                    pass 

                details = {"link": link}
        
                details["tanggal_posting"] = dict_tgl_posting.get(link, "Unknown")
                details["waktu_scrape"] = datetime.datetime.now().strftime("%Y-%m-%d")

                info_container = soup.find("div", class_=re.compile("infoContainer"))
        
                if info_container:
                    nama_elem = info_container.find("h1", class_=re.compile("title"))
                    details["nama_motor"] = nama_elem.get_text(strip=True) if nama_elem else "Unknown"
            
                    merk_elem = info_container.find("p", class_=re.compile("subtitle"))
                    details["merk"] = merk_elem.get_text(strip=True) if merk_elem else "Unknown"
            
                    lokasi_elem = info_container.find("div", class_=re.compile("locationWrapper"))
                    details["lokasi"] = lokasi_elem.get_text(strip=True) if lokasi_elem else "Unknown"
                
                    detail_items = info_container.find_all("div", class_=re.compile("detailItem"))
                    for item in detail_items:
                        label = item.find("p", class_="caption")
                        value = item.find("p", class_=re.compile("subtitle"))
                
                        if label and value:
                            key_name = label.get_text(strip=True) 
                            val_text = value.get_text(strip=True) 
                            details[key_name] = val_text

                harga_final = "Unknown"
                harga_desktop = soup.find("h2", class_=re.compile("priceDesktop"))
                if harga_desktop:
                    harga_final = harga_desktop.get_text(strip=True)
                else:
                    harga_mobile = soup.find("p", class_=re.compile("priceMobile"))
                    if harga_mobile:
                        harga_final = harga_mobile.get_text(strip=True)
                
                details["harga"] = harga_final

                data_baru.append(details)
                link_sudah_selesai.append(link)
                count_disimpan += 1
        
                if count_disimpan % 20 == 0:
                    print(f"\n   ✅ [Auto-Save] {count_disimpan} data motor berhasil diekstrak dan disimpan...")
                    df_temp_baru = pd.DataFrame(data_baru)
            
                    if os.path.exists(FILE_HASIL):
                        df_lama = pd.read_csv(FILE_HASIL, low_memory=False)
                        df_gabungan = pd.concat([df_lama, df_temp_baru], ignore_index=True)
                        save_scrape(df_gabungan, FILE_HASIL)
                    else:
                        save_scrape(df_temp_baru, FILE_HASIL)
                    data_baru = []
            
                if count_disimpan % 100 == 0:
                    print("\n   ⏳ 15 sec break ...\n")
                    time.sleep(15) 
            
            except AccessControlError:
                raise
            except Exception as e:
                pass 

        if len(data_baru) > 0:
            df_temp_baru = pd.DataFrame(data_baru)
            if os.path.exists(FILE_HASIL):
                df_lama = pd.read_csv(FILE_HASIL, low_memory=False)
                df_gabungan = pd.concat([df_lama, df_temp_baru], ignore_index=True)
                save_scrape(df_gabungan, FILE_HASIL)
            else:
                save_scrape(df_temp_baru, FILE_HASIL)

        data_baru = []
        print("\n\nEkstraksi Momotor Selesai Total!")
    finally:
        try:
            if data_baru:
                save_scrape(pd.DataFrame(data_baru), FILE_HASIL, append=True)
        finally:
            driver.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scraper Momotor; hanya jalankan dengan izin penggunaan situs.')
    parser.add_argument('--batch', default=default_batch())
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
