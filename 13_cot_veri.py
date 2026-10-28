"""
Adım 13 - CFTC Commitments of Traders (COT) verisi (ücretsiz)
-------------------------------------------------------------
CFTC her hafta, vadeli piyasalarda kimin ne kadar uzun/kısa pozisyon taşıdığını
yayımlar. "Disaggregated Futures Only" raporundan pamuk, buğday ve bakır için:
  - managed money (fonlar) net pozisyon / açık pozisyon
  - producer/merchant (üretici-tüccar, hedge eden) net pozisyon / açık pozisyon
  - açık pozisyon (open interest)
çıkarır ve data/cot.csv olarak kaydeder.

VERİ KAYNAĞI (sırayla denenir):
  1) data/cot_ham/ klasöründeki dosyalar (elle indirdiysen): .zip, .txt, .csv, .xls(x)
  2) 'cot_reports' Python paketi (pip install cot_reports): yıl yıl indirir
Elle indirme: cftc.gov -> Market Reports -> Commitments of Traders -> Historical
Compressed -> "Disaggregated Futures Only" -> yıllık zip dosyaları (2011-2026).
Hepsini data/cot_ham/ klasörüne koy (zip açmana gerek yok).

ZAMANLAMA (sızıntı önlemi): Rapor Salı pozisyonlarını gösterir ama Cuma günü
(~15:30 ET) yayımlanır. Bu yüzden her satıra "kullanilabilir" = rapor tarihi + 3 gün
yazıyoruz; modelde bir Cuma kapanışında SADECE o Cumadan ÖNCE yayımlanmış raporlar
kullanılır (14 numaralı script bunu uygular).

Gerekli paketler: pandas (+ opsiyonel cot_reports)
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CIKTI = Path("data/cot.csv")
YEREL = Path("data/cot_ham")
YILLAR = list(range(2011, 2027))

# ad -> piyasa adında geçmesi gereken ifadeler
PIYASALAR = {
    "pamuk": ["COTTON NO. 2"],
    "bugday": ["WHEAT-SRW"],
    "bakir": ["COPPER"],
}


def norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def tarih_cevir(seri: pd.Series) -> pd.Series:
    """Farklı tarih biçimlerini (MM/DD/YYYY, YYYY-MM-DD, YYMMDD) dener, en iyi uyanı seçer."""
    s = seri.astype(str).str.strip()
    en_iyi, en_oran = None, -1.0
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%y%m%d"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        oran = d.notna().mean()
        if oran > en_oran:
            en_iyi, en_oran = d, oran
    return en_iyi


def zip_oku(veri) -> pd.DataFrame:
    with zipfile.ZipFile(veri) as z:
        ad = z.namelist()[0]
        return pd.read_csv(z.open(ad), low_memory=False)


def yerel_oku() -> list:
    tum = []
    if not YEREL.exists():
        return tum
    for p in sorted(YEREL.iterdir()):
        s = p.suffix.lower()
        try:
            if s == ".zip":
                tum.append(zip_oku(p))
            elif s in (".txt", ".csv"):
                tum.append(pd.read_csv(p, low_memory=False))
            elif s in (".xls", ".xlsx"):
                tum.append(pd.read_excel(p))
            else:
                continue
            print(f"  okundu: {p.name}")
        except Exception as e:
            print(f"  okunamadı: {p.name} ({e})")
    return tum


def paketten_indir() -> list:
    try:
        import cot_reports as cot
    except ImportError:
        print("  'cot_reports' paketi kurulu değil (pip install cot_reports)")
        return []
    tum = []
    for y in YILLAR:
        try:
            try:
                d = cot.cot_year(year=y, cot_report_type="disaggregated_fut",
                                 store_txt=False, verbose=False)
            except TypeError:
                d = cot.cot_year(year=y, cot_report_type="disaggregated_fut")
            tum.append(d)
            print(f"  {y}: {len(d)} satır")
        except Exception as e:
            print(f"  {y} indirilemedi: {e}")
    return tum


def ayikla(ham: pd.DataFrame) -> pd.DataFrame:
    """Ham COT tablosundan üç piyasa için özellikleri çıkarır (sütun adlarına toleranslı)."""
    kol = {norm(c): c for c in ham.columns}

    def bul(*adaylar):
        for a in adaylar:
            if a in kol:
                return kol[a]
        raise KeyError(f"Sütun bulunamadı: {adaylar}\nMevcut sütunlar: {list(ham.columns)}")

    c_ad = bul("marketandexchangenames")
    c_tar = bul("reportdateasmmddyyyy", "reportdateasyyyymmdd", "asofdateinformyyyymmdd",
                "asofdateinformyymmdd")
    c_oi = bul("openinterestall")
    c_mml = bul("mmoneypositionslongall", "moneymanagerpositionslongall", "moneymanagerlongall")
    c_mms = bul("mmoneypositionsshortall", "moneymanagerpositionsshortall", "moneymanagershortall")
    c_pl = bul("prodmercpositionslongall", "producermerchantprocessoruserpositionslongall",
               "producermerchantprocessoruserlongall")
    c_ps = bul("prodmercpositionsshortall", "producermerchantprocessoruserpositionsshortall",
               "producermerchantprocessorusershortall")

    ham = ham.copy()
    ham["_tar"] = tarih_cevir(ham[c_tar])
    parcalar = []
    for ad, ifadeler in PIYASALAR.items():
        m = ham[c_ad].astype(str).str.upper()
        sec = ham[np.logical_and.reduce([m.str.contains(i, regex=False) for i in ifadeler])]
        if ad == "bakir":                     # COMEX bakır (mini/diğer borsalar hariç)
            sec = sec[sec[c_ad].astype(str).str.upper().str.contains("COMMODITY EXCHANGE")]
        if sec.empty:
            print(f"  UYARI: {ad} için satır bulunamadı")
            continue
        sec = sec.sort_values("_tar").drop_duplicates("_tar", keep="last").set_index("_tar")
        oi = pd.to_numeric(sec[c_oi], errors="coerce")
        out = pd.DataFrame(index=sec.index)
        out[f"{ad}_mm_net_pct"] = (pd.to_numeric(sec[c_mml], errors="coerce")
                                   - pd.to_numeric(sec[c_mms], errors="coerce")) / oi
        out[f"{ad}_prod_net_pct"] = (pd.to_numeric(sec[c_pl], errors="coerce")
                                     - pd.to_numeric(sec[c_ps], errors="coerce")) / oi
        out[f"{ad}_oi"] = oi
        parcalar.append(out)
    return pd.concat(parcalar, axis=1) if parcalar else pd.DataFrame()


def main():
    Path("data").mkdir(exist_ok=True)
    print("data/cot_ham/ klasörü kontrol ediliyor...")
    tum = yerel_oku()
    if not tum:
        print("Yerel dosya yok, 'cot_reports' paketiyle indiriliyor...")
        tum = paketten_indir()
    if not tum:
        raise SystemExit(
            "\nHiç veri alınamadı. İki yol:\n"
            "  (a) pip install cot_reports   (ve SSL hatası alırsan: pip install --upgrade certifi pip-system-certs)\n"
            "  (b) CFTC sitesinden yıllık 'Disaggregated Futures Only' zip dosyalarını elle indirip data/cot_ham/ içine koy.")

    parcalar = [ayikla(d) for d in tum]
    parcalar = [p for p in parcalar if not p.empty]
    if not parcalar:
        raise SystemExit("Dosyalar okundu ama pamuk/buğday/bakır satırı çıkarılamadı (piyasa adlarını kontrol et).")
    cot = pd.concat(parcalar).sort_index()
    cot = cot[~cot.index.duplicated(keep="last")]
    cot = cot[cot.index >= "2011-01-01"]
    cot.index.name = "tarih"
    cot["kullanilabilir"] = cot.index + pd.Timedelta(days=3)     # Cuma yayım günü
    cot.to_csv(CIKTI)

    print(f"\nKaydedildi: {CIKTI} ({len(cot)} haftalık rapor, {cot.index.min().date()} -> {cot.index.max().date()})")
    print("\nEksik değer sayıları:\n", cot.drop(columns="kullanilabilir").isna().sum().to_string())
    print("\nSon 3 rapor:\n", cot.tail(3).round(3).to_string())


if __name__ == "__main__":
    main()
