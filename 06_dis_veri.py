"""
Adım 6 - Dış değişkenleri hazırlama
-----------------------------------
Tahmin modeline yardımcı olabilecek ilgili emtiaları Pink Sheet'ten çeker
ve data/dis_degiskenler.csv olarak kaydeder.

ÖNEMLİ: Pink Sheet Excel'ini daha önce sütun silerek küçülttüysen, bu adım için
ORİJİNAL (tam) dosyaya ihtiyacın var. World Bank'tan "Monthly prices" dosyasını
yeniden indir ve data/CMO-Historical-Data-Monthly.xlsx olarak koy.
(01_veri_hazirla.py tam dosyayla da aynı şekilde çalışır, fiyatlar.csv değişmez.)

Gerekli paketler: pandas, openpyxl
"""

from pathlib import Path

import pandas as pd

GIRDI = Path("data/CMO-Historical-Data-Monthly.xlsx")
CIKTI = Path("data/dis_degiskenler.csv")

# ad -> aranacak anahtar kelimeler (küçük harfle, sırayla denenir)
DIS = {
    "petrol_brent": ["crude oil, brent", "brent"],
    "dogalgaz_ab": ["natural gas, europe"],
    "komur": ["coal, australian"],
    "misir": ["maize"],
    "soya": ["soybeans"],
    "ure": ["urea"],
    "bakir": ["copper"],
}


def baslik_satirini_bul(ham):
    for i in range(min(15, len(ham))):
        satir = " ".join(str(x).lower() for x in ham.iloc[i].values)
        if "cotton" in satir and "wheat" in satir:
            return i
    raise ValueError("Sütun adlarının olduğu satır bulunamadı.")


def sutun_bul(sutunlar, anahtarlar):
    for a in anahtarlar:
        for c in sutunlar:
            if a in str(c).lower():
                return c
    return None


def main():
    if not GIRDI.exists():
        raise FileNotFoundError(f"{GIRDI} yok. Tam Pink Sheet dosyasını data/ klasörüne koy.")
    xl = pd.ExcelFile(GIRDI)
    sayfa = next((s for s in xl.sheet_names if "monthly" in s.lower() and "price" in s.lower()), None)
    if sayfa is None:
        raise ValueError(f"'Monthly Prices' sayfası yok: {xl.sheet_names}")

    ham = xl.parse(sayfa, header=None)
    bs = baslik_satirini_bul(ham)
    df = ham.iloc[bs + 1:].copy()
    df.columns = [str(c).strip() for c in ham.iloc[bs].values]
    t = df.columns[0]
    df = df[df[t].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)]
    tarih = pd.to_datetime(df[t].str.replace("M", "-", regex=False) + "-01")

    sonuc = pd.DataFrame({"tarih": tarih.values})
    print("Seçilen sütunlar (kontrol et!):")
    for ad, anahtarlar in DIS.items():
        c = sutun_bul(df.columns[1:], anahtarlar)
        if c is None:
            print(f"  {ad:13s} -> BULUNAMADI (atlandı)")
            continue
        print(f"  {ad:13s} -> {c}")
        sonuc[ad] = pd.to_numeric(df[c].values, errors="coerce")

    sonuc = sonuc.sort_values("tarih").set_index("tarih").interpolate(limit=2)
    sonuc = sonuc.loc["1990-01-01":]
    sonuc.to_csv(CIKTI)
    print(f"\nKaydedildi: {CIKTI} ({len(sonuc)} satır)")
    print(sonuc.tail())
    print("\nEksik değer sayısı:\n", sonuc.isna().sum())
    print("\nEksik değerleri çok olan bir seri varsa (özellikle 2000 sonrası) bana yaz.")


if __name__ == "__main__":
    main()
