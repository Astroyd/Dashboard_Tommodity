"""
Adım 1 - Fiyat verisini hazırlama
---------------------------------


Gerekli paketler: pandas, openpyxl
    pip install pandas openpyxl
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
GIRDI = DATA_DIR / "CMO-Historical-Data-Monthly.xlsx"
CIKTI = DATA_DIR / "fiyatlar.csv"

# Pink Sheet'te sütun adları sürümden sürüme biraz değişebiliyor.
# Bu yüzden tam eşleşme yerine anahtar kelimeyle arıyoruz.
HAMMADDELER = {
    "pamuk": ["cotton"],
    "aluminyum": ["aluminum", "aluminium"],
    "bugday": ["wheat"],
}
# Buğdayda birden fazla seri var (US HRW, US SRW...). Tercih sırası:
TERCIH = {"bugday": ["hrw", "us"]}


def baslik_satirini_bul(ham: pd.DataFrame) -> int:
    """İçinde 'cotton' geçen satır, sütun adlarının olduğu satırdır."""
    for i in range(min(15, len(ham))):
        satir = " ".join(str(x).lower() for x in ham.iloc[i].values)
        if "cotton" in satir and "wheat" in satir:
            return i
    raise ValueError("Sütun adlarının olduğu satır bulunamadı. Excel yapısı değişmiş olabilir.")


def sutun_sec(sutunlar, anahtarlar, tercih=None):
    adaylar = [c for c in sutunlar if any(a in str(c).lower() for a in anahtarlar)]
    if not adaylar:
        return None
    if tercih:
        for t in tercih:
            for c in adaylar:
                if t in str(c).lower():
                    return c
    return adaylar[0]


def main():
    if not GIRDI.exists():
        raise FileNotFoundError(f"{GIRDI} bulunamadı. Excel dosyasını data/ klasörüne koy.")

    xl = pd.ExcelFile(GIRDI)
    sayfa = next((s for s in xl.sheet_names if "monthly" in s.lower() and "price" in s.lower()), None)
    if sayfa is None:
        raise ValueError(f"'Monthly Prices' sayfası yok. Mevcut sayfalar: {xl.sheet_names}")

    ham = xl.parse(sayfa, header=None)
    bs = baslik_satirini_bul(ham)

    df = ham.iloc[bs + 1:].copy()
    df.columns = [str(c).strip() for c in ham.iloc[bs].values]

    # İlk sütun tarih: "1960M01" formatında
    tarih_sutunu = df.columns[0]
    df = df[df[tarih_sutunu].astype(str).str.match(r"^\d{4}M\d{2}$", na=False)]
    df["tarih"] = pd.to_datetime(
        df[tarih_sutunu].str.replace("M", "-", regex=False) + "-01"
    )

    sonuc = pd.DataFrame({"tarih": df["tarih"].values})
    kullanilan = {}
    for ad, anahtarlar in HAMMADDELER.items():
        col = sutun_sec(df.columns, anahtarlar, TERCIH.get(ad))
        if col is None:
            print(f"UYARI: {ad} için sütun bulunamadı.")
            continue
        kullanilan[ad] = col
        sonuc[ad] = pd.to_numeric(df[col].values, errors="coerce")

    sonuc = sonuc.sort_values("tarih").set_index("tarih")

    # Eksik değerleri doldur (kısa boşluklar için), uzun boşlukları bırak
    sonuc = sonuc.interpolate(limit=2)

    # 1990 sonrası yeterli: hem veri kalitesi iyi hem 2022 şoku içinde
    sonuc = sonuc.loc["1990-01-01":]

    DATA_DIR.mkdir(exist_ok=True)
    sonuc.to_csv(CIKTI)

    print("Kullanılan sütunlar:")
    for k, v in kullanilan.items():
        print(f"  {k:10s} -> {v}")
    print(f"\nKaydedildi: {CIKTI}  ({len(sonuc)} satır)")
    print(sonuc.tail())
    print("\nEksik değer sayısı:\n", sonuc.isna().sum())


if __name__ == "__main__":
    main()