"""
Adım 11 - ETF vekil serileri (roll sıçramasız günlük veri)
-----------------------------------------------------------
Sorun: Yahoo'daki "=F" serilerinde kontrat değişim sıçramaları var, tek tek
kontratların geçmişi ise sadece son ~2,7 yıl için duruyor (10 numaralı script
bunu gösterdi). 2012'den beri temiz günlük veri için ikinci yol: emtia ETF'leri.

ETF'ler vadeli kontratları kademeli olarak değiştirdiği için fiyat serilerinde
roll sıçraması olmaz, ve düzeltilmiş kapanış (temettü/bölünme) Yahoo'da bulunur.

ÖNEMLİ SINIRLAR:
  - ETF getirisi, spot fiyatın değil "vadeli kontrat portföyünün" getirisidir
    (taşıma/roll getirisi içerir). Araştırma vekili olarak makul, bir ürün için
    gerçek vadeli veri sağlayıcısı gerekir.
  - BAL (pamuk) için ETN'in hâlâ işlem görüp görmediğini bilmiyorum; rapor gösterecek.

Çıktı: data/gunluk_etf.csv (düzeltilmiş kapanışlar) + ekrana doğrulama raporu.
Rapor, ETF getirilerini =F serileriyle (data/gunluk.csv) karşılaştırır.

Gerekli paketler: pandas, numpy, yfinance
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BAS = "2011-06-01"
CIKTI = Path("data/gunluk_etf.csv")
GUNLUK = Path("data/gunluk.csv")

HEDEF = {"pamuk": "BAL", "bugday": "WEAT", "bakir": "CPER"}
ADAY = {  # yardımcı / alternatif
    "misir_etf": "CORN",
    "soya_etf": "SOYB",
    "seker_etf": "CANE",
    "tarim_sepeti": "DBA",
    "baz_metal": "DBB",
}


def main():
    Path("data").mkdir(exist_ok=True)
    semboller = {**HEDEF, **ADAY}
    ham = yf.download(list(semboller.values()), start=BAS, auto_adjust=True,
                      progress=False, group_by="column")["Close"]
    ham.columns = [k for s in ham.columns for k, v in semboller.items() if v == s]
    ham.index = pd.to_datetime(ham.index).tz_localize(None)
    ham = ham.sort_index()

    print("=== Kapsam ===")
    for c in ham.columns:
        s = ham[c].dropna()
        if s.empty:
            print(f"{c:14s} ({semboller[c]}): VERİ YOK")
            continue
        print(f"{c:14s} ({semboller[c]}): {s.index.min().date()} -> {s.index.max().date()} ({len(s)} gün)")

    print("\n=== En büyük günlük hareketler (hedef seriler) ===")
    for c in HEDEF:
        if c not in ham.columns or ham[c].dropna().empty:
            continue
        r = np.log(ham[c].dropna()).diff().dropna()
        mad = 1.4826 * (r - r.median()).abs().median()
        print(f"\n{c}: std={r.std()*100:.2f}%  5xMAD'ı aşan gün={int((r.abs() > 5*mad).sum())}")
        for t in r.abs().sort_values(ascending=False).head(6).index:
            print(f"  {t.date()}  {r[t]*100:+.1f}%")

    if GUNLUK.exists():
        print("\n=== ETF vs =F günlük getiri karşılaştırması (ortak günler) ===")
        f = pd.read_csv(GUNLUK, parse_dates=[0], index_col=0)
        for c in HEDEF:
            if c not in ham.columns or c not in f.columns or ham[c].dropna().empty:
                continue
            re = np.log(ham[c].dropna()).diff()
            rf = np.log(f[c].dropna()).diff()
            j = pd.concat([re, rf], axis=1, keys=["etf", "F"]).dropna()
            ust = j.F.abs().quantile(0.99)             # =F'in en uç %1'ini (roll şüphesi) çıkar
            j2 = j[j.F.abs() < ust]
            print(f"{c:8s}: korelasyon tüm gün={j.corr().iloc[0,1]:.2f} | uç %1 hariç={j2.corr().iloc[0,1]:.2f} | n={len(j)}")

    ham.to_csv(CIKTI)
    print(f"\nKaydedildi: {CIKTI}")


if __name__ == "__main__":
    main()
