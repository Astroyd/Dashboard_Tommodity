"""
Adım 9 - Günlük/haftalık veri (Yahoo Finance)
---------------------------------------------
Pamuk, buğday ve alüminyum vadeli işlem fiyatlarını ve ilgili piyasaları günlük
çeker; data/gunluk.csv (günlük) ve data/haftalik.csv (haftalık, cuma kapanışı)
olarak kaydeder. Ekrana veri kalitesi raporu basar. O raporu bana yapıştır.

Gerekli paketler: pandas, numpy, yfinance
    pip install yfinance

UYARI (veri kalitesi):
  Yahoo'daki "CT=F, ZW=F, ALI=F" gibi sürekli kontratlar, kontrat değişim
  (roll) günlerinde geri düzeltilmez. O günlerde fiyat gerçek bir hareket olmadan
  sıçrayabilir. Rapordaki "en büyük günlük hareketler" listesi bu yüzden önemli:
  tek günde %X sıçrama haber değil kontrat değişimi olabilir.
  ALI=F (COMEX alüminyum) az işlem gören bir kontrat; geçmişi eksik olabilir.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BAS = "2012-01-01"
CIKTI_G = Path("data/gunluk.csv")
CIKTI_H = Path("data/haftalik.csv")

HEDEF = {  # ad -> Yahoo sembolü
    "pamuk": "CT=F",
    "bugday": "ZW=F",
    "aluminyum": "ALI=F",
}
YARDIMCI = {
    "petrol_wti": "CL=F",
    "petrol_brent": "BZ=F",
    "misir": "ZC=F",
    "soya": "ZS=F",
    "dogalgaz": "NG=F",
    "bakir": "HG=F",
    "seker": "SB=F",
    "dolar_endeksi": "DX-Y.NYB",
    "usd_try": "TRY=X",
    "vix": "^VIX",
    "sp500": "^GSPC",
}


def indir(semboller: dict) -> pd.DataFrame:
    ham = yf.download(list(semboller.values()), start=BAS, auto_adjust=False,
                      progress=False, group_by="column")
    kapanis = ham["Close"].copy()
    kapanis.columns = [k for s in kapanis.columns for k, v in semboller.items() if v == s]
    return kapanis.sort_index()


def main():
    Path("data").mkdir(exist_ok=True)
    tum = {**HEDEF, **YARDIMCI}
    g = indir(tum)
    g.index = pd.to_datetime(g.index).tz_localize(None)

    print("=== Kapsam raporu ===")
    for c in g.columns:
        s = g[c].dropna()
        if s.empty:
            print(f"{c:14s}: VERİ YOK")
            continue
        print(f"{c:14s}: {s.index.min().date()} -> {s.index.max().date()}  "
              f"({len(s)} gün, eksik oran {g[c].isna().mean():.1%})")

    print("\n=== En büyük günlük hareketler (kontrat değişimi şüphesi için bak) ===")
    for c in HEDEF:
        if c not in g.columns:
            continue
        r = np.log(g[c].dropna()).diff().dropna()
        en = r.abs().sort_values(ascending=False).head(8)
        print(f"\n{c}:")
        for t in en.index:
            print(f"  {t.date()}  {r[t]*100:+.1f}%")

    g.to_csv(CIKTI_G)
    h = g.resample("W-FRI").last()
    h.to_csv(CIKTI_H)
    print(f"\nKaydedildi: {CIKTI_G} ({len(g)} gün), {CIKTI_H} ({len(h)} hafta)")


if __name__ == "__main__":
    main()
