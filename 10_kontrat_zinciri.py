"""
Adım 10 - Kontrat zincirleme (roll sıçramalarını yok etme)
----------------------------------------------------------
Sorun: Yahoo'nun "CT=F, ZW=F" gibi sürekli serileri, kontrat değiştikçe
geri düzeltilmiyor. Her geçişte fiyatta yapay bir sıçrama kalıyor.

Çözüm: Tek tek kontratları (ör. CTZ25.NYB) indirip, günlük getiriyi HEP AYNI
kontratın iki ardışık günü arasından hesaplarız. Kontrat vadesine yaklaşınca
(ROLL_GUN işlem günü kala) bir sonraki kontrata geçeriz, ama geçiş gününün
getirisi de yeni kontratın kendi dünkü fiyatına göre hesaplanır. Böylece iki
kontrat arasındaki fark (spread) hiçbir zaman getiri olarak görünmez.

Çıktı:
  data/gunluk_zincir.csv   -> <ad>_getiri (günlük log getiri), <ad>_endeks (zincirli seviye)
  Ekrana kapsam raporu.

Gerekli paketler: pandas, numpy, yfinance

UYARI: Yahoo'nun süresi dolmuş tek tek kontratların geçmişini ne kadar tuttuğunu
önceden bilemiyorum. Rapor kapsam eksiklerini gösterecek; yeterli değilse
alternatif veri kaynağı konuşuruz.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BAS_YIL, SON_YIL = 2011, 2026
ROLL_GUN = 20           # vadeden kaç işlem günü önce yeni kontrata geç
CIKTI = Path("data/gunluk_zincir.csv")

# ad: (sembol_öneki, vade_ayı_kodları, borsa_soneki)
URUNLER = {
    "pamuk": ("CT", "HKNVZ", ".NYB"),      # ICE Pamuk: Mar, May, Tem, Eki, Ara
    "bugday": ("ZW", "HKNUZ", ".CBT"),     # CBOT Buğday: Mar, May, Tem, Eyl, Ara
    "bakir": ("HG", "HKNUZ", ".CMX"),      # COMEX Bakır: Mar, May, Tem, Eyl, Ara
}


def zincirle(kontratlar: dict, roll_gun: int = ROLL_GUN) -> pd.Series:
    """kontratlar: {ad: fiyat Series}. Günlük log getiri Series döndürür (roll sıçraması yok)."""
    seri = {k: v.dropna() for k, v in kontratlar.items() if v.dropna().size > roll_gun + 5}
    if not seri:
        return pd.Series(dtype=float)
    # Her kontratın "son kullanım" günü: vadeden roll_gun işlem günü önceki gözlem
    kesim = {k: s.index[-(roll_gun + 1)] for k, s in seri.items()}
    tum_gunler = sorted(set().union(*[s.index for s in seri.values()]))
    getiri = {}
    for t in tum_gunler:
        adaylar = sorted((k for k in seri if kesim[k] >= t), key=lambda k: kesim[k])
        for k in adaylar:                      # vadesi en yakın olandan başla
            s = seri[k]
            if t not in s.index:
                continue
            onceki = s.index[s.index < t]
            if len(onceki) == 0:
                continue
            getiri[t] = float(np.log(s.loc[t] / s.loc[onceki[-1]]))
            break
    return pd.Series(getiri).sort_index()


def indir_kontratlar(onek, aylar, sonek):
    import yfinance as yf

    semboller = [f"{onek}{a}{str(y)[-2:]}{sonek}"
                 for y in range(BAS_YIL, SON_YIL + 1) for a in aylar]
    kontratlar = {}
    for i in range(0, len(semboller), 20):
        grup = semboller[i:i + 20]
        try:
            ham = yf.download(grup, start=f"{BAS_YIL - 1}-01-01", auto_adjust=False,
                              progress=False, group_by="column")["Close"]
        except Exception as e:
            print(f"  indirme hatası ({grup[0]}...): {e}")
            continue
        if isinstance(ham, pd.Series):
            ham = ham.to_frame(grup[0])
        for s in ham.columns:
            x = ham[s].dropna()
            if len(x) > ROLL_GUN + 5:
                x.index = pd.to_datetime(x.index).tz_localize(None)
                kontratlar[s] = x
    return semboller, kontratlar


def main():
    Path("data").mkdir(exist_ok=True)
    sonuc = pd.DataFrame()
    for ad, (onek, aylar, sonek) in URUNLER.items():
        print(f"\n=== {ad} ===")
        semboller, kontratlar = indir_kontratlar(onek, aylar, sonek)
        print(f"  Denenen kontrat: {len(semboller)} | verisi bulunan: {len(kontratlar)}")
        if not kontratlar:
            print("  UYARI: hiç kontrat verisi alınamadı.")
            continue
        r = zincirle(kontratlar)
        if r.empty:
            continue
        r = r[r.index >= "2012-01-01"]
        bos = pd.bdate_range(r.index.min(), r.index.max()).difference(r.index)
        print(f"  Getiri serisi: {r.index.min().date()} -> {r.index.max().date()} ({len(r)} gün)")
        print(f"  Aralıklardaki iş günü boşluğu: {len(bos)} gün (tatiller dahil)")
        buyuk = r[r.abs() > 4 * 1.4826 * (r - r.median()).abs().median()]
        print(f"  4xMAD'ı aşan gün sayısı: {len(buyuk)} (gerçek haber şoku olmalı)")
        for t, v in buyuk.abs().sort_values(ascending=False).head(6).items():
            print(f"    {t.date()}  {r[t]*100:+.1f}%")
        sonuc[f"{ad}_getiri"] = r
        sonuc[f"{ad}_endeks"] = 100 * np.exp(r.cumsum())

    sonuc.sort_index().to_csv(CIKTI)
    print(f"\nKaydedildi: {CIKTI}")


if __name__ == "__main__":
    main()
