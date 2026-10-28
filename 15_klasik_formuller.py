"""
Adım 15 - Profesyonellerin kullandığı klasik formüller (aylık, walk-forward)
---------------------------------------------------------------------------
Makine öğrenmesi yok, her model 1-2 parametreli, finansta/emtia analizinde bilinen formüller:

  naif    : fiyat aynı kalır (baseline)
  drift   : tarihsel ortalama aylık getiri x h
  mr60    : ortalamaya dönüş: tahmin = beta x (log fiyat - 60 ay ortalaması)
  mr120   : aynısı, 120 aylık ortalama (uzun vadeli denge fiyatı)
  tsmom12 : trend takibi (time-series momentum): tahmin = beta x isaret(12 aylık getiri)
  macross : hareketli ortalama kesişimi: tahmin = beta x isaret(MA3 - MA12)

beta, her origin'de sadece o güne kadar bilinen veriyle en küçük kareler (OLS) ile
tahmin edilir (gelecek sızmaz).

Değerlendirme iki pencerede ayrı raporlanır:
  gorulmus : 2012-01 -> 2024-12 (önceki deneylerde zaten bakılmış dönem)
  yeni     : 2025-01 ve sonrası (Pink Sheet'i güncellersen oluşur; bakılmamış dönem)

Çoklu test: bu projede önceki deneylerde yaklaşık 72 karşılaştırma yapıldı. Düzeltilmiş
p-değeri = p x (TOPLAM_ONCEKI + bu script'teki test sayısı). Şans eseri kazananı eler.

Gerekli paketler: pandas, numpy, scipy
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DB = Path("data/envanter.db")
FIYAT_CSV = Path("data/fiyatlar.csv")      # 01_veri_hazirla.py çıktısı (varsa bu kullanılır)
HORIZONLAR = [1, 3, 6, 12]
TEST_BAS = "2012-01-01"
YENI_BAS = "2025-01-01"
TOPLAM_ONCEKI = 72           # projede bu scripte kadar yapılan karşılaştırma sayısı (yaklaşık)


def sinyaller(lp: pd.Series) -> pd.DataFrame:
    s = pd.DataFrame(index=lp.index)
    s["mr60"] = lp - lp.rolling(60).mean()
    s["mr120"] = lp - lp.rolling(120).mean()
    s["tsmom12"] = np.sign(lp - lp.shift(12))
    ma3, ma12 = lp.rolling(3).mean(), lp.rolling(12).mean()
    s["macross"] = np.sign(ma3 - ma12)
    return s


def tahminle(lp: pd.Series, h: int) -> pd.DataFrame:
    n = len(lp)
    S = sinyaller(lp)
    y = (lp.shift(-h) - lp)
    r1 = lp.diff()
    kayit = []
    for i in range(n):
        t = lp.index[i]
        if t < pd.Timestamp(TEST_BAS) or i + h >= n:
            continue
        tr = slice(0, i - h + 1)                       # hedefi t'ye kadar bilinenler
        yt = y.iloc[tr]
        tah = {"naif": 0.0, "drift": float(r1.iloc[1: i + 1].mean() * h)}
        for ad in S.columns:
            x = S[ad].iloc[tr]
            ok = x.notna() & yt.notna()
            if ok.sum() < 36 or pd.isna(S[ad].iloc[i]):
                continue
            beta = float((x[ok] * yt[ok]).sum() / (x[ok] ** 2).sum())
            tah[ad] = beta * float(S[ad].iloc[i])
        gercek = float(y.iloc[i])
        for ad, v in tah.items():
            kayit.append(dict(tarih=t, model=ad, tahmin=v, gercek=gercek))
    return pd.DataFrame(kayit)


def dm_p(d: np.ndarray, h: int) -> float:
    n = len(d)
    if n < max(8, 2 * h):          # çok az gözlemde DM testi güvenilmez/tanımsız
        return np.nan
    m = d.mean()
    g = [((d - m)[k:] * (d - m)[: n - k]).mean() for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    return float(2 * (1 - stats.norm.cdf(abs(m / np.sqrt(var))))) if var > 0 else np.nan


def ozetle(d: pd.DataFrame, h: int, hid: str, pencere: str) -> list:
    p = d.pivot(index="tarih", columns="model", values="tahmin").dropna()
    g = d.drop_duplicates("tarih").set_index("tarih")["gercek"].reindex(p.index)
    if pencere == "gorulmus":
        mask = p.index < pd.Timestamp(YENI_BAS)
    else:
        mask = p.index >= pd.Timestamp(YENI_BAS)
    p, g = p[mask], g[mask]
    if len(p) < 8:
        return []
    e0 = (p["naif"] - g) ** 2
    satirlar = []
    for m in p.columns:
        if m == "naif":
            continue
        e = (p[m] - g) ** 2
        satirlar.append(dict(
            pencere=pencere, hammadde=hid, ufuk_ay=h, model=m, n=len(p),
            naife_oran=round(float(np.sqrt(e.mean() / e0.mean())), 3),
            DM_p=round(dm_p((e0 - e).values, h), 3),
            yon_dogrulugu=round(float((np.sign(p[m]) == np.sign(g)).mean()), 3),
            gercek_artis_orani=round(float((g > 0).mean()), 3)))
    return satirlar


def fiyatlari_oku() -> pd.DataFrame:
    """Önce data/fiyatlar.csv (01 çıktısı, en güncel), yoksa veri tabanındaki fiyat tablosu."""
    if FIYAT_CSV.exists():
        df = pd.read_csv(FIYAT_CSV, parse_dates=["tarih"])
        fiy = df.melt(id_vars="tarih", var_name="hammadde_id", value_name="fiyat_usd_ton").dropna()
        print(f"Kaynak: {FIYAT_CSV} (son tarih: {df['tarih'].max().date()})")
        return fiy
    with sqlite3.connect(DB) as con:
        fiy = pd.read_sql("SELECT tarih, hammadde_id, fiyat_usd_ton FROM fiyat", con, parse_dates=["tarih"])
    print(f"Kaynak: {DB} (son tarih: {fiy['tarih'].max().date()})")
    return fiy


def main():
    fiy = fiyatlari_oku()
    satirlar = []
    for hid, g in fiy.groupby("hammadde_id"):
        lp = np.log(g.set_index("tarih")["fiyat_usd_ton"].sort_index())
        print(f"{hid}: {lp.index.min().date()} -> {lp.index.max().date()}")
        for h in HORIZONLAR:
            d = tahminle(lp, h)
            for pencere in ("gorulmus", "yeni"):
                satirlar += ozetle(d, h, hid, pencere)
    s = pd.DataFrame(satirlar)
    sayi = s[s.pencere == "gorulmus"].shape[0]
    s["DM_p_duzeltilmis"] = (s["DM_p"] * (TOPLAM_ONCEKI + sayi)).clip(upper=1).round(3)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    print(f"\nBu scriptte {sayi} test (görülmüş pencere); düzeltme çarpanı = {TOPLAM_ONCEKI + sayi}")
    print("naife_oran < 1 iyi. DM_p_duzeltilmis < 0.05 olmadan 'bulundu' denmez.\n")
    print(s.sort_values(["pencere", "hammadde", "ufuk_ay", "model"]).to_string(index=False))
    s.to_csv("data/klasik_formuller_sonuc.csv", index=False)
    print("\nKaydedildi: data/klasik_formuller_sonuc.csv")


if __name__ == "__main__":
    main()
