"""
Adım 7 - Dış değişkenlerle tahmin (walk-forward)
------------------------------------------------
04'teki modelin aynısı, ama her hammadde için İLGİLİ dış değişkenler de eklenir.
Hangi değişkenin hangi hammaddeye eklendiği veri bakmadan, ekonomik mantıkla
önceden belirlendi (aşağıdaki ALAKALI sözlüğü). Böylece "her şeyi deneyip tutanı
seçme" (veri avcılığı) riskinden kaçınıyoruz.

Eklenen modeller: ridge_dis, gbm_dis  (tahmin tablosuna eklenir)
Sonra 05_simulasyon.py'de TAHMIN_MODELLERI listesine bu isimleri ekleyip
simülasyonu yeniden çalıştırabilirsin.

Gerekli paketler: pandas, numpy, scikit-learn, scipy
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DB = Path("data/envanter.db")
DIS_CSV = Path("data/dis_degiskenler.csv")
TEST_BASLANGIC = "2012-01-01"
HORIZONLAR = [1, 3]
SOK_BAS, SOK_BIT = "2021-06-01", "2023-06-01"

# Ekonomik gerekçe:
#  pamuk     : petrol (polyester ikamesi + nakliye), mısır/soya (ekim alanı rekabeti), üre (gübre)
#  alüminyum : enerji (üretim çok elektrik yoğun): doğalgaz, kömür, petrol; bakır (sanayi talebi)
#  buğday    : mısır/soya (ikame ve alan rekabeti), petrol (girdi/biyoyakıt), üre (gübre)
ALAKALI = {
    "pamuk": ["petrol_brent", "misir", "soya", "ure"],
    "aluminyum": ["petrol_brent", "dogalgaz_ab", "komur", "bakir"],
    "bugday": ["misir", "soya", "petrol_brent", "ure"],
}


def ozellikler(logp: pd.Series, dis: pd.DataFrame) -> pd.DataFrame:
    r = logp.diff()
    f = pd.DataFrame(index=logp.index)
    for k in range(1, 7):
        f[f"r_lag{k}"] = r.shift(k - 1)
    f["mom3"] = logp - logp.shift(3)
    f["mom12"] = logp - logp.shift(12)
    f["sapma12"] = logp - logp.rolling(12).mean()
    f["vol6"] = r.rolling(6).std()
    ay = logp.index.month
    f["ay_sin"] = np.sin(2 * np.pi * ay / 12)
    f["ay_cos"] = np.cos(2 * np.pi * ay / 12)
    for c in dis.columns:                      # dış değişken özellikleri (hepsi t anında bilinir)
        x = np.log(dis[c])
        f[f"{c}_r1"] = x.diff()
        f[f"{c}_mom3"] = x - x.shift(3)
        f[f"{c}_mom12"] = x - x.shift(12)
        f[f"{c}_sapma12"] = x - x.rolling(12).mean()
    return f


def yeni_modeller():
    return {
        "ridge_dis": make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=[1, 10, 100, 1000], cv=TimeSeriesSplit(5))),
        "gbm_dis": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }


def walk_forward(fiyat: pd.Series, dis: pd.DataFrame, h: int) -> pd.DataFrame:
    logp = np.log(fiyat)
    X = ozellikler(logp, dis.reindex(fiyat.index))
    y = logp.shift(-h) - logp
    test_idx = X.index[X.index >= TEST_BASLANGIC]
    test_idx = test_idx[: len(test_idx) - h]
    pos = {d: i for i, d in enumerate(X.index)}

    kayit, modeller, son_yil = [], None, None
    for t in test_idx:
        i = pos[t]
        if son_yil != t.year:
            Xe, ye = X.iloc[: i - h + 1], y.iloc[: i - h + 1]
            ok = Xe.notna().all(axis=1) & ye.notna()
            modeller = yeni_modeller()
            for m in modeller.values():
                m.fit(Xe[ok], ye[ok])
            son_yil = t.year
        x = X.iloc[[i]]
        if x.isna().any(axis=None):
            continue
        p0, gercek = fiyat.iloc[i], fiyat.iloc[i + h]
        for ad, m in modeller.items():
            dlog = float(m.predict(x)[0])
            kayit.append(dict(
                tarih=t.strftime("%Y-%m-%d"), hedef_tarih=fiyat.index[i + h].strftime("%Y-%m-%d"),
                ufuk=h, model=ad, tahmin_log_degisim=dlog, tahmin_fiyat=p0 * np.exp(dlog),
                gercek_fiyat=gercek, gercek_log_degisim=float(y.iloc[i])))
    return pd.DataFrame(kayit)


def dm_p(d: np.ndarray, h: int) -> float:
    """Diebold-Mariano p-değeri (HAC, h-1 gecikme). d = naif hata^2 - model hata^2."""
    n, m = len(d), d.mean()
    g = [((d - m)[k:] * (d - m)[: n - k]).mean() for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0:
        return np.nan
    return float(2 * (1 - stats.norm.cdf(abs(m / np.sqrt(var)))))


def ozet(con) -> pd.DataFrame:
    tum = pd.read_sql("SELECT * FROM tahmin", con, parse_dates=["tarih"])
    modeller = ["ridge", "gbm", "ridge_dis", "gbm_dis"]
    satirlar = []
    for (hid, h), g in tum.groupby(["hammadde_id", "ufuk"]):
        w = g.pivot(index="tarih", columns="model", values="tahmin_log_degisim")
        gercek = g.drop_duplicates("tarih").set_index("tarih")["gercek_log_degisim"]
        w = w.dropna(subset=["naif"] + [m for m in modeller if m in w.columns])
        gercek = gercek.reindex(w.index)                 # ortak tarihlerde adil kıyas
        sok = (w.index >= SOK_BAS) & (w.index <= SOK_BIT)
        e_naif = (w["naif"] - gercek) ** 2
        for m in modeller:
            if m not in w.columns:
                continue
            e = (w[m] - gercek) ** 2
            satirlar.append(dict(
                hammadde=hid, ufuk_ay=h, model=m, n=len(w),
                naife_oran=round(np.sqrt(e.mean() / e_naif.mean()), 3),
                sok_naife_oran=round(np.sqrt(e[sok].mean() / e_naif[sok].mean()), 3) if sok.sum() else np.nan,
                DM_p=round(dm_p((e_naif - e).values, h), 3),
                yon_dogrulugu=round((np.sign(w[m]) == np.sign(gercek)).mean(), 3)))
    return pd.DataFrame(satirlar)


def main():
    if not DB.exists() or not DIS_CSV.exists():
        raise FileNotFoundError("Önce 03, 04 ve 06 numaralı scriptleri çalıştır.")
    dis_tum = pd.read_csv(DIS_CSV, parse_dates=["tarih"], index_col="tarih")

    with sqlite3.connect(DB) as con:
        fiy = pd.read_sql("SELECT tarih, hammadde_id, fiyat_usd_ton FROM fiyat", con,
                          parse_dates=["tarih"])
        yeni = []
        for hid, g in fiy.groupby("hammadde_id"):
            s = g.set_index("tarih")["fiyat_usd_ton"].sort_index()
            kolonlar = [c for c in ALAKALI[hid] if c in dis_tum.columns]
            eksik = set(ALAKALI[hid]) - set(kolonlar)
            if eksik:
                print(f"UYARI: {hid} için bulunamayan dış değişkenler: {sorted(eksik)}")
            dis = dis_tum[kolonlar]
            for h in HORIZONLAR:
                print(f"Eğitiliyor: {hid}, {h} ay, dış değişkenler: {kolonlar}")
                t = walk_forward(s, dis, h)
                t["hammadde_id"] = hid
                yeni.append(t)
        yeni = pd.concat(yeni, ignore_index=True)

        con.execute("DELETE FROM tahmin WHERE model IN ('ridge_dis','gbm_dis')")
        yeni.to_sql("tahmin", con, if_exists="append", index=False)
        sonuc = ozet(con)

    pd.set_option("display.width", 200)
    print("\n=== Ortak tarihlerde karşılaştırma ===")
    print("naife_oran < 1: naiften iyi | DM_p < 0.05: fark istatistiksel olarak anlamlı\n")
    print(sonuc.to_string(index=False))
    print("\n'tahmin' tablosuna ridge_dis ve gbm_dis eklendi.")


if __name__ == "__main__":
    main()
