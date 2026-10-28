"""
Adım 12 - Haftalık tahmin deneyi (GELİŞTİRME aşaması)
-----------------------------------------------------
Veri : data/gunluk_etf.csv (buğday=WEAT, bakır=CPER, mısır/soya ETF'leri)
       data/gunluk.csv (dolar endeksi, Brent, VIX, S&P500: roll sorunu olmayan seriler)
Hedef: h hafta sonraki log getiri, h = 1 ve 4
Modeller: naif (0), ridge, ridge_yari (0.5 x ridge), gbm

TEST DÖNEMİ KİLİDİ:
  Bu script yalnızca GELISTIRME_SON (2023-12-29) tarihine kadar olan veriyi okur.
  2024 ve sonrası bu aşamada HİÇ kullanılmaz; o dönem, nihai veriyle (satın alınacak
  geri düzeltilmiş vadeli kontratlar) TEK SEFER çalıştırılacak.
  Geliştirme: eğitim 2011-2017, değerlendirme 2018-2023 (walk-forward).

Çoklu test: test sayısıyla Bonferroni düzeltmesi raporlanır.

Gerekli paketler: pandas, numpy, scikit-learn, scipy
"""

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

ETF = Path("data/gunluk_etf.csv")
FUT = Path("data/gunluk.csv")
GELISTIRME_SON = "2023-12-29"
DEGERLENDIRME_BAS = "2018-01-01"
HORIZONLAR = [1, 4]
YENILEME_HAFTA = 26

# Önceden belirlenmiş (sonuca bakmadan) çapraz özellikler
CAPRAZ = {
    "bugday": ["dolar_endeksi", "petrol_brent", "misir_etf", "soya_etf", "vix"],
    "bakir": ["dolar_endeksi", "petrol_brent", "sp500", "vix"],
}


def haftalik_veri():
    etf = pd.read_csv(ETF, parse_dates=[0], index_col=0)
    fut = pd.read_csv(FUT, parse_dates=[0], index_col=0)
    # CPER'deki hatalı fiyat kayıtlarını (art arda zıt yönlü >%25 sıçrama) temizle
    r = np.log(etf["bakir"]).diff()
    kotu = r.abs() > 0.25
    etf.loc[kotu | kotu.shift(-1, fill_value=False), "bakir"] = np.nan
    etf["bakir"] = etf["bakir"].ffill(limit=2)
    gun = pd.concat([etf[["bugday", "bakir", "misir_etf", "soya_etf"]],
                     fut[["dolar_endeksi", "petrol_brent", "sp500", "vix"]]], axis=1)
    gun = gun[gun.index <= GELISTIRME_SON]
    return gun.resample("W-FRI").last()


def ozellikler(w: pd.DataFrame, hedef: str) -> pd.DataFrame:
    lp = np.log(w[hedef])
    r = lp.diff()
    f = pd.DataFrame(index=w.index)
    for k in range(1, 5):
        f[f"r_lag{k}"] = r.shift(k - 1)
    for n in (4, 13, 26, 52):
        f[f"mom{n}"] = lp - lp.shift(n)
    f["sapma52"] = lp - lp.rolling(52).mean()
    f["vol8"] = r.rolling(8).std()
    f["ay_sin"] = np.sin(2 * np.pi * w.index.month / 12)
    f["ay_cos"] = np.cos(2 * np.pi * w.index.month / 12)
    for c in CAPRAZ[hedef]:
        x = np.log(w[c])
        f[f"{c}_r1"] = x.diff()
        f[f"{c}_mom4"] = x - x.shift(4)
        f[f"{c}_mom13"] = x - x.shift(13)
    return f


def modeller():
    return {
        "ridge": make_pipeline(StandardScaler(),
                               RidgeCV(alphas=[1, 10, 100, 1000], cv=TimeSeriesSplit(5))),
        "gbm": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=42),
    }


def walk_forward(w, hedef, h):
    lp = np.log(w[hedef])
    X = ozellikler(w, hedef)
    y = lp.shift(-h) - lp
    idx = [i for i, d in enumerate(w.index) if d >= pd.Timestamp(DEGERLENDIRME_BAS) and i + h < len(w)]
    kayit, mdl, son = [], None, None
    for i in idx:
        if son is None or i - son >= YENILEME_HAFTA:
            Xe, ye = X.iloc[: i - h + 1], y.iloc[: i - h + 1]
            ok = Xe.notna().all(axis=1) & ye.notna()
            mdl = modeller()
            for m in mdl.values():
                m.fit(Xe[ok], ye[ok])
            son = i
        x = X.iloc[[i]]
        if x.isna().any(axis=None) or pd.isna(y.iloc[i]):
            continue
        tah = {"naif": 0.0}
        tah.update({k: float(m.predict(x)[0]) for k, m in mdl.items()})
        tah["ridge_yari"] = 0.5 * tah["ridge"]
        for ad, v in tah.items():
            kayit.append(dict(tarih=w.index[i], model=ad, tahmin=v, gercek=float(y.iloc[i])))
    return pd.DataFrame(kayit)


def dm_p(d, h):
    n, m = len(d), d.mean()
    g = [((d - m)[k:] * (d - m)[: n - k]).mean() for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    return float(2 * (1 - stats.norm.cdf(abs(m / np.sqrt(var))))) if var > 0 else np.nan


def main():
    w = haftalik_veri()
    print(f"Haftalık veri: {w.index.min().date()} -> {w.index.max().date()} ({len(w)} hafta)")
    print(f"(Kilit: {GELISTIRME_SON} sonrası veri okunmadı)\n")
    satirlar = []
    for hedef in CAPRAZ:
        for h in HORIZONLAR:
            d = walk_forward(w, hedef, h)
            p = d.pivot(index="tarih", columns="model", values="tahmin")
            g = d.drop_duplicates("tarih").set_index("tarih")["gercek"].reindex(p.index)
            e0 = (p["naif"] - g) ** 2
            for m in ["ridge", "ridge_yari", "gbm"]:
                e = (p[m] - g) ** 2
                satirlar.append(dict(
                    hammadde=hedef, ufuk_hafta=h, model=m, n=len(p),
                    naife_oran=round(float(np.sqrt(e.mean() / e0.mean())), 3),
                    DM_p=round(dm_p((e0 - e).values, h), 3),
                    yon_dogrulugu=round(float((np.sign(p[m]) == np.sign(g)).mean()), 3),
                    gercek_artis_orani=round(float((g > 0).mean()), 3)))
    s = pd.DataFrame(satirlar)
    s["DM_p_bonferroni"] = (s["DM_p"] * len(s)).clip(upper=1).round(3)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    s.to_csv("data/haftalik_gelistirme_sonuc.csv", index=False)
    print("\nnaife_oran < 1 iyi | yon_dogrulugu, gercek_artis_orani ile (hep 'yükselir' deseydik) kıyaslanmalı")


if __name__ == "__main__":
    main()
