"""
Adım 4 - Fiyat tahmini (walk-forward) ve baseline karşılaştırması
-----------------------------------------------------------------
Hedef: t anında, h ay sonraki fiyatın log değişimini tahmin etmek
       (h = 1 ve 3 ay).

Modeller:
  - naif   : "fiyat değişmez" (random walk). Geçilmesi gereken baseline.
  - ridge  : doğrusal model
  - gbm    : gradient boosting

Walk-forward: Her yılın ilk ayında model, SADECE o güne kadar bilinen
veriyle yeniden eğitilir. Gelecek bilgisi sızmaz (data leakage yok).

Çıktılar:
  - data/envanter.db içine 'tahmin' tablosu
  - ekrana model karşılaştırma tablosu

Gerekli paketler: pandas, numpy, scikit-learn
    pip install scikit-learn
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DB = Path("data/envanter.db")
TEST_BASLANGIC = "2012-01-01"
HORIZONLAR = [1, 3]
SOK_BAS, SOK_BIT = "2021-06-01", "2023-06-01"  # şok dönemi (ayrıca raporlanır)


def ozellikler(logp: pd.Series) -> pd.DataFrame:
    """t anında bilinen bilgilerle özellik üretir (gelecek kullanılmaz)."""
    r = logp.diff()
    f = pd.DataFrame(index=logp.index)
    for k in range(1, 7):
        f[f"r_lag{k}"] = r.shift(k - 1)          # r_lag1 = bu ayki getiri
    f["mom3"] = logp - logp.shift(3)
    f["mom12"] = logp - logp.shift(12)
    f["sapma12"] = logp - logp.rolling(12).mean()  # 12 aylık ortalamadan sapma
    f["vol6"] = r.rolling(6).std()
    ay = logp.index.month
    f["ay_sin"] = np.sin(2 * np.pi * ay / 12)
    f["ay_cos"] = np.cos(2 * np.pi * ay / 12)
    return f


def yeni_modeller():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "gbm": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }


def walk_forward(fiyat: pd.Series, h: int) -> pd.DataFrame:
    logp = np.log(fiyat)
    X = ozellikler(logp)
    y = logp.shift(-h) - logp                      # h ay sonraki log değişim
    test_idx = X.index[X.index >= TEST_BASLANGIC]
    test_idx = test_idx[: len(test_idx) - h]       # gerçeği henüz bilinmeyenleri at
    pos = {d: i for i, d in enumerate(X.index)}

    kayit, modeller, son_yil = [], None, None
    for t in test_idx:
        i = pos[t]
        if son_yil != t.year:                      # yılda bir yeniden eğit
            Xe, ye = X.iloc[: i - h + 1], y.iloc[: i - h + 1]   # y'si t'ye kadar bilinenler
            ok = Xe.notna().all(axis=1) & ye.notna()
            modeller = yeni_modeller()
            for m in modeller.values():
                m.fit(Xe[ok], ye[ok])
            son_yil = t.year
        x = X.iloc[[i]]
        if x.isna().any(axis=None):
            continue
        tahminler = {"naif": 0.0}
        tahminler.update({ad: float(m.predict(x)[0]) for ad, m in modeller.items()})
        p0 = fiyat.iloc[i]
        gercek = fiyat.iloc[i + h]
        for ad, dlog in tahminler.items():
            kayit.append(dict(
                tarih=t, hedef_tarih=fiyat.index[i + h], ufuk=h, model=ad,
                tahmin_log_degisim=dlog, tahmin_fiyat=p0 * np.exp(dlog),
                gercek_fiyat=gercek, gercek_log_degisim=float(y.iloc[i])))
    return pd.DataFrame(kayit)


def ozet(tahmin: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for (hid, h), g in tahmin.groupby(["hammadde_id", "ufuk"]):
        hata = lambda d: np.sqrt(np.mean((d.tahmin_log_degisim - d.gercek_log_degisim) ** 2))
        naif_all = hata(g[g.model == "naif"])
        sok = g[(g.tarih >= SOK_BAS) & (g.tarih <= SOK_BIT)]
        naif_sok = hata(sok[sok.model == "naif"])
        for m, gm in g.groupby("model"):
            yon = (np.sign(gm.tahmin_log_degisim) == np.sign(gm.gercek_log_degisim)).mean()
            sm = sok[sok.model == m]
            satirlar.append(dict(
                hammadde=hid, ufuk_ay=h, model=m,
                RMSE=round(hata(gm), 4),
                naife_oran=round(hata(gm) / naif_all, 3),      # <1 ise naifi geçiyor
                sok_donemi_naife_oran=round(hata(sm) / naif_sok, 3) if len(sm) else np.nan,
                yon_dogrulugu=round(yon, 3) if m != "naif" else np.nan))
    return pd.DataFrame(satirlar)


def main():
    if not DB.exists():
        raise FileNotFoundError("Önce 03_envanter.py'yi çalıştır (data/envanter.db yok).")
    with sqlite3.connect(DB) as con:
        fiy = pd.read_sql("SELECT tarih, hammadde_id, fiyat_usd_ton FROM fiyat", con,
                          parse_dates=["tarih"])
        tum = []
        for hid, g in fiy.groupby("hammadde_id"):
            s = g.set_index("tarih")["fiyat_usd_ton"].sort_index()
            for h in HORIZONLAR:
                print(f"Eğitiliyor: {hid}, {h} ay ufku ...")
                t = walk_forward(s, h)
                t["hammadde_id"] = hid
                tum.append(t)
        tahmin = pd.concat(tum, ignore_index=True)

        out = tahmin.copy()
        out["tarih"] = out["tarih"].dt.strftime("%Y-%m-%d")
        out["hedef_tarih"] = out["hedef_tarih"].dt.strftime("%Y-%m-%d")
        out.to_sql("tahmin", con, if_exists="replace", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Model karşılaştırması (naife_oran < 1 -> naif baseline'dan iyi) ===")
    print(ozet(tahmin).to_string(index=False))
    print("\nTahminler data/envanter.db -> 'tahmin' tablosuna kaydedildi.")


if __name__ == "__main__":
    main()
