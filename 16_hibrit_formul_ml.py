"""
Adım 16 - Hibrit Finansal Formül + ML Tahmin Motoru (Walk-Forward) [GÜNCELLENDİ]
----------------------------------------------------------------------------------
Şampiyon ridge_dis (dış değişkenli model) ve hibrit formül bileşimlerini
data/envanter.db -> 'tahmin' tablosuna kaydeder.
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DB = Path("data/envanter.db")
DIS_CSV = Path("data/dis_degiskenler.csv")
CIKTI_CSV = Path("data/hibrit_tahmin_sonuc.csv")

TEST_BASLANGIC = "2012-01-01"
HORIZONLAR = [1, 3, 6]
SOK_BAS, SOK_BIT = "2021-06-01", "2023-06-01"

MALIYET_SEPETI = {
    "pamuk": {"petrol_brent": 0.35, "soya": 0.25, "misir": 0.20, "ure": 0.20},
    "aluminyum": {"dogalgaz_ab": 0.30, "komur": 0.30, "petrol_brent": 0.20, "bakir": 0.20},
    "bugday": {"misir": 0.35, "soya": 0.25, "ure": 0.25, "petrol_brent": 0.15},
}

ALAKALI_DIS = {
    "pamuk": ["petrol_brent", "misir", "soya", "ure"],
    "aluminyum": ["petrol_brent", "dogalgaz_ab", "komur", "bakir"],
    "bugday": ["misir", "soya", "petrol_brent", "ure"],
}


def maliyet_endeksi_uret(lp: pd.Series, dis: pd.DataFrame, hid: str) -> pd.Series:
    sepet = MALIYET_SEPETI.get(hid, {})
    mevcut = {k: w for k, w in sepet.items() if k in dis.columns and dis[k].notna().sum() > 60}
    if not mevcut:
        return lp.rolling(60, min_periods=24).mean()
    toplam_w = sum(mevcut.values())
    endeks = pd.Series(0.0, index=lp.index)
    for k, w in mevcut.items():
        x = np.log(dis[k].reindex(lp.index)).ffill().bfill()
        mu = x.rolling(60, min_periods=24).mean()
        sd = x.rolling(60, min_periods=24).std().replace(0, np.nan)
        endeks += (w / toplam_w) * ((x - mu) / sd).fillna(0.0)
    return endeks


def finansal_sinyaller(fiyat: pd.Series, dis: pd.DataFrame, hid: str) -> pd.DataFrame:
    lp = np.log(fiyat)
    r1 = lp.diff()
    vol6 = r1.rolling(6, min_periods=3).std().clip(lower=0.01)
    vol12 = r1.rolling(12, min_periods=6).std().clip(lower=0.01)

    s = pd.DataFrame(index=fiyat.index)
    ma60 = lp.rolling(60, min_periods=24).mean()
    sd60 = lp.rolling(60, min_periods=24).std().clip(lower=0.02)
    ma120 = lp.rolling(120, min_periods=36).mean()
    sd120 = lp.rolling(120, min_periods=36).std().clip(lower=0.02)

    s["mr60_z"] = -((lp - ma60) / sd60).clip(-3.0, 3.0)
    s["mr120_z"] = -((lp - ma120) / sd120).clip(-3.0, 3.0)
    maliyet_z = maliyet_endeksi_uret(lp, dis, hid)
    fiyat_z = ((lp - ma60) / sd60).fillna(0.0)
    s["cost_spread_z"] = (maliyet_z - fiyat_z).clip(-3.0, 3.0)

    ma3 = lp.rolling(3).mean()
    ma12 = lp.rolling(12).mean()
    s["vol_macd"] = ((ma3 - ma12) / (vol12 * np.sqrt(6))).clip(-3.0, 3.0)
    s["mom3_sharpe"] = ((lp - lp.shift(3)) / (vol6 * np.sqrt(3))).clip(-3.0, 3.0)
    s["dis_ivme"] = maliyet_z.diff(3).clip(-3.0, 3.0)

    ay = fiyat.index.month
    s["ay_sin"] = np.sin(2 * np.pi * ay / 12)
    s["ay_cos"] = np.cos(2 * np.pi * ay / 12)

    if not dis.empty:
        kolonlar = [c for c in ALAKALI_DIS.get(hid, []) if c in dis.columns]
        for c in kolonlar:
            x = np.log(dis[c].reindex(fiyat.index)).ffill().bfill()
            s[f"{c}_r1"] = x.diff()
            s[f"{c}_mom3"] = x - x.shift(3)

    return s


def tekil_formul_ols(x: np.ndarray, y: np.ndarray, x_now: float, shrinkage: float = 0.65) -> float:
    X_mat = np.column_stack([np.ones_like(x), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        ham_tahmin = float(beta[0] + beta[1] * x_now)
        return shrinkage * ham_tahmin
    except Exception:
        return 0.0


def walk_forward_hibrit(fiyat: pd.Series, dis: pd.DataFrame, hid: str, h: int) -> pd.DataFrame:
    lp = np.log(fiyat)
    S = finansal_sinyaller(fiyat, dis, hid)
    y = lp.shift(-h) - lp

    test_idx = S.index[S.index >= TEST_BASLANGIC]
    test_idx = test_idx[: len(test_idx) - h]
    pos = {d: i for i, d in enumerate(S.index)}

    uzman_kolonlar = ["mr60_z", "mr120_z", "cost_spread_z", "vol_macd", "mom3_sharpe", "dis_ivme"]
    kayit = []

    for t in test_idx:
        i = pos[t]
        tr_bitis = i - h + 1
        if tr_bitis < 48:
            continue

        Xe = S.iloc[:tr_bitis]
        ye = y.iloc[:tr_bitis]
        ok = Xe.notna().all(axis=1) & ye.notna()
        if ok.sum() < 40 or S.iloc[[i]].isna().any(axis=None):
            continue

        Xe_ok, ye_ok = Xe.loc[ok], ye.loc[ok]
        x_now = S.iloc[[i]]
        buzulme = {1: 0.35, 3: 0.55, 6: 0.75}.get(h, 0.60)

        uzman_tahmin = {}
        uzman_hata = {}
        for col in uzman_kolonlar:
            if col not in Xe_ok.columns:
                continue
            xv = Xe_ok[col].values
            yv = ye_ok.values
            uzman_tahmin[col] = tekil_formul_ols(xv, yv, float(x_now[col].iloc[0]), shrinkage=buzulme)
            son_k = min(36, len(xv))
            X_mat = np.column_stack([np.ones(len(xv)), xv])
            beta, _, _, _ = np.linalg.lstsq(X_mat, yv, rcond=None)
            pred_gecmis = buzulme * (beta[0] + beta[1] * xv[-son_k:])
            uzman_hata[col] = float(np.mean((yv[-son_k:] - pred_gecmis) ** 2)) + 1e-6

        aktif_uzmanlar = [k for k in uzman_kolonlar if k in uzman_tahmin]
        oncelik = {k: 1.0 for k in aktif_uzmanlar}
        inv_mse = {k: oncelik[k] / uzman_hata[k] for k in aktif_uzmanlar}
        w_top = sum(inv_mse.values()) if inv_mse else 1.0
        pred_ens = sum((inv_mse[k] / w_top) * uzman_tahmin[k] for k in aktif_uzmanlar) if inv_mse else 0.0

        dis_kolonlar = [c for c in S.columns if c not in uzman_kolonlar and c not in ("ay_sin", "ay_cos")]
        ridge_cols = ["cost_spread_z", "vol_macd"] + dis_kolonlar
        ridge_cols = [c for c in ridge_cols if c in Xe_ok.columns]

        if ridge_cols:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(Xe_ok[ridge_cols])
            x_now_scaled = scaler.transform(x_now[ridge_cols])
            ridge = RidgeCV(alphas=[10.0, 50.0, 150.0, 500.0])
            ridge.fit(X_scaled, ye_ok.values)
            pred_ridge_dis = float(ridge.predict(x_now_scaled)[0]) * buzulme
        else:
            pred_ridge_dis = pred_ens

        pred_mr = 0.5 * uzman_tahmin.get("mr120_z", 0) + 0.5 * uzman_tahmin.get("cost_spread_z", 0)

        max_degisim = 1.5 * float(ye_ok.std())
        tahminler = {
            "naif": 0.0,
            "mr_maliyet": float(np.clip(pred_mr, -max_degisim, max_degisim)),
            "ridge_dis": float(np.clip(pred_ridge_dis, -max_degisim, max_degisim)),
            "hibrit_ensemble": float(np.clip(0.5 * pred_ens + 0.5 * pred_ridge_dis, -max_degisim, max_degisim)),
        }

        p0 = float(fiyat.iloc[i])
        gercek_fiyat = float(fiyat.iloc[i + h])
        gercek_deg = float(y.iloc[i])

        for m_ad, dlog in tahminler.items():
            kayit.append(dict(
                tarih=t.strftime("%Y-%m-%d"),
                hedef_tarih=fiyat.index[i + h].strftime("%Y-%m-%d"),
                hammadde_id=hid,
                ufuk=h,
                model=m_ad,
                tahmin_log_degisim=dlog,
                tahmin_fiyat=p0 * np.exp(dlog),
                gercek_fiyat=gercek_fiyat,
                gercek_log_degisim=gercek_deg,
            ))

    return pd.DataFrame(kayit)


def dm_p_bartlett(d: np.ndarray, h: int) -> float:
    n = len(d)
    if n < max(10, h + 2):
        return np.nan
    m = d.mean()
    g0 = np.mean((d - m) ** 2)
    g_sum = sum(2.0 * (1.0 - k / h) * np.mean((d - m)[k:] * (d - m)[: n - k]) for k in range(1, h))
    var = (g0 + g_sum) / n
    if var <= 0:
        return np.nan
    dm_stat = m / np.sqrt(var)
    hln_factor = np.sqrt(max(0.1, (n + 1 - 2 * h + h * (h - 1) / n) / n))
    return float(2 * (1 - stats.t.cdf(abs(dm_stat * hln_factor), df=n - 1)))


def performans_ozeti(con) -> pd.DataFrame:
    tum = pd.read_sql("SELECT * FROM tahmin", con, parse_dates=["tarih"])
    aday_modeller = ["ridge", "gbm", "mr_maliyet", "ridge_dis", "hibrit_ridge", "hibrit_ensemble"]
    satirlar = []

    for (hid, h), g in tum.groupby(["hammadde_id", "ufuk"]):
        w = g.pivot(index="tarih", columns="model", values="tahmin_log_degisim")
        if "naif" not in w.columns:
            continue
        mevcut_m = [m for m in aday_modeller if m in w.columns]
        w = w.dropna(subset=["naif"] + [m for m in ["hibrit_ensemble"] if m in w.columns])
        gercek = g.drop_duplicates("tarih").set_index("tarih")["gercek_log_degisim"].reindex(w.index)

        sok_mask = (w.index >= SOK_BAS) & (w.index <= SOK_BIT)
        e_naif = (w["naif"] - gercek) ** 2

        for m in mevcut_m:
            valid = w[m].notna() & gercek.notna()
            if valid.sum() < 12:
                continue
            em = (w.loc[valid, m] - gercek.loc[valid]) ** 2
            en = e_naif.loc[valid]
            sok_v = sok_mask & valid

            satirlar.append(dict(
                hammadde=hid,
                ufuk_ay=h,
                model=m,
                n=int(valid.sum()),
                RMSE=round(float(np.sqrt(em.mean())), 4),
                naife_oran=round(float(np.sqrt(em.mean() / en.mean())), 3),
                sok_naife_oran=round(float(np.sqrt(em.loc[sok_v].mean() / en.loc[sok_v].mean())), 3) if sok_v.sum() else np.nan,
                DM_p=round(dm_p_bartlett((en - em).values, h), 3),
                yon_dogrulugu=round(float((np.sign(w.loc[valid, m]) == np.sign(gercek.loc[valid])).mean()), 3),
                gercek_artis_orani=round(float((gercek.loc[valid] > 0).mean()), 3),
            ))

    df_ozet = pd.DataFrame(satirlar)
    df_ozet["DM_p_grup_duzeltilmis"] = df_ozet.groupby(["hammadde", "ufuk_ay"])["DM_p"].transform(
        lambda p: (p * len(p)).clip(upper=1.0).round(3)
    )
    return df_ozet.sort_values(["ufuk_ay", "hammadde", "naife_oran"])


def main():
    if not DB.exists():
        raise FileNotFoundError("Önce 03_envanter.py'yi çalıştır.")
    dis_tum = pd.read_csv(DIS_CSV, parse_dates=["tarih"], index_col="tarih") if DIS_CSV.exists() else pd.DataFrame()

    with sqlite3.connect(DB) as con:
        fiy = pd.read_sql("SELECT tarih, hammadde_id, fiyat_usd_ton FROM fiyat", con, parse_dates=["tarih"])
        yeni_tahminler = []
        for hid, g in fiy.groupby("hammadde_id"):
            s = g.set_index("tarih")["fiyat_usd_ton"].sort_index()
            for h in HORIZONLAR:
                print(f"Hibrit model eğitiliyor: {hid:10s} | Ufuk: {h} ay ...")
                t_df = walk_forward_hibrit(s, dis_tum, hid, h)
                yeni_tahminler.append(t_df)

        yeni_df = pd.concat(yeni_tahminler, ignore_index=True)
        con.execute("DELETE FROM tahmin WHERE model IN ('mr_maliyet', 'ridge_dis', 'hibrit_ensemble')")
        con.execute("DELETE FROM tahmin WHERE ufuk = 6 AND model = 'naif'")
        guncel_ek = yeni_df[
            yeni_df["model"].isin(["mr_maliyet", "ridge_dis", "hibrit_ensemble"])
            | ((yeni_df["ufuk"] == 6) & (yeni_df["model"] == "naif"))
        ]
        guncel_ek.to_sql("tahmin", con, if_exists="append", index=False)
        ozet_tablo = performans_ozeti(con)

    ozet_tablo.to_csv(CIKTI_CSV, index=False)
    pd.set_option("display.width", 220)
    print("\n=== GÜNCELLENMİŞ HİBRİT + RIDGE_DIS TAHMİN ÖZETİ ===")
    print(ozet_tablo.to_string(index=False))
    print(f"\nKaydedildi: {CIKTI_CSV}")


if __name__ == "__main__":
    main()