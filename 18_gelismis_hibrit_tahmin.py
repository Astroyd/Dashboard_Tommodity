"""
Adım 18 - Çoklu Frekanslı (MIDAS) ve Olasılıksal Yön Odaklı Tahmin Motoru
-------------------------------------------------------------------------
data/ klasöründeki TÜM veri kaynaklarını (fiyatlar.csv, dis_degiskenler.csv,
cot.csv, gunluk.csv, gunluk_etf.csv) veri sızıntısı olmadan (walk-forward)
birleştirir:

  1) CFTC COT 'Smart Money' (Üretici vs Fon) pozisyon sıkışması sinyali
  2) Günlük veriden hesaplanan Ay İçi Gerçekleşen Oynaklık (Realized Vol),
     Dolar Endeksi (DXY), VIX ve S&P 500 makro sinyalleri
  3) Sızıntısız (Out-of-Fold) Tarihsel Hasat/Ay Medyan Anomalisi
  4) Aykırı şoklara dayanıklı Huber Regresyon + L2 Lojistik Yön Sınıflandırıcı
     ('quant_yon_hibrit' ve 'midas_ensemble')

Çıktılar:
  - data/envanter.db -> 'tahmin' tablosuna yeni modelleri ekler
  - data/hibrit_tahmin_sonuc.csv -> Dashboard ve analiz için güncel özet tablo
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LogisticRegression, RidgeCV
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

DB = Path("data/envanter.db")
DIS_CSV = Path("data/dis_degiskenler.csv")
COT_CSV = Path("data/cot.csv")
GUNLUK_CSV = Path("data/gunluk.csv")
ETF_CSV = Path("data/gunluk_etf.csv")
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


# ---------------------------------------------------------------------------
# 1. ÇOKLU FREKANS (GÜNLÜK VE HAFTALIK -> AYLIK) SIZINTISIZ VERİ BİRLEŞTİRME
# ---------------------------------------------------------------------------
def gunluk_ve_makro_ozet_uret(aylik_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    gunluk.csv ve gunluk_etf.csv dosyalarından her ayın son iş günü itibarıyla
    DXY, VIX, SP500 ve ay içi gerçekleşen oynaklık (Realized Volatility) çıkarır.
    Pink Sheet tarihleri YYYY-MM-01 olduğu için eşleştirme o ayın son gününe göre yapılır.
    """
    out = pd.DataFrame(index=aylik_index)
    if not GUNLUK_CSV.exists():
        return out

    g = pd.read_csv(GUNLUK_CSV, parse_dates=[0], index_col=0).sort_index()
    g.index = pd.to_datetime(g.index).tz_localize(None)

    if ETF_CSV.exists():
        etf = pd.read_csv(ETF_CSV, parse_dates=[0], index_col=0).sort_index()
        etf.index = pd.to_datetime(etf.index).tz_localize(None)
        for col in etf.columns:
            g[f"etf_{col}"] = etf[col]

    # Aylık periyot bazında grupla (YYYY-MM)
    g["ym"] = g.index.to_period("M")
    ay_sonu = g.groupby("ym").last()

    # Makro seriler (Dolar Endeksi, VIX, S&P500)
    ym_idx = aylik_index.to_period("M")

    if "dolar_endeksi" in ay_sonu.columns:
        dxy = np.log(ay_sonu["dolar_endeksi"].replace(0, np.nan)).ffill()
        out["dxy_r1"] = dxy.diff(1).reindex(ym_idx).values
        out["dxy_mom3"] = dxy.diff(3).reindex(ym_idx).values

    if "vix" in ay_sonu.columns:
        vix = ay_sonu["vix"].ffill()
        vix_z = (vix - vix.rolling(36, min_periods=12).mean()) / vix.rolling(36, min_periods=12).std()
        out["vix_z"] = vix_z.reindex(ym_idx).values

    if "sp500" in ay_sonu.columns:
        sp = np.log(ay_sonu["sp500"].replace(0, np.nan)).ffill()
        out["sp500_mom3"] = sp.diff(3).reindex(ym_idx).values

    # Her hedef emtia için ay içi gerçekleşen oynaklık (Realized Volatility) ve son 5 gün ivmesi
    for hid in ["pamuk", "aluminyum", "bugday"]:
        seri_adi = f"etf_{hid}" if f"etf_{hid}" in g.columns and g[f"etf_{hid}"].notna().sum() > 500 else hid
        if seri_adi in g.columns:
             gun_r = np.log(g[seri_adi].replace(0, np.nan)).diff()
             # Roll sıçramalarını (%15 üstü tek gün) filtrele
             gun_r = gun_r.where(gun_r.abs() < 0.15, np.nan)
             df_r = pd.DataFrame({"r": gun_r, "ym": g.index.to_period("M")})
             rv = df_r.groupby("ym")["r"].std() * np.sqrt(21)
             son5 = df_r.groupby("ym")["r"].apply(lambda x: x.dropna().iloc[-5:].sum() if len(x.dropna()) >= 5 else 0.0)
             out[f"{hid}_rv"] = rv.reindex(ym_idx).values
             out[f"{hid}_son5_ivme"] = son5.reindex(ym_idx).values

    return out.fillna(0.0)


def cot_aylik_ozet_uret(aylik_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    cot.csv dosyasından her ayın son günü itibarıyla YAYIMLANMIŞ (kullanilabilir <= ay_sonu)
    en son CFTC raporunu sızıntısız olarak çeker.
    """
    out = pd.DataFrame(index=aylik_index)
    if not COT_CSV.exists():
        return out

    cot = pd.read_csv(COT_CSV, parse_dates=["tarih", "kullanilabilir"]).sort_values("kullanilabilir")
    # Ayın son gününü referans al (Pink Sheet'teki YYYY-MM-01 o ayın tamamını temsil eder)
    hedef_df = pd.DataFrame({
        "ay_basi": aylik_index,
        "ay_sonu": aylik_index + pd.offsets.MonthEnd(1),
    }).sort_values("ay_sonu")

    m = pd.merge_asof(
        hedef_df, cot,
        left_on="ay_sonu", right_on="kullanilabilir",
        direction="backward", allow_exact_matches=True,
    ).set_index("ay_basi")

    # Alüminyum CFTC'de doğrudan olmadığı için baz metal vekili olarak Bakır COT verisi kullanılır
    eslesme = {"pamuk": "pamuk", "bugday": "bugday", "aluminyum": "bakir"}
    for hid, cot_ad in eslesme.items():
        mm_col = f"{cot_ad}_mm_net_pct"
        pr_col = f"{cot_ad}_prod_net_pct"
        if mm_col in m.columns and pr_col in m.columns:
            mm = m[mm_col].ffill()
            pr = m[pr_col].ffill()
            # Smart Money (Üretici) vs Managed Money (Fon) Spread Z-Skoru
            spread = pr - mm
            sp_z = (spread - spread.rolling(36, min_periods=12).mean()) / spread.rolling(36, min_periods=12).std().replace(0, np.nan)
            out[f"{hid}_cot_smart_z"] = sp_z.clip(-3.0, 3.0).fillna(0.0)
            out[f"{hid}_cot_mm_d3"] = mm.diff(3).fillna(0.0)

    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# 2. KANTİTATİF FORMÜLLER VE ÖZELLİK MÜHENDİSLİĞİ
# ---------------------------------------------------------------------------
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


def sızıntısız_mevsimsel_anomali(lp: pd.Series, h: int) -> pd.Series:
    """
    Her t adımında SADECE t-h öncesindeki geçmiş yıllara bakarak, ilgili takvim ayının
    tarihsel medyan h-aylık getirisini hesaplar (Gelecek bilgisi sızmaz).
    """
    y_gecmis = lp - lp.shift(h)  # t anında bilinen tamamlanmış h-aylık değişim
    anomali = pd.Series(0.0, index=lp.index)
    aylar = lp.index.month
    for i in range(48, len(lp)):
        hedef_ay = (lp.index[i].month + h - 1) % 12 + 1
        gecmis_mask = (aylar[:i] == hedef_ay) & y_gecmis.iloc[:i].notna()
        if gecmis_mask.sum() >= 3:
            anomali.iloc[i] = float(y_gecmis.iloc[:i][gecmis_mask].median())
    sd = anomali.expanding(min_periods=24).std().replace(0, 0.01)
    return (anomali / sd).clip(-2.5, 2.5).fillna(0.0)


def finansal_sinyaller(
    fiyat: pd.Series, dis: pd.DataFrame, makro_gun: pd.DataFrame, cot_df: pd.DataFrame, hid: str, h: int
) -> pd.DataFrame:
    lp = np.log(fiyat)
    r1 = lp.diff()
    vol6 = r1.rolling(6, min_periods=3).std().clip(lower=0.01)
    vol12 = r1.rolling(12, min_periods=6).std().clip(lower=0.01)

    s = pd.DataFrame(index=fiyat.index)
    ma60 = lp.rolling(60, min_periods=24).mean()
    sd60 = lp.rolling(60, min_periods=24).std().clip(lower=0.02)
    ma120 = lp.rolling(120, min_periods=36).mean()
    sd120 = lp.rolling(120, min_periods=36).std().clip(lower=0.02)

    # 1) Ortalamaya Dönüş ve Maliyet Marjı Formülleri
    s["mr60_z"] = -((lp - ma60) / sd60).clip(-3.0, 3.0)
    s["mr120_z"] = -((lp - ma120) / sd120).clip(-3.0, 3.0)
    maliyet_z = maliyet_endeksi_uret(lp, dis, hid)
    fiyat_z = ((lp - ma60) / sd60).fillna(0.0)
    s["cost_spread_z"] = (maliyet_z - fiyat_z).clip(-3.0, 3.0)

    # 2) Trend ve Momentum Formülleri
    ma3 = lp.rolling(3).mean()
    ma12 = lp.rolling(12).mean()
    s["vol_macd"] = ((ma3 - ma12) / (vol12 * np.sqrt(6))).clip(-3.0, 3.0)
    s["mom1_sharpe"] = (r1 / vol6).clip(-3.0, 3.0)
    s["mom3_sharpe"] = ((lp - lp.shift(3)) / (vol6 * np.sqrt(3))).clip(-3.0, 3.0)
    s["dis_ivme"] = maliyet_z.diff(3).clip(-3.0, 3.0)

    # 3) Sızıntısız Hasat / Takvim Mevsimselliği
    s["mevsim_anomali"] = sızıntısız_mevsimsel_anomali(lp, h)
    ay = fiyat.index.month
    s["ay_sin"] = np.sin(2 * np.pi * ay / 12)
    s["ay_cos"] = np.cos(2 * np.pi * ay / 12)

    # 4) Dış Değişken Kısa Vadeli Yayılım (Spillover)
    if not dis.empty:
        kolonlar = [c for c in ALAKALI_DIS.get(hid, []) if c in dis.columns]
        for c in kolonlar:
            x = np.log(dis[c].reindex(fiyat.index)).ffill().bfill()
            s[f"{c}_r1"] = x.diff()
            s[f"{c}_mom3"] = x - x.shift(3)

    # 5) CFTC COT ve Günlük Makro Sinyaller (2012 sonrası aktifleşir)
    if f"{hid}_cot_smart_z" in cot_df.columns:
        s["cot_smart_z"] = cot_df[f"{hid}_cot_smart_z"].reindex(fiyat.index).fillna(0.0)
        s["cot_mm_d3"] = cot_df[f"{hid}_cot_mm_d3"].reindex(fiyat.index).fillna(0.0)

    for m_col in ["dxy_r1", "dxy_mom3", "vix_z", "sp500_mom3"]:
        if m_col in makro_gun.columns:
            s[m_col] = makro_gun[m_col].reindex(fiyat.index).fillna(0.0)

    if f"{hid}_son5_ivme" in makro_gun.columns:
        s["ay_sonu_ivme"] = makro_gun[f"{hid}_son5_ivme"].reindex(fiyat.index).fillna(0.0)

    return s


def tekil_formul_ols(x: np.ndarray, y: np.ndarray, x_now: float, shrinkage: float = 0.65) -> float:
    if np.std(x) < 1e-6:
        return 0.0
    X_mat = np.column_stack([np.ones_like(x), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        ham_tahmin = float(beta[0] + beta[1] * x_now)
        return shrinkage * ham_tahmin
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 3. WALK-FORWARD EĞİTİM VE OLASILIKSAL YÖN MİMARİSİ
# ---------------------------------------------------------------------------
def walk_forward_gelismis(
    fiyat: pd.Series, dis: pd.DataFrame, makro_gun: pd.DataFrame, cot_df: pd.DataFrame, hid: str, h: int
) -> pd.DataFrame:
    lp = np.log(fiyat)
    S = finansal_sinyaller(fiyat, dis, makro_gun, cot_df, hid, h)
    y = lp.shift(-h) - lp

    test_idx = S.index[S.index >= TEST_BASLANGIC]
    test_idx = test_idx[: len(test_idx) - h]
    pos = {d: i for i, d in enumerate(S.index)}

    uzman_kolonlar = [
        "mr60_z", "mr120_z", "cost_spread_z", "vol_macd",
        "mom1_sharpe", "mom3_sharpe", "dis_ivme", "mevsim_anomali"
    ]
    kayit = []

    for t in test_idx:
        i = pos[t]
        tr_bitis = i - h + 1
        if tr_bitis < 48:
            continue

        Xe = S.iloc[:tr_bitis].copy()
        ye = y.iloc[:tr_bitis].copy()
        ok = Xe.notna().all(axis=1) & ye.notna()
        if ok.sum() < 40 or S.iloc[[i]].isna().any(axis=None):
            continue

        Xe_ok, ye_ok = Xe.loc[ok], ye.loc[ok]
        x_now = S.iloc[[i]]
        buzulme = {1: 0.38, 3: 0.58, 6: 0.75}.get(h, 0.60)

        # 1) Tekil Kantitatif Uzmanların OLS Tahminleri
        uzman_tahmin, uzman_hata = {}, {}
        for col in uzman_kolonlar:
            if col not in Xe_ok.columns:
                continue
            xv, yv = Xe_ok[col].values, ye_ok.values
            uzman_tahmin[col] = tekil_formul_ols(xv, yv, float(x_now[col].iloc[0]), shrinkage=buzulme)
            son_k = min(36, len(xv))
            X_mat = np.column_stack([np.ones(len(xv)), xv])
            beta, _, _, _ = np.linalg.lstsq(X_mat, yv, rcond=None)
            pred_gecmis = buzulme * (beta[0] + beta[1] * xv[-son_k:])
            uzman_hata[col] = float(np.mean((yv[-son_k:] - pred_gecmis) ** 2)) + 1e-6

        aktif_uzmanlar = [k for k in uzman_kolonlar if k in uzman_tahmin]
        inv_mse = {k: 1.0 / uzman_hata[k] for k in aktif_uzmanlar}
        w_top = sum(inv_mse.values()) if inv_mse else 1.0
        pred_ens = sum((inv_mse[k] / w_top) * uzman_tahmin[k] for k in aktif_uzmanlar) if inv_mse else 0.0

        # 2) ridge_dis (Makro Dış Değişkenli Şampiyon Ridge)
        dis_kolonlar = [c for c in S.columns if "_r1" in c or "_mom3" in c]
        ridge_cols = ["cost_spread_z", "vol_macd", "mevsim_anomali"] + dis_kolonlar
        ridge_cols = [c for c in ridge_cols if c in Xe_ok.columns and Xe_ok[c].std() > 1e-6]

        scaler_r = StandardScaler()
        Xr_sc = scaler_r.fit_transform(Xe_ok[ridge_cols])
        xr_now_sc = scaler_r.transform(x_now[ridge_cols])
        ridge_m = RidgeCV(alphas=[5.0, 25.0, 100.0, 300.0])
        ridge_m.fit(Xr_sc, ye_ok.values)
        pred_ridge_dis = float(ridge_m.predict(xr_now_sc)[0]) * buzulme

        # 3) YENİ: Olasılıksal Yön + Huber Regresyon Hibriti (quant_yon_hibrit)
        # Aykırı şoklara (2022 krizi) karşı RobustScaler + HuberRegressor + L2 Lojistik Sınıflandırıcı
        yon_cols = [c for c in S.columns if Xe_ok[c].std() > 1e-6]
        r_scaler = RobustScaler()
        Xy_sc = r_scaler.fit_transform(Xe_ok[yon_cols])
        xy_now_sc = r_scaler.transform(x_now[yon_cols])

        # Son yıllara üstel artan örneklem ağırlığı
        n_obs = len(ye_ok)
        ornek_agirlik = np.exp(np.linspace(-1.2, 0.0, n_obs))

        # (a) Büyüklük tahmini (Huber kayıp fonksiyonu uç şoklardan bozulmaz)
        try:
            huber = HuberRegressor(alpha=25.0, epsilon=1.35, max_iter=300)
            huber.fit(Xy_sc, ye_ok.values, sample_weight=ornek_agirlik)
            pred_huber = float(huber.predict(xy_now_sc)[0]) * buzulme
        except Exception:
            pred_huber = pred_ridge_dis

        # (b) Doğrudan Yön Olasılığı Tahmini P(y > 0)
        y_bin = (ye_ok.values > 0).astype(int)
        if len(np.unique(y_bin)) == 2:
            clf = LogisticRegression(C=0.15, penalty="l2", solver="lbfgs", max_iter=200)
            clf.fit(Xy_sc, y_bin, sample_weight=ornek_agirlik)
            prob_up = float(clf.predict_proba(xy_now_sc)[0, 1])
            # Olasılığı [-1, +1] aralığında yön sinyaline çevir ve medyan mutlak değişimle ölçekle
            med_buyukluk = float(np.median(np.abs(ye_ok.values)))
            pred_lojistik = (2.0 * prob_up - 1.0) * med_buyukluk * 0.85
        else:
            pred_lojistik = pred_huber

        # Huber, Lojistik Yön ve Ridge_Dis kombinasyonu
        pred_quant_yon = 0.40 * pred_lojistik + 0.35 * pred_ridge_dis + 0.25 * pred_huber

        # 4) mr_maliyet (Maliyet Düzeltilmiş Ortalamaya Dönüş)
        pred_mr = 0.50 * uzman_tahmin.get("mr120_z", 0.0) + 0.50 * uzman_tahmin.get("cost_spread_z", 0.0)

        max_degisim = 1.5 * float(ye_ok.std())
        tahminler = {
            "naif": 0.0,
            "mr_maliyet": float(np.clip(pred_mr, -max_degisim, max_degisim)),
            "ridge_dis": float(np.clip(pred_ridge_dis, -max_degisim, max_degisim)),
            "hibrit_ensemble": float(np.clip(0.45 * pred_ens + 0.55 * pred_ridge_dis, -max_degisim, max_degisim)),
            "quant_yon_hibrit": float(np.clip(pred_quant_yon, -max_degisim, max_degisim)),
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
    aday_modeller = ["ridge", "gbm", "mr_maliyet", "ridge_dis", "hibrit_ridge", "hibrit_ensemble", "quant_yon_hibrit"]
    satirlar = []

    for (hid, h), g in tum.groupby(["hammadde_id", "ufuk"]):
        w = g.pivot(index="tarih", columns="model", values="tahmin_log_degisim")
        if "naif" not in w.columns:
            continue
        mevcut_m = [m for m in aday_modeller if m in w.columns]
        w = w.dropna(subset=["naif"] + [m for m in ["quant_yon_hibrit"] if m in w.columns])
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
    return df_ozet.sort_values(["ufuk_ay", "hammadde", "yon_dogrulugu"], ascending=[True, True, False])


def main():
    if not DB.exists():
        raise FileNotFoundError("Önce 03_envanter.py'yi çalıştır.")
    dis_tum = pd.read_csv(DIS_CSV, parse_dates=["tarih"], index_col="tarih") if DIS_CSV.exists() else pd.DataFrame()

    with sqlite3.connect(DB) as con:
        fiy = pd.read_sql("SELECT tarih, hammadde_id, fiyat_usd_ton FROM fiyat", con, parse_dates=["tarih"])
        ortak_idx = pd.DatetimeIndex(sorted(fiy["tarih"].unique()))

        print("Çoklu frekanslı veriler (COT, Günlük Makro, ETF Volatilite) sızıntısız birleştiriliyor...")
        makro_gun = gunluk_ve_makro_ozet_uret(ortak_idx)
        cot_df = cot_aylik_ozet_uret(ortak_idx)

        yeni_tahminler = []
        for hid, g in fiy.groupby("hammadde_id"):
            s = g.set_index("tarih")["fiyat_usd_ton"].sort_index()
            for h in HORIZONLAR:
                print(f"Gelişmiş MIDAS + Olasılıksal Yön modeli eğitiliyor: {hid:10s} | Ufuk: {h} ay ...")
                t_df = walk_forward_gelismis(s, dis_tum, makro_gun, cot_df, hid, h)
                yeni_tahminler.append(t_df)

        yeni_df = pd.concat(yeni_tahminler, ignore_index=True)
        con.execute("DELETE FROM tahmin WHERE model IN ('mr_maliyet', 'ridge_dis', 'hibrit_ensemble', 'quant_yon_hibrit')")
        con.execute("DELETE FROM tahmin WHERE ufuk = 6 AND model = 'naif'")
        guncel_ek = yeni_df[
            yeni_df["model"].isin(["mr_maliyet", "ridge_dis", "hibrit_ensemble", "quant_yon_hibrit"])
            | ((yeni_df["ufuk"] == 6) & (yeni_df["model"] == "naif"))
        ]
        guncel_ek.to_sql("tahmin", con, if_exists="append", index=False)
        ozet_tablo = performans_ozeti(con)

    ozet_tablo.to_csv(CIKTI_CSV, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 120)
    print("\n=== GELİŞMİŞ ÇOKLU FREKANS (MIDAS + COT + LOJİSTİK YÖN) TAHMİN SONUÇLARI ===")
    print(ozet_tablo.to_string(index=False))
    print(f"\nKaydedildi: {CIKTI_CSV}")


if __name__ == "__main__":
    main()