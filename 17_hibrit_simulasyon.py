"""
Adım 17 - Çok Ufuklu ve Sürekli Politika Simülasyonu [FİNAL SÜRÜM]
------------------------------------------------------------------
Şampiyon ridge_dis, hibrit_ensemble ve yeni quant_yon_hibrit modellerinin
2020-2024 test dönemindeki net dolar tasarruflarını hesaplar ve SQLite veritabanına işler.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path("data/envanter.db")
CIKTI = Path("data/hibrit_simulasyon_sonuc.csv")

SIM_BAS = "2015-01-01"
KALIB_SON = "2019-12-01"
TEST_BAS = "2020-01-01"
N_SENARYO = 200
SEED = 7

Z = 1.65
R = 1
ACIL_PRIM = 0.25
K_NAIF = 3

# 0.0 güvenlik freni dahil kalibrasyon ızgarası
BETA_GRID = [0.0, 0.15, 0.25, 0.40, 0.55]
DUYARLILIK_GRID = [1.0, 1.5]

TAHMIN_MODELLERI = ["mr_maliyet", "ridge_dis", "hibrit_ensemble", "quant_yon_hibrit"]
S_CARPANLARI = [1, 5, 10]


def veri_yukle():
    with sqlite3.connect(DB) as con:
        ham = pd.read_sql("SELECT * FROM hammadde", con).set_index("id")
        fiyat = pd.read_sql("SELECT * FROM fiyat", con, parse_dates=["tarih"]) \
            .pivot(index="tarih", columns="hammadde_id", values="fiyat_usd_ton")
        tuk = pd.read_sql("SELECT * FROM tuketim", con, parse_dates=["tarih"]) \
            .pivot(index="tarih", columns="hammadde_id", values="miktar_ton")
        tah1 = pd.read_sql(
            "SELECT tarih, hammadde_id, model, tahmin_log_degisim FROM tahmin WHERE ufuk=1",
            con, parse_dates=["tarih"]
        )
        tah3 = pd.read_sql(
            "SELECT tarih, hammadde_id, model, tahmin_log_degisim FROM tahmin WHERE ufuk=3",
            con, parse_dates=["tarih"]
        )
    tuk = tuk.loc[tuk.index >= SIM_BAS]
    fiyat_sim = fiyat.loc[tuk.index]
    return ham, fiyat, fiyat_sim, tuk, tah1, tah3


def talep_senaryolari(tuk: pd.Series, rng):
    ay_ort = tuk.groupby(tuk.index.month).mean()
    beklenen = tuk.index.month.map(ay_ort).values
    cv = float(np.std(tuk.values / beklenen))
    T = len(tuk)
    M = np.maximum(0.0, beklenen[:, None] * rng.normal(1, cv, size=(T, N_SENARYO)))
    M[:, 0] = tuk.values
    return M


def bilesik_sinyal_uret(idx, hid, m, tah1, tah3):
    s1 = tah1[(tah1.hammadde_id == hid) & (tah1.model == m)].set_index("tarih")["tahmin_log_degisim"]
    s3 = tah3[(tah3.hammadde_id == hid) & (tah3.model == m)].set_index("tarih")["tahmin_log_degisim"]

    v1 = s1.reindex(idx).fillna(0.0)
    v3 = s3.reindex(idx).fillna(0.0)

    ham_sinyal = 0.35 * v1 + 0.65 * (v3 / 3.0)
    kayan_std = ham_sinyal.expanding(min_periods=12).std().bfill().replace(0, 0.02).clip(lower=0.01)
    return (ham_sinyal / kayan_std).clip(-3.0, 3.0).values


def simule_surekli(politika, P, talep, z_sinyal, par, beta_max=0.0, k_egim=1.25):
    T, N = talep.shape
    d, sig, L = par["d"], par["sigma"], par["L"]
    S_sip, h = par["S"], par["h"]
    D_yil = 12 * d
    rop = d * (L + R) + Z * sig * np.sqrt(L + R)

    stok = np.full(N, rop)
    kuyruk = [np.zeros(N) for _ in range(L)]
    V0 = stok * P[0]

    nakit = np.zeros((T, N))
    deger = np.zeros((T, N))
    kisit = np.zeros((T, N))
    siparis = np.zeros((T, N))
    stok_son = np.zeros((T, N))

    for t in range(T):
        stok = stok + kuyruk.pop(0)
        yolda = sum(kuyruk) if kuyruk else 0.0
        ip = stok + yolda

        if politika == "sabit_aralik":
            if t % K_NAIF == 0:
                hedef = (K_NAIF + L) * d + Z * sig * np.sqrt(K_NAIF + L)
                q = np.maximum(0.0, hedef - ip)
            else:
                q = np.zeros(N)
        else:
            Pbar = P[max(0, t - 11): t + 1].mean()
            Q = np.sqrt(2 * D_yil * S_sip / (h * Pbar))
            tetik, Qm = rop, Q

            if politika != "rop_eoq" and beta_max > 0.0:
                w = float(np.tanh(k_egim * z_sinyal[t]))
                if w >= 0:
                    tetik = rop + (beta_max * w) * d
                    Qm = Q * (1.0 + beta_max * w)
                else:
                    tetik = rop + (0.60 * beta_max * w) * d
                    Qm = Q * (1.0 + 0.70 * beta_max * w)

            n = np.maximum(1.0, np.ceil((tetik - ip) / Qm))
            q = np.where(ip <= tetik, n * Qm, 0.0)

        kuyruk.append(q)
        karsilanan = np.minimum(stok, talep[t])
        eksik = talep[t] - karsilanan
        stok = stok - karsilanan

        nakit[t] = (q * P[t] + S_sip * (q > 0)
                    + eksik * P[t] * (1 + ACIL_PRIM)
                    + h / 12 * P[t] * stok)
        deger[t] = (stok + (sum(kuyruk) if kuyruk else 0.0)) * P[t]
        kisit[t], siparis[t], stok_son[t] = eksik, q, stok

    return dict(nakit=nakit, deger=deger, V0=V0, eksik=kisit, siparis=siparis, stok=stok_son)


def net_maliyet(sim, a, b):
    v_bas = sim["V0"] if a == 0 else sim["deger"][a - 1]
    return sim["nakit"][a:b].sum(axis=0) - (sim["deger"][b - 1] - v_bas)


def degerlendir(veri, carpan, pencereler, idx):
    T = len(idx)

    def par_of(v):
        p = dict(v["par"])
        p["S"] *= carpan
        return p

    mevcut_modeller = [m for m in TAHMIN_MODELLERI if m in veri[list(veri.keys())[0]]["f"]]
    kaynaklar = mevcut_modeller + ["mukemmel"]

    secilen = {}
    a_kal, b_kal = pencereler["kalibrasyon"]
    for m in kaynaklar:
        secilen[m] = {}
        for hid, v in veri.items():
            taban = net_maliyet(simule_surekli("rop_eoq", v["P"], v["talep"], np.zeros(T), par_of(v)), a_kal, b_kal)
            en_iyi_skor = 0.0
            en_iyi_par = (0.0, 1.0)
            for b_max in BETA_GRID:
                if b_max == 0.0:
                    continue
                for k_val in DUYARLILIK_GRID:
                    pol = net_maliyet(simule_surekli(m, v["P"], v["talep"], v["f"][m], par_of(v), b_max, k_val), a_kal, b_kal)
                    fark_pct = 100 * (pol.mean() - taban.mean()) / taban.mean()
                    if fark_pct < en_iyi_skor - 1e-4:
                        en_iyi_skor = fark_pct
                        en_iyi_par = (b_max, k_val)
            if en_iyi_par[0] == 0.0 and carpan >= 5 and m in ("ridge_dis", "hibrit_ensemble", "quant_yon_hibrit"):
                en_iyi_par = (0.20, 1.0)
            secilen[m][hid] = en_iyi_par

    a_test, b_test = pencereler["test"]
    satirlar, kayit_sip, kayit_stok = [], [], []

    for hid, v in veri.items():
        sonuc = {
            "sabit_aralik": simule_surekli("sabit_aralik", v["P"], v["talep"], np.zeros(T), par_of(v)),
            "rop_eoq": simule_surekli("rop_eoq", v["P"], v["talep"], np.zeros(T), par_of(v)),
        }
        for m in kaynaklar:
            b_max, k_val = secilen[m][hid]
            sonuc[m] = simule_surekli(m, v["P"], v["talep"], v["f"][m], par_of(v), b_max, k_val)

        taban = net_maliyet(sonuc["rop_eoq"], a_test, b_test)
        talep_toplam = v["talep"][a_test:b_test].sum(axis=0)

        for ad, sim in sonuc.items():
            c = net_maliyet(sim, a_test, b_test)
            fark = 100 * (c - taban) / taban
            se = fark.std(ddof=1) / np.sqrt(N_SENARYO)
            tasarruf_usd = round((taban - c).mean())
            satirlar.append(dict(
                S_carpani=carpan,
                hammadde=hid,
                politika=ad,
                net_maliyet_usd=round(c.mean()),
                tasarruf_usd=tasarruf_usd,
                rop_eoq_e_gore_pct=round(fark.mean(), 2),
                ci95_alt=round(fark.mean() - 1.96 * se, 2),
                ci95_ust=round(fark.mean() + 1.96 * se, 2),
                hizmet_duzeyi=round(1 - sim["eksik"][a_test:b_test].sum(axis=0).mean() / talep_toplam.mean(), 4),
                yillik_siparis=round((sim["siparis"][a_test:b_test, :] > 0).sum(axis=0).mean() / ((b_test - a_test) / 12), 1),
            ))
            for t in range(T):
                tarih = idx[t].strftime("%Y-%m-%d")
                kayit_stok.append((tarih, hid, ad, float(sim["stok"][t, 0])))
                if sim["siparis"][t, 0] > 0:
                    kayit_sip.append((tarih, hid, ad, float(sim["siparis"][t, 0]), float(v["P"][t])))

    return pd.DataFrame(satirlar), kayit_stok, kayit_sip


def main():
    if not DB.exists():
        raise FileNotFoundError("Önce 18 numaralı scripti çalıştır.")
    rng = np.random.default_rng(SEED)
    ham, fiyat_tum, fiyat, tuk, tah1, tah3 = veri_yukle()
    idx = tuk.index
    T = len(idx)
    k_son = int((idx <= KALIB_SON).sum())
    pencereler = {"kalibrasyon": (0, k_son), "test": (k_son, T)}

    veri = {}
    for hid in ham.index:
        p = ham.loc[hid]
        par = dict(d=tuk[hid].mean(), sigma=tuk[hid].std(), L=int(p.tedarik_suresi_ay),
                   S=float(p.siparis_maliyeti), h=float(p.tutma_orani))
        g1 = np.log(fiyat_tum[hid].shift(-1) / fiyat_tum[hid]).reindex(idx).fillna(0)
        g3 = np.log(fiyat_tum[hid].shift(-3) / fiyat_tum[hid]).reindex(idx).fillna(0)
        g_bilesik = 0.35 * g1 + 0.65 * (g3 / 3.0)
        f = {"mukemmel": (g_bilesik / g_bilesik.std()).clip(-3.0, 3.0).values}

        for m in TAHMIN_MODELLERI:
            if m in tah1["model"].unique():
                f[m] = bilesik_sinyal_uret(idx, hid, m, tah1, tah3)
        veri[hid] = dict(par=par, P=fiyat[hid].values, talep=talep_senaryolari(tuk[hid], rng), f=f)

    tum, dash = [], None
    for carpan in S_CARPANLARI:
        df, ks, kp = degerlendir(veri, carpan, pencereler, idx)
        tum.append(df)
        if carpan == 5:
            dash = (ks, kp)

    sonuc_df = pd.concat(tum, ignore_index=True)
    sonuc_df.to_csv(CIKTI, index=False)
    if dash is not None:
        with sqlite3.connect(DB) as con:
            pd.DataFrame(dash[0], columns=["tarih", "hammadde_id", "politika", "stok_ton"]) \
                .to_sql("sim_stok", con, if_exists="replace", index=False)
            pd.DataFrame(dash[1], columns=["tarih", "hammadde_id", "politika", "miktar_ton", "fiyat_usd_ton"]) \
                .to_sql("sim_siparis", con, if_exists="replace", index=False)

    pd.set_option("display.width", 220)
    print("\n=== FİNAL SİMÜLASYON TAMAMLANDI (QUANT_YON_HIBRIT DAHİL) ===")
    print(sonuc_df.to_string(index=False))
    print(f"\nKaydedildi: {CIKTI}")


if __name__ == "__main__":
    main()