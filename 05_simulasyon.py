"""
Adım 5 - Politika simülasyonu ve backtest
-----------------------------------------
Soru: Fiyat tahmini, sipariş kararını gerçekten iyileştiriyor mu?

Aynı geçmiş fiyat yolu (2015-2024) ve aynı talep senaryoları üzerinde
şu politikaları çalıştırır:

  1) sabit_aralik : 3 ayda bir "order-up-to" siparişi (naif)
  2) rop_eoq      : klasik (s,Q) politikası  <-- ana kıyas noktası
  3) ridge / gbm  : rop_eoq + tahmine göre zamanlama/miktar ayarı
  4) mukemmel     : rop_eoq + GERÇEK gelecek fiyatı bilen (üst sınır, referans)

Adil değerlendirme için:
  - theta (tahmin eşiği) SADECE 2015-2019'da seçilir (kalibrasyon),
    sonuçlar 2020-2024'te raporlanır (test; 2022 şoku burada).
  - 200 talep senaryosu aynı sırayla her politikaya uygulanır (eşleştirilmiş
    karşılaştırma), farklar için %95 güven aralığı verilir.
  - Net maliyet = nakit maliyetler - (dönem sonu stok değeri - dönem başı stok değeri)

Duyarlılık: sipariş maliyeti S, 03_envanter.py'deki değerin 1x, 5x ve 10x'i için
ayrı ayrı çalıştırılır. S küçükken EOQ yaklaşık 1 aylık talep kadar çıkar, yani zaten
neredeyse her ay sipariş verilir ve zamanlamayla kazanılacak yer kalmaz.

Zaman çizelgesi (aylık):
  ay başı: gelen sipariş stoğa girer -> politika sipariş verir (fiyat o anki fiyat,
  teslim L ay sonra) -> ay içinde talep karşılanır; karşılanamayan kısım
  acil alımla (spot fiyat + %25 prim) karşılanır -> ay sonu stok tutma maliyeti.

Gerekli paketler: pandas, numpy
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path("data/envanter.db")
CIKTI = Path("data/simulasyon_sonuc.csv")

SIM_BAS = "2015-01-01"
KALIB_SON = "2019-12-01"      # kalibrasyon: SIM_BAS .. KALIB_SON
TEST_BAS = "2020-01-01"       # test: TEST_BAS .. veri sonu
N_SENARYO = 200
SEED = 7

Z = 1.65                      # %95 hizmet düzeyi
R = 1                         # gözden geçirme periyodu (ay)
ACIL_PRIM = 0.25              # acil alım primi
K_NAIF = 3                    # naif politika: kaç ayda bir sipariş
BETA = 0.5                    # tahmine göre miktar ayarı (+/- %50)
THETA_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, np.inf]   # inf = tahmini kullanma
TAHMIN_MODELLERI = ["ridge", "gbm"]
# Sipariş maliyeti (S) en belirsiz varsayımımız. Sonuç buna çok duyarlı, bu yüzden
# 3 farklı değerle çalıştırıp hepsini raporluyoruz (1 = 03_envanter.py'deki değerler).
S_CARPANLARI = [1, 5, 10]


# ---------------------------------------------------------------- veri ----
def veri_yukle():
    with sqlite3.connect(DB) as con:
        ham = pd.read_sql("SELECT * FROM hammadde", con).set_index("id")
        fiyat = pd.read_sql("SELECT * FROM fiyat", con, parse_dates=["tarih"]) \
            .pivot(index="tarih", columns="hammadde_id", values="fiyat_usd_ton")
        tuk = pd.read_sql("SELECT * FROM tuketim", con, parse_dates=["tarih"]) \
            .pivot(index="tarih", columns="hammadde_id", values="miktar_ton")
        tah = pd.read_sql(
            "SELECT tarih, hammadde_id, model, tahmin_log_degisim FROM tahmin WHERE ufuk=3",
            con, parse_dates=["tarih"])
    tuk = tuk.loc[tuk.index >= SIM_BAS]
    fiyat_sim = fiyat.loc[tuk.index]
    return ham, fiyat, fiyat_sim, tuk, tah


def talep_senaryolari(tuk: pd.Series, rng):
    """Senaryo 0 = veri tabanındaki talep; diğerleri aynı dağılımdan üretilir."""
    ay_ort = tuk.groupby(tuk.index.month).mean()
    beklenen = tuk.index.month.map(ay_ort).values
    cv = float(np.std(tuk.values / beklenen))
    T = len(tuk)
    M = np.maximum(0.0, beklenen[:, None] * rng.normal(1, cv, size=(T, N_SENARYO)))
    M[:, 0] = tuk.values
    return M


# ------------------------------------------------------------ simülasyon ---
def simule(politika, P, talep, f3, par, theta=np.inf):
    """P: (T,) fiyat | talep: (T,N) | f3: (T,) tahmin edilen 3 aylık log değişim."""
    T, N = talep.shape
    d, sig, L = par["d"], par["sigma"], par["L"]
    S_sip, h = par["S"], par["h"]
    D_yil = 12 * d
    rop = d * (L + R) + Z * sig * np.sqrt(L + R)

    stok = np.full(N, rop)
    kuyruk = [np.zeros(N) for _ in range(L)]      # yoldaki siparişler
    V0 = stok * P[0]

    nakit = np.zeros((T, N)); deger = np.zeros((T, N))
    kisit = np.zeros((T, N)); siparis = np.zeros((T, N)); stok_son = np.zeros((T, N))

    for t in range(T):
        stok = stok + kuyruk.pop(0)
        yolda = sum(kuyruk) if kuyruk else 0.0
        ip = stok + yolda                           # envanter pozisyonu

        if politika == "sabit_aralik":
            if t % K_NAIF == 0:
                hedef = (K_NAIF + L) * d + Z * sig * np.sqrt(K_NAIF + L)
                q = np.maximum(0.0, hedef - ip)
            else:
                q = np.zeros(N)
        else:                                       # (s,Q) ve türevleri
            Pbar = P[max(0, t - 11): t + 1].mean()
            Q = np.sqrt(2 * D_yil * S_sip / (h * Pbar))
            tetik, Qm = rop, Q
            if politika != "rop_eoq" and np.isfinite(theta):
                s = f3[t]
                if s > theta:        # fiyat artacak: erken ve fazla al
                    tetik, Qm = rop + d, Q * (1 + BETA)
                elif s < -theta:     # fiyat düşecek: geç ve az al
                    tetik, Qm = rop - d, Q * (1 - BETA)
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
    """[a, b) ay penceresi için senaryo başına net maliyet (N,)."""
    v_bas = sim["V0"] if a == 0 else sim["deger"][a - 1]
    return sim["nakit"][a:b].sum(axis=0) - (sim["deger"][b - 1] - v_bas)


# ------------------------------------------------------------------ ana ---
def degerlendir(veri, carpan, pencereler, idx):
    T = len(idx)

    def par_of(v):
        p = dict(v["par"])
        p["S"] *= carpan
        return p

    kaynaklar = TAHMIN_MODELLERI + ["mukemmel"]
    secilen = {}
    print(f"\n########## Sipariş maliyeti çarpanı: {carpan}x ##########")
    print("--- Kalibrasyon (2015-2019): theta seçimi ---")
    for m in kaynaklar:
        skor = {}
        for th in THETA_GRID:
            yuzde = []
            for hid, v in veri.items():
                a, b = pencereler["kalibrasyon"]
                taban = net_maliyet(simule("rop_eoq", v["P"], v["talep"], np.zeros(T), par_of(v)), a, b)
                pol = net_maliyet(simule(m, v["P"], v["talep"], v["f"][m], par_of(v), th), a, b)
                yuzde.append(100 * (pol.mean() - taban.mean()) / taban.mean())
            skor[th] = np.mean(yuzde)
        en_iyi = min(skor, key=lambda th: (skor[th], th == 0))
        if skor[en_iyi] > skor[np.inf] - 1e-9:
            en_iyi = np.inf                       # hiçbir theta işe yaramadı: tahmini kullanma
        secilen[m] = en_iyi
        print(f"{m:9s}: theta={en_iyi}  (kalibrasyonda ortalama fark {skor[en_iyi]:+.2f}%)")

    politikalar = [("sabit_aralik", "sabit_aralik", None), ("rop_eoq", "rop_eoq", None)] + \
                  [(m, m, secilen[m]) for m in kaynaklar]
    a, b = pencereler["test"]
    satirlar, kayit_sip, kayit_stok = [], [], []
    for hid, v in veri.items():
        sonuc = {}
        for ad, pol, th in politikalar:
            f = v["f"].get(pol, np.zeros(T))
            sonuc[ad] = simule(pol, v["P"], v["talep"], f, par_of(v), np.inf if th is None else th)
        taban = net_maliyet(sonuc["rop_eoq"], a, b)
        talep_toplam = v["talep"][a:b].sum(axis=0)
        for ad, sim in sonuc.items():
            c = net_maliyet(sim, a, b)
            fark = 100 * (c - taban) / taban
            se = fark.std(ddof=1) / np.sqrt(N_SENARYO)
            satirlar.append(dict(
                S_carpani=carpan, hammadde=hid, politika=ad,
                net_maliyet_usd=round(c.mean()),
                rop_eoq_e_gore_pct=round(fark.mean(), 2),
                ci95_alt=round(fark.mean() - 1.96 * se, 2),
                ci95_ust=round(fark.mean() + 1.96 * se, 2),
                hizmet_duzeyi=round(1 - sim["eksik"][a:b].sum(axis=0).mean() / talep_toplam.mean(), 4),
                yillik_siparis=round((sim["siparis"][a:b, :] > 0).sum(axis=0).mean() / ((b - a) / 12), 1)))
            for t in range(T):                     # senaryo 0 -> dashboard için
                tarih = idx[t].strftime("%Y-%m-%d")
                kayit_stok.append((tarih, hid, ad, float(sim["stok"][t, 0])))
                if sim["siparis"][t, 0] > 0:
                    kayit_sip.append((tarih, hid, ad, float(sim["siparis"][t, 0]), float(v["P"][t])))
    df = pd.DataFrame(satirlar)
    print(f"--- Test dönemi ({idx[a].date()} - {idx[b-1].date()}), {N_SENARYO} senaryo ---")
    print("(rop_eoq_e_gore_pct < 0: politika rop_eoq'dan UCUZ; güven aralığı 0'ı içeriyorsa fark anlamlı değil)")
    print(df.drop(columns="S_carpani").to_string(index=False))
    return df, kayit_stok, kayit_sip


def main():
    if not DB.exists():
        raise FileNotFoundError("Önce 03 ve 04 numaralı scriptleri çalıştır.")
    rng = np.random.default_rng(SEED)
    ham, fiyat_tum, fiyat, tuk, tah = veri_yukle()
    idx = tuk.index
    T = len(idx)
    k_son = int((idx <= KALIB_SON).sum())
    pencereler = {"kalibrasyon": (0, k_son), "test": (k_son, T)}

    veri = {}
    for hid in ham.index:
        p = ham.loc[hid]
        par = dict(d=tuk[hid].mean(), sigma=tuk[hid].std(), L=int(p.tedarik_suresi_ay),
                   S=float(p.siparis_maliyeti), h=float(p.tutma_orani))
        gelecek = np.log(fiyat_tum[hid].shift(-3) / fiyat_tum[hid]).reindex(idx).fillna(0).values
        f = {"mukemmel": gelecek}
        for m in TAHMIN_MODELLERI:
            s = tah[(tah.hammadde_id == hid) & (tah.model == m)].set_index("tarih")["tahmin_log_degisim"]
            f[m] = s.reindex(idx).fillna(0).values
        veri[hid] = dict(par=par, P=fiyat[hid].values, talep=talep_senaryolari(tuk[hid], rng), f=f)

    pd.set_option("display.width", 220)
    tum, dash = [], None
    for carpan in S_CARPANLARI:
        df, ks, kp = degerlendir(veri, carpan, pencereler, idx)
        tum.append(df)
        if dash is None:                           # dashboard için ilk (temel) çarpan
            dash = (ks, kp)

    pd.concat(tum).to_csv(CIKTI, index=False)
    with sqlite3.connect(DB) as con:
        pd.DataFrame(dash[0], columns=["tarih", "hammadde_id", "politika", "stok_ton"]) \
            .to_sql("sim_stok", con, if_exists="replace", index=False)
        pd.DataFrame(dash[1], columns=["tarih", "hammadde_id", "politika", "miktar_ton", "fiyat_usd_ton"]) \
            .to_sql("sim_siparis", con, if_exists="replace", index=False)
    print(f"\nKaydedildi: {CIKTI} ve {DB} -> sim_stok, sim_siparis (dashboard için, {S_CARPANLARI[0]}x)")


if __name__ == "__main__":
    main()
